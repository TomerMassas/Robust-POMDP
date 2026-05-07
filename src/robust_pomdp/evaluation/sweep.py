"""
Experiment sweep infrastructure.

`run_config` runs `n_trials` episodes for one configuration and returns summary
statistics. `run_sweep` runs many configs, writes summary + raw CSVs per sweep,
and appends a description line to INDEX.md.

Two planner backends are dispatched on `planner_label`:
    "robust_pomcp" — our `robust_pomcp_plan` + the discrete `bayes_update`.
    "BasicPOMCP"   — pomdp_py's POMCP via the adapter (label kept for CSV
                     schema continuity).
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from robust_pomdp.core.pomdp_types import TabularPOMDP
from robust_pomdp.evaluation.belief_update import bayes_update
from robust_pomdp.evaluation.episode import EpisodeStep, run_episode
from robust_pomdp.evaluation.perturbation import build_perturbed_tiger
from robust_pomdp.evaluation.pomdp_py_adapter import (
    make_basic_pomcp_planner,
    make_basicpomcp_belief_updater,
)
from robust_pomdp.solvers.robust_pomcp import robust_pomcp_plan
from robust_pomdp.uncertainty.uncertainty_sets import (
    TVDistance,
    UncertaintySets,
    uniform_uncertainty,
)


logger = logging.getLogger(__name__)


# Default results directory: <repo>/experiments/tiger/results/
# sweep.py lives at <repo>/src/robust_pomdp/evaluation/sweep.py, so go up 3.
DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "experiments" / "tiger" / "results"
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """One row in a sweep.

    Same struct describes both BasicPOMCP and robust runs; fields not relevant
    to a backend (e.g., rho_T for BasicPOMCP) are simply ignored.
    """
    experiment_id:   str
    date:            str
    description:     str
    planner_label:   str
    rho_T:           float = 0.0
    rho_Z:           float = 0.0
    ucb_mode:        str = "nominal"
    budget:          int = 500
    sims_per_backup: int = 5
    horizon:         int = 5
    eta_world_obs:   float = 0.0
    eta_world_trans: float = 0.0
    n_trials:        int = 100
    seed_base:       int = 1
    listen_accuracy: float = 0.85


# ---------------------------------------------------------------------------
# Planner factories — each returns a (belief) -> action closure
# ---------------------------------------------------------------------------

def make_robust_planner(model_planner: TabularPOMDP,
                        uncertainty: UncertaintySets,
                        cfg: ExperimentConfig
                        ) -> Callable[[np.ndarray], int]:
    """Closure for the robust solver."""
    def _plan(belief: np.ndarray) -> int:
        action, _ = robust_pomcp_plan(belief,
                                      model_planner,
                                      uncertainty,
                                      cfg.horizon,
                                      cfg.budget,
                                      cfg.sims_per_backup,
                                      ucb_mode=cfg.ucb_mode,
                                      logger=None
                                      )
        return action
    return _plan


def make_robust_belief_updater(model_planner: TabularPOMDP
                               ) -> Callable[[np.ndarray, int, int], np.ndarray]:
    return lambda b, a, o: bayes_update(b, model_planner, a, o)


# ---------------------------------------------------------------------------
# Per-trial trajectory diagnostics (Tiger-specific)
# ---------------------------------------------------------------------------

def tiger_action_counts(trajectory: list[EpisodeStep]) -> tuple[int, int, int]:
    """Extract (n_correct_open, n_wrong_open, n_listen) from one trajectory.

    Tiger 0-based ids:
        action 0 (open-left)  is correct iff state == 1 (tiger-right).
        action 1 (open-right) is correct iff state == 0 (tiger-left).
        action 2 (listen)     just increments the listen count.
    """
    n_correct = 0
    n_wrong = 0
    n_listen = 0
    for step in trajectory:
        if step.action == 0:        # open-left
            if step.state == 1:
                n_correct += 1
            else:
                n_wrong += 1
        elif step.action == 1:      # open-right
            if step.state == 0:
                n_correct += 1
            else:
                n_wrong += 1
        elif step.action == 2:      # listen
            n_listen += 1
    return n_correct, n_wrong, n_listen


# ---------------------------------------------------------------------------
# run_config
# ---------------------------------------------------------------------------

# CSV column order (matches Julia exactly — viz/plot_results.py contract).
SUMMARY_COLUMNS = [
    "experiment_id", "date", "description", "planner_label",
    "rho_T", "rho_Z", "ucb_mode", "budget", "sims_per_backup", "horizon",
    "eta_world_obs", "eta_world_trans",
    "n_trials", "seed_base",
    "mean_return", "std_return", "sem_return", "mean_runtime_s",
    "n_correct_open", "n_wrong_open", "n_listen_actions",
]
RAW_COLUMNS = [
    "experiment_id", "date", "description", "planner_label",
    "rho_T", "rho_Z", "ucb_mode", "budget", "sims_per_backup", "horizon",
    "eta_world_obs", "eta_world_trans",
    "trial_seed", "episode_return", "runtime_s",
    "n_correct_open", "n_wrong_open", "n_listen_actions",
]


def run_config(cfg: ExperimentConfig,
               model_nominal: TabularPOMDP
               ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run cfg.n_trials episodes for one configuration.

    Returns (summary_row, per_trial_rows) — both as dicts with column names.
    """
    # Build the *true* world (may be perturbed).
    model_true = build_perturbed_tiger(model_nominal,
                                       cfg.eta_world_obs,
                                       eta_trans=cfg.eta_world_trans
                                       )

    # Build planner uncertainty (only used by robust planner).
    uncertainty = uniform_uncertainty(model_nominal.n_states,
                                      model_nominal.n_actions,
                                      cfg.rho_T,
                                      cfg.rho_Z,
                                      TVDistance()
                                      )

    # Dispatch on planner_label.
    if cfg.planner_label == "robust_pomcp":
        planner_act = make_robust_planner(model_nominal, uncertainty, cfg)
        belief_update_fn = make_robust_belief_updater(model_nominal)
    elif cfg.planner_label == "BasicPOMCP":
        planner_act = make_basic_pomcp_planner(model_nominal, cfg)
        belief_update_fn = make_basicpomcp_belief_updater(model_nominal)
    else:
        raise ValueError(f"Unknown planner_label: {cfg.planner_label!r}")

    belief_init = np.full(
        model_nominal.n_states, 1.0 / model_nominal.n_states,
    )

    returns: list[float] = []
    runtimes: list[float] = []
    n_correct_total = 0
    n_wrong_total = 0
    n_listen_total = 0
    per_trial_rows: list[dict[str, Any]] = []

    for trial in range(cfg.n_trials):
        seed = cfg.seed_base + trial
        rng = np.random.default_rng(seed)
        # pomdp_py's POMCP uses Python's stdlib `random` module; seed it too
        # so the BasicPOMCP path is reproducible per trial.
        random.seed(seed)
        t0 = time.perf_counter()
        total_reward, trajectory = run_episode(planner_act,
                                               belief_update_fn,
                                               model_true,
                                               belief_init,
                                               cfg.horizon,
                                               rng
                                               )
        elapsed = time.perf_counter() - t0

        returns.append(total_reward)
        runtimes.append(elapsed)
        nc, nw, nl = tiger_action_counts(trajectory)
        n_correct_total += nc
        n_wrong_total += nw
        n_listen_total += nl

        per_trial_rows.append({
            "experiment_id":    cfg.experiment_id,
            "date":             cfg.date,
            "description":      cfg.description,
            "planner_label":    cfg.planner_label,
            "rho_T":            cfg.rho_T,
            "rho_Z":            cfg.rho_Z,
            "ucb_mode":         cfg.ucb_mode,
            "budget":           cfg.budget,
            "sims_per_backup":  cfg.sims_per_backup,
            "horizon":          cfg.horizon,
            "eta_world_obs":    cfg.eta_world_obs,
            "eta_world_trans":  cfg.eta_world_trans,
            "trial_seed":       seed,
            "episode_return":   total_reward,
            "runtime_s":        elapsed,
            "n_correct_open":   nc,
            "n_wrong_open":     nw,
            "n_listen_actions": nl,
        })

    n = len(returns)
    returns_arr = np.asarray(returns)
    mean_r = float(returns_arr.mean())
    std_r = float(returns_arr.std(ddof=1)) if n > 1 else 0.0
    sem_r = std_r / np.sqrt(n) if n > 1 else 0.0

    summary_row = {
        "experiment_id":    cfg.experiment_id,
        "date":             cfg.date,
        "description":      cfg.description,
        "planner_label":    cfg.planner_label,
        "rho_T":            cfg.rho_T,
        "rho_Z":            cfg.rho_Z,
        "ucb_mode":         cfg.ucb_mode,
        "budget":            cfg.budget,
        "sims_per_backup":  cfg.sims_per_backup,
        "horizon":           cfg.horizon,
        "eta_world_obs":    cfg.eta_world_obs,
        "eta_world_trans":  cfg.eta_world_trans,
        "n_trials":         cfg.n_trials,
        "seed_base":         cfg.seed_base,
        "mean_return":      mean_r,
        "std_return":        std_r,
        "sem_return":        sem_r,
        "mean_runtime_s":   float(np.mean(runtimes)),
        "n_correct_open":   n_correct_total,
        "n_wrong_open":     n_wrong_total,
        "n_listen_actions": n_listen_total,
    }
    return summary_row, per_trial_rows


# ---------------------------------------------------------------------------
# CSV / INDEX bookkeeping
# ---------------------------------------------------------------------------

def sweep_filename(experiment_id: str, date: str, description: str) -> str:
    return f"{experiment_id}_{date}_{description}.csv"


def sweep_raw_filename(experiment_id: str, date: str, description: str) -> str:
    return f"{experiment_id}_{date}_{description}_raw.csv"


def append_to_index(filename: str,
                    oneliner: str,
                    *,
                    results_dir: Path = DEFAULT_RESULTS_DIR
                    ) -> None:
    """Append a one-line description for this sweep to INDEX.md.

    Creates the file if needed. Idempotent: skips if filename already listed.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    index_path = results_dir / "INDEX.md"
    if not index_path.is_file():
        index_path.write_text("# Experiment results index\n\n")
    existing = index_path.read_text()
    if filename in existing:
        return
    with index_path.open("a") as f:
        f.write(f"- `{filename}` — {oneliner}\n")


# ---------------------------------------------------------------------------
# run_sweep
# ---------------------------------------------------------------------------

def run_sweep(experiment_id: str,
              date: str,
              description: str,
              configs: list[ExperimentConfig],
              model_nominal: TabularPOMDP,
              *,
              index_oneliner: str | None = None,
              results_dir: Path = DEFAULT_RESULTS_DIR
              ) -> pd.DataFrame:
    """Run a list of configs, write summary + raw CSVs, append to INDEX.md.

    Returns the summary DataFrame (in-memory copy of what was written).
    """
    summary_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []

    for i, cfg in enumerate(configs, start=1):
        logger.info(
            "[%s] config %d / %d: %s rho_Z=%s eta_obs=%s",
            experiment_id, i, len(configs),
            cfg.planner_label, cfg.rho_Z, cfg.eta_world_obs,
        )
        summary_row, per_trial_rows = run_config(cfg, model_nominal)
        summary_rows.append(summary_row)
        raw_rows.extend(per_trial_rows)

    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    raw_df = pd.DataFrame(raw_rows, columns=RAW_COLUMNS)

    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / sweep_filename(experiment_id, date, description)
    raw_path = results_dir / sweep_raw_filename(experiment_id, date, description)
    summary_df.to_csv(summary_path, index=False)
    raw_df.to_csv(raw_path, index=False)

    append_to_index(summary_path.name,
                    index_oneliner if index_oneliner is not None else description,
                    results_dir=results_dir
                    )

    logger.info("Sweep written: %s", summary_path)
    logger.info("Raw data:      %s", raw_path)
    return summary_df
