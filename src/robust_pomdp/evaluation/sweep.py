"""
Experiment sweep infrastructure.

`run_config` runs `n_trials` episodes for one configuration and returns summary
statistics. `run_sweep` runs many configs, writes summary + raw CSVs per sweep,
and appends a description line to INDEX.md.

Two planner backends are dispatched on `planner_label`:
    "robust_pomcp" — our `robust_pomcp_plan` + the discrete `bayes_update`.
    "BasicPOMCP"   — pomdp_py's POMCP via the adapter (label kept for CSV
                     schema continuity).

Per-toy plumbing (perturbation, per-trial metrics, results directory) lives in
`Scenario` — see `scenario.py`. This module is scenario-agnostic.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from robust_pomdp.core.pomdp_types import TabularPOMDP
from robust_pomdp.evaluation.belief_update import bayes_update
from robust_pomdp.evaluation.episode import run_episode
from robust_pomdp.evaluation.pomdp_py_adapter import (
    make_basic_pomcp_planner,
    make_basicpomcp_belief_updater,
)
from robust_pomdp.evaluation.scenario import Scenario
from robust_pomdp.solvers.robust_pomcp import robust_pomcp_plan
from robust_pomdp.uncertainty.uncertainty_sets import UncertaintySets


logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Planner factories — each returns a (belief) -> action closure
# ---------------------------------------------------------------------------

def make_robust_planner(model_planner: TabularPOMDP,
                        uncertainty: UncertaintySets,
                        cfg: ExperimentConfig,
                        rng: np.random.Generator,
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
                                      logger=None,
                                      rng=rng
                                      )
        return action
    return _plan


def make_robust_belief_updater(model_planner: TabularPOMDP
                               ) -> Callable[[np.ndarray, int, int], np.ndarray]:
    return lambda b, a, o: bayes_update(b, model_planner, a, o)


# ---------------------------------------------------------------------------
# CSV schema (base columns common to all scenarios; per-scenario metric
# columns are appended at write time)
# ---------------------------------------------------------------------------

SUMMARY_BASE_COLUMNS = [
    "experiment_id", "date", "description", "planner_label",
    "rho_T", "rho_Z", "ucb_mode", "budget", "sims_per_backup", "horizon",
    "eta_world_obs", "eta_world_trans",
    "n_trials", "seed_base",
    "mean_return", "std_return", "sem_return", "mean_runtime_s",
]
RAW_BASE_COLUMNS = [
    "experiment_id", "date", "description", "planner_label",
    "rho_T", "rho_Z", "ucb_mode", "budget", "sims_per_backup", "horizon",
    "eta_world_obs", "eta_world_trans",
    "trial_seed", "episode_return", "runtime_s",
]


# ---------------------------------------------------------------------------
# run_config
# ---------------------------------------------------------------------------

def run_config(cfg: ExperimentConfig,
               scenario: Scenario
               ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run cfg.n_trials episodes for one configuration.

    Per-trial metrics from `scenario.metrics_fn` are summed across trials for
    the summary row; the raw CSV stores them per trial.

    Returns (summary_row, per_trial_rows) — both as dicts with column names.
    """
    model_nominal = scenario.nominal_model
    model_true = scenario.perturb_world(model_nominal,
                                        cfg.eta_world_obs,
                                        cfg.eta_world_trans
                                        )

    # Build planner uncertainty (only used by robust planner).
    uncertainty = scenario.uncertainty_factory(cfg.rho_T, cfg.rho_Z)

    belief_init = np.full(
        model_nominal.n_states, 1.0 / model_nominal.n_states,
    )

    returns: list[float] = []
    runtimes: list[float] = []
    metric_totals: dict[str, float] = {col: 0.0 for col in scenario.metric_columns}
    per_trial_rows: list[dict[str, Any]] = []

    for trial in range(cfg.n_trials):
        seed = cfg.seed_base + trial
        # Two independent RNG streams per trial: one for the world, one for the
        # planner. spawn(2) derives them deterministically from a single seed,
        # so changing one side's randomness consumption can't perturb the other.
        world_rng, planner_rng = np.random.default_rng(seed).spawn(2)
        # pomdp_py's POMCP uses Python's stdlib `random` module; seed it too
        # so the BasicPOMCP path is reproducible per trial.
        random.seed(seed)

        # Build planner + belief updater inside the loop so the planner closure
        # captures this trial's `planner_rng`. (Belief updater rebuild is
        # trivial; kept here for symmetry.)
        if cfg.planner_label == "robust_pomcp":
            planner_act = make_robust_planner(model_nominal, uncertainty, cfg, planner_rng)
            belief_update_fn = make_robust_belief_updater(model_nominal)
        elif cfg.planner_label == "BasicPOMCP":
            planner_act = make_basic_pomcp_planner(model_nominal, cfg)
            belief_update_fn = make_basicpomcp_belief_updater(model_nominal)
        else:
            raise ValueError(f"Unknown planner_label: {cfg.planner_label!r}")

        t0 = time.perf_counter()
        total_reward, trajectory = run_episode(planner_act,
                                               belief_update_fn,
                                               model_true,
                                               belief_init,
                                               cfg.horizon,
                                               world_rng
                                               )
        elapsed = time.perf_counter() - t0

        returns.append(total_reward)
        runtimes.append(elapsed)

        metrics = scenario.metrics_fn(trajectory)
        for k, v in metrics.items():
            metric_totals[k] += v

        per_trial_row = {
            "experiment_id":   cfg.experiment_id,
            "date":            cfg.date,
            "description":     cfg.description,
            "planner_label":   cfg.planner_label,
            "rho_T":           cfg.rho_T,
            "rho_Z":           cfg.rho_Z,
            "ucb_mode":        cfg.ucb_mode,
            "budget":          cfg.budget,
            "sims_per_backup": cfg.sims_per_backup,
            "horizon":         cfg.horizon,
            "eta_world_obs":   cfg.eta_world_obs,
            "eta_world_trans": cfg.eta_world_trans,
            "trial_seed":      seed,
            "episode_return":  total_reward,
            "runtime_s":       elapsed,
        }
        per_trial_row.update(metrics)
        per_trial_rows.append(per_trial_row)

    n = len(returns)
    returns_arr = np.asarray(returns)
    mean_r = float(returns_arr.mean())
    std_r = float(returns_arr.std(ddof=1)) if n > 1 else 0.0
    sem_r = std_r / np.sqrt(n) if n > 1 else 0.0

    summary_row = {
        "experiment_id":   cfg.experiment_id,
        "date":            cfg.date,
        "description":     cfg.description,
        "planner_label":   cfg.planner_label,
        "rho_T":           cfg.rho_T,
        "rho_Z":           cfg.rho_Z,
        "ucb_mode":        cfg.ucb_mode,
        "budget":          cfg.budget,
        "sims_per_backup": cfg.sims_per_backup,
        "horizon":         cfg.horizon,
        "eta_world_obs":   cfg.eta_world_obs,
        "eta_world_trans": cfg.eta_world_trans,
        "n_trials":        cfg.n_trials,
        "seed_base":       cfg.seed_base,
        "mean_return":     mean_r,
        "std_return":      std_r,
        "sem_return":      sem_r,
        "mean_runtime_s":  float(np.mean(runtimes)),
    }
    summary_row.update(metric_totals)
    return summary_row, per_trial_rows


# ---------------------------------------------------------------------------
# CSV / INDEX bookkeeping
# ---------------------------------------------------------------------------

def sweep_filename(experiment_id: str, date: str, description: str) -> str:
    return f"{experiment_id}_{date}_{description}.csv"


def sweep_raw_filename(experiment_id: str, date: str, description: str) -> str:
    return f"{experiment_id}_{date}_{description}_raw.csv"


def append_to_index(relative_path: str,
                    oneliner: str,
                    *,
                    results_dir: Path
                    ) -> None:
    """Append a one-line description for this sweep to INDEX.md.

    `relative_path` is the path of the summary CSV relative to `results_dir`
    (e.g. ``"E1/csv/E1_2026-05-03_baseline_*.csv"``). It is both the link
    target and the idempotency key. Creates INDEX.md if absent.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    index_path = results_dir / "INDEX.md"
    if not index_path.is_file():
        index_path.write_text("# Experiment results index\n\n")
    existing = index_path.read_text()
    if relative_path in existing:
        return
    with index_path.open("a") as f:
        f.write(f"- `{relative_path}` — {oneliner}\n")


# ---------------------------------------------------------------------------
# run_sweep
# ---------------------------------------------------------------------------

def run_sweep(experiment_id: str,
              date: str,
              description: str,
              configs: list[ExperimentConfig],
              scenario: Scenario,
              *,
              index_oneliner: str | None = None
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
        summary_row, per_trial_rows = run_config(cfg, scenario)
        summary_rows.append(summary_row)
        raw_rows.extend(per_trial_rows)

    summary_columns = SUMMARY_BASE_COLUMNS + scenario.metric_columns
    raw_columns = RAW_BASE_COLUMNS + scenario.metric_columns

    summary_df = pd.DataFrame(summary_rows, columns=summary_columns)
    raw_df = pd.DataFrame(raw_rows, columns=raw_columns)

    csv_dir = scenario.results_dir / experiment_id / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    summary_path = csv_dir / sweep_filename(experiment_id, date, description)
    raw_path = csv_dir / sweep_raw_filename(experiment_id, date, description)
    summary_df.to_csv(summary_path, index=False)
    raw_df.to_csv(raw_path, index=False)

    append_to_index(f"{experiment_id}/csv/{summary_path.name}",
                    index_oneliner if index_oneliner is not None else description,
                    results_dir=scenario.results_dir
                    )

    logger.info("Sweep written: %s", summary_path)
    logger.info("Raw data:      %s", raw_path)
    return summary_df
