"""
Main experiment runner.

Usage:
    python experiments/tiger/run_experiments.py                 # all experiments
    python experiments/tiger/run_experiments.py --only E1       # just E1
    python experiments/tiger/run_experiments.py --only E1,E3    # E1 and E3

Each experiment writes one summary CSV + one `_raw.csv` under
`experiments/tiger/results/` and appends a line to INDEX.md. Filename
convention: {Ek}_{ISO_date}_{short_description}.csv.
"""

from __future__ import annotations

import argparse
import datetime
import logging

from robust_pomdp import ExperimentConfig, run_sweep

from tiger_problem import make_tiger_pomdp


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Common setup
# ---------------------------------------------------------------------------

TODAY = datetime.date.today().isoformat()  # "yyyy-mm-dd"
TIGER = make_tiger_pomdp(listen_accuracy=0.85)

N_TRIALS_DEFAULT = 10
BUDGET_DEFAULT = 100
HORIZON_DEFAULT = 5


# ---------------------------------------------------------------------------
# E1 — Baseline equivalence (sanity)
#   BasicPOMCP vs robust_pomcp_plan(rho=0) on the nominal Tiger world.
# ---------------------------------------------------------------------------

def experiment_E1() -> None:
    cfgs = [
        ExperimentConfig(experiment_id="E1",
                         date=TODAY,
                         description="baseline_basicpomcp_vs_rho0",
                         planner_label="BasicPOMCP",
                         n_trials=N_TRIALS_DEFAULT,
                         budget=BUDGET_DEFAULT,
                         horizon=HORIZON_DEFAULT
                         ),
        ExperimentConfig(experiment_id="E1",
                         date=TODAY,
                         description="baseline_basicpomcp_vs_rho0",
                         planner_label="robust_pomcp",
                         rho_T=0.0, rho_Z=0.0,
                         ucb_mode="nominal",
                         n_trials=N_TRIALS_DEFAULT,
                         budget=BUDGET_DEFAULT,
                         sims_per_backup=5,
                         horizon=HORIZON_DEFAULT
                         ),
    ]
    run_sweep("E1", TODAY, "baseline_basicpomcp_vs_rho0", cfgs, TIGER,
              index_oneliner="Baseline equivalence: BasicPOMCP vs robust_pomcp(rho=0) on nominal Tiger"
              )


# ---------------------------------------------------------------------------
# E2 — rho_Z sweep, nominal world
# ---------------------------------------------------------------------------

def experiment_E2() -> None:
    cfgs = [
        ExperimentConfig(experiment_id="E2",
                         date=TODAY,
                         description="rho_z_sweep_nominal",
                         planner_label="robust_pomcp",
                         rho_Z=rho_z,
                         ucb_mode="nominal",
                         n_trials=N_TRIALS_DEFAULT,
                         budget=BUDGET_DEFAULT,
                         sims_per_backup=5,
                         horizon=HORIZON_DEFAULT
                         )
        for rho_z in (0.0, 0.05, 0.1, 0.2, 0.3)
    ]
    run_sweep("E2", TODAY, "rho_z_sweep_nominal", cfgs, TIGER,
              index_oneliner="rho_Z sweep (0..0.3) on nominal Tiger world"
              )


# ---------------------------------------------------------------------------
# E3 — rho_Z sweep, perturbed world (eta=0.20)
# ---------------------------------------------------------------------------

def experiment_E3() -> None:
    cfgs = [
        ExperimentConfig(experiment_id="E3",
                         date=TODAY,
                         description="rho_z_sweep_perturbed_eta0.20",
                         planner_label="robust_pomcp",
                         rho_Z=rho_z,
                         ucb_mode="nominal",
                         eta_world_obs=0.20,
                         n_trials=N_TRIALS_DEFAULT,
                         budget=BUDGET_DEFAULT,
                         sims_per_backup=5,
                         horizon=HORIZON_DEFAULT
                         )
        for rho_z in (0.0, 0.05, 0.1, 0.2, 0.3)
    ]
    run_sweep("E3", TODAY, "rho_z_sweep_perturbed_eta0.20", cfgs, TIGER,
              index_oneliner="rho_Z sweep (0..0.3) on eta=0.20 perturbed Tiger (headline)"
              )


# ---------------------------------------------------------------------------
# E4 — Adversarial eta sweep, fixed planner pair (vanilla vs robust rho_Z=0.2)
# ---------------------------------------------------------------------------

def experiment_E4() -> None:
    cfgs = []
    for eta in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3):
        for rho_z in (0.0, 0.2):
            cfgs.append(ExperimentConfig(experiment_id="E4",
                                         date=TODAY,
                                         description="eta_sweep_paired",
                                         planner_label="robust_pomcp",
                                         rho_Z=rho_z,
                                         ucb_mode="nominal",
                                         eta_world_obs=eta,
                                         n_trials=N_TRIALS_DEFAULT,
                                         budget=BUDGET_DEFAULT,
                                         sims_per_backup=5,
                                         horizon=HORIZON_DEFAULT
                                         ))
    run_sweep("E4", TODAY, "eta_sweep_paired", cfgs, TIGER,
              index_oneliner="eta sweep on world; vanilla(rho=0) vs robust(rho_Z=0.2)"
              )


# ---------------------------------------------------------------------------
# E5 — sims_per_backup sweep (nominal + perturbed -> 2 CSVs)
# ---------------------------------------------------------------------------

def experiment_E5() -> None:
    for suffix, eta_obs, oneliner in (
        ("nominal", 0.0, "sims_per_backup sweep on nominal world"),
        ("perturbed_eta0.20", 0.20, "sims_per_backup sweep on eta=0.20 perturbed world"),
    ):
        cfgs = [
            ExperimentConfig(experiment_id="E5",
                             date=TODAY,
                             description=f"simsperbackup_sweep_{suffix}",
                             planner_label="robust_pomcp",
                             rho_Z=0.1,
                             ucb_mode="nominal",
                             eta_world_obs=eta_obs,
                             n_trials=N_TRIALS_DEFAULT,
                             budget=BUDGET_DEFAULT,
                             sims_per_backup=K,
                             horizon=HORIZON_DEFAULT
                             )
            for K in (1, 5, 25, 100, 500)
        ]
        run_sweep("E5", TODAY, f"simsperbackup_sweep_{suffix}", cfgs, TIGER,
                  index_oneliner=oneliner
                  )


# ---------------------------------------------------------------------------
# E6 — UCB mode comparison ("nominal" vs "robust" at rho_Z=0.1)
# ---------------------------------------------------------------------------

def experiment_E6() -> None:
    for suffix, eta_obs, oneliner in (
        ("nominal", 0.0, "ucb_mode compare on nominal world"),
        ("perturbed_eta0.20", 0.20, "ucb_mode compare on eta=0.20 perturbed world"),
    ):
        cfgs = [
            ExperimentConfig(experiment_id="E6",
                             date=TODAY,
                             description=f"ucbmode_compare_{suffix}",
                             planner_label="robust_pomcp",
                             rho_Z=0.1,
                             ucb_mode=mode,
                             eta_world_obs=eta_obs,
                             n_trials=N_TRIALS_DEFAULT,
                             budget=BUDGET_DEFAULT,
                             sims_per_backup=5,
                             horizon=HORIZON_DEFAULT
                             )
            for mode in ("nominal", "robust")
        ]
        run_sweep("E6", TODAY, f"ucbmode_compare_{suffix}", cfgs, TIGER,
                  index_oneliner=oneliner
                  )


# ---------------------------------------------------------------------------
# E7 — Horizon sweep (optional)
# ---------------------------------------------------------------------------

def experiment_E7() -> None:
    for suffix, eta_obs, oneliner in (
        ("nominal", 0.0, "horizon sweep on nominal world"),
        ("perturbed_eta0.20", 0.20, "horizon sweep on eta=0.20 perturbed world"),
    ):
        cfgs = [
            ExperimentConfig(experiment_id="E7",
                             date=TODAY,
                             description=f"horizon_sweep_{suffix}",
                             planner_label="robust_pomcp",
                             rho_Z=0.1,
                             ucb_mode="nominal",
                             eta_world_obs=eta_obs,
                             n_trials=N_TRIALS_DEFAULT,
                             budget=BUDGET_DEFAULT,
                             sims_per_backup=5,
                             horizon=h
                             )
            for h in (3, 5, 7, 10)
        ]
        run_sweep("E7", TODAY, f"horizon_sweep_{suffix}", cfgs, TIGER,
                  index_oneliner=oneliner
                  )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    ("E1", experiment_E1),
    ("E2", experiment_E2),
    ("E3", experiment_E3),
    ("E4", experiment_E4),
    ("E5", experiment_E5),
    ("E6", experiment_E6),
    ("E7", experiment_E7),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated experiment ids (e.g. E1,E3). Default: all."
                        )
    args = parser.parse_args()

    selected = (
        {s.strip() for s in args.only.split(",")} if args.only is not None
        else None
    )

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    for eid, fn in EXPERIMENTS:
        if selected is None or eid in selected:
            logger.info("==== Running %s ====", eid)
            fn()
        else:
            logger.info("Skipping %s", eid)

    logger.info("All requested experiments complete.")


if __name__ == "__main__":
    main()
