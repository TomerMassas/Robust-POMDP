"""
Tiger POMDP — Solver A end-to-end test script.

Runs the solver with a logger enabled, reports a summary at the root,
and exports the entire run (tree + simulations + backups) to a JSON file
at experiments/tiger/tiger_run.json.

The JSON file is consumed by the Python Dash UI in viz/app.py.

Parameters (edit these for different runs):
    budget: number of simulations.
    horizon: planning depth.
    rho_T, rho_Z: uncertainty radii.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robust_pomdp import (
    SolverLog,
    TVDistance,
    export_log_to_json,
    num_events,
    robust_pomcp_plan,
    tree_size,
    uniform_uncertainty,
)

from tiger_problem import make_tiger_pomdp


def main() -> None:
    print("=" * 70)
    print("Tiger POMDP — Robust POMCP")
    print("=" * 70)

    tiger = make_tiger_pomdp(listen_accuracy=0.85)

    rho_T_val = 0.0
    rho_Z_val = 0.1
    uncertainty = uniform_uncertainty(tiger.n_states,
                                      tiger.n_actions,
                                      rho_T_val,
                                      rho_Z_val,
                                      TVDistance()
                                      )

    belief = np.array([0.5, 0.5])
    horizon = 5
    budget = 500
    sims_per_backup = 5

    print("Parameters:")
    print(f"  belief           = {belief.tolist()}")
    print(f"  horizon          = {horizon}")
    print(f"  budget           = {budget}")
    print(f"  sims_per_backup  = {sims_per_backup}")
    print(f"  rho_T            = {rho_T_val}")
    print(f"  rho_Z            = {rho_Z_val}")
    print()

    print("Running robust_pomcp_plan...")
    log = SolverLog()
    rng = np.random.default_rng(0)

    action, root = robust_pomcp_plan(belief,
                                     tiger,
                                     uncertainty,
                                     horizon,
                                     budget,
                                     sims_per_backup,
                                     ucb_mode="nominal",
                                     exploration_constant=50,
                                     logger=log,
                                     rng=rng,
                                     )

    action_names = ["open-left", "open-right", "listen"]
    print(f"Solver returned action {action} ({action_names[action]})")
    print()

    print("Q values at root:")
    print("  Action           Q_nominal     Q_robust      N_visits")
    print("  " + "-" * 55)
    for a in range(tiger.n_actions):
        if a in root.children:
            an = root.children[a]
            print(f"  {action_names[a]:<16} {an.Q_nominal:<13.3f} "
                  f"{an.Q_robust:<13.3f} {an.N}")
        else:
            print(f"  {action_names[a]:<16} (not expanded)")
    print()

    n_h, n_a = tree_size(root)
    print(f"Tree: {n_h} HistoryNodes, {n_a} ActionNodes")
    print(f"Log:  {num_events(log)} events")
    print()

    output_path = Path(__file__).parent / "tiger_run.json"
    export_log_to_json(log, output_path,
                       metadata={
                           "problem":         "Tiger",
                           "n_states":        tiger.n_states,
                           "n_actions":       tiger.n_actions,
                           "n_obs":           tiger.n_obs,
                           "horizon":         horizon,
                           "budget":          budget,
                           "sims_per_backup": sims_per_backup,
                           "ucb_mode":        "nominal",
                           "rho_T":           rho_T_val,
                           "rho_Z":           rho_Z_val,
                           "belief":          belief.tolist(),
                           "action_names":    action_names,
                           "state_names":     ["tiger-left", "tiger-right"],
                           "obs_names":       ["hear-left", "hear-right"],
                       }
                       )

    print(f"JSON exported: {output_path}")
    print()
    print(f"To explore: python viz/app.py --json {output_path}")


if __name__ == "__main__":
    main()
