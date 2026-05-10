"""
FrozenLake smoke test.

Verifies the scenario integrates with the robust_pomcp planner and the
episode harness end to end. Not an experiment — just a "does it run."
"""

from __future__ import annotations

import numpy as np

from robust_pomdp import robust_pomcp_plan, run_episode, bayes_update

from frozenlake_problem import (
    make_frozenlake_scenario,
    START_INDEX, GOAL_INDEX, TERMINAL_STATE,
    N_STATES, N_ACTIONS,
)


def main() -> None:
    scenario = make_frozenlake_scenario(epsilon=0.05, p=0.6)
    model = scenario.nominal_model

    # Initial belief: concentrated at start cell
    belief = np.zeros(N_STATES)
    belief[START_INDEX] = 1.0

    # 1. Run a single planner call
    uncertainty = scenario.uncertainty_factory(rho_T_val=0.0, rho_Z_val=0.0)
    rng = np.random.default_rng(42)
    action, root = robust_pomcp_plan(belief,
                                     model,
                                     uncertainty,
                                     horizon=10,
                                     budget=50,
                                     sims_per_backup=5,
                                     ucb_mode="nominal",
                                     logger=None,
                                     rng=rng
                                     )
    assert 0 <= action < N_ACTIONS, f"Action out of range: {action}"
    print(f"Planner call: action={action}, |children|={len(root.children)}")

    # 2. Run a full episode at horizon=30
    rng_world = np.random.default_rng(0)
    rng_planner = np.random.default_rng(1)

    def planner_act(b: np.ndarray) -> int:
        a, _ = robust_pomcp_plan(b,
                                 model,
                                 uncertainty,
                                 horizon=10,
                                 budget=50,
                                 sims_per_backup=5,
                                 ucb_mode="nominal",
                                 logger=None,
                                 rng=rng_planner
                                 )
        return a

    def belief_update_fn(b: np.ndarray, a: int, o: int) -> np.ndarray:
        return bayes_update(b, model, a, o)

    total_reward, trajectory = run_episode(planner_act,
                                           belief_update_fn,
                                           scenario.nominal_model,
                                           belief,
                                           horizon=30,
                                           rng=rng_world,
                                           initial_state=START_INDEX
                                           )
    assert len(trajectory) == 30
    print(f"Episode: total_reward={total_reward:.4f}, trajectory length={len(trajectory)}")

    # 3. Metrics
    metrics = scenario.metrics_fn(trajectory)
    print(f"Metrics: {metrics}")
    assert set(metrics.keys()) == set(scenario.metric_columns)

    # Outcome interpretation
    if metrics["n_reached_goal"]:
        print("  Result: REACHED GOAL")
    elif metrics["n_fell_in_hole"]:
        print("  Result: fell in hole")
    else:
        print("  Result: timed out (still active or absorbed without reaching goal)")


if __name__ == "__main__":
    main()
