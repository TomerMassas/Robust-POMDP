"""
Generic episode harness for evaluating any planner on a tabular POMDP.

Drives the standard POMDP loop:
    belief -> plan -> execute (true world) -> observe -> Bayes update -> repeat

The planner sees ONLY the nominal model it was built around (captured inside
`planner_act`). The world transitions and emits observations under the *true*
model, which may differ from the nominal one in perturbed-world experiments.
This decoupling is what makes robustness experiments meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from robust_pomdp.core.pomdp_types import TabularPOMDP


@dataclass
class EpisodeStep:
    """One row of the trajectory log."""
    t: int
    state: int
    belief: np.ndarray
    action: int
    reward: float
    obs: int
    next_state: int


def run_episode(planner_act: Callable[[np.ndarray], int],
                belief_update: Callable[[np.ndarray, int, int], np.ndarray],
                model_true: TabularPOMDP,
                belief_init: np.ndarray,
                horizon: int,
                rng: np.random.Generator,
                *,
                initial_state: int | None = None
                ) -> tuple[float, list[EpisodeStep]]:
    """Run a single episode for `horizon` steps.

    Args:
        planner_act: closure capturing the planner config; (belief) -> action.
        belief_update: agent's belief filter; (belief, action, obs) -> new belief.
        model_true: the *real* environment dynamics (may be perturbed).
        belief_init: agent's starting belief vector.
        horizon: number of timesteps.
        rng: NumPy random generator (for reproducibility).
        initial_state: if None, sampled from `belief_init`.

    Returns:
        (total_reward, trajectory).
    """
    belief_init = np.asarray(belief_init, dtype=np.float64)
    if initial_state is None:
        state = int(rng.choice(len(belief_init), p=belief_init))
    else:
        state = initial_state

    belief = belief_init.copy()
    total_reward = 0.0
    trajectory: list[EpisodeStep] = []

    for t in range(horizon):
        action = planner_act(belief)
        r = model_true.reward(state, action)
        next_state = model_true.sample_transition(state, action, rng)
        obs = model_true.sample_observation(next_state, rng)

        trajectory.append(EpisodeStep(
            t=t,
            state=state,
            belief=belief.copy(),
            action=action,
            reward=r,
            obs=obs,
            next_state=next_state,
        ))

        total_reward += r
        state = next_state
        belief = belief_update(belief, action, obs)

    return total_reward, trajectory
