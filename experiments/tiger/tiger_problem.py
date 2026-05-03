"""
Tiger POMDP problem definition (0-based ids).

Classic Kaelbling et al. (1998) Tiger problem:
  - 2 states: tiger-left (s=0), tiger-right (s=1)
  - 3 actions: open-left (a=0), open-right (a=1), listen (a=2)
  - 2 observations: hear-left (z=0), hear-right (z=1)

Rewards:
  - Listen: -1
  - Open correct door: +10
  - Open wrong door: -100

Listening accuracy: 85% correct observation (configurable).
After opening a door, the state resets uniformly (game restarts).

NOTE: ids are 0-based here, vs 1-based in the Julia version (Tiger
constants in Julia were 1, 2, 3). See plan §4.7.
"""

from __future__ import annotations

import numpy as np

from robust_pomdp import TabularPOMDP


# Named constants (0-based)
TIGER_ACTION_OPEN_LEFT = 0
TIGER_ACTION_OPEN_RIGHT = 1
TIGER_ACTION_LISTEN = 2

TIGER_STATE_LEFT = 0
TIGER_STATE_RIGHT = 1

TIGER_OBS_HEAR_LEFT = 0
TIGER_OBS_HEAR_RIGHT = 1


def make_tiger_pomdp(listen_accuracy: float = 0.85) -> TabularPOMDP:
    """Build the Tiger POMDP as a TabularPOMDP.

    Args:
        listen_accuracy: probability of hearing the tiger correctly (default 0.85).

    Returns:
        TabularPOMDP with n_states=2, n_actions=3, n_obs=2.
    """
    n_states = 2
    n_actions = 3
    n_obs = 2

    # T[s, a, s_next]
    T = np.zeros((n_states, n_actions, n_states))

    # Open-left or Open-right: state resets uniformly
    for s in range(n_states):
        T[s, TIGER_ACTION_OPEN_LEFT, :] = 0.5
        T[s, TIGER_ACTION_OPEN_RIGHT, :] = 0.5

    # Listen: state unchanged
    for s in range(n_states):
        T[s, TIGER_ACTION_LISTEN, s] = 1.0

    # O[s_next, z]
    O = np.zeros((n_states, n_obs))
    O[TIGER_STATE_LEFT, TIGER_OBS_HEAR_LEFT] = listen_accuracy
    O[TIGER_STATE_LEFT, TIGER_OBS_HEAR_RIGHT] = 1.0 - listen_accuracy
    O[TIGER_STATE_RIGHT, TIGER_OBS_HEAR_LEFT] = 1.0 - listen_accuracy
    O[TIGER_STATE_RIGHT, TIGER_OBS_HEAR_RIGHT] = listen_accuracy

    # R[s, a]
    R = np.zeros((n_states, n_actions))
    R[TIGER_STATE_LEFT, TIGER_ACTION_OPEN_LEFT] = -100.0
    R[TIGER_STATE_RIGHT, TIGER_ACTION_OPEN_LEFT] = 10.0
    R[TIGER_STATE_LEFT, TIGER_ACTION_OPEN_RIGHT] = 10.0
    R[TIGER_STATE_RIGHT, TIGER_ACTION_OPEN_RIGHT] = -100.0
    R[TIGER_STATE_LEFT, TIGER_ACTION_LISTEN] = -1.0
    R[TIGER_STATE_RIGHT, TIGER_ACTION_LISTEN] = -1.0

    return TabularPOMDP.from_matrices(T, O, R)
