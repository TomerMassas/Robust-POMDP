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

from robust_pomdp.evaluation.episode import EpisodeStep
from robust_pomdp.evaluation.scenario import Scenario, default_results_dir
from robust_pomdp.uncertainty.uncertainty_sets import (
    TVDistance,
    UncertaintySets,
    uniform_uncertainty,
)


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


# ---------------------------------------------------------------------------
# Scenario plumbing — perturbation, per-trial metrics, scenario factory
# ---------------------------------------------------------------------------

def build_perturbed_tiger(model_nominal: TabularPOMDP,
                          eta_obs: float,
                          eta_trans: float
                          ) -> TabularPOMDP:
    """Tiger perturbation: reduce listen accuracy by `eta_obs`.

    Tiger has degenerate transitions (open resets, listen stays); `eta_trans`
    must be 0. Kept in the signature for parity with the generic
    `PerturbWorldFn` interface in `Scenario`.
    """
    if eta_trans != 0.0:
        raise ValueError("Tiger transitions are degenerate; eta_trans must be 0")
    if eta_obs == 0.0:
        return model_nominal
    O_pert = model_nominal.O.copy()
    O_pert[TIGER_STATE_LEFT,  TIGER_OBS_HEAR_LEFT]  -= eta_obs
    O_pert[TIGER_STATE_LEFT,  TIGER_OBS_HEAR_RIGHT] += eta_obs
    O_pert[TIGER_STATE_RIGHT, TIGER_OBS_HEAR_LEFT]  += eta_obs
    O_pert[TIGER_STATE_RIGHT, TIGER_OBS_HEAR_RIGHT] -= eta_obs
    assert np.all((0.0 <= O_pert) & (O_pert <= 1.0)), \
        f"Perturbation eta_obs={eta_obs} drove probabilities out of [0,1]"
    return TabularPOMDP.from_matrices(model_nominal.T, O_pert, model_nominal.R)


def tiger_metrics(trajectory: list[EpisodeStep]) -> dict[str, float]:
    """Tiger per-trial diagnostics: correct-open / wrong-open / listen counts."""
    n_correct = n_wrong = n_listen = 0
    for step in trajectory:
        if step.action == TIGER_ACTION_OPEN_LEFT:
            if step.state == TIGER_STATE_RIGHT:
                n_correct += 1
            else:
                n_wrong += 1
        elif step.action == TIGER_ACTION_OPEN_RIGHT:
            if step.state == TIGER_STATE_LEFT:
                n_correct += 1
            else:
                n_wrong += 1
        elif step.action == TIGER_ACTION_LISTEN:
            n_listen += 1
    return {"n_correct_open":   n_correct,
            "n_wrong_open":     n_wrong,
            "n_listen_actions": n_listen}


def make_tiger_scenario(listen_accuracy: float = 0.85) -> Scenario:
    """Bundle the Tiger plumbing into a Scenario for `run_sweep`."""
    nominal = make_tiger_pomdp(listen_accuracy=listen_accuracy)

    def uncertainty_factory(rho_T: float, rho_Z: float) -> UncertaintySets:
        return uniform_uncertainty(nominal.n_states,
                                   nominal.n_actions,
                                   rho_T,
                                   rho_Z,
                                   TVDistance()
                                   )

    return Scenario(name="tiger",
                    nominal_model=nominal,
                    perturb_world=build_perturbed_tiger,
                    metrics_fn=tiger_metrics,
                    metric_columns=["n_correct_open", "n_wrong_open", "n_listen_actions"],
                    results_dir=default_results_dir("tiger"),
                    uncertainty_factory=uncertainty_factory
                    )
