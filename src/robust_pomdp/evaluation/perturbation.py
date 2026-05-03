"""
World perturbation utilities for robustness experiments.

For Tiger we perturb the observation model by reducing the listen accuracy:
the *true* world's observation distribution is shifted, while the planner
keeps believing the nominal accuracy. This creates the model misspecification
that the robust planner is supposed to handle gracefully.

Transition perturbations are stubbed (Tiger uses rho_T = 0 in v1 experiments).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from robust_pomdp.core.pomdp_types import TabularPOMDP


ObsPerturbMode = Literal["reduce_accuracy"]
TransPerturbMode = Literal["identity"]


def perturb_observation_model(O: np.ndarray,
                              eta: float,
                              *,
                              mode: ObsPerturbMode = "reduce_accuracy"
                              ) -> np.ndarray:
    """Return a new observation matrix with the listen accuracy reduced by `eta`.

    Tiger convention: O[0, 0] = O[1, 1] = alpha (listen accuracy), rows sum to 1.
    After perturbation: O[0, 0] = O[1, 1] = alpha - eta; off-diagonals fill in to
    preserve row-sums of 1.

    Modes:
        "reduce_accuracy" — symmetric reduction in diagonal entries (Tiger-only).
    """
    if mode == "reduce_accuracy":
        assert O.shape == (2, 2), \
            "reduce_accuracy mode is Tiger-specific (2 states, 2 obs)"
        O_new = O.copy()
        O_new[0, 0] -= eta
        O_new[0, 1] += eta
        O_new[1, 0] += eta
        O_new[1, 1] -= eta
        assert np.all((0.0 <= O_new) & (O_new <= 1.0)), \
            f"Perturbation eta={eta} drove probabilities out of [0,1]"
        return O_new
    raise ValueError(f"Unsupported perturbation mode: {mode}")


def perturb_transition_model(
    T: np.ndarray, eta: float, *, mode: TransPerturbMode = "identity",
) -> np.ndarray:
    """Stub. Returns `T` unchanged when `mode == 'identity'` or eta == 0."""
    if mode == "identity" or eta == 0.0:
        return T
    raise ValueError(
        f"Unsupported transition perturbation mode: {mode} (only 'identity' supported)"
    )


def build_perturbed_tiger(model_nominal: TabularPOMDP,
                          eta_obs: float,
                          *,
                          eta_trans: float = 0.0
                          ) -> TabularPOMDP:
    """Convenience: produce a perturbed Tiger world with reduced listen accuracy.

    The reward matrix is left untouched.
    """
    O_pert = (model_nominal.O if eta_obs == 0.0
              else perturb_observation_model(model_nominal.O, eta_obs))
    T_pert = (model_nominal.T if eta_trans == 0.0
              else perturb_transition_model(model_nominal.T, eta_trans))
    return TabularPOMDP.from_matrices(T_pert, O_pert, model_nominal.R)
