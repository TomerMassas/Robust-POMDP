"""
Discrete Bayes belief update for tabular POMDPs.

Used by the episode harness to refresh the agent's belief between successive
planner calls, given the action taken and the observation seen.

Formula:
    b'(s') ∝ O(s', z) · Σ_s T(s, a, s') · b(s)

If the denominator is zero (z is impossible under (b, a)), we fall back to the
prior over s' weighted by the transition kernel — guards against numerical edge
cases that should not happen with any well-formed POMDP.
"""

from __future__ import annotations

import numpy as np

from robust_pomdp.core.pomdp_types import TabularPOMDP


def bayes_update(
    belief: np.ndarray,
    model: TabularPOMDP,
    action: int,
    obs: int,
) -> np.ndarray:
    """Bayes filter step.

    Args:
        belief: probability vector over states, length n_states.
        model: TabularPOMDP supplying T and O.
        action: action taken.
        obs: observation received.

    Returns:
        New probability vector over states.
    """
    belief = np.asarray(belief, dtype=np.float64)

    # Predicted prior: pred[s'] = Σ_s T(s, a, s') b(s)
    pred = belief @ model.T[:, action, :]
    new_belief = model.O[:, obs] * pred

    Z = new_belief.sum()
    if Z > 0.0:
        return new_belief / Z

    # Pathological: observation impossible under (b, a). Fall back to predicted prior.
    Zp = pred.sum()
    return pred / Zp if Zp > 0.0 else pred
