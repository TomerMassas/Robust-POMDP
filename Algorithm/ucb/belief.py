"""
Projected-Bayes beliefs (option B) — deterministic, over the sampled support.

  root:  b_hat_0 = restrict(b0, S_in(root))         UNNORMALIZED (paper Eq. 10)
  child: b_hat'(s') ∝ Σ_{s∈S_in} b_hat(s)·T(s,a,s')·O(s',z),  RENORMALIZED

Only the root is unnormalized (that carries the depth-0 out-of-support leakage);
renormalizing children keeps each observation branch's value conditional on z,
so the robust backup's observation weighting isn't double-counted.
"""

from __future__ import annotations

import numpy as np

from Algorithm.core.pomdp import TabularPOMDP


def restrict(b0: np.ndarray, S_in: list[int]) -> np.ndarray:
    """Unnormalized restriction of b0 to S_in (sorted). The root projected belief."""
    b0 = np.asarray(b0, dtype=np.float64)
    return b0[list(S_in)].copy()


def projected_bayes(belief: np.ndarray,
                    model: TabularPOMDP,
                    a: int,
                    z: int,
                    S_in_parent: list[int],
                    S_in_child: list[int],
                    ) -> np.ndarray:
    """Projected Bayes update onto the child support (renormalized).

    Args:
        belief: length-|S_in_parent| vector aligned with sorted S_in_parent.
        a, z: action taken and observation received.
        S_in_parent, S_in_child: sorted supports of the parent and child.

    Returns a length-|S_in_child| normalized vector aligned with sorted S_in_child.
    Uniform fallback if the in-support obs likelihood is 0.
    """
    belief = np.asarray(belief, dtype=np.float64)
    T_sub = model.T[list(S_in_parent)][:, a][:, list(S_in_child)]  # (|parent|, |child|)
    O_sub = model.O[list(S_in_child), z]                            # (|child|,)
    posterior = (belief @ T_sub) * O_sub
    total = posterior.sum()
    if total <= 0.0:
        return np.full(len(S_in_child), 1.0 / len(S_in_child))
    return posterior / total
