"""
Certificate helpers — the Theorem-1 leakage bound and the per-root-action bracket
used for CR-UCT's anytime early stop (CERTIFIED_BEST_ACTION in pseudocode.md).

Canonical home for the bound  eps = R_max * (H - sum_{s in S_in} b0(s)*c[s])  and
R_max; the post-hoc experiment (fixed_policy_tree) imports from here so the math
lives in one place.

The bracket [Q_rob(a) - eps_a, Q_rob(a) + eps_a] uses the ACTION node's survival
c (an.c), the root's sampled support, and the TRUE initial belief b0 (unnormalized
restriction is implicit: states outside S_in(root) contribute 0).
"""

from __future__ import annotations

import numpy as np

from Algorithm.core.pomdp import TabularPOMDP
from Algorithm.core.tree import ActionNode, HistoryNode


def r_max(model: TabularPOMDP) -> float:
    """R_max = max(|R.min|, |R.max|), per the paper."""
    return float(max(abs(model.R.min()), abs(model.R.max())))


def eps_for_c(c: dict[int, float],
              b0: np.ndarray,
              S_in,
              H: int,
              R_max: float,
              ) -> float:
    """Theorem-1 leakage bound  eps = R_max * (H - sum_{s in S_in} b0[s]*c[s])."""
    b0 = np.asarray(b0, dtype=np.float64)
    survived = sum(b0[s] * c[s] for s in S_in)
    return R_max * (H - survived)


def action_bracket(an: ActionNode,
                   b0: np.ndarray,
                   S_in_root,
                   H: int,
                   R_max: float,
                   ) -> tuple[float, float, float]:
    """(L, U, eps) for one root action: [Q_rob - eps, Q_rob + eps] with eps from an.c."""
    eps = eps_for_c(an.c, b0, S_in_root, H, R_max)
    return an.Q_rob - eps, an.Q_rob + eps, eps


def certified_best_action(root: HistoryNode,
                          b0: np.ndarray,
                          H: int,
                          R_max: float,
                          n_actions: int,
                          ) -> int | None:
    """Return the certified best root action, or None if not yet certifiable.

    Certifiable only once every root action is expanded and one action's lower
    bound clears every other action's upper bound (CERTIFIED_BEST_ACTION).
    """
    if len(root.children) < n_actions:                 # not all actions expanded
        return None
    S_in = sorted(root.S_in)
    L: dict[int, float] = {}
    U: dict[int, float] = {}
    for a, an in root.children.items():
        L[a], U[a], _ = action_bracket(an, b0, S_in, H, R_max)
    a_star = max(L, key=L.get)
    if all(L[a_star] >= U[a] for a in U if a != a_star):
        return a_star
    return None
