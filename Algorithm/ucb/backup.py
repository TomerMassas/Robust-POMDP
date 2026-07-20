"""
Robust Bellman + survival backup at one HistoryNode (children already backed up).

Sets, per node:
  - each expanded action's `Q_rob` and survival `c` (dict s -> value);
  - `node.V_rob = max_a Q_rob`, `node.c = greedy-action's c` (the greedy policy
    the certificate follows).

Node/action cases (the whole leakage story):
  - frontier node (no expanded actions): V_rob = best immediate reward;
    c[s] = 1  (survives its own depth; the unexpanded tail leaks).
  - terminal action (a == abort_action, or depth == H-1): Q_rob = immediate
    reward; c[s] = H - depth  (survives all remaining depths, no leakage —
    valid because a terminal action contributes 0 future reward).
  - interior action: robust_q + survival_c over S_next.

Convention: H reward steps at depths 0..H-1; certificate sums depths 0..H-1.
`belief` is aligned with sorted(node.S_in). `robust_q` and `survival_c` share
`lp_cache`.
"""

from __future__ import annotations

import numpy as np

from Algorithm.core.pomdp import TabularPOMDP
from Algorithm.core.projected_tv_ball import ProjectedTVBall
from Algorithm.core.robust_q import robust_q
from Algorithm.core.tree import HistoryNode
from Algorithm.core.uncertainty import UncertaintySets
from Algorithm.ucb.leakage import leakage_c


def _immediate(model: TabularPOMDP, a: int, belief: np.ndarray, S_in: list[int]) -> float:
    """Σ_s belief[s] · R[s, a] over the (sorted) support."""
    return float(sum(belief[i] * model.R[s, a] for i, s in enumerate(S_in)))


def backup(node: HistoryNode,
           belief: np.ndarray,
           model: TabularPOMDP,
           uncertainty: UncertaintySets,
           H: int,
           abort_action: int | None,
           lp_cache: dict[int, ProjectedTVBall] | None = None,
           ) -> None:
    """Backup `node` in place. Children must already be backed up (post-order)."""
    if lp_cache is None:
        lp_cache = {}
    S_in = sorted(node.S_in)
    belief = np.asarray(belief, dtype=np.float64)

    if node.is_frontier():
        node.V_rob = max(_immediate(model, a, belief, S_in) for a in range(model.n_actions))
        node.c = {int(s): 1.0 for s in S_in}
        return

    for a, an in node.children.items():
        terminal = (abort_action is not None and a == abort_action) or (node.depth == H - 1)
        if terminal:
            an.Q_rob = _immediate(model, a, belief, S_in)
            an.c = {int(s): float(H - node.depth) for s in S_in}
        elif not an.children:                              # expanded, not yet deepened -> tail leaks
            an.Q_rob = _immediate(model, a, belief, S_in)
            an.c = {int(s): 1.0 for s in S_in}
        else:                                              # interior
            Z_in = sorted(an.Z_in)
            S_next = sorted(an.S_next)
            V_child = {z: an.children[z].V_rob for z in Z_in}
            child_c = {z: an.children[z].c for z in Z_in}
            Q, _, _ = robust_q(model, uncertainty, a, belief, S_in, Z_in, S_next, V_child, lp_cache)
            an.Q_rob = Q
            an.c = leakage_c(model, uncertainty, a, S_in, Z_in, S_next, child_c, lp_cache)

    best_a = max(node.children, key=lambda a: node.children[a].Q_rob)
    node.V_rob = node.children[best_a].Q_rob
    node.c = dict(node.children[best_a].c)
