"""
Robust Bellman + continuation-survival backup at one HistoryNode (children already
backed up).

Sets, per node:
  - each expanded action's `Q_rob` and continuation-survival `delta` (paper δ, Eq. 50);
  - `node.V_rob = max_a Q_rob`, `node.delta = greedy-action's delta` (the greedy
    policy the certificate follows).

Node/action cases:
  - frontier node (no expanded actions): V_rob = best immediate reward;
    delta[s] = 0  (own depth is exact; the entire unexpanded continuation leaks).
  - terminal action (a == abort_action, or depth == H-1): Q_rob = immediate
    reward; delta[s] = H - 1 - depth  (all remaining continuation depths survive,
    no leakage — a terminal action contributes 0 future reward). At depth H-1
    this is δ = 0 (Eq. 51).
  - interior action: robust_q value + leakage_delta over S_next.

The immediate reward is the action's exact `r_exact` at ROOT nodes (exact belief),
else the projected immediate over S_in; the continuation is Σ_{s∈S_in} belief[s]·σ[s]
from robust_q. Certificate (Eq. 49): eps = R_max·((H-1) - Σ_s b0(s)·δ_root(s)).

Convention: H reward steps at depths 0..H-1. `belief` is aligned with
sorted(node.S_in). `robust_q` and `leakage_delta` share `lp_cache`.
"""

from __future__ import annotations

import numpy as np

from Algorithm.core.pomdp import TabularPOMDP
from Algorithm.core.projected_tv_ball import ProjectedTVBall
from Algorithm.core.robust_q import robust_q
from Algorithm.core.tree import HistoryNode
from Algorithm.core.uncertainty import UncertaintySets
from Algorithm.ucb.leakage import leakage_delta


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
        node.delta = {int(s): 0.0 for s in S_in}          # unexpanded continuation all leaks
        return

    for a, an in node.children.items():
        # Root action nodes carry the EXACT immediate reward (exact belief there);
        # elsewhere the projected immediate over S_in.
        imm = an.r_exact if an.r_exact is not None else _immediate(model, a, belief, S_in)
        terminal = (abort_action is not None and a == abort_action) or (node.depth == H - 1)
        if terminal:
            an.Q_rob = imm
            an.delta = {int(s): float(H - 1 - node.depth) for s in S_in}  # continuation survives
        elif not an.children:                              # expanded, not yet deepened -> tail leaks
            an.Q_rob = imm
            an.delta = {int(s): 0.0 for s in S_in}
        else:                                              # interior
            Z_in = sorted(an.Z_in)
            S_next = sorted(an.S_next)
            V_child = {z: an.children[z].V_rob for z in Z_in}
            child_delta = {z: an.children[z].delta for z in Z_in}
            _, _, sigma = robust_q(model, uncertainty, a, belief, S_in, Z_in, S_next, V_child, lp_cache)
            an.Q_rob = imm + float(belief @ sigma)         # immediate + robust continuation
            an.delta = leakage_delta(model, uncertainty, a, S_in, Z_in, S_next, child_delta, lp_cache)

    best_a = max(node.children, key=lambda a: node.children[a].Q_rob)
    node.V_rob = node.children[best_a].Q_rob
    node.delta = dict(node.children[best_a].delta)
