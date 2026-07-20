"""
Full-support robust expectimax — optimal robust value + frozen greedy policy.

Built on Algorithm.core.robust_q + Algorithm.ucb.belief (NOT a copy of
a1/greedy_solver), so conventions match the new code: H reward depths 0..H-1,
terminal at depth H-1, ABORT terminal, and at full support robust_q reduces to
the true robust Bellman backup.

Returns:
  V_full             : optimal robust value at the root.
  q_root[a]          : Q_full^rob(h0, a) = value of "take a at root, then greedy".
  pi_by_root_action  : {a: pi_a}, where pi_a is the FROZEN policy "take a at the
                       root, then argmax_Q_rob below", keyed by observation-path.

The freeze is the point: extract the greedy ONCE here, then evaluate pi_a on any
projected support (fixed_policy_tree) — never re-derive greedy per tree, or the
full and projected trees would follow different policies and the certificate
would not apply.
"""

from __future__ import annotations

import numpy as np

from Algorithm.core.pomdp import TabularPOMDP
from Algorithm.core.robust_q import robust_q
from Algorithm.core.tree import ActionNode, HistoryNode, NodeIdCounter
from Algorithm.core.uncertainty import UncertaintySets
from Algorithm.ucb.belief import projected_bayes


def solve_full(model: TabularPOMDP,
               uncertainty: UncertaintySets,
               b0: np.ndarray,
               H: int,
               abort_action: int | None = None,
               lp_cache: dict | None = None,
               ) -> tuple[float, dict[int, float], dict[int, dict[tuple, int]]]:
    """Return (V_full, q_root, pi_by_root_action). Full-support robust expectimax."""
    if lp_cache is None:
        lp_cache = {}
    S_all = list(range(model.n_states))
    Z_all = list(range(model.n_obs))
    ctr = NodeIdCounter()
    argmax_action: dict[int, int] = {}

    def _solve(depth: int, belief: np.ndarray) -> tuple[dict[int, float], HistoryNode]:
        node = HistoryNode(ctr.next_id(), depth)
        node.S_in = set(S_all)
        q: dict[int, float] = {}
        for a in range(model.n_actions):
            terminal = (depth == H - 1) or (abort_action is not None and a == abort_action)
            if terminal:
                q[a] = float(sum(belief[s] * model.R[s, a] for s in S_all))
            else:
                an = ActionNode(ctr.next_id())
                an.Z_in = set(Z_all)
                an.S_next = set(S_all)
                for z in Z_all:
                    child_belief = projected_bayes(belief, model, a, z, S_all, S_all)
                    q_child, child = _solve(depth + 1, child_belief)
                    child.V_rob = max(q_child.values())
                    an.children[z] = child
                node.children[a] = an
                V_child = {z: an.children[z].V_rob for z in Z_all}
                q[a], _, _ = robust_q(model, uncertainty, a, belief,
                                      S_all, Z_all, S_all, V_child, lp_cache)
        best = max(q, key=q.get)
        argmax_action[node.id] = best
        node.V_rob = q[best]
        return q, node

    b0 = np.asarray(b0, dtype=np.float64)
    q_root, root = _solve(0, b0)
    V_full = max(q_root.values())
    pi_by_root_action = {a: _extract_policy(root, a, argmax_action) for a in q_root}
    return V_full, q_root, pi_by_root_action


def _extract_policy(root: HistoryNode,
                    a: int,
                    argmax_action: dict[int, int],
                    ) -> dict[tuple, int]:
    """Frozen 'take a at root, then argmax below' policy, keyed by observation-path.

    pi[()] = a; deeper entries follow the full-solve argmax. A terminal greedy
    action (abort / depth H-1) has no expanded children, so the walk stops there.
    """
    pi_a: dict[tuple, int] = {(): a}
    if a in root.children:
        def _walk(node: HistoryNode, obs_path: tuple) -> None:
            best = argmax_action[node.id]
            pi_a[obs_path] = best
            if best in node.children:
                for z, child in node.children[best].children.items():
                    _walk(child, obs_path + (z,))
        for z, child in root.children[a].children.items():
            _walk(child, (z,))
    return pi_a
