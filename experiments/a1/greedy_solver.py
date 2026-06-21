"""
Full-robust-greedy solver for A1 (Phase A).

Builds a FULL-support expectimax tree (every history branches all actions, all
observations, full models), does a robust backup with max-over-actions, and
returns:
  - V_full: the optimal robust value at the root (= V^{pi_full}, the constant
    reference the certificate is measured against).
  - pi_full: the optimal robust policy, as a dict {obs_path_tuple -> action}
    along the greedy trajectory. obs_path = (z1, ..., zk) is the observation
    sequence from the root; along the greedy path the actions are determined by
    pi_full, so obs_path uniquely keys the history.

Robustness: reuses compute_robust_q per (history, action). With full support,
compute_robust_q uses the corrected full-TV-ball backup (full_support=True).

Cost: ~(|A_cont|*|Z|)^H histories (ABORT terminates, so |A_cont| excludes it).
One-time solve, reused across the projection sweep.
"""

from __future__ import annotations

import numpy as np

from robust_pomdp import TabularPOMDP
from robust_pomdp.core.tree import HistoryNode, ActionNode, NodeIdCounter
from robust_pomdp.solvers.robust_pomcp import compute_robust_q
from robust_pomdp.uncertainty.uncertainty_sets import UncertaintySets

from tree_evaluator import project_belief, projected_bayes_update


def solve_full_greedy(model: TabularPOMDP,
                      uncertainty: UncertaintySets,
                      b0_full: np.ndarray,
                      H: int,
                      *,
                      abort_action: int | None = None,
                      lp_cache: dict | None = None,
                      ) -> tuple[float, dict[tuple[int, ...], int]]:
    """Return (V_full, pi_full). pi_full maps obs-path tuples to the greedy
    robust action along the optimal trajectory."""
    if lp_cache is None:
        lp_cache = {}
    S_full = list(range(model.n_states))
    Z_full = list(range(model.n_obs))
    id_counter = NodeIdCounter()
    argmax_action: dict[int, int] = {}   # node.id -> greedy action

    def _solve(depth: int, belief: np.ndarray) -> tuple[float, HistoryNode]:
        node = HistoryNode(id_counter.next_id(), depth)
        node.S_in = set(S_full)
        q: dict[int, float] = {}
        for a in range(model.n_actions):
            if depth == H or a == abort_action:
                # Terminal action: immediate reward only, no future, no model
                # uncertainty (deterministic given the belief).
                q[a] = float(sum(belief[s] * model.R[s, a] for s in S_full))
            else:
                action_node = ActionNode(id_counter.next_id())
                action_node.Z_in = set(Z_full)
                for z in Z_full:
                    child_belief = projected_bayes_update(belief, model, a, z, S_full)
                    v_child, child = _solve(depth + 1, child_belief)
                    child.V_robust = v_child
                    action_node.children[z] = child
                node.children[a] = action_node
                q_val, _ = compute_robust_q(h=node, a=a, action_node=action_node,
                                            model=model, uncertainty=uncertainty,
                                            logger=None, backup_counter=0,
                                            previous_Q_robust=0.0, lp_cache=lp_cache,
                                            belief_override=np.asarray(belief, dtype=np.float64))
                q[a] = q_val
        best_a = max(q, key=q.get)
        argmax_action[node.id] = best_a
        node.V_robust = q[best_a]
        return q[best_a], node

    b0_in = project_belief(np.asarray(b0_full, dtype=np.float64), S_full)  # = b0 at full support
    v_full, root = _solve(0, b0_in)

    # Extract pi_full along the greedy trajectory, keyed by obs-path.
    pi_full: dict[tuple[int, ...], int] = {}

    def _walk(node: HistoryNode, obs_path: tuple[int, ...]) -> None:
        a = argmax_action[node.id]
        pi_full[obs_path] = a
        if a in node.children:   # continuing action -> descend its obs children
            for z, child in node.children[a].children.items():
                _walk(child, obs_path + (z,))

    _walk(root, ())
    return v_full, pi_full