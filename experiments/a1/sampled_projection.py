"""
Per-node sampled-projection tree builder (Phase 2 / B2-2).

Walks pi_full's tree; at each node samples a per-node support from the projected
belief: S_in(child_z) = K next-states drawn from the one-step posterior given z,
Z_in(ha) = K observations drawn from the obs marginal P(z|b,a). Sampling is WITH
replacement then deduped, so |support| <= K (peaked beliefs yield fewer unique).

Belief is projected-Bayes propagated over the per-node supports (root =
unnormalized restriction of b0 to S_in(root); children = posterior restricted to
S_in(child), renormalized). This is the belief that makes V_proj equal the
paper's V_hat (the renormalizer Sum_{s' in S_in(child)} post(s') is exactly the
in-support obs likelihood P_hat(z|b,a), so the value recursion's cancellation
holds), and it is what a real projected planner computes from samples.

Produces the same (root, node_data) structure as build_policy_tree_by_path, so
evaluate_robust_value and the per-node certificate run on it unchanged.
"""

from __future__ import annotations

import numpy as np

from robust_pomdp import TabularPOMDP
from robust_pomdp.core.tree import HistoryNode, ActionNode, NodeIdCounter
from tree_evaluator import NodeData


def build_sampled_projection_tree(model: TabularPOMDP,
                                  pi_full: dict,
                                  b0_full: np.ndarray,
                                  H: int,
                                  K: int,
                                  rng: np.random.Generator,
                                  *,
                                  abort_action: int | None = None,
                                  id_counter: NodeIdCounter | None = None,
                                  ) -> tuple[HistoryNode, dict[int, NodeData]]:
    if id_counter is None:
        id_counter = NodeIdCounter()
    N = model.n_states
    b0_full = np.asarray(b0_full, dtype=np.float64)
    node_data: dict[int, NodeData] = {}

    def _sample_support(probs: np.ndarray, k: int) -> list[int]:
        """Sample k indices ~ probs (normalized) WITH replacement, dedup, sort."""
        total = float(probs.sum())
        if total <= 0.0:
            return []
        draws = rng.choice(len(probs), size=k, replace=True, p=probs / total)
        return sorted(set(int(d) for d in draws))

    def _expand(depth: int, obs_path: tuple, S_in_sorted: list[int],
                belief_vec: np.ndarray) -> HistoryNode:
        # belief_vec aligned with S_in_sorted (root: unnormalized restriction of
        # b0; children: normalized over S_in).
        node = HistoryNode(id_counter.next_id(), depth)
        node.S_in = set(S_in_sorted)
        a = pi_full[obs_path]
        is_terminal = (depth >= H) or (abort_action is not None and a == abort_action)
        node_data[node.id] = NodeData(belief=belief_vec, policy_action=a,
                                      is_terminal=is_terminal, depth=depth)
        if is_terminal:
            return node

        b_full = np.zeros(N, dtype=np.float64)
        b_full[S_in_sorted] = belief_vec
        next_marginal = b_full @ model.T[:, a, :]     # Sum_s b(s) T(s,a,s')   (length N)
        obs_marginal = next_marginal @ model.O        # Sum_s' next(s') O(s',z) (length M)

        action_node = ActionNode(id_counter.next_id())
        node.children[a] = action_node
        Z_in = _sample_support(obs_marginal, K)
        action_node.Z_in = set(Z_in)

        for z in Z_in:
            post_full = next_marginal * model.O[:, z]   # Sum_s b(s) T(s,a,s') O(s',z)
            S_in_child = _sample_support(post_full, K)
            if not S_in_child:
                continue
            child_vec = post_full[S_in_child]
            child_vec = child_vec / child_vec.sum()     # renormalize over S_in(child)
            child = _expand(depth + 1, obs_path + (z,), S_in_child, child_vec)
            action_node.children[z] = child
        return node

    S_in_root = _sample_support(b0_full, K)
    root_vec = b0_full[S_in_root].copy()                # unnormalized restriction
    root = _expand(0, (), S_in_root, root_vec)
    return root, node_data