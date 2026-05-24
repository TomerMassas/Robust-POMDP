"""
Tree builder + projected robust value evaluator for A1.

Reuses HistoryNode / ActionNode / NodeIdCounter from `core.tree` and
`compute_robust_q` from `solvers.robust_pomcp`. The fixed-policy tree builder
takes a deterministic history-conditional policy, an initial belief, and the
desired (S_in, Z_in) supports, then expands the full tree under the policy.
`evaluate_robust_value` runs a post-order backup using exact belief vectors
(via `belief_override` on `compute_robust_q`).

The same code handles both full evaluation (S_in = full S, Z_in = full Z) and
projected evaluation (S_in ⊂ S, Z_in ⊂ Z) — only the support args differ.
Branches with observations outside Z_in are simply not expanded; their absence
is what the certificate's leakage term accounts for (Phase 6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from robust_pomdp import TabularPOMDP
from robust_pomdp.core.tree import HistoryNode, ActionNode, NodeIdCounter
from robust_pomdp.solvers.robust_pomcp import compute_robust_q
from robust_pomdp.uncertainty.projected_sets import ProjectedTVBall
from robust_pomdp.uncertainty.uncertainty_sets import UncertaintySets


Policy = Callable[[int, "int | None", int], int]


@dataclass
class NodeData:
    """A1-specific metadata living parallel to the existing HistoryNode."""
    belief: np.ndarray   # length len(S_in), aligned with sorted(S_in)
    policy_action: int
    is_terminal: bool
    depth: int


def project_belief(b_full: np.ndarray, S_in_sorted: list[int]) -> np.ndarray:
    """Restrict a full-S belief to S_in and renormalize.

    Uniform fallback if all original mass lies outside S_in.
    """
    b_proj = np.asarray(b_full)[S_in_sorted]
    total = b_proj.sum()
    if total <= 0:
        return np.full(len(S_in_sorted), 1.0 / len(S_in_sorted))
    return b_proj / total


def projected_bayes_update(belief: np.ndarray,
                           model: TabularPOMDP,
                           a: int,
                           z: int,
                           S_in_sorted: list[int]
                           ) -> np.ndarray:
    """Projected Bayes update on the sampled support:
       b'(s') ∝ Σ_{s ∈ S_in} b(s) · T(s, a, s') · O(s', z),   for s' ∈ S_in.
    Returns a length-|S_in| renormalized vector. Uniform fallback if z is
    impossible under (belief, a) restricted to S_in.
    """
    T_proj = model.T[S_in_sorted, a, :][:, S_in_sorted]   # (|S_in|, |S_in|)
    O_proj = model.O[S_in_sorted, z]                      # (|S_in|,)
    new = (belief @ T_proj) * O_proj
    total = new.sum()
    if total <= 0:
        return np.full(len(S_in_sorted), 1.0 / len(S_in_sorted))
    return new / total


def build_fixed_policy_tree(model: TabularPOMDP,
                            policy: Policy,
                            b0_full: np.ndarray,
                            H: int,
                            S_in: list[int],
                            Z_in: list[int],
                            *,
                            abort_action: int | None = None,
                            id_counter: NodeIdCounter | None = None,
                            ) -> tuple[HistoryNode, dict[int, NodeData]]:
    """Build the full finite-horizon fixed-policy tree.

    Same code for full evaluation (S_in = full S, Z_in = full Z) and projected
    (S_in ⊂ S, Z_in ⊂ Z). The tree expands through depth H inclusive — depth-H
    nodes are terminal leaves whose V is the immediate expected reward of the
    policy's action at that depth.

    `abort_action` (e.g. ABORT in the synthetic POMDP) makes branches end
    early when the policy selects it. None disables that trigger.

    Returns (root, node_data) where node_data maps node.id -> NodeData.
    """
    if id_counter is None:
        id_counter = NodeIdCounter()
    S_in_sorted = sorted(S_in)
    Z_in_sorted = sorted(Z_in)
    M = model.n_obs
    node_data: dict[int, NodeData] = {}

    def _expand(depth: int, last_obs: int | None, belief: np.ndarray) -> HistoryNode:
        node = HistoryNode(id_counter.next_id(), depth)
        node.S_in = set(S_in_sorted)

        a = policy(depth, last_obs, M)
        is_terminal = (depth >= H) or (abort_action is not None and a == abort_action)
        node_data[node.id] = NodeData(belief=belief,
                                      policy_action=a,
                                      is_terminal=is_terminal,
                                      depth=depth)
        if is_terminal:
            return node

        action_node = ActionNode(id_counter.next_id())
        action_node.Z_in = set(Z_in_sorted)
        node.children[a] = action_node

        for z in Z_in_sorted:
            child_belief = projected_bayes_update(belief, model, a, z, S_in_sorted)
            child = _expand(depth + 1, z, child_belief)
            action_node.children[z] = child

        return node

    b0_in = project_belief(b0_full, S_in_sorted)
    root = _expand(depth=0, last_obs=None, belief=b0_in)
    return root, node_data


def evaluate_robust_value(root: HistoryNode,
                          node_data: dict[int, NodeData],
                          model: TabularPOMDP,
                          uncertainty: UncertaintySets,
                          *,
                          lp_cache: dict[int, ProjectedTVBall] | None = None,
                          ) -> float:
    """Post-order robust backup over a fixed-policy tree.

    Leaf: V = Σ_s b(s)·R(s, π(node)). No LP work.
    Interior: V = Q_robust(π(node)) via `compute_robust_q` with
              `belief_override = node_data[node.id].belief`.

    Pass an empty dict for `lp_cache` to share persistent LPs across the
    whole tree (recommended).
    """
    if lp_cache is None:
        lp_cache = {}
    return _eval(root, node_data, model, uncertainty, lp_cache)


def _eval(node: HistoryNode,
          node_data: dict[int, NodeData],
          model: TabularPOMDP,
          uncertainty: UncertaintySets,
          lp_cache: dict[int, ProjectedTVBall]
          ) -> float:
    data = node_data[node.id]
    S_in_sorted = sorted(node.S_in)

    if data.is_terminal:
        R_a = model.R[S_in_sorted, data.policy_action]
        v = float(np.dot(data.belief, R_a))
        node.V_robust = v
        return v

    action_node = node.children[data.policy_action]
    for child in action_node.children.values():
        _eval(child, node_data, model, uncertainty, lp_cache)

    q_val, _ = compute_robust_q(h=node,
                                a=data.policy_action,
                                action_node=action_node,
                                model=model,
                                uncertainty=uncertainty,
                                logger=None,
                                backup_counter=0,
                                previous_Q_robust=0.0,
                                lp_cache=lp_cache,
                                belief_override=data.belief)
    node.V_robust = q_val
    action_node.Q_robust = q_val
    return q_val
