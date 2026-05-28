"""
Certificate aggregator for paper Eq. 74 (Phase 6 sub-step 3).

For a fixed-policy projected tree (Phase 4's tree_evaluator), combines:
  - Delta_b (Phase 6's belief mismatch, closed form)
  - DeltaK_rob_split per action (Phase 5's DeltaKComputer)
  - phi_j (Phase 6 sub-step 2's inside-trajectory probabilities)

into the per-depth bound:

    cert_0 = R_max * delta_b
    cert_j = R_max * [delta_b + max-of-sum-of-delta_K-over-paths-to-depth-j + (1 - phi_j)]
             for j in 1..H

    certificate = Sum_{j=0..H} cert_j

The per-depth delta_K aggregator is **max-of-sum over root-to-depth-j paths**:
for each depth-j node in the tree, compute Sum_{k<j} delta_K_rob_split(pi(h_k))
along the root-to-node path, then take the max. Tighter than sum-of-max-per-
depth (which stitches deltas from non-co-occurring histories).

j=0 drops the leakage term because the depth-0 reward depends only on the
initial belief (no transitions, no observations have happened yet), so the
mismatch is bounded by R_max * delta_b alone. Including (1 - phi_0) on top
would double-count the out-of-support initial belief mass that's already in
delta_b.

R_max is auto-derived from model.R as max(|R.min()|, |R.max()|).

Aborted paths only contribute to max_path_sum at their abort depth (the path
ends there); they do not contribute to depths beyond, consistent with the
I_terminal = 1 convention in the leakage path.
"""

from __future__ import annotations

import numpy as np

from robust_pomdp import TabularPOMDP
from robust_pomdp.bounds.belief_mismatch import delta_b as compute_delta_b
from robust_pomdp.bounds.delta_k import DeltaKComputer
from robust_pomdp.bounds.inside_trajectory import compute_inside_traj_probs
from robust_pomdp.bounds.leakage import LeakageComputer
from robust_pomdp.core.tree import HistoryNode
from robust_pomdp.uncertainty.uncertainty_sets import UncertaintySets


def compute_certificate(root: HistoryNode,
                        node_data: dict,
                        model: TabularPOMDP,
                        uncertainty: UncertaintySets,
                        S_in: list[int],
                        Z_in: list[int],
                        b0_full: np.ndarray,
                        H: int,
                        ) -> dict:
    """Compute Eq. 74 certificate components and the total bound on
    |V_full - V_proj|. Returns dict with keys:
      - 'certificate': total upper bound.
      - 'delta_b': belief mismatch (closed form).
      - 'max_path_sum_delta_K': list of length H+1; index j is max over root-
            to-depth-j paths of Sum_{k<j} delta_K_rob_split(pi(h_k)).
      - 'phi': inside-trajectory probabilities, length H+1.
      - 'leakage_per_depth': [1 - phi_j], length H+1.
      - 'per_depth_bound': per-j contribution to certificate, length H+1.
      - 'R_max': auto-derived from model.R.
      - 'm_in': in-support belief mass.
    """
    b0_full = np.asarray(b0_full, dtype=np.float64)
    R_max = float(max(abs(model.R.min()), abs(model.R.max())))

    db = compute_delta_b(b0_full, S_in)
    m_in = float(sum(b0_full[s] for s in S_in))

    dk_computer = DeltaKComputer(model, uncertainty, S_in, Z_in)
    leakage_comp = LeakageComputer(model, uncertainty, S_in, Z_in)

    max_path_sum = _max_path_sum_delta_k(root, node_data, dk_computer, H)
    phi = compute_inside_traj_probs(root, node_data, leakage_comp, b0_full, H)
    leakage_per_depth = [1.0 - p for p in phi]

    per_depth_bound: list[float] = []
    for j in range(H + 1):
        if j == 0:
            per_depth_bound.append(R_max * db)
        else:
            per_depth_bound.append(
                R_max * (db + max_path_sum[j] + leakage_per_depth[j])
            )

    return {
        "certificate": float(sum(per_depth_bound)),
        "delta_b": db,
        "max_path_sum_delta_K": max_path_sum,
        "phi": phi,
        "leakage_per_depth": leakage_per_depth,
        "per_depth_bound": per_depth_bound,
        "R_max": R_max,
        "m_in": m_in,
    }


def _max_path_sum_delta_k(root: HistoryNode,
                          node_data: dict,
                          dk_computer: DeltaKComputer,
                          H: int,
                          ) -> list[float]:
    """For each j in 0..H, returns max over root-to-depth-j paths in the tree
    of Sum_{k=0..j-1} delta_K_rob_split(pi(h_k)).

    Aborted paths contribute only at their abort depth (path ends there).
    Indices that no path reaches stay at 0 (the empty-sum lower bound).
    """
    max_sum = [0.0] * (H + 1)

    def _walk(node: HistoryNode, current_sum: float) -> None:
        data = node_data[node.id]
        k = data.depth
        if current_sum > max_sum[k]:
            max_sum[k] = current_sum
        if data.is_terminal or k >= H:
            return
        a = data.policy_action
        contrib = dk_computer.delta_k_rob_split(a)
        action_node = node.children[a]
        for child in action_node.children.values():
            _walk(child, current_sum + contrib)

    _walk(root, 0.0)
    return max_sum
