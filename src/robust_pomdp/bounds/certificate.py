"""
Certificate aggregator for paper Eq. 74 (Phase 6 sub-step 3).

For a fixed-policy projected tree (Phase 4's tree_evaluator), combines:
  - Delta_b (Phase 6's belief mismatch, closed form)
  - DeltaK_rob_split per action (Phase 5's DeltaKComputer)
  - phi_j (Phase 6 sub-step 2's inside-trajectory probabilities)

into the per-depth bound:

    cert_j = R_max * [max-of-sum-of-delta_K-over-paths-to-depth-j + (1 - phi_j)]
             for j in 0..H

    certificate = Sum_{j=0..H} cert_j

Delta_b = 0: the projected belief is the UNNORMALIZED restriction of b to the
support (b_hat = b on S_in), so the in-support belief mismatch vanishes. All
belief leakage is carried by the omitted-trajectory term (1 - phi_j), now
charged at every depth j=0..H. The j=0 term is exactly the initial out-of-
support mass (1 - phi_0 = 1 - m_in); at j=0 there is no kernel step so
max_path_sum[0] = 0.

The per-depth delta_K aggregator is **max-of-sum over root-to-depth-j paths**:
for each depth-j node in the tree, compute Sum_{k<j} delta_K_rob(pi(h_k))
along the root-to-node path, then take the max. Tighter than sum-of-max-per-
depth (which stitches deltas from non-co-occurring histories).

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
from robust_pomdp.bounds.delta_k_exact import DeltaKExactOrthant
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
                        *,
                        delta_k_mode: str = "split",
                        delta_k_max_free_bits: int = 16,
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
      - 'delta_b_contribution': total R_max-scaled delta_b mass in the
            certificate (charged at every depth j=0..H).
      - 'delta_K_contribution': total R_max-scaled kernel-mismatch mass
            (depths j=1..H).
      - 'leakage_contribution': total R_max-scaled omitted-trajectory mass
            (depths j=1..H). These three sum to 'certificate'.
      - 'R_max': auto-derived from model.R.
      - 'm_in': in-support belief mass.
    """
    b0_full = np.asarray(b0_full, dtype=np.float64)
    R_max = float(max(abs(model.R.min()), abs(model.R.max())))

    # In-support belief mismatch vs the projected root belief the tree actually
    # used (node_data[root.id].belief). Computed explicitly so it stays correct
    # under any projected-belief definition: 0 for the unnormalized restriction,
    # the renorm gap (1 - m_in) if renormalized. Out-of-support mass is leakage.
    db = compute_delta_b(b0_full, node_data[root.id].belief, S_in)
    m_in = float(sum(b0_full[s] for s in S_in))

    if delta_k_mode == "split":
        dk = DeltaKComputer(model, uncertainty, S_in, Z_in)
        delta_k_rob_fn = dk.delta_k_rob_split
    elif delta_k_mode == "exact":
        dk = DeltaKExactOrthant(model, uncertainty, S_in, Z_in,
                                max_free_bits=delta_k_max_free_bits)
        delta_k_rob_fn = dk.delta_k_rob_exact
    else:
        raise ValueError(f"delta_k_mode must be 'split' or 'exact', got {delta_k_mode!r}")
    leakage_comp = LeakageComputer(model, uncertainty, S_in, Z_in)

    max_path_sum = _max_path_sum_delta_k(root, node_data, delta_k_rob_fn, H)
    phi = compute_inside_traj_probs(root, node_data, leakage_comp, b0_full, H)
    leakage_per_depth = [1.0 - p for p in phi]

    # Per-depth bound. db (in-support belief mismatch) is charged at every depth
    # j=0..H (it propagates through the D_k recursion); leakage (1 - phi_j) also
    # at every j -- the j=0 leakage is the initial out-of-support mass 1 - m_in,
    # and at j=0 there is no kernel step so max_path_sum[0] = 0. Under the
    # unnormalized-restriction projected belief db == 0, so this reduces to
    # delta_K + leakage; db is kept in the formula so it stays correct if the
    # projected-belief definition changes.
    per_depth_bound: list[float] = [
        R_max * (db + max_path_sum[j] + leakage_per_depth[j]) for j in range(H + 1)
    ]

    # Component contributions (these three sum to 'certificate').
    delta_b_contribution = R_max * db * (H + 1)
    delta_K_contribution = R_max * float(sum(max_path_sum[1:]))  # max_path_sum[0] is 0
    leakage_contribution = R_max * float(sum(leakage_per_depth))  # all j incl. j=0

    return {
        "certificate": float(sum(per_depth_bound)),
        "delta_b": db,
        "max_path_sum_delta_K": max_path_sum,
        "phi": phi,
        "leakage_per_depth": leakage_per_depth,
        "per_depth_bound": per_depth_bound,
        "delta_b_contribution": delta_b_contribution,
        "delta_K_contribution": delta_K_contribution,
        "leakage_contribution": leakage_contribution,
        "R_max": R_max,
        "m_in": m_in,
    }


def _max_path_sum_delta_k(root: HistoryNode,
                          node_data: dict,
                          delta_k_rob_fn,
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
        contrib = delta_k_rob_fn(a)
        action_node = node.children[a]
        for child in action_node.children.values():
            _walk(child, current_sum + contrib)

    _walk(root, 0.0)
    return max_sum
