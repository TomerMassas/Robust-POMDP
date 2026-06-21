"""
Certificate aggregator for the robust projected-value bound (leakage-only form).

For a fixed-policy projected tree (tree_evaluator), combines:
  - Delta_b (belief mismatch; 0 under the unnormalized-restriction projected belief)
  - phi_j (inside-trajectory probabilities, per-node support, inside_trajectory.py)

into the per-depth bound:

    cert_j = R_max * [ delta_b + (1 - phi_j) ]   for j in 0..H
    certificate = Sum_{j=0..H} cert_j

This is the adjusted theory: the inside-support kernel-mismatch term (Delta_K) is
gone -- all approximation error is carried by the omitted-trajectory / leakage
term (1 - phi_j). The j=0 term (1 - phi_0 = 1 - m_in) is the initial out-of-
support belief mass.

Delta_b = 0 under the unnormalized-restriction projected belief (b_hat = b on the
support); it is computed explicitly (not hardcoded) so it stays correct if the
belief definition changes. R_max is auto-derived from model.R.

NOTE: the legacy kernel-mismatch path (delta_k.py / delta_k_exact.py) is no
longer used here; those modules are kept standalone for reference only.
"""

from __future__ import annotations

import numpy as np

from robust_pomdp import TabularPOMDP
from robust_pomdp.bounds.belief_mismatch import delta_b as compute_delta_b
from robust_pomdp.bounds.inside_trajectory import compute_inside_traj_probs
from robust_pomdp.bounds.leakage import LeakageComputer
from robust_pomdp.core.tree import HistoryNode
from robust_pomdp.uncertainty.uncertainty_sets import UncertaintySets


def compute_certificate(root: HistoryNode,
                        node_data: dict,
                        model: TabularPOMDP,
                        uncertainty: UncertaintySets,
                        b0_full: np.ndarray,
                        H: int,
                        ) -> dict:
    """Compute the leakage-only certificate components and the total bound on
    |V_full - V_proj|. Reads per-node supports (S_in(h), Z_in(ha)) from the tree.
    Returns dict with keys:
      - 'certificate': total upper bound.
      - 'delta_b': in-support belief mismatch (0 under unnormalized restriction).
      - 'phi': inside-trajectory probabilities, length H+1.
      - 'leakage_per_depth': [1 - phi_j], length H+1.
      - 'per_depth_bound': per-j contribution to certificate, length H+1.
      - 'delta_b_contribution': total R_max-scaled delta_b mass (all depths).
      - 'leakage_contribution': total R_max-scaled omitted-trajectory mass.
            delta_b_contribution + leakage_contribution = certificate.
      - 'R_max': auto-derived from model.R.
      - 'm_in': in-support belief mass at the root.
    """
    b0_full = np.asarray(b0_full, dtype=np.float64)
    R_max = float(max(abs(model.R.min()), abs(model.R.max())))

    # In-support belief mismatch vs the projected root belief the tree actually
    # used (node_data[root.id].belief), over the ROOT's own support. Computed
    # explicitly so it stays correct under any projected-belief definition:
    # 0 for the unnormalized restriction, the renorm gap (1 - m_in) if renormalized.
    root_S_in = sorted(root.S_in)
    db = compute_delta_b(b0_full, node_data[root.id].belief, root_S_in)
    m_in = float(sum(b0_full[s] for s in root_S_in))

    leakage_comp = LeakageComputer(model, uncertainty)
    phi = compute_inside_traj_probs(root, node_data, leakage_comp, b0_full, H)
    leakage_per_depth = [1.0 - p for p in phi]

    # Per-depth bound: db charged at every depth j=0..H; leakage (1 - phi_j) at
    # every j (the j=0 leakage is the initial out-of-support mass 1 - m_in).
    per_depth_bound: list[float] = [
        R_max * (db + leakage_per_depth[j]) for j in range(H + 1)
    ]

    delta_b_contribution = R_max * db * (H + 1)
    leakage_contribution = R_max * float(sum(leakage_per_depth))

    return {
        "certificate": float(sum(per_depth_bound)),
        "delta_b": db,
        "phi": phi,
        "leakage_per_depth": leakage_per_depth,
        "per_depth_bound": per_depth_bound,
        "delta_b_contribution": delta_b_contribution,
        "leakage_contribution": leakage_contribution,
        "R_max": R_max,
        "m_in": m_in,
    }