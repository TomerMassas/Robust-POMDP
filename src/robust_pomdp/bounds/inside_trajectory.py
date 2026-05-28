"""
Inside-trajectory probability tree DP (Phase 6 sub-step 2, paper Eq. 73).

For a fixed-policy tree (Phase 4's tree_evaluator), computes phi_j for each
j in 0..H -- the worst-case probability that the trajectory stays inside
(S_in, Z_in) up to depth j, weighted by the initial belief restricted to
S_in (NOT renormalized):

    phi_j = inf over admissible model sequence of
              Sum_{s_0 in S_in} b_full(s_0 | h_0) *
              Sum over in-support (z_{1:j}, s_{1:j}) of
                  Prod_{k=0..j-1} P_T,k(s_{k+1} | a_k, s_k) * P_Z,k(z_{k+1} | s_{k+1})

Under rectangular uncertainty, the inf factors per (s, s_next, step), so we
run a backward DP via two nested TVBallLinearMinLP calls per node (inner over
P_Z per s_next, outer over P_T per s). One DP pass per j.

Terminal-node convention: at an abort-terminated h_k with k < j, set
I_k^j(s) = 1 for all s in S_in. Tight reward-agnostic choice (option (a)
from the design discussion): aborted paths have zero V mismatch at depth
j > k since the path ended -- they contribute 0 to leakage there, which
I = 1 -> (1 - I) = 0 encodes. The depth-k V mismatch at the abort node is
handled separately by the inside-mismatch path (Delta_b + sum Delta_K^rob),
not by phi_j.

Depth-H natural terminations need no convention: the recursion stops at
k = j, so depth-H leaves are only visited when j = H (base case k = j).
"""

from __future__ import annotations

import numpy as np

from robust_pomdp.bounds.leakage import LeakageComputer
from robust_pomdp.core.tree import HistoryNode


def compute_inside_traj_probs(root: HistoryNode,
                              node_data: dict,
                              leakage: LeakageComputer,
                              b0_full: np.ndarray,
                              H: int,
                              ) -> list[float]:
    """phi[j] for j in 0..H. b0_full is the full-S initial belief; the j=0
    value is m_in = Sum_{s in S_in} b_full(s) (NOT renormalized)."""
    S_in = leakage.S_in
    b0_full = np.asarray(b0_full, dtype=np.float64)
    if b0_full.shape != (leakage.model.n_states,):
        raise ValueError(f"b0_full shape {b0_full.shape} != ({leakage.model.n_states},)")

    phi: list[float] = [float(sum(b0_full[s] for s in S_in))]
    for j in range(1, H + 1):
        I_at_root = _backward_dp(root, node_data, leakage, j)
        phi.append(float(sum(b0_full[s] * I_at_root[s] for s in S_in)))
    return phi


def _backward_dp(node: HistoryNode,
                 node_data: dict,
                 leakage: LeakageComputer,
                 j: int,
                 ) -> dict[int, float]:
    """Compute I_k^j(s) for each s in S_in, where k = node.depth.
    Post-order DFS. Base case k = j returns 1; abort-terminal at k < j also
    returns 1 (option (a) convention)."""
    data = node_data[node.id]
    k = data.depth
    S_in = leakage.S_in
    Z_in = leakage.Z_in

    if k == j:
        return {s: 1.0 for s in S_in}

    if data.is_terminal:
        return {s: 1.0 for s in S_in}

    a = data.policy_action
    action_node = node.children[a]
    children_I = {z: _backward_dp(child, node_data, leakage, j)
                  for z, child in action_node.children.items()}

    # Inner: J(s_p) = min_{P_Z(.|s_p)} Sum_{z in Z_in} P_Z(z|s_p) * I_{k+1}^j(s_p; child_z).
    n_obs = leakage.model.n_obs
    J: dict[int, float] = {}
    for s_p in S_in:
        c = np.zeros(n_obs, dtype=np.float64)
        for z in Z_in:
            c[z] = children_I[z][s_p]
        J[s_p] = leakage.obs_min_over_TV_ball(s_p, c)

    # Outer: I(s) = min_{P_T(.|s, a)} Sum_{s' in S_in} P_T(s'|s, a) * J(s').
    n_states = leakage.model.n_states
    I: dict[int, float] = {}
    for s in S_in:
        c = np.zeros(n_states, dtype=np.float64)
        for s_p in S_in:
            c[s_p] = J[s_p]
        I[s] = leakage.state_min_over_TV_ball(s, a, c)

    return I
