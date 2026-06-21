"""
Inside-trajectory probability tree DP (paper Eq. 27 / 21, omitted-trajectory term).

For a fixed-policy tree (tree_evaluator), computes phi_j for each j in 0..H --
the worst-case probability that the trajectory stays inside the sampled support
up to depth j, weighted by the initial belief restricted to the root support
(NOT renormalized):

    phi_j = inf over admissible model sequence of
              Sum_{s_0 in S_in(h_0)} b_full(s_0 | h_0) *
              Sum over in-support (z_{1:j}, s_{1:j}) of
                  Prod_{k=0..j-1} P_T,k(s_{k+1} | a_k, s_k) * P_Z,k(z_{k+1} | s_{k+1})

PER-NODE SUPPORT: the support varies per node -- S_in(h_k) = node.S_in,
Z_in(h_k a_k) = action_node.Z_in. A trajectory step (s -> s', observe z) from
h_k is IN-SUPPORT iff z in Z_in(h_k a_k) AND s' in S_in(child_z) (the support
of the child history reached by z). Uniform support is the special case where
every node shares the same S_in / Z_in.

Under rectangular uncertainty the inf factors per (s, s', step), so we run a
backward DP via two nested full-TV-ball minimizations per node (inner over P_Z
per next-state s', outer over P_T per current-state s). The per-node / per-child
support is encoded as zero coefficients (out-of-support entries contribute 0).
One DP pass per j.

Terminal-node convention: at an abort-terminated h_k with k < j, I_k^j(s) = 1
for all s in S_in(h_k). Tight reward-agnostic choice: aborted paths have zero V
mismatch at depth j > k (the path ended), so they contribute 0 to leakage there,
which I = 1 -> (1 - I) = 0 encodes. Depth-H natural terminations need no
convention (the recursion stops at k = j, base case I = 1).
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
    """phi[j] for j in 0..H. b0_full is the full-S initial belief; the j=0 value
    is m_in = Sum_{s in S_in(root)} b_full(s) (NOT renormalized)."""
    b0_full = np.asarray(b0_full, dtype=np.float64)
    if b0_full.shape != (leakage.model.n_states,):
        raise ValueError(f"b0_full shape {b0_full.shape} != ({leakage.model.n_states},)")

    root_S_in = sorted(root.S_in)
    phi: list[float] = [float(sum(b0_full[s] for s in root_S_in))]
    for j in range(1, H + 1):
        I_at_root = _backward_dp(root, node_data, leakage, j)
        phi.append(float(sum(b0_full[s] * I_at_root[s] for s in root_S_in)))
    return phi


def _backward_dp(node: HistoryNode,
                 node_data: dict,
                 leakage: LeakageComputer,
                 j: int,
                 ) -> dict[int, float]:
    """Compute I_k^j(s) for each s in S_in(h_k), where k = node.depth.
    Post-order DFS. Base case k = j returns 1; abort-terminal at k < j also
    returns 1 (the I_terminal = 1 convention)."""
    data = node_data[node.id]
    k = data.depth
    S_in = sorted(node.S_in)

    if k == j or data.is_terminal:
        return {s: 1.0 for s in S_in}

    a = data.policy_action
    action_node = node.children[a]
    Z_in = sorted(action_node.Z_in)
    children_I = {z: _backward_dp(child, node_data, leakage, j)
                  for z, child in action_node.children.items()}

    # Next-state support reachable IN-support via some observation in Z_in.
    next_states: set[int] = set()
    for z in Z_in:
        next_states.update(children_I[z].keys())   # keys = S_in(child_z)
    next_states_sorted = sorted(next_states)

    # Inner: J(s') = min_{P_Z(.|s')} Sum_{z in Z_in, s' in S_in(child_z)}
    #                                  P_Z(z|s') * I_{k+1}^j(s'; child_z).
    n_obs = leakage.model.n_obs
    J: dict[int, float] = {}
    for s_p in next_states_sorted:
        c = np.zeros(n_obs, dtype=np.float64)
        for z in Z_in:
            child_I = children_I[z]
            if s_p in child_I:          # (z, s') in-support; else coefficient 0 (leak)
                c[z] = child_I[s_p]
        J[s_p] = leakage.obs_min_over_TV_ball(s_p, c)

    # Outer: I(s) = min_{P_T(.|s, a)} Sum_{s'} P_T(s'|s, a) * J(s'), s' over
    # next_states (0 elsewhere -> leaking to out-of-support next-states is free).
    n_states = leakage.model.n_states
    I: dict[int, float] = {}
    for s in S_in:
        c = np.zeros(n_states, dtype=np.float64)
        for s_p in next_states_sorted:
            c[s_p] = J[s_p]
        I[s] = leakage.state_min_over_TV_ball(s, a, c)

    return I