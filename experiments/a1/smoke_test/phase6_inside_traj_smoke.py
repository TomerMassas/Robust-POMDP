"""Inside-trajectory tree DP smoke (per-node support).

Verifies compute_inside_traj_probs against an independent per-node nominal
forward DP and structural properties:

1. rho=0, uniform tree with aborts: backward DP == nominal forward DP.
2. Full support, rho>0: phi_j = 1.0 for all j.
3. Partial (uniform) support, rho>0: worst-case <= nominal; monotone in j.
4. PER-NODE varying support (gold check): manually shrink S_in / Z_in per node,
   then backward DP == per-node nominal forward DP at rho=0. Exercises the
   per-child next-state union and the (z in Z_in AND s' in S_in(child_z)) mask
   that uniform support never reaches.

Run from anywhere:
    python experiments/a1/smoke_test/phase6_inside_traj_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(_SCRIPT.parents[1]))  # experiments/a1/
sys.path.insert(0, str(_SCRIPT.parents[3]))  # repo root

from robust_pomdp.bounds.inside_trajectory import compute_inside_traj_probs
from robust_pomdp.bounds.leakage import LeakageComputer
from robust_pomdp.uncertainty.uncertainty_sets import TVDistance, uniform_uncertainty

from experiments.a1.synthetic_pomdp import (ABORT, PROCEED,
                                             make_initial_belief, make_synthetic_pomdp)
from experiments.a1.tree_evaluator import build_policy_tree_by_path


def _const_policy(obs_path):
    return PROCEED


def nominal_forward(root, node_data, model, b0_full, H) -> list[float]:
    """Per-node nominal (rho=0) forward DP. weights[node.id][s] = nominal prob of
    reaching (node, s) in-support. Terminal/abort mass is frozen and counted for
    all future j (matches the backward DP's I_terminal = 1 convention). Reads
    per-node S_in(h) = node.S_in and Z_in(ha) = action_node.Z_in."""
    b0_full = np.asarray(b0_full, dtype=np.float64)
    root_S_in = sorted(root.S_in)
    phi = [float(sum(b0_full[s] for s in root_S_in))]
    weights = {root.id: {s: float(b0_full[s]) for s in root_S_in}}
    frozen_through = [0.0] * (H + 1)
    nodes_at_depth = {0: [root]}

    for k in range(H):
        nodes_at_depth[k + 1] = []
        for node in nodes_at_depth[k]:
            data = node_data[node.id]
            S_in = sorted(node.S_in)
            if data.is_terminal:
                frozen_here = sum(weights[node.id][s] for s in S_in)
                for j_fut in range(k + 1, H + 1):
                    frozen_through[j_fut] += frozen_here
                continue
            a = data.policy_action
            action_node = node.children[a]
            for z in sorted(action_node.Z_in):
                child = action_node.children[z]
                child_S_in = sorted(child.S_in)
                w_child = {}
                for s_p in child_S_in:
                    val = 0.0
                    for s in S_in:
                        val += weights[node.id][s] * model.T[s, a, s_p] * model.O[s_p, z]
                    w_child[s_p] = val
                weights[child.id] = w_child
                nodes_at_depth[k + 1].append(child)
        alive = sum(sum(weights[n.id][s] for s in sorted(n.S_in))
                    for n in nodes_at_depth[k + 1])
        phi.append(float(alive + frozen_through[k + 1]))
    return phi


def _shrink_supports(root, node_data):
    """Create per-node variation: at depth>=1 drop the highest state from S_in;
    at depth>=2 drop another; at depth>=1 drop the highest obs from Z_in (+ its
    child). Keeps every support non-empty. Mutates the tree in place."""
    def _walk(node):
        data = node_data[node.id]
        d = data.depth
        S = sorted(node.S_in)
        if d >= 1 and len(S) > 1:
            node.S_in = set(S[:-1])
        if d >= 2 and len(node.S_in) > 1:
            node.S_in = set(sorted(node.S_in)[:-1])
        if data.is_terminal:
            return
        an = node.children[data.policy_action]
        Z = sorted(an.Z_in)
        if d >= 1 and len(Z) > 1:
            drop = Z[-1]
            an.Z_in = set(Z[:-1])
            an.children.pop(drop, None)
        for child in an.children.values():
            _walk(child)
    _walk(root)


def main() -> None:
    N, M, H = 4, 3, 3
    model = make_synthetic_pomdp(N=N, M=M)
    b0 = make_initial_belief(N=N)
    A = model.n_actions
    S_full = list(range(N))
    Z_full = list(range(M))

    # ---- Test 1: rho=0, uniform tree with aborts: backward == forward ----
    root_a, nd_a = build_policy_tree_by_path(
        model, _const_policy, b0, H, S_in=[0, 1], Z_in=Z_full, abort_action=ABORT)
    leak0 = LeakageComputer(model, uniform_uncertainty(N, A, 0.0, 0.0, TVDistance()))
    phi_back = compute_inside_traj_probs(root_a, nd_a, leak0, b0, H)
    phi_fwd = nominal_forward(root_a, nd_a, model, b0, H)
    for j in range(H + 1):
        assert abs(phi_back[j] - phi_fwd[j]) < 1e-9, (
            f"rho=0 j={j}: backward={phi_back[j]:.9f}, forward={phi_fwd[j]:.9f}")
    print(f"  (1) rho=0 uniform w/ aborts: backward == forward  OK  {[f'{p:.4f}' for p in phi_back]}")

    # ---- Test 2: full support, rho>0: phi_j = 1.0 ----
    root_f, nd_f = build_policy_tree_by_path(
        model, _const_policy, b0, H, S_in=S_full, Z_in=Z_full, abort_action=ABORT)
    leak_full = LeakageComputer(model, uniform_uncertainty(N, A, 0.2, 0.2, TVDistance()))
    phi_full = compute_inside_traj_probs(root_f, nd_f, leak_full, b0, H)
    for j, p in enumerate(phi_full):
        assert abs(p - 1.0) < 1e-9, f"full support phi_{j} = {p}, expected 1.0"
    print(f"  (2) full support, rho=0.2: phi all 1.0  OK")

    # ---- Test 3: partial (uniform) support, rho>0: robust <= nominal, monotone ----
    root_p, nd_p = build_policy_tree_by_path(
        model, _const_policy, b0, H, S_in=[0, 1], Z_in=[0, 1], abort_action=ABORT)
    leak_p = LeakageComputer(model, uniform_uncertainty(N, A, 0.1, 0.1, TVDistance()))
    phi_rob = compute_inside_traj_probs(root_p, nd_p, leak_p, b0, H)
    phi_nom = nominal_forward(root_p, nd_p, model, b0, H)
    for j in range(H + 1):
        assert phi_rob[j] <= phi_nom[j] + 1e-9, f"j={j}: robust {phi_rob[j]} > nominal {phi_nom[j]}"
    for j in range(H):
        assert phi_rob[j + 1] <= phi_rob[j] + 1e-9, "phi not monotone in j"
    print(f"  (3) partial uniform, rho=0.1: robust <= nominal, monotone  OK")

    # ---- Test 4 (GOLD): per-node varying support, rho=0: backward == forward ----
    root_v, nd_v = build_policy_tree_by_path(
        model, _const_policy, b0, H, S_in=S_full, Z_in=Z_full, abort_action=ABORT)
    _shrink_supports(root_v, nd_v)
    leak0v = LeakageComputer(model, uniform_uncertainty(N, A, 0.0, 0.0, TVDistance()))
    phi_back_v = compute_inside_traj_probs(root_v, nd_v, leak0v, b0, H)
    phi_fwd_v = nominal_forward(root_v, nd_v, model, b0, H)
    for j in range(H + 1):
        assert abs(phi_back_v[j] - phi_fwd_v[j]) < 1e-9, (
            f"per-node rho=0 j={j}: backward={phi_back_v[j]:.9f}, forward={phi_fwd_v[j]:.9f}")
    # phi_0 = full root mass (root support untouched) = 1; deeper < 1 (shrinkage leaks)
    assert abs(phi_back_v[0] - 1.0) < 1e-9, f"per-node phi_0 = {phi_back_v[0]}, expected 1"
    assert phi_back_v[H] < phi_back_v[0] - 1e-9, "per-node shrinkage should reduce phi with depth"
    print(f"  (4) per-node varying support, rho=0: backward == forward  OK")
    print(f"        backward: {[f'{p:.4f}' for p in phi_back_v]}")
    print(f"        forward:  {[f'{p:.4f}' for p in phi_fwd_v]}")

    print("\nInside-trajectory tree DP OK (uniform regression + per-node gold check)")


if __name__ == "__main__":
    main()