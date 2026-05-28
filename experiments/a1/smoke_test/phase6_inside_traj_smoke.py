"""Phase 6 sub-step 2 - inside-trajectory tree DP smoke.

Builds the A1 synthetic POMDP + fixed-policy tree, then verifies
compute_inside_traj_probs against:

1. rho=0 + tree with aborts (full Z_in -> z=2 triggers ABORT at depth 1):
   backward DP matches a nominal forward DP that freezes aborted mass.
2. Full support (S_in=S, Z_in=Z), rho > 0: phi_j = 1.0 for all j.
3. Partial support, rho > 0: worst-case <= nominal.
4. Monotonicity: phi_{j+1} <= phi_j.

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

from experiments.a1.synthetic_pomdp import ABORT, make_initial_belief, make_synthetic_pomdp
from experiments.a1.fixed_policy import warning_threshold_policy
from experiments.a1.tree_evaluator import build_fixed_policy_tree


def compute_nominal_traj_probs(root, node_data, leakage, b0_full, H) -> list[float]:
    """Nominal forward DP -- aborted paths' mass is frozen (added to phi_j for
    all j > abort_depth), matching option (a)'s I_terminal = 1 convention."""
    S_in = leakage.S_in
    model = leakage.model
    b0_full = np.asarray(b0_full, dtype=np.float64)

    phi: list[float] = [float(sum(b0_full[s] for s in S_in))]
    weights: dict[int, dict[int, float]] = {root.id: {s: float(b0_full[s]) for s in S_in}}
    frozen_through: list[float] = [0.0] * (H + 1)
    nodes_at_depth: dict[int, list] = {0: [root]}

    for k in range(H):
        nodes_at_depth[k + 1] = []
        for node in nodes_at_depth[k]:
            data = node_data[node.id]
            if data.is_terminal:
                frozen_here = sum(weights[node.id][s] for s in S_in)
                for j_fut in range(k + 1, H + 1):
                    frozen_through[j_fut] += frozen_here
                continue
            a = data.policy_action
            action_node = node.children[a]
            for z, child in action_node.children.items():
                w_child = {}
                for s_p in S_in:
                    val = 0.0
                    for s in S_in:
                        val += weights[node.id][s] * model.T[s, a, s_p] * model.O[s_p, z]
                    w_child[s_p] = val
                weights[child.id] = w_child
                nodes_at_depth[k + 1].append(child)
        alive = sum(sum(weights[n.id][s] for s in S_in) for n in nodes_at_depth[k + 1])
        phi.append(float(alive + frozen_through[k + 1]))
    return phi


def main() -> None:
    N, M, H = 4, 3, 3
    model = make_synthetic_pomdp(N=N, M=M)
    b0 = make_initial_belief(N=N)
    A = model.n_actions
    S_full = list(range(N))
    Z_full = list(range(M))

    # ---- Test 1: rho=0, full Z_in (z=2 triggers abort): backward DP = forward DP ----
    root_a, nd_a = build_fixed_policy_tree(
        model, warning_threshold_policy, b0, H, S_in=[0, 1], Z_in=Z_full,
        abort_action=ABORT,
    )
    unc_zero = uniform_uncertainty(N, A, 0.0, 0.0, TVDistance())
    leak_zero = LeakageComputer(model, unc_zero, S_in=[0, 1], Z_in=Z_full)
    phi_back = compute_inside_traj_probs(root_a, nd_a, leak_zero, b0, H)
    phi_fwd = compute_nominal_traj_probs(root_a, nd_a, leak_zero, b0, H)
    for j in range(H + 1):
        assert abs(phi_back[j] - phi_fwd[j]) < 1e-9, (
            f"rho=0 j={j}: backward={phi_back[j]:.9f}, forward={phi_fwd[j]:.9f}"
        )
    print("  rho=0 (S_in=[0,1], Z_in=full -> aborts present): backward DP == nominal forward DP")
    print(f"        backward: {[f'{p:.6f}' for p in phi_back]}")
    print(f"        forward:  {[f'{p:.6f}' for p in phi_fwd]}")

    # ---- Test 2: full support: phi_j = 1.0 for all j ----
    root_f, nd_f = build_fixed_policy_tree(
        model, warning_threshold_policy, b0, H, S_in=S_full, Z_in=Z_full,
        abort_action=ABORT,
    )
    unc_full = uniform_uncertainty(N, A, 0.2, 0.2, TVDistance())
    leak_full = LeakageComputer(model, unc_full, S_in=S_full, Z_in=Z_full)
    phi_full = compute_inside_traj_probs(root_f, nd_f, leak_full, b0, H)
    for j, p in enumerate(phi_full):
        assert abs(p - 1.0) < 1e-9, f"full support phi_{j} = {p}, expected 1.0"
    print(f"  full support, rho=0.2: phi = {[f'{p:.6f}' for p in phi_full]}  (all 1.0)")

    # ---- Test 3: partial S_in + partial Z_in, rho > 0: robust <= nominal ----
    root_p, nd_p = build_fixed_policy_tree(
        model, warning_threshold_policy, b0, H, S_in=[0, 1], Z_in=[0, 1],
        abort_action=ABORT,
    )
    unc_p = uniform_uncertainty(N, A, 0.1, 0.1, TVDistance())
    leak_p = LeakageComputer(model, unc_p, S_in=[0, 1], Z_in=[0, 1])
    phi_robust = compute_inside_traj_probs(root_p, nd_p, leak_p, b0, H)
    phi_nom = compute_nominal_traj_probs(root_p, nd_p, leak_p, b0, H)
    for j in range(H + 1):
        assert phi_robust[j] <= phi_nom[j] + 1e-9, (
            f"j={j}: robust {phi_robust[j]:.6f} > nominal {phi_nom[j]:.6f}"
        )
    print("  partial (S_in=[0,1], Z_in=[0,1]), rho=0.1:")
    print(f"        robust:  {[f'{p:.6f}' for p in phi_robust]}")
    print(f"        nominal: {[f'{p:.6f}' for p in phi_nom]}  (robust <= nominal)")

    # ---- Test 4: monotonicity: phi_{j+1} <= phi_j ----
    for j in range(H):
        assert phi_robust[j + 1] <= phi_robust[j] + 1e-9, (
            f"monotonicity: phi_{j+1}={phi_robust[j+1]:.6f} > phi_{j}={phi_robust[j]:.6f}"
        )
    print("  monotonicity: phi non-increasing in j  OK")

    print("\nInside-trajectory tree DP OK (rho=0 backward=forward w/ aborts + full + sanity + mono)")


if __name__ == "__main__":
    main()
