"""Phase 2 / B2-2 - per-node sampled-projection tree builder smoke.

V1. Reproducibility: same seed -> identical tree (V_proj, cert, per-node supports).
V2. Dense-model reduction: on a DENSE POMDP (all T,O > 0), large K samples the
    full support at every node -> V_proj == V_full and cert == 0.
V3. Validity (key): on the sparse synthetic POMDP, cert >= true_error across K/seeds.
V4. Monotone trend: larger K -> larger avg support fraction -> smaller cert (mean).

Run from anywhere:
    python experiments/a1/smoke_test/phaseB2_sampler_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(_SCRIPT.parents[1]))  # experiments/a1/
sys.path.insert(0, str(_SCRIPT.parents[3]))  # repo root

from robust_pomdp import TabularPOMDP
from robust_pomdp.bounds.certificate import compute_certificate
from robust_pomdp.uncertainty.uncertainty_sets import TVDistance, uniform_uncertainty

from synthetic_pomdp import ABORT, make_initial_belief, make_synthetic_pomdp
from tree_evaluator import evaluate_robust_value
from greedy_solver import solve_full_greedy
from sampled_projection import build_sampled_projection_tree


def _avg_support_fraction(node_data, N, M):
    """Mean over history nodes of |S_in|/N, and over action-supported nodes of
    |Z_in|/M (returns the average of the two means)."""
    s_fracs, z_fracs = [], []
    # node_data holds depth/belief; supports live on the tree nodes -- but we
    # only need a summary, so recompute from belief lengths and is_terminal.
    for nd in node_data.values():
        s_fracs.append(len(nd.belief) / N)
    return float(np.mean(s_fracs))


def _cert(root, node_data, model, unc, b0, H):
    """Leakage-only certificate + V_proj on a (per-node) tree following pi_full."""
    res = compute_certificate(root, node_data, model, unc, b0, H)
    v_proj = evaluate_robust_value(root, node_data, model, unc)
    return v_proj, res["certificate"]


def test_v1_reproducibility():
    model = make_synthetic_pomdp(N=6, M=4)
    b0 = make_initial_belief(N=6)
    unc = uniform_uncertainty(6, model.n_actions, 0.1, 0.1, TVDistance())
    V_full, pi_full = solve_full_greedy(model, unc, b0, 3, abort_action=ABORT)
    r1, n1 = build_sampled_projection_tree(model, pi_full, b0, 3, K=3,
                                           rng=np.random.default_rng(7), abort_action=ABORT)
    r2, n2 = build_sampled_projection_tree(model, pi_full, b0, 3, K=3,
                                           rng=np.random.default_rng(7), abort_action=ABORT)
    v1, c1 = _cert(r1, n1, model, unc, b0, 3)
    v2, c2 = _cert(r2, n2, model, unc, b0, 3)
    assert abs(v1 - v2) < 1e-12 and abs(c1 - c2) < 1e-12, "same seed must reproduce"
    assert len(n1) == len(n2), "same seed must give same tree size"
    print(f"  V1 reproducibility: seed 7 -> identical (V_proj={v1:.4f}, cert={c1:.4f}, "
          f"{len(n1)} nodes)  OK")


def test_v2_dense_reduction():
    # Dense POMDP: every T, O entry > 0. Large K -> full support -> cert = 0.
    T = np.array([
        [[0.5, 0.3, 0.2], [0.2, 0.5, 0.3]],
        [[0.3, 0.4, 0.3], [0.3, 0.3, 0.4]],
        [[0.2, 0.3, 0.5], [0.4, 0.3, 0.3]],
    ], dtype=np.float64)
    O = np.array([[0.6, 0.4], [0.5, 0.5], [0.3, 0.7]], dtype=np.float64)
    R = np.array([[1.0, -1.0], [0.0, 2.0], [-1.0, 1.0]], dtype=np.float64)
    model = TabularPOMDP.from_matrices(T, O, R)
    b0 = np.array([0.4, 0.3, 0.3], dtype=np.float64)
    unc = uniform_uncertainty(3, model.n_actions, 0.15, 0.15, TVDistance())
    H = 3
    V_full, pi_full = solve_full_greedy(model, unc, b0, H, abort_action=None)
    root, nd = build_sampled_projection_tree(model, pi_full, b0, H, K=4000,
                                             rng=np.random.default_rng(0), abort_action=None)
    v_proj, cert = _cert(root, nd, model, unc, b0, H)
    assert abs(v_proj - V_full) < 1e-6, f"dense large-K: V_proj={v_proj} != V_full={V_full}"
    assert abs(cert) < 1e-6, f"dense large-K: cert={cert}, expected 0"
    print(f"  V2 dense reduction (K=4000): V_proj={v_proj:.4f} == V_full={V_full:.4f}, "
          f"cert={cert:.2e}  OK")


def test_v3_validity():
    model = make_synthetic_pomdp(N=8, M=4)
    b0 = make_initial_belief(N=8)
    unc = uniform_uncertainty(8, model.n_actions, 0.1, 0.1, TVDistance())
    H = 3
    V_full, pi_full = solve_full_greedy(model, unc, b0, H, abort_action=ABORT)
    worst = -1.0
    for seed in range(8):
        for K in (2, 3, 5):
            root, nd = build_sampled_projection_tree(model, pi_full, b0, H, K=K,
                                                     rng=np.random.default_rng(seed),
                                                     abort_action=ABORT)
            v_proj, cert = _cert(root, nd, model, unc, b0, H)
            true_err = abs(V_full - v_proj)
            assert true_err <= cert + 1e-6, (
                f"seed={seed} K={K}: true_error {true_err:.4f} > cert {cert:.4f}")
            worst = max(worst, true_err - cert)
    print(f"  V3 validity (8 seeds x K in 2,3,5): cert >= true_error always  OK "
          f"(worst margin {worst:.4f})")


def test_v4_monotone_trend():
    model = make_synthetic_pomdp(N=8, M=4)
    b0 = make_initial_belief(N=8)
    unc = uniform_uncertainty(8, model.n_actions, 0.1, 0.1, TVDistance())
    H = 3
    V_full, pi_full = solve_full_greedy(model, unc, b0, H, abort_action=ABORT)
    print("  V4 monotone trend (mean over 12 seeds):")
    prev_frac, prev_cert = -1.0, 1e18
    for K in (1, 2, 4, 8):
        fracs, certs = [], []
        for seed in range(12):
            root, nd = build_sampled_projection_tree(model, pi_full, b0, H, K=K,
                                                     rng=np.random.default_rng(100 + seed),
                                                     abort_action=ABORT)
            _, cert = _cert(root, nd, model, unc, b0, H)
            fracs.append(_avg_support_fraction(nd, 8, 4))
            certs.append(cert)
        mf, mc = float(np.mean(fracs)), float(np.mean(certs))
        print(f"        K={K}: avg_support_frac={mf:.3f}  mean_cert={mc:.3f}")
        assert mf >= prev_frac - 1e-9, "avg support fraction should grow with K"
        assert mc <= prev_cert + 1e-9, "mean cert should shrink with K"
        prev_frac, prev_cert = mf, mc


def main():
    print("Per-node sampled-projection builder smoke:")
    test_v1_reproducibility()
    test_v2_dense_reduction()
    test_v3_validity()
    test_v4_monotone_trend()
    print("\nB2-2 sampled-projection builder OK.")


if __name__ == "__main__":
    main()