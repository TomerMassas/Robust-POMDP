"""Phase 5 - DeltaKExactOrthant verification (ground-truth Delta_K^rob, Eq. 71).

Four independent checks:
  (a) rho=0 -> exact equals the closed-form nominal kernel mismatch (no optimizer).
  (b) exact <= split always (DeltaKComputer is a proven upper bound).
  (c) monotonic non-decreasing in rho_T and rho_Z.
  (d) gold check: brute-force grid on a 2x2 single-support instance (definitional
      projected set, shares no machinery with the LP) + random sampling at 3x3.

Run from anywhere:
    python experiments/a1/smoke_test/phase5_delta_k_exact_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(_SCRIPT.parents[3]))  # repo root

from robust_pomdp import TabularPOMDP
from robust_pomdp.bounds.delta_k import DeltaKComputer
from robust_pomdp.bounds.delta_k_exact import DeltaKExactOrthant
from robust_pomdp.uncertainty.uncertainty_sets import TVDistance, uniform_uncertainty


def build_tiny_pomdp() -> TabularPOMDP:
    """4 states, 2 actions, 3 obs (same as the existing phase5 smokes)."""
    T = np.array([
        [[0.70, 0.20, 0.10, 0.00], [0.10, 0.60, 0.20, 0.10]],
        [[0.10, 0.70, 0.20, 0.00], [0.20, 0.40, 0.30, 0.10]],
        [[0.05, 0.15, 0.60, 0.20], [0.10, 0.10, 0.60, 0.20]],
        [[0.00, 0.10, 0.30, 0.60], [0.00, 0.20, 0.40, 0.40]],
    ], dtype=np.float64)
    O = np.array([
        [0.70, 0.20, 0.10],
        [0.40, 0.40, 0.20],
        [0.20, 0.40, 0.40],
        [0.10, 0.30, 0.60],
    ], dtype=np.float64)
    R = np.zeros((4, 2), dtype=np.float64)
    return TabularPOMDP.from_matrices(T, O, R)


def check_a_rho_zero():
    """At rho=0 the full and projected one-step kernels coincide on the shared
    support (both nominal), so the in-support mismatch is exactly 0."""
    model = build_tiny_pomdp()
    N, A = model.n_states, model.n_actions
    S_in, Z_in = [0, 1], [0, 1]
    unc = uniform_uncertainty(N, A, 0.0, 0.0, TVDistance())
    exact = DeltaKExactOrthant(model, unc, S_in, Z_in)
    worst = 0.0
    for s in S_in:
        for a in range(A):
            got = exact.delta_k_exact(s, a)
            worst = max(worst, abs(got))
            assert abs(got) < 1e-9, f"(s={s},a={a}): rho=0 exact={got:.8f}, expected 0"
    print(f"  (a) rho=0 -> exact = 0 (in-support kernels identical)  OK (max |val| {worst:.2e})")


def check_b_exact_le_split():
    model = build_tiny_pomdp()
    N, A = model.n_states, model.n_actions
    S_in, Z_in = [0, 1], [0, 1]
    print("  (b) exact <= split:")
    for rho in (0.0, 0.1, 0.2, 0.4):
        unc = uniform_uncertainty(N, A, rho, rho, TVDistance())
        split = DeltaKComputer(model, unc, S_in, Z_in)
        exact = DeltaKExactOrthant(model, unc, S_in, Z_in)
        for a in range(A):
            for s in S_in:
                e, sp = exact.delta_k_exact(s, a), split.row(s, a)
                assert e <= sp + 1e-6, f"rho={rho} (s={s},a={a}): exact {e} > split {sp}"
            er, spr = exact.delta_k_rob_exact(a), split.delta_k_rob_split(a)
            assert er <= spr + 1e-6, f"rho={rho} a={a}: exact {er} > split {spr}"
            ratio = er / spr if spr > 1e-12 else 1.0
            print(f"        rho={rho:.1f} a={a}: exact={er:.4f}  split={spr:.4f}  "
                  f"(exact/split={ratio:.3f})")


def check_c_monotonic():
    model = build_tiny_pomdp()
    N, A = model.n_states, model.n_actions
    S_in, Z_in = [0, 1], [0, 1]
    # Monotone in rho_T (fix rho_Z=0.1).
    prev = -1.0
    for rho_t in (0.0, 0.1, 0.2, 0.3, 0.5):
        unc = uniform_uncertainty(N, A, rho_t, 0.1, TVDistance())
        v = DeltaKExactOrthant(model, unc, S_in, Z_in).delta_k_exact(0, 0)
        assert v >= prev - 1e-7, f"non-monotone in rho_T at {rho_t}: {v} < {prev}"
        prev = v
    # Monotone in rho_Z (fix rho_T=0.1).
    prev = -1.0
    for rho_z in (0.0, 0.1, 0.2, 0.3, 0.5):
        unc = uniform_uncertainty(N, A, 0.1, rho_z, TVDistance())
        v = DeltaKExactOrthant(model, unc, S_in, Z_in).delta_k_exact(0, 0)
        assert v >= prev - 1e-7, f"non-monotone in rho_Z at {rho_z}: {v} < {prev}"
        prev = v
    print("  (c) monotonic non-decreasing in rho_T and rho_Z  OK")


def check_d_bruteforce_2x2():
    """Gold check: 2x2, S_in=[0], Z_in=[0], source s=0. Brute-force 4 free scalars
    (t, q0, t', q0') over their 1-D ball intervals (definitional projected set)."""
    # Clean 2-state, 2-obs, 1-action model.
    T = np.zeros((2, 1, 2), dtype=np.float64)
    T[0, 0, :] = [0.6, 0.4]
    T[1, 0, :] = [0.3, 0.7]
    O = np.array([[0.7, 0.3], [0.2, 0.8]], dtype=np.float64)
    R = np.zeros((2, 1), dtype=np.float64)
    model = TabularPOMDP.from_matrices(T, O, R)

    rho_t, rho_z = 0.2, 0.2
    unc = uniform_uncertainty(2, 1, rho_t, rho_z, TVDistance())
    exact = DeltaKExactOrthant(model, unc, S_in=[0], Z_in=[0]).delta_k_exact(0, 0)

    # 1-D intervals (TV ball: |x - x0| <= rho/2, clipped to [0,1]).
    def interval(x0, rho, G):
        lo, hi = max(0.0, x0 - rho / 2), min(1.0, x0 + rho / 2)
        return np.linspace(lo, hi, G)

    G = 41
    t = interval(0.6, rho_t, G)
    q0 = interval(0.7, rho_z, G)
    tp = interval(0.6, rho_t, G)    # t' = P_T^in(0), independent full-ball restriction
    q0p = interval(0.7, rho_z, G)   # q0' = P_Z^in(0|0), independent
    T_, Q0_, TP_, Q0P_ = np.meshgrid(t, q0, tp, q0p, indexing="ij")
    # In-support sum is the single entry (z=0, s'=0): |t*q0 - t'*q0'|.
    norm = np.abs(T_ * Q0_ - TP_ * Q0P_)
    brute = float(norm.max())

    assert brute <= exact + 1e-6, f"grid {brute} exceeds exact {exact}"
    assert exact <= brute + 0.02, f"exact {exact} far above grid {brute} (G={G})"
    print(f"  (d) 2x2 gold: exact={exact:.6f}  brute-grid(G={G})={brute:.6f}  "
          f"(gap {exact-brute:.2e})  OK")


def _sample_in_ball(p_o, rho, rng):
    """Sample a full distribution in TV-ball(p_o, rho): interpolate nominal toward
    a random simplex point, scaled to stay within the L1 budget."""
    n = len(p_o)
    p = rng.dirichlet(np.ones(n))
    d = float(np.sum(np.abs(p - p_o)))
    lam = rng.uniform(0.0, 1.0 if d <= rho else rho / d)
    return (1 - lam) * p_o + lam * p


def check_d_random_3x3(n_samples=200_000, seed=0):
    """Random-sampling cross-check at 3x3, S_in=[0,1], Z_in=[0,1]. One-sided:
    every sampled feasible (full, projected) pair must give norm <= exact."""
    T = np.zeros((3, 1, 3), dtype=np.float64)
    T[0, 0, :] = [0.5, 0.3, 0.2]
    T[1, 0, :] = [0.2, 0.5, 0.3]
    T[2, 0, :] = [0.1, 0.3, 0.6]
    O = np.array([[0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]], dtype=np.float64)
    R = np.zeros((3, 1), dtype=np.float64)
    model = TabularPOMDP.from_matrices(T, O, R)
    S_in, Z_in = [0, 1], [0, 1]
    rho_t, rho_z = 0.2, 0.2
    unc = uniform_uncertainty(3, 1, rho_t, rho_z, TVDistance())
    exact = DeltaKExactOrthant(model, unc, S_in, Z_in).delta_k_exact(0, 0)

    rng = np.random.default_rng(seed)
    s = 0
    best = 0.0
    for _ in range(n_samples):
        P_T = _sample_in_ball(model.T[s, 0, :], rho_t, rng)
        P_T_ext = _sample_in_ball(model.T[s, 0, :], rho_t, rng)
        norm = 0.0
        # In-support sum only: s' in S_in, z in Z_in. P_T/P_Z are full dists;
        # we use only their in-support components.
        for s_p in S_in:
            P_Z = _sample_in_ball(model.O[s_p, :], rho_z, rng)
            P_Z_ext = _sample_in_ball(model.O[s_p, :], rho_z, rng)
            for z in Z_in:
                k = P_T[s_p] * P_Z[z]
                k_hat = P_T_ext[s_p] * P_Z_ext[z]
                norm += abs(k - k_hat)
        best = max(best, norm)
    assert best <= exact + 1e-6, f"random sample {best} exceeds exact {exact}"
    print(f"  (d) 3x3 random ({n_samples} samples): exact={exact:.6f}  "
          f"sampled_max={best:.6f}  (gap {exact-best:.4f})  OK")


def main():
    print("DeltaKExactOrthant verification:")
    check_a_rho_zero()
    check_b_exact_le_split()
    check_c_monotonic()
    check_d_bruteforce_2x2()
    check_d_random_3x3()
    print("\nAll DeltaKExactOrthant checks passed.")


if __name__ == "__main__":
    main()