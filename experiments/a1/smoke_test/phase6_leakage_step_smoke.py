"""Phase 6 sub-step 1 - leakage primitives smoke.

Verifies wZ_in_min and robust_inside_prob_step at closed-form configs:

wZ_in_min checks:
- rho_Z = 0: equals nominal in-support mass.
- Z_in = full Z: equals 1.0 for any rho (mass can't leave the obs space).
- Partial Z_in, rho_Z > 0: equals max(0, baseline - rho_Z/2).
- Low-baseline clipping at 0.

robust_inside_prob_step checks:
- rho_T = rho_Z = 0: equals nominal one-step inside prob.
- Full support (S_in=S, Z_in=Z): equals 1.0 for any rho.
- General: robust value <= nominal (worst-case is below or equal).

Run from anywhere:
    python experiments/a1/smoke_test/phase6_leakage_step_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from robust_pomdp import TabularPOMDP
from robust_pomdp.bounds.leakage import LeakageComputer
from robust_pomdp.bounds.tv_ball_lp import TVBallLinearMinLP
from robust_pomdp.uncertainty.uncertainty_sets import TVDistance, uniform_uncertainty


def build_tiny_pomdp() -> TabularPOMDP:
    """4 states, 2 actions, 3 observations. (Same as Phase 5 delta_k smoke.)"""
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


def main() -> None:
    model = build_tiny_pomdp()
    N, A, M = model.n_states, model.n_actions, model.n_obs

    # ============ wZ_in_min checks ============

    # 1. rho_Z = 0: nominal baseline.
    uncertainty_zero = uniform_uncertainty(N, A, 0.0, 0.0, TVDistance())
    leak_zero = LeakageComputer(model, uncertainty_zero, S_in=[0, 1], Z_in=[0, 1])
    for s_next in range(N):
        expected = float(model.O[s_next, 0] + model.O[s_next, 1])
        got = leak_zero.wZ_in_min(s_next)
        assert abs(got - expected) < 1e-9, (
            f"rho_Z=0 s_next={s_next}: expected {expected}, got {got}"
        )
    print("  rho_Z=0: wZ_in_min(s_next) == nominal baseline for all s_next  OK")

    # 2. Z_in = full Z: returns 1.0 always.
    uncertainty = uniform_uncertainty(N, A, 0.0, 0.2, TVDistance())
    leak_full = LeakageComputer(model, uncertainty, S_in=[0, 1], Z_in=list(range(M)))
    for s_next in range(N):
        got = leak_full.wZ_in_min(s_next)
        assert abs(got - 1.0) < 1e-9, (
            f"Z_in=full s_next={s_next}: expected 1.0, got {got}"
        )
    print("  Z_in=full Z, rho_Z=0.2: wZ_in_min(s_next) = 1.0 for all s_next  OK")

    # 3. Partial Z_in, rho_Z > 0: closed-form max(0, baseline - rho_Z/2).
    leak_partial = LeakageComputer(model, uncertainty, S_in=[0, 1], Z_in=[0, 1])
    for s_next in range(N):
        baseline = float(model.O[s_next, 0] + model.O[s_next, 1])
        expected = max(0.0, baseline - 0.1)            # rho_Z/2 = 0.1
        got = leak_partial.wZ_in_min(s_next)
        assert abs(got - expected) < 1e-6, (
            f"s_next={s_next}: expected {expected:.6f}, got {got:.6f}"
        )
        print(f"  Z_in=[0,1], s_next={s_next}, rho_Z=0.2: wZ_in_min = {got:.6f} "
              f"(baseline={baseline:.2f} - 0.1)")

    # 4. Low-baseline clipping (use the raw LP to construct the edge case).
    lp = TVBallLinearMinLP(n=3)
    p = np.array([0.05, 0.50, 0.45], dtype=np.float64)
    c = np.array([1.0, 0.0, 0.0], dtype=np.float64)    # indicator(Z_in={0})
    got = lp.solve(p, rho=0.2, c=c)
    expected = max(0.0, 0.05 - 0.1)                    # = 0 (clipped)
    assert abs(got - expected) < 1e-9, f"low baseline: expected {expected}, got {got}"
    print(f"  Low baseline p[0]=0.05, rho_Z=0.2: wZ_in_min clipped to {got:.6f} (= 0)  OK")

    # ============ robust_inside_prob_step checks ============

    # 5. rho_T = rho_Z = 0: equals nominal one-step inside prob.
    leak_zero_step = LeakageComputer(model, uncertainty_zero, S_in=[0, 1], Z_in=[0, 1])
    for s in [0, 1]:
        for a in range(A):
            nominal = sum(
                float(model.T[s, a, s_p]) * float(model.O[s_p, 0] + model.O[s_p, 1])
                for s_p in [0, 1]
            )
            got = leak_zero_step.robust_inside_prob_step(s, a)
            assert abs(got - nominal) < 1e-9, (
                f"rho=0 (s,a)=({s},{a}): expected {nominal}, got {got}"
            )
    print("  rho_T=rho_Z=0: robust_inside_prob_step(s,a) == nominal for all (s,a)  OK")

    # 6. Full support: returns 1.0 (mass can't leave the universe).
    leak_full_support = LeakageComputer(
        model, uncertainty, S_in=list(range(N)), Z_in=list(range(M))
    )
    for s in range(N):
        for a in range(A):
            got = leak_full_support.robust_inside_prob_step(s, a)
            assert abs(got - 1.0) < 1e-9, (
                f"full support (s,a)=({s},{a}): expected 1.0, got {got}"
            )
    print("  Full (S_in=S, Z_in=Z): robust_inside_prob_step = 1.0 for all (s,a)  OK")

    # 7. General: robust value <= nominal.
    uncertainty_mid = uniform_uncertainty(N, A, 0.1, 0.1, TVDistance())
    leak_mid = LeakageComputer(model, uncertainty_mid, S_in=[0, 1], Z_in=[0, 1])
    print()
    print("  General (rho_T=rho_Z=0.1, S_in=[0,1], Z_in=[0,1]):")
    for s in [0, 1]:
        for a in range(A):
            nominal = sum(
                float(model.T[s, a, s_p]) * float(model.O[s_p, 0] + model.O[s_p, 1])
                for s_p in [0, 1]
            )
            got = leak_mid.robust_inside_prob_step(s, a)
            assert got <= nominal + 1e-9, (
                f"(s,a)=({s},{a}): robust {got:.6f} > nominal {nominal:.6f}"
            )
            print(f"    (s={s}, a={a}): robust={got:.6f}  nominal={nominal:.6f}  "
                  f"(robust <= nominal)")

    print("\nLeakage primitives OK (closed-form + monotonicity checks)")


if __name__ == "__main__":
    main()
