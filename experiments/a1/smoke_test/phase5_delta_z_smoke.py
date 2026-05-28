"""Phase 5 - step 3 - delta_z_inside_max MILP smoke.

Verifies:
- rho_Z = 0: returns 0 (fast path).
- rho_Z very small (forces MILP path): returns ~0.
- Uniform P_Z_o, Z_in = full Z, rho_Z in [0.1, ..., 0.8]: returns 2*rho_Z exactly
  (diameter of joint TV-ball; symmetric-opposite construction is feasible within
  the simplex up to rho_Z = 4/n for uniform P on n elements).
- Non-uniform P_Z_o at large rho_Z: simplex positivity clips the diameter below
  2*rho_Z. Only range-checked here (0 <= r <= 2*rho_Z); orthant cross-check in
  step 4 will verify mid-range values exactly.

Run from anywhere:
    python experiments/a1/smoke_test/phase5_delta_z_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from robust_pomdp.bounds.l1_max import ProjectedL1MaxMILP


def main() -> None:

    milp_partial = ProjectedL1MaxMILP(n=3, support=[0])
    r = milp_partial.solve(np.array([0.05, 0.5, 0.45]), rho=0.3)

    M = 5
    p_skewed = np.array([0.05, 0.15, 0.6, 0.15, 0.05])

    assert abs(p_skewed.sum() - 1.0) < 1e-12

    # 1. rho=0 (fast path).
    milp_partial = ProjectedL1MaxMILP(n=M, support=[0, 1, 2])
    r = milp_partial.solve(p_skewed, rho=0.0)
    assert r == 0.0
    print(f"rho=0, Z_in=[0,1,2]: delta_z_inside_max = {r:.6f}  (fast path)")

    # 2. rho very small (forces MILP path): returns ~0.
    r = milp_partial.solve(p_skewed, rho=1e-9)
    assert r < 1e-6, f"expected ~0 at rho=1e-9, got {r}"
    print(f"rho=1e-9, Z_in=[0,1,2]: delta_z_inside_max = {r:.3e}  (MILP path, ~0)")

    # 3. Uniform P, Z_in = full Z, rho in [0.1, ..., 0.8]: diameter = 2*rho exactly.
    #    Range 0.8 is the largest where the simplex positivity constraint doesn't
    #    clip the symmetric-opposite construction for n=5.
    p_uniform = np.full(M, 1.0 / M)
    milp_full = ProjectedL1MaxMILP(n=M, support=list(range(M)))
    for rho in [0.1, 0.2, 0.4, 0.6, 0.8]:
        r = milp_full.solve(p_uniform, rho=rho)
        expected = 2 * rho
        assert abs(r - expected) < 1e-6, f"rho={rho} expected {expected}, got {r}"
    print(f"Uniform P, Z_in=full at rho in [0.1, 0.2, 0.4, 0.6, 0.8]: delta_z = 2*rho")

    # 4. Non-uniform P at large rho: positivity clips below 2*rho. Range-check only.
    #    (Exact mid-range values cross-checked via orthant enumeration in step 4.)
    for rho in [0.5, 1.0]:
        r = milp_full.solve(p_skewed, rho=rho)
        assert 0.0 <= r <= 2 * rho + 1e-6, f"out of range at rho={rho}: r = {r}"
        print(f"Non-uniform P, rho={rho}: delta_z = {r:.6f}  (in [0, {2*rho:.1f}])")

    # 5. Reuse with uniform P at small rho - verify LP state cleanly resets.
    r = milp_full.solve(p_uniform, rho=0.3)
    assert abs(r - 0.6) < 1e-6, f"reuse uniform rho=0.3: expected 0.6, got {r}"
    print(f"reuse: p_uniform rho=0.3: delta_z = {r:.6f}  (= 2*rho)")

    print("delta_z_inside_max MILP OK (sanity values; orthant cross-check pending step 4)")


if __name__ == "__main__":
    main()