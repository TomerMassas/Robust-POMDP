
"""Phase 5 - step 2 - wZ_max LP smoke.

Verifies:
- rho=0 returns Sum_{z in Z_in} P_Z_o(z) exactly.
- rho > 0 with feasible expansion returns baseline + rho/2 (since each unit
  of mass moved costs 2 in our no-1/2 TV convention).
- rho large enough saturates at 1.0.
- Z_in = full Z always returns 1.0 regardless of rho.
- Z_in = [] returns 0.0.

Run from anywhere:
    python experiments/a1/smoke_test/phase5_wz_max_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from robust_pomdp.bounds.wz_max import WZMaxLP


def main() -> None:
    M = 5
    p_nominal = np.array([0.05, 0.15, 0.6, 0.15, 0.05])
    assert abs(p_nominal.sum() - 1.0) < 1e-12

    lp = WZMaxLP(n=M)

    # 1. rho=0: returns exact baseline.
    Z_in = [0, 1]
    baseline = p_nominal[0] + p_nominal[1]            # = 0.2
    r = lp.solve(p_nominal, rho=0.0, Z_in=Z_in)
    assert abs(r - baseline) < 1e-9, f"rho=0 expected {baseline}, got {r}"
    print(f"rho=0,  Z_in={Z_in}: wZ_max = {r:.6f}  (baseline = {baseline:.6f})")

    # 2. rho=0.2: baseline + rho/2 = 0.3 (mass-outside is 0.8, feasible push).
    r = lp.solve(p_nominal, rho=0.2, Z_in=Z_in)
    assert abs(r - 0.30) < 1e-9, f"rho=0.2 expected 0.30, got {r}"
    print(f"rho=0.2, Z_in={Z_in}: wZ_max = {r:.6f}  (baseline + rho/2)")

    # 3. rho=2.0: would push 1.0 inside, clamped at 1.0.
    r = lp.solve(p_nominal, rho=2.0, Z_in=Z_in)
    assert abs(r - 1.0) < 1e-9, f"rho=2 expected 1.0 (saturated), got {r}"
    print(f"rho=2.0, Z_in={Z_in}: wZ_max = {r:.6f}  (saturated)")

    # 4. Z_in = full Z: always 1.0.
    for rho in [0.0, 0.1, 1.0]:
        r = lp.solve(p_nominal, rho=rho, Z_in=list(range(M)))
        assert abs(r - 1.0) < 1e-9, f"Z_in=full at rho={rho}: expected 1.0, got {r}"
    print(f"Z_in=full at rho in [0, 0.1, 1.0]: wZ_max = 1.0 (always)")

    # 5. Z_in = []: returns 0.0.
    r = lp.solve(p_nominal, rho=0.5, Z_in=[])
    assert r == 0.0
    print(f"Z_in=[]: wZ_max = 0.0")

    # 6. Reuse the same solver instance for a different p_nominal - verify LP
    #    state is correctly reset per solve.
    p2 = np.array([0.4, 0.1, 0.2, 0.2, 0.1])
    Z_in_2 = [3, 4]
    baseline_2 = p2[3] + p2[4]                         # = 0.3
    r = lp.solve(p2, rho=0.0, Z_in=Z_in_2)
    assert abs(r - baseline_2) < 1e-9, f"reuse rho=0 expected {baseline_2}, got {r}"
    r = lp.solve(p2, rho=0.2, Z_in=Z_in_2)
    assert abs(r - 0.4) < 1e-9, f"reuse rho=0.2 expected 0.4, got {r}"
    print(f"reuse OK: p2 Z_in_2 rho=0.2 wZ_max = {r:.6f}")

    print("wZ_max LP OK")


if __name__ == "__main__":
    main()