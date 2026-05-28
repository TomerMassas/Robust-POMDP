"""Phase 5 - step 4 - orthant cross-check of delta_z_inside_max MILP.

For small problems (|Z_in| <= ~5), the MILP answer must equal the brute-force
orthant-enumeration answer to machine precision. This catches big-M / sign /
formulation bugs in the MILP that sanity values alone would miss.

Test grid covers:
- Various Z_in subset sizes (2..5).
- Various p_nominal shapes (skewed, uniform, edge-heavy).
- Various rho values (small, medium, large; up to clipping regime).

Run from anywhere:
    python experiments/a1/smoke_test/phase5_delta_z_orthant_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from robust_pomdp.bounds.l1_max import (
    ProjectedL1MaxMILP,
    projected_l1_max_orthant,
)


def main() -> None:
    # v_orthant = projected_l1_max_orthant(np.array([0.3, 0.25, 0.2, 0.15, 0.1]), 0.3, [0,1,2], n=5)

    p_skewed = np.array([0.05, 0.15, 0.6, 0.15, 0.05])
    p_uniform = np.full(5, 0.2)
    p_edge = np.array([0.4, 0.1, 0.2, 0.2, 0.1])

    configs = [
        # (n, Z_in, p_nominal, rho)
        (5, [0, 1, 2],          p_skewed,  0.1),
        (5, [0, 1, 2],          p_skewed,  0.3),
        (5, [0, 1, 2],          p_skewed,  0.7),
        (5, [0, 1],             p_uniform, 0.2),
        (5, [3, 4],             p_uniform, 0.5),
        (5, [1, 2, 3],          p_edge,    0.3),
        (5, list(range(5)),     p_skewed,  1.0),       # the 1.8 case
        (4, [0, 1, 2, 3],       np.array([0.1, 0.4, 0.4, 0.1]),  0.5),
        (4, [0, 3],             np.full(4, 0.25),                 0.4),
    ]

    for i, (n, Z_in, p_nom, rho) in enumerate(configs):
        milp = ProjectedL1MaxMILP(n=n, support=Z_in)
        v_milp = milp.solve(p_nom, rho=rho)
        v_orthant = projected_l1_max_orthant(p_nom, rho, Z_in, n=n)
        diff = abs(v_milp - v_orthant)
        assert diff < 1e-6, (
            f"config {i}: MILP={v_milp:.8f} vs orthant={v_orthant:.8f}, diff={diff:.2e}"
        )
        print(f"config {i}: n={n} |Z_in|={len(Z_in)} rho={rho:.2f}  ->  "
              f"MILP={v_milp:.6f}  orthant={v_orthant:.6f}  diff={diff:.2e}")

    print(f"\northant cross-check OK ({len(configs)} configs; MILP matches orthant to 1e-6)")


if __name__ == "__main__":
    main()