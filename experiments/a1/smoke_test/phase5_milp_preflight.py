"""Pre-flight check: verify highspy MIP API works on this install.

Phase 5 needs binary integer variables + big-M linearization for max-L1
problems. This smoke confirms `changeColIntegrality` + `kInteger` is the
correct API path on the installed highspy version.

Test: maximize |x| for x in [-0.7, +0.5] via a 1-binary MILP with big-M=2.
Expected optimum: x=-0.7, u=0.7, w=0.

Run from anywhere:
    python experiments/a1/smoke_test/phase5_milp_preflight.py
"""

from __future__ import annotations

import highspy


def main() -> None:
    h = highspy.Highs()
    h.silent()

    INF = highspy.kHighsInf

    # Variables:
    #   col 0: x in [-0.7, 0.5]   (continuous, free target)
    #   col 1: u in [0, INF)       (= |x|, the quantity we maximize)
    #   col 2: w in {0, 1}         (sign indicator: 1 if x >= 0, 0 if x < 0)
    h.addCol(0.0, -0.7, 0.5, 0, [], [])    # x
    h.addCol(1.0,  0.0, INF, 0, [], [])    # u (objective coefficient = 1.0)
    h.addCol(0.0,  0.0, 1.0, 0, [], [])    # w (will be marked integer)

    h.changeColIntegrality(2, highspy.HighsVarType.kInteger)

    # Big-M = 2 linearization of u = |x|:
    #   u >= +x                              (row 0:  u - x >= 0)
    #   u >= -x                              (row 1:  u + x >= 0)
    #   u <= +x + 2 * (1 - w)                (row 2:  u - x + 2w <= 2)
    #   u <= -x + 2 * w                      (row 3:  u + x - 2w <= 0)
    h.addRow(0.0,  INF, 2, [1, 0],    [1.0, -1.0])           # row 0
    h.addRow(0.0,  INF, 2, [1, 0],    [1.0,  1.0])           # row 1
    h.addRow(-INF, 2.0, 3, [1, 0, 2], [1.0, -1.0,  2.0])     # row 2
    h.addRow(-INF, 0.0, 3, [1, 0, 2], [1.0,  1.0, -2.0])     # row 3

    h.changeObjectiveSense(highspy.ObjSense.kMaximize)

    h.run()

    status = h.getModelStatus()
    if status != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"MILP did not solve to optimality: status = {status}")

    sol = h.getSolution()
    x_val = sol.col_value[0]
    u_val = sol.col_value[1]
    w_val = sol.col_value[2]

    print(f"x = {x_val:+.6f}, u = {u_val:.6f}, w = {w_val:.6f}")
    assert abs(u_val - 0.7) < 1e-6, f"expected u = 0.7, got {u_val}"
    assert abs(x_val - (-0.7)) < 1e-6, f"expected x = -0.7, got {x_val}"
    assert abs(w_val - 0.0) < 1e-6, f"expected w = 0, got {w_val}"

    print("preflight MILP OK | highspy MIP API works (changeColIntegrality + kInteger)")


if __name__ == "__main__":
    main()