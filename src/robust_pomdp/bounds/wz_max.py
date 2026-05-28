"""
WZMaxLP - max-in-support mass primitive for Phase 5 (paper Eq. 47 + 71).

Computes wZ_max(s_next) = sup over P_Z in TV-ball(P_Z_o(.|s_next), rho_Z) of
Sum_{z in Z_in} P_Z(z|s_next). Geometrically: the largest amount of observation
mass the adversary can push inside the projected observation support.

Generic form: sup over P in TV-ball(p_nominal, rho) of Sum_{i in support} P(i).

In Phase 5 this is the precomputed weight on each |T - T_hat| term inside the
transition_part MILP (see l1_max.py). The DeltaKComputer orchestrator caches
wZ_max(s_next) per s_next.

TV convention: codebase uses ||P-Q||_1 (no 1/2 factor), so rho values are 2x
textbook TV.
"""

from __future__ import annotations

import highspy
import numpy as np


class WZMaxLP:
    """Persistent LP solver for wZ_max(s_next) = sup_P_Z  Sum_{z in Z_in} P_Z(z|s_next).

    Build once for a given n = |Z|; solve many times across (s_next, P_Z_o, rho_Z, Z_in).

    Variable layout (2n columns):
        d_plus[z]  = column z         (>= 0)
        d_minus[z] = column n + z     (>= 0)
    P_Z is implicit: P_Z(z) = p_nominal[z] + d_plus[z] - d_minus[z].

    Constraint layout (n + 2 rows):
        row z in 0..n-1:  d_plus[z] - d_minus[z] >= -p_nominal[z]   (positivity of P_Z)
        row n:            Sum (d_plus - d_minus) = 0                (mass balance: Sum P_Z = 1)
        row n + 1:        Sum (d_plus + d_minus) <= rho              (TV budget)

    Mass balance is structural (never updated). Positivity row bounds and the TV
    budget row bound update per solve.
    """

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.n = n
        self._build()

    def _build(self) -> None:
        n = self.n
        h = highspy.Highs()
        h.silent()
        INF = highspy.kHighsInf

        for _ in range(2 * n):
            h.addCol(0.0, 0.0, INF, 0, [], [])

        for z in range(n):
            h.addRow(0.0, INF, 2, [z, n + z], [1.0, -1.0])

        mb_idx = list(range(n)) + list(range(n, 2 * n))
        mb_val = [1.0] * n + [-1.0] * n
        h.addRow(0.0, 0.0, len(mb_idx), mb_idx, mb_val)

        all_idx = list(range(2 * n))
        all_one = [1.0] * (2 * n)
        h.addRow(-INF, 0.0, len(all_idx), all_idx, all_one)

        h.changeObjectiveSense(highspy.ObjSense.kMaximize)

        self._h = h
        self._INF = INF

    def solve(self,
              p_nominal: np.ndarray,
              rho: float,
              Z_in: list[int]
              ) -> float:
        """Sup over P_Z in TV-ball(p_nominal, rho) of Sum_{z in Z_in} P_Z(z)."""
        n = self.n
        p_nominal = np.asarray(p_nominal, dtype=np.float64)
        if p_nominal.shape != (n,):
            raise ValueError(f"p_nominal shape {p_nominal.shape} != ({n},)")
        if rho < 0:
            raise ValueError(f"rho must be >= 0, got {rho}")
        if not Z_in:
            return 0.0

        Z_in_array = np.asarray(sorted(set(int(z) for z in Z_in)), dtype=int)

        if rho == 0.0:
            return float(p_nominal[Z_in_array].sum())

        h = self._h
        INF = self._INF
        Z_in_set = set(Z_in_array.tolist())

        for z in range(n):
            h.changeRowBounds(z, -float(p_nominal[z]), INF)

        h.changeRowBounds(n + 1, -INF, float(rho))

        for z in range(n):
            if z in Z_in_set:
                h.changeColCost(z, 1.0)
                h.changeColCost(n + z, -1.0)
            else:
                h.changeColCost(z, 0.0)
                h.changeColCost(n + z, 0.0)

        h.run()
        status = h.getModelStatus()
        if status != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"WZMaxLP did not solve to optimality: status = {status}")

        sol = h.getSolution()
        col_value = np.asarray(sol.col_value, dtype=np.float64)
        d_plus = col_value[:n]
        d_minus = col_value[n:2 * n]
        P_Z = p_nominal + d_plus - d_minus

        result = float(P_Z[Z_in_array].sum())
        return max(0.0, min(1.0, result))
