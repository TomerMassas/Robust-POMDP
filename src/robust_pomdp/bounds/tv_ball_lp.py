"""
TVBallLinearMinLP - linear-objective minimization over a full TV-ball.

Computes:
    min over P in TV-ball(p_nominal, rho) of  Sum_i c[i] * P(i)

where P is a length-n full probability distribution and c is a per-solve
coefficient vector.

Used in Phase 6's leakage / omitted-trajectory machinery (see leakage.py):
  - wZ_in_min(s_next): c = indicator(Z_in).
  - robust_inside_prob_step(s, a): c[s'] = wZ_in_min(s') if s' in S_in else 0.

(WZMaxLP is the max-direction sibling with indicator coefficients, used by
Phase 5's inside-mismatch path.)

TV convention: codebase uses ||P-Q||_1 (no 1/2 factor), so rho values are 2x
textbook TV.
"""

from __future__ import annotations

import highspy
import numpy as np


class TVBallLinearMinLP:
    """Persistent LP for  min Sum p_i * c_i  over P in TV-ball(p_nominal, rho).

    Build once for n; solve many times with (p_nominal, rho, c).

    Variable layout (2n columns):
        d_plus[i]  = column i         (>= 0)
        d_minus[i] = column n + i     (>= 0)
    P(i) is implicit: P(i) = p_nominal[i] + d_plus[i] - d_minus[i].

    Constraint layout (n + 2 rows):
        row i in 0..n-1:  d_plus[i] - d_minus[i] >= -p_nominal[i]   (P positivity)
        row n:            Sum (d_plus - d_minus) = 0                (mass balance: Sum P = 1)
        row n + 1:        Sum (d_plus + d_minus) <= rho              (TV budget)

    Mass balance is structural (never updated). Positivity row bounds, the TV
    budget row bound, and the objective coefficients update per solve.
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

        for i in range(n):
            h.addRow(0.0, INF, 2, [i, n + i], [1.0, -1.0])

        mb_idx = list(range(n)) + list(range(n, 2 * n))
        mb_val = [1.0] * n + [-1.0] * n
        h.addRow(0.0, 0.0, len(mb_idx), mb_idx, mb_val)

        all_idx = list(range(2 * n))
        all_one = [1.0] * (2 * n)
        h.addRow(-INF, 0.0, len(all_idx), all_idx, all_one)

        h.changeObjectiveSense(highspy.ObjSense.kMinimize)

        self._h = h
        self._INF = INF

    def solve(self,
              p_nominal: np.ndarray,
              rho: float,
              c: np.ndarray
              ) -> float:
        """min Sum_i c[i] * P(i) over P in TV-ball(p_nominal, rho)."""
        n = self.n
        p_nominal = np.asarray(p_nominal, dtype=np.float64)
        c = np.asarray(c, dtype=np.float64)
        if p_nominal.shape != (n,):
            raise ValueError(f"p_nominal shape {p_nominal.shape} != ({n},)")
        if c.shape != (n,):
            raise ValueError(f"c shape {c.shape} != ({n},)")
        if rho < 0:
            raise ValueError(f"rho must be >= 0, got {rho}")

        # Fast path: rho = 0 forces P = p_nominal exactly.
        if rho == 0.0:
            return float(np.dot(p_nominal, c))

        h = self._h
        INF = self._INF

        for i in range(n):
            h.changeRowBounds(i, -float(p_nominal[i]), INF)
        h.changeRowBounds(n + 1, -INF, float(rho))

        # Objective constant part Sum_i c[i] * p_nominal[i] is added back below.
        # Set per-solve costs: on d_plus[i] = c[i], on d_minus[i] = -c[i].
        for i in range(n):
            h.changeColCost(i, float(c[i]))
            h.changeColCost(n + i, -float(c[i]))

        h.run()
        status = h.getModelStatus()
        if status != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"TVBallLinearMinLP did not solve to optimality: status = {status}")

        sol = h.getSolution()
        col_value = np.asarray(sol.col_value, dtype=np.float64)
        d_plus = col_value[:n]
        d_minus = col_value[n:2 * n]
        P = p_nominal + d_plus - d_minus
        return float(np.dot(P, c))
