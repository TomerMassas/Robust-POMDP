"""
Projected uncertainty set optimization for TV distance.

DUPLICATED from src/robust_pomdp/uncertainty/projected_sets.py (logic identical)
so Algorithm/ is self-contained. Only change: a defensive optimal-status check
after solve (does not affect returned values for the always-feasible projected
ball).

Solves:
  min  sum p_i * c_i
  s.t. p in hat{P}(p_o, rho)          (projected TV ball)

The projected TV ball is characterized by:
  p_i >= 0
  sum p_i <= 1                                 (sub-probability)
  sum |p_i - p_o_i| + |sum p_i - m_o| <= rho   (projected TV constraint)

A persistent Highs instance is built once (fixed variables / sparsity /
mass-balance equality); solve() updates only the changing row bounds and
objective coefficients.
"""

from __future__ import annotations

import highspy  # imported at module top to lock load order before pandas
import numpy as np


class ProjectedTVBall:
    """Reusable LP solver for the projected TV ball minimization.

    Build once, solve many times with different (p_nominal, rho, objective_coeffs).

    Variable layout (2n + 2 columns):
      d_plus[i]   = column i           for i in 0..n-1   (>= 0)
      d_minus[i]  = column n + i       for i in 0..n-1   (>= 0)
      delta_plus  = column 2n                            (>= 0)
      delta_minus = column 2n + 1                        (>= 0)

    Row layout (n + 3 rows):
      pos[i] = row i           d_plus[i] - d_minus[i] >= -p_nominal[i]
      tv     = row n           sum(d_plus + d_minus) + delta_plus + delta_minus <= rho
      mb     = row n + 1       sum(d_plus - d_minus) - delta_plus + delta_minus == 0
      sub    = row n + 2       sum(d_plus - d_minus) <= 1 - m_nominal
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

        for _ in range(2 * n + 2):
            h.addCol(0.0, 0.0, INF, 0, [], [])

        for i in range(n):
            h.addRow(0.0, INF, 2, [i, n + i], [1.0, -1.0])

        all_idx = list(range(2 * n + 2))
        all_one = [1.0] * (2 * n + 2)
        h.addRow(-INF, 0.0, len(all_idx), all_idx, all_one)

        mb_idx = list(range(n)) + list(range(n, 2 * n)) + [2 * n, 2 * n + 1]
        mb_val = [1.0] * n + [-1.0] * n + [-1.0, 1.0]
        h.addRow(0.0, 0.0, len(mb_idx), mb_idx, mb_val)

        sub_idx = list(range(n)) + list(range(n, 2 * n))
        sub_val = [1.0] * n + [-1.0] * n
        h.addRow(-INF, 0.0, len(sub_idx), sub_idx, sub_val)

        h.changeObjectiveSense(highspy.ObjSense.kMinimize)

        self._h = h
        self._INF = INF

    def solve(self,
              p_nominal,
              rho: float,
              objective_coeffs,
              full_support: bool = False,
              ) -> tuple[float, np.ndarray]:
        """Minimize sum p_i * c_i over the projected TV ball.

        Args:
            p_nominal: nominal probability vector restricted to the support
                (length n; need not sum to 1, the rest is out-of-support mass).
            rho: TV distance radius (rho >= 0).
            objective_coeffs: coefficients c_i for the objective sum p_i * c_i.
            full_support: True when the support is the FULL space. Then leakage
                is impossible and the ball reduces to the full TV ball (Sum P = 1).

        Returns:
            (optimal_value, optimal_p) with optimal_p a length-n sub-probability
            vector (a full distribution when full_support=True).
        """
        n = self.n
        p_nominal = np.asarray(p_nominal, dtype=np.float64)
        c = np.asarray(objective_coeffs, dtype=np.float64)
        if p_nominal.shape != (n,):
            raise ValueError(f"p_nominal must have length {n}, got shape {p_nominal.shape}")
        if c.shape != (n,):
            raise ValueError(f"objective_coeffs must have length {n}, got shape {c.shape}")

        # Fast path: rho <= 0 means no deviation allowed; optimal_p = p_nominal.
        if rho <= 0.0:
            optimal_p = p_nominal.copy()
            return float(np.dot(optimal_p, c)), optimal_p

        h = self._h
        INF = self._INF
        m_nominal = float(np.sum(p_nominal))

        for i in range(n):
            h.changeRowBounds(i, -float(p_nominal[i]), INF)
        h.changeRowBounds(n, -INF, float(rho))          # TV budget
        h.changeRowBounds(n + 2, -INF, 1.0 - m_nominal)  # sub-probability

        # delta_plus/delta_minus model mass leaking out of support; cap at 0 for
        # full support so the ball reduces to the full TV ball (Sum P = 1).
        if full_support:
            h.changeColBounds(2 * n, 0.0, 0.0)
            h.changeColBounds(2 * n + 1, 0.0, 0.0)
        else:
            h.changeColBounds(2 * n, 0.0, INF)
            h.changeColBounds(2 * n + 1, 0.0, INF)

        for i in range(n):
            h.changeColCost(i, float(c[i]))
            h.changeColCost(n + i, -float(c[i]))

        h.run()

        # Defensive: the projected ball is always feasible & bounded, so this
        # should never fire; catch solver regressions early instead of returning
        # silent garbage. (Only addition over the src version.)
        status = h.getModelStatus()
        if status != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"ProjectedTVBall did not solve to optimality: status = {status}")

        col_value = np.asarray(h.getSolution().col_value, dtype=np.float64)
        optimal_p = p_nominal + col_value[:n] - col_value[n : 2 * n]
        return float(np.dot(optimal_p, c)), optimal_p
