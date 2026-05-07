"""
Projected uncertainty set optimization for TV distance.

Solves the core LP used by the robust Bellman backup:
  min  sum p_i * c_i
  s.t. p in hat{P}(p_o, rho)   (projected TV ball)

The projected TV ball is characterized by:
  p_i >= 0
  sum p_i <= 1                              (sub-probability)
  sum |p_i - p_o_i| + |sum p_i - m_o| <= rho  (projected TV constraint)

where m_o = sum p_o_i (nominal in-support mass).

Used by both:
  - SolveObservationLP: p_o = P_Z^o(z|s') on Z_in, c = V_children
  - SolveTransitionLP:  p_o = P_T^o(s'|s,a) on S_in, c = w(s')


PERFORMANCE NOTE — LP reuse
---------------------------
A naive implementation rebuilds the whole LP on every call. At ~200K LP
solves per E2 config that was the dominant cost (~6 s/episode at rho>0).

This implementation keeps a persistent `Highs` instance: __init__ builds the
LP structure once; solve() updates only the changing parts (some row bounds,
the objective coefficients) and re-solves. The LP topology (variables,
sparsity pattern, the mass-balance equality) is fixed forever.

The public API is the `ProjectedTVBall.solve` method — callers don't see
highspy. If highspy becomes a bottleneck at larger scale, swap in Gurobi or
OR-Tools by reimplementing this one class.
"""

from __future__ import annotations

import highspy  # imported at module top to lock load order before pandas
import numpy as np


class ProjectedTVBall:
    """Reusable LP solver for the projected TV ball minimization.

    Build once, solve many times with different (p_nominal, rho, objective_coeffs).
    The LP structure (variables, constraint sparsity, mass-balance equality) is fixed;
    only constraint right-hand-sides and objective coefficients change per solve.

    Variable layout (2n + 2 columns):
      d_plus[i]   = column i           for i in 0..n-1   (>= 0)
      d_minus[i]  = column n + i       for i in 0..n-1   (>= 0)
      delta_plus  = column 2n                            (>= 0)
      delta_minus = column 2n + 1                        (>= 0)

    Row layout (n + 3 rows):
      pos[i] = row i           for i in 0..n-1     d_plus[i] - d_minus[i] >= -p_nominal[i]
      tv     = row n                                sum(d_plus + d_minus) + delta_plus + delta_minus <= rho
      mb     = row n + 1                            sum(d_plus - d_minus) - delta_plus + delta_minus == 0
      sub    = row n + 2                            sum(d_plus - d_minus) <= 1 - m_nominal

    The mass-balance row (mb) is structural and never updated.
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

        # Add 2n + 2 columns: all non-negative, initial cost 0 (set per solve).
        for _ in range(2 * n + 2):
            h.addCol(0.0, 0.0, INF, 0, [], [])

        # Positivity rows: d_plus[i] - d_minus[i] >= -p_nominal[i]
        # Initial bounds (will be updated per solve): lower=0, upper=INF.
        for i in range(n):
            h.addRow(0.0, INF, 2, [i, n + i], [1.0, -1.0])

        # TV budget row: sum(d_plus + d_minus) + delta_plus + delta_minus <= rho
        # Coefficients: 1 on every column. Initial upper=0 (placeholder, updated per solve).
        all_idx = list(range(2 * n + 2))
        all_one = [1.0] * (2 * n + 2)
        h.addRow(-INF, 0.0, len(all_idx), all_idx, all_one)

        # Mass balance (equality, structural — never updated):
        # sum(d_plus - d_minus) - delta_plus + delta_minus == 0
        mb_idx = list(range(n)) + list(range(n, 2 * n)) + [2 * n, 2 * n + 1]
        mb_val = [1.0] * n + [-1.0] * n + [-1.0, 1.0]
        h.addRow(0.0, 0.0, len(mb_idx), mb_idx, mb_val)

        # Sub-probability row: sum(d_plus - d_minus) <= 1 - m_nominal
        sub_idx = list(range(n)) + list(range(n, 2 * n))
        sub_val = [1.0] * n + [-1.0] * n
        h.addRow(-INF, 0.0, len(sub_idx), sub_idx, sub_val)

        h.changeObjectiveSense(highspy.ObjSense.kMinimize)

        self._h = h
        self._INF = INF

    def solve(self,
              p_nominal,
              rho: float,
              objective_coeffs
              ) -> tuple[float, np.ndarray]:
        """Minimize sum p_i * c_i over the projected TV ball.

        Args:
            p_nominal: nominal probability vector restricted to the support
                (length n; need not sum to 1, the rest is out-of-support mass).
            rho: TV distance radius (rho >= 0).
            objective_coeffs: coefficients c_i for the objective sum p_i * c_i.

        Returns:
            (optimal_value, optimal_p) where optimal_p is a length-n sub-probability vector.
        """
        n = self.n
        p_nominal = np.asarray(p_nominal, dtype=np.float64)
        c = np.asarray(objective_coeffs, dtype=np.float64)
        if p_nominal.shape != (n,):
            raise ValueError(f"p_nominal must have length {n}, got shape {p_nominal.shape}")
        if c.shape != (n,):
            raise ValueError(f"objective_coeffs must have length {n}, got shape {c.shape}")

        # Fast path: rho <= 0 means no deviation allowed; optimal_p = p_nominal.
        # Mirrors the Julia short-circuit and avoids a wasted LP call.
        if rho <= 0.0:
            optimal_p = p_nominal.copy()
            optimal_value = float(np.dot(optimal_p, c))
            return optimal_value, optimal_p

        h = self._h
        INF = self._INF
        m_nominal = float(np.sum(p_nominal))

        # Update positivity rows: lower bound = -p_nominal[i].
        for i in range(n):
            h.changeRowBounds(i, -float(p_nominal[i]), INF)

        # Update TV budget row: upper bound = rho.
        h.changeRowBounds(n, -INF, float(rho))

        # Mass balance row (n + 1) is structural — no update.

        # Update sub-probability row: upper bound = 1 - m_nominal.
        h.changeRowBounds(n + 2, -INF, 1.0 - m_nominal)

        # Update objective coefficients:
        # objective is sum (d_plus[i] - d_minus[i]) * c[i], so cost on
        # d_plus[i] is c[i] and cost on d_minus[i] is -c[i]. delta_plus and
        # delta_minus stay at cost 0.
        for i in range(n):
            h.changeColCost(i, float(c[i]))
            h.changeColCost(n + i, -float(c[i]))

        h.run()

        sol = h.getSolution()
        col_value = sol.col_value

        # Reconstruct optimal_p_i = p_nominal_i + d_plus_i - d_minus_i.
        # Use np.array() to convert from highspy's container to a plain ndarray,
        # which lets us slice safely regardless of the exact return type.
        col_value = np.asarray(col_value, dtype=np.float64)
        optimal_p = p_nominal + col_value[:n] - col_value[n : 2 * n]

        # Compute the full objective value (avoids depending on highspy's
        # constant offset handling and stays consistent with the Julia version).
        optimal_value = float(np.dot(optimal_p, c))

        return optimal_value, optimal_p
