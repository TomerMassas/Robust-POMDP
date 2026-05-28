"""
Generic projected weighted-L1-max primitive for Phase 5 (paper Eq. 47 + 71).

Computes:
    sup over (P in TV-ball(p_nominal, rho), P_hat in projected ball)
        Sum_{i in support}  weights[i] * |P(s_i) - P_hat(s_i)|

where P is a length-n full distribution and P_hat is a sub-prob on `support`,
both centered at p_nominal with the same TV radius rho.

Used by Phase 5 in two ways:
  - delta_z_inside_max(s_next): weights = ones, support = Z_in, n = |Z|,
    p_nominal = P_Z_o(.|s_next), rho = rho_Z.
  - transition_part(s, a): weights = wZ_max(s') for s' in S_in,
    support = S_in, n = |S|, p_nominal = T_o(.|s, a), rho = rho_T(s, a).

The DeltaKComputer orchestrator (in delta_k.py) instantiates ProjectedL1MaxMILP
once per (n, support) tuple and reuses it across (p_nominal, rho, weights) calls.

ProjectedL1MaxOrthant + projected_l1_max_orthant provide a brute-force orthant-
enumeration baseline that gives the exact same value via 2^|support| standard
LPs - used to cross-check the MILP on small problems (|support| <= ~12).

TV convention: codebase uses ||P-Q||_1 (no 1/2 factor), so rho values are 2x
textbook TV.
"""

from __future__ import annotations

import highspy
import numpy as np


class ProjectedL1MaxMILP:
    """Persistent MILP for the generic projected weighted-L1-max primitive:
        sup over (P in TV-ball(p_nominal, rho), P_hat in projected ball)
            Sum_{i in support}  weights[i] * |P(s_i) - P_hat(s_i)|

    Build once for fixed (n, support); solve many times with (p_nominal, rho,
    weights). weights=None defaults to ones (= delta_z_inside_max when applied
    to obs distributions). weights=wZ_max_array gives transition_part.

    Variable layout (2n + 4k + 2 cols, k = |support|):
        cols 0..n-1:             d_P_plus[z]                                  (>= 0)
        cols n..2n-1:            d_P_minus[z]                                 (>= 0)
        cols 2n..2n+k-1:         d_Ph_plus[i]    for i in 0..k-1              (>= 0)
        cols 2n+k..2n+2k-1:      d_Ph_minus[i]                                (>= 0)
        col  2n+2k:              delta_plus      (projected mass slack)       (>= 0)
        col  2n+2k+1:            delta_minus                                  (>= 0)
        cols 2n+2k+2..2n+3k+1:   u[i]            (= |delta[i]|, cost set per solve from weights)
        cols 2n+3k+2..2n+4k+1:   w[i]            (sign indicator, binary)     {0, 1}

    Implicit values:
        P(z)        = p_nominal[z]    + d_P_plus[z]  - d_P_minus[z]            for z in 0..n-1
        P_hat(s_i)  = p_nominal[s_i]  + d_Ph_plus[i] - d_Ph_minus[i]           for s_i in support
        delta[i]    = P(s_i) - P_hat(s_i)
                    = (d_P_plus[s_i] - d_P_minus[s_i]) - (d_Ph_plus[i] - d_Ph_minus[i])

    Constraint layout (n + 5k + 5 rows):
        rows 0..n-1:           d_P_plus[z] - d_P_minus[z] >= -p_nominal[z]    (P positivity)
        row  n:                Sum (d_P_plus - d_P_minus) = 0                  (Sum P = 1)
        row  n+1:              Sum (d_P_plus + d_P_minus) <= rho                (P TV budget)
        rows n+2..n+k+1:       d_Ph_plus[i] - d_Ph_minus[i] >= -p_nominal[s_i] (P_hat positivity)
        row  n+k+2:            Sum (d_Ph_plus - d_Ph_minus) - delta_plus + delta_minus = 0
                                                                              (projected mass balance with slack)
        row  n+k+3:            Sum (d_Ph_plus + d_Ph_minus) + delta_plus + delta_minus <= rho
                                                                              (projected TV budget)
        row  n+k+4:            Sum (d_Ph_plus - d_Ph_minus) <= 1 - m_in_o     (sub-prob bound on P_hat)
        rows n+k+5..n+5k+4:    4 linearization rows per i in 0..k-1:
            (a) u[i] >= +delta[i]
            (b) u[i] >= -delta[i]
            (c) u[i] <= +delta[i] + 2*(1 - w[i])   (big-M = 2)
            (d) u[i] <= -delta[i] + 2*w[i]
    Objective: max Sum_{i} weights[i] * u[i].
    """

    def __init__(self, n: int, support: list[int]) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        support_sorted = sorted(set(int(z) for z in support))
        if not support_sorted:
            raise ValueError("support must be non-empty")
        for z in support_sorted:
            if not 0 <= z < n:
                raise ValueError(f"support contains invalid index {z} for n = {n}")
        self.n = n              # full space size
        self.support = support_sorted
        self.k = len(support_sorted)
        self._build()

    def _build(self) -> None:
        n, k = self.n, self.k
        support = self.support
        h = highspy.Highs()
        h.silent()
        INF = highspy.kHighsInf

        # Columns.
        for _ in range(2 * n):                       # d_P_plus, d_P_minus
            h.addCol(0.0, 0.0, INF, 0, [], [])
        for _ in range(2 * k):                       # d_Ph_plus, d_Ph_minus
            h.addCol(0.0, 0.0, INF, 0, [], [])
        h.addCol(0.0, 0.0, INF, 0, [], [])           # delta_plus
        h.addCol(0.0, 0.0, INF, 0, [], [])           # delta_minus
        for _ in range(k):                           # u[i] - cost set per solve (weights)
            h.addCol(0.0, 0.0, INF, 0, [], [])
        for _ in range(k):                           # w[i] - binary
            h.addCol(0.0, 0.0, 1.0, 0, [], [])

        w_start = 2 * n + 3 * k + 2
        for i in range(k):
            h.changeColIntegrality(w_start + i, highspy.HighsVarType.kInteger)

        # Rows.
        # P positivity (n rows). Lower bound updated per solve.
        for z in range(n):
            h.addRow(0.0, INF, 2, [z, n + z], [1.0, -1.0])

        # P mass balance.
        mb_P_idx = list(range(n)) + list(range(n, 2 * n))
        mb_P_val = [1.0] * n + [-1.0] * n
        h.addRow(0.0, 0.0, 2 * n, mb_P_idx, mb_P_val)

        # P TV budget. Upper updated per solve.
        tv_P_idx = list(range(2 * n))
        tv_P_val = [1.0] * (2 * n)
        h.addRow(-INF, 0.0, 2 * n, tv_P_idx, tv_P_val)

        # P_hat positivity (k rows). Lower bound updated per solve.
        for i in range(k):
            h.addRow(0.0, INF, 2, [2 * n + i, 2 * n + k + i], [1.0, -1.0])

        # P_hat projected mass balance with slack.
        d_ph_plus_cols = list(range(2 * n, 2 * n + k))
        d_ph_minus_cols = list(range(2 * n + k, 2 * n + 2 * k))
        delta_plus_col = 2 * n + 2 * k
        delta_minus_col = 2 * n + 2 * k + 1
        mb_Ph_idx = d_ph_plus_cols + d_ph_minus_cols + [delta_plus_col, delta_minus_col]
        mb_Ph_val = [1.0] * k + [-1.0] * k + [-1.0, 1.0]
        h.addRow(0.0, 0.0, len(mb_Ph_idx), mb_Ph_idx, mb_Ph_val)

        # P_hat projected TV budget. Upper updated per solve.
        tv_Ph_idx = d_ph_plus_cols + d_ph_minus_cols + [delta_plus_col, delta_minus_col]
        tv_Ph_val = [1.0] * (2 * k + 2)
        h.addRow(-INF, 0.0, len(tv_Ph_idx), tv_Ph_idx, tv_Ph_val)

        # P_hat sub-prob bound. Upper updated per solve.
        sub_idx = d_ph_plus_cols + d_ph_minus_cols
        sub_val = [1.0] * k + [-1.0] * k
        h.addRow(-INF, 0.0, len(sub_idx), sub_idx, sub_val)

        # MILP absolute-value linearization (4k rows).
        for i in range(k):
            s_i = support[i]
            col_d_P_plus_si = s_i
            col_d_P_minus_si = n + s_i
            col_d_Ph_plus_i = 2 * n + i
            col_d_Ph_minus_i = 2 * n + k + i
            col_u_i = 2 * n + 2 * k + 2 + i
            col_w_i = w_start + i

            # (a) u[i] - delta[i] >= 0
            h.addRow(0.0, INF, 5,
                     [col_u_i, col_d_P_plus_si, col_d_P_minus_si, col_d_Ph_plus_i, col_d_Ph_minus_i],
                     [1.0, -1.0, 1.0, 1.0, -1.0])
            # (b) u[i] + delta[i] >= 0
            h.addRow(0.0, INF, 5,
                     [col_u_i, col_d_P_plus_si, col_d_P_minus_si, col_d_Ph_plus_i, col_d_Ph_minus_i],
                     [1.0, 1.0, -1.0, -1.0, 1.0])
            # (c) u[i] - delta[i] + 2*w[i] <= 2
            h.addRow(-INF, 2.0, 6,
                     [col_u_i, col_d_P_plus_si, col_d_P_minus_si, col_d_Ph_plus_i, col_d_Ph_minus_i, col_w_i],
                     [1.0, -1.0, 1.0, 1.0, -1.0, 2.0])
            # (d) u[i] + delta[i] - 2*w[i] <= 0
            h.addRow(-INF, 0.0, 6,
                     [col_u_i, col_d_P_plus_si, col_d_P_minus_si, col_d_Ph_plus_i, col_d_Ph_minus_i, col_w_i],
                     [1.0, 1.0, -1.0, -1.0, 1.0, -2.0])

        h.changeObjectiveSense(highspy.ObjSense.kMaximize)
        self._h = h
        self._INF = INF

    def solve(self,
              p_nominal: np.ndarray,
              rho: float,
              weights: np.ndarray | None = None
              ) -> float:
        """sup over (P, P_hat) of Sum_{i in support} weights[i] * |P(s_i) - P_hat(s_i)|.

        weights=None -> unit weights (= delta_z_inside_max when applied to obs).
        """
        n, k = self.n, self.k
        support = self.support
        p_nominal = np.asarray(p_nominal, dtype=np.float64)
        if p_nominal.shape != (n,):
            raise ValueError(f"p_nominal shape {p_nominal.shape} != ({n},)")
        if rho < 0:
            raise ValueError(f"rho must be >= 0, got {rho}")

        if weights is None:
            weights_arr = np.ones(k, dtype=np.float64)
        else:
            weights_arr = np.asarray(weights, dtype=np.float64)
            if weights_arr.shape != (k,):
                raise ValueError(f"weights shape {weights_arr.shape} != ({k},)")
            if np.any(weights_arr < 0):
                raise ValueError("weights must be >= 0")

        # Fast path: rho = 0 forces P = p_nominal and P_hat = p_nominal restricted -> delta = 0.
        if rho == 0.0:
            return 0.0

        h = self._h
        INF = self._INF
        m_in_o = float(sum(p_nominal[z] for z in support))

        for z in range(n):
            h.changeRowBounds(z, -float(p_nominal[z]), INF)
        h.changeRowBounds(n + 1, -INF, float(rho))
        for i, s_i in enumerate(support):
            h.changeRowBounds(n + 2 + i, -float(p_nominal[s_i]), INF)
        h.changeRowBounds(n + k + 3, -INF, float(rho))
        h.changeRowBounds(n + k + 4, -INF, 1.0 - m_in_o)

        # Set per-solve u[i] objective costs from weights.
        u_start = 2 * n + 2 * k + 2
        for i in range(k):
            h.changeColCost(u_start + i, float(weights_arr[i]))

        h.run()
        status = h.getModelStatus()
        if status != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"ProjectedL1MaxMILP did not solve to optimality: status = {status}")

        sol = h.getSolution()
        col_value = np.asarray(sol.col_value, dtype=np.float64)
        u_values = col_value[u_start:u_start + k]
        result = float((weights_arr * u_values).sum())
        upper = 2.0 * float(weights_arr.max()) if k > 0 else 0.0
        return max(0.0, min(upper, result))


class ProjectedL1MaxOrthant:
    """Internal-only LP for orthant-enumeration cross-check of ProjectedL1MaxMILP.

    For each sign pattern sigma in {-1, +1}^|support|, solves the *standard LP*:
        max  Sum_{i} weights[i] * sigma_i * (P(s_i) - P_hat(s_i))
    over the same TV-ball + projected-TV-ball polytopes as ProjectedL1MaxMILP.
    By the L1 dual (weights >= 0), max over all 2^|support| sign patterns
    equals the weighted L1 max exactly. Used to verify the MILP formulation.
    """

    def __init__(self, n: int, support: list[int]) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        support_sorted = sorted(set(int(z) for z in support))
        if not support_sorted:
            raise ValueError("support must be non-empty")
        for z in support_sorted:
            if not 0 <= z < n:
                raise ValueError(f"support contains invalid index {z} for n = {n}")
        self.n = n
        self.support = support_sorted
        self.k = len(support_sorted)
        self._build()

    def _build(self) -> None:
        n, k = self.n, self.k
        h = highspy.Highs()
        h.silent()
        INF = highspy.kHighsInf

        for _ in range(2 * n):                       # d_P_plus, d_P_minus
            h.addCol(0.0, 0.0, INF, 0, [], [])
        for _ in range(2 * k):                       # d_Ph_plus, d_Ph_minus
            h.addCol(0.0, 0.0, INF, 0, [], [])
        h.addCol(0.0, 0.0, INF, 0, [], [])           # delta_plus
        h.addCol(0.0, 0.0, INF, 0, [], [])           # delta_minus

        for z in range(n):
            h.addRow(0.0, INF, 2, [z, n + z], [1.0, -1.0])
        mb_P_idx = list(range(n)) + list(range(n, 2 * n))
        mb_P_val = [1.0] * n + [-1.0] * n
        h.addRow(0.0, 0.0, 2 * n, mb_P_idx, mb_P_val)
        tv_P_idx = list(range(2 * n))
        tv_P_val = [1.0] * (2 * n)
        h.addRow(-INF, 0.0, 2 * n, tv_P_idx, tv_P_val)
        for i in range(k):
            h.addRow(0.0, INF, 2, [2 * n + i, 2 * n + k + i], [1.0, -1.0])
        d_ph_plus_cols = list(range(2 * n, 2 * n + k))
        d_ph_minus_cols = list(range(2 * n + k, 2 * n + 2 * k))
        delta_plus_col = 2 * n + 2 * k
        delta_minus_col = 2 * n + 2 * k + 1
        mb_Ph_idx = d_ph_plus_cols + d_ph_minus_cols + [delta_plus_col, delta_minus_col]
        mb_Ph_val = [1.0] * k + [-1.0] * k + [-1.0, 1.0]
        h.addRow(0.0, 0.0, len(mb_Ph_idx), mb_Ph_idx, mb_Ph_val)
        tv_Ph_idx = d_ph_plus_cols + d_ph_minus_cols + [delta_plus_col, delta_minus_col]
        tv_Ph_val = [1.0] * (2 * k + 2)
        h.addRow(-INF, 0.0, len(tv_Ph_idx), tv_Ph_idx, tv_Ph_val)
        sub_idx = d_ph_plus_cols + d_ph_minus_cols
        sub_val = [1.0] * k + [-1.0] * k
        h.addRow(-INF, 0.0, len(sub_idx), sub_idx, sub_val)

        h.changeObjectiveSense(highspy.ObjSense.kMaximize)
        self._h = h
        self._INF = INF

    def solve_orthant(self,
                      p_nominal: np.ndarray,
                      rho: float,
                      signs: np.ndarray,
                      weights: np.ndarray | None = None
                      ) -> float:
        """LP max of  Sum_i weights[i] * sigma_i * (P(s_i) - P_hat(s_i)) for given signs."""
        n, k = self.n, self.k
        support = self.support
        p_nominal = np.asarray(p_nominal, dtype=np.float64)
        if p_nominal.shape != (n,):
            raise ValueError(f"p_nominal shape {p_nominal.shape} != ({n},)")
        if signs.shape != (k,):
            raise ValueError(f"signs shape {signs.shape} != ({k},)")

        if weights is None:
            weights_arr = np.ones(k, dtype=np.float64)
        else:
            weights_arr = np.asarray(weights, dtype=np.float64)
            if weights_arr.shape != (k,):
                raise ValueError(f"weights shape {weights_arr.shape} != ({k},)")
            if np.any(weights_arr < 0):
                raise ValueError("weights must be >= 0")

        h = self._h
        INF = self._INF
        m_in_o = float(sum(p_nominal[z] for z in support))

        for z in range(n):
            h.changeRowBounds(z, -float(p_nominal[z]), INF)
        h.changeRowBounds(n + 1, -INF, float(rho))
        for i, s_i in enumerate(support):
            h.changeRowBounds(n + 2 + i, -float(p_nominal[s_i]), INF)
        h.changeRowBounds(n + k + 3, -INF, float(rho))
        h.changeRowBounds(n + k + 4, -INF, 1.0 - m_in_o)

        # Objective: Sum_i weights[i] * sigma_i * (P(s_i) - P_hat(s_i))
        for z in range(n):
            h.changeColCost(z, 0.0)
            h.changeColCost(n + z, 0.0)
        for i in range(k):
            h.changeColCost(2 * n + i, 0.0)
            h.changeColCost(2 * n + k + i, 0.0)
        for i, s_i in enumerate(support):
            ws = float(weights_arr[i]) * float(signs[i])
            h.changeColCost(s_i, ws)                # d_P_plus[s_i]
            h.changeColCost(n + s_i, -ws)           # d_P_minus[s_i]
            h.changeColCost(2 * n + i, -ws)         # d_Ph_plus[i]
            h.changeColCost(2 * n + k + i, ws)      # d_Ph_minus[i]
        h.changeColCost(2 * n + 2 * k, 0.0)
        h.changeColCost(2 * n + 2 * k + 1, 0.0)

        h.run()
        status = h.getModelStatus()
        if status != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"ProjectedL1MaxOrthant did not solve to optimality: status = {status}")

        sol = h.getSolution()
        col_value = np.asarray(sol.col_value, dtype=np.float64)
        d_P_plus = col_value[:n]
        d_P_minus = col_value[n:2 * n]
        d_Ph_plus = col_value[2 * n:2 * n + k]
        d_Ph_minus = col_value[2 * n + k:2 * n + 2 * k]
        P = p_nominal + d_P_plus - d_P_minus

        total = 0.0
        for i, s_i in enumerate(support):
            P_hat_si = float(p_nominal[s_i]) + float(d_Ph_plus[i]) - float(d_Ph_minus[i])
            total += float(weights_arr[i]) * float(signs[i]) * (float(P[s_i]) - P_hat_si)
        return total


def projected_l1_max_orthant(p_nominal: np.ndarray,
                             rho: float,
                             support: list[int],
                             n: int,
                             weights: np.ndarray | None = None
                             ) -> float:
    """Brute-force orthant-enumeration baseline for ProjectedL1MaxMILP.

    Enumerates 2^|support| sign patterns; for each, one standard LP via
    ProjectedL1MaxOrthant. Returns the max over sign patterns. Exact answer.
    Tractable only for |support| <= ~12.

    weights=None gives unit weights (= delta_z_inside_max when applied to obs).
    """
    support_sorted = sorted(set(int(z) for z in support))
    k = len(support_sorted)
    if k == 0:
        return 0.0
    if k > 12:
        raise ValueError(f"orthant enumeration impractical at |support| = {k} (2^{k} LPs)")

    if weights is None:
        weights_arr = np.ones(k, dtype=np.float64)
    else:
        weights_arr = np.asarray(weights, dtype=np.float64)
        if weights_arr.shape != (k,):
            raise ValueError(f"weights shape {weights_arr.shape} != ({k},)")

    lp = ProjectedL1MaxOrthant(n=n, support=support_sorted)
    best = 0.0
    for sign_idx in range(2 ** k):
        signs = np.array([1.0 if (sign_idx >> i) & 1 else -1.0 for i in range(k)],
                         dtype=np.float64)
        val = lp.solve_orthant(p_nominal, rho, signs, weights=weights_arr)
        if val > best:
            best = val
    upper = 2.0 * float(weights_arr.max())
    return max(0.0, min(upper, best))
