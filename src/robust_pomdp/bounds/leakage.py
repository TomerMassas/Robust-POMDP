"""
LeakageComputer - inside-trajectory primitives for Phase 6 (paper Eq. 73).

For a fixed (model, uncertainty, S_in, Z_in), computes the two leakage-side
quantities under rectangular uncertainty:

  wZ_in_min(s_next) = inf over P_Z in TV-ball(P_Z_o(.|s_next), rho_Z) of
                       Sum_{z in Z_in} P_Z(z | s_next)
    Worst-case in-support obs mass; the dual of wZ_max (Phase 5).
    Closed-form: 1.0 when Z_in = Z (mass can't leave the obs space);
                 max(0, baseline - rho_Z/2) when Z_in is proper subset.

  robust_inside_prob_step(s, a) = inf over (P_T, P_Z) of one-step inside-
                                  support probability from (s, a):
        Sum_{s' in S_in} P_T(s' | s, a) * Sum_{z in Z_in} P_Z(z | s')
    By rectangularity, the inner inf over P_Z(.|s') for each s' independently
    gives wZ_in_min(s'); the outer inf over P_T then becomes a linear-objective
    min over the full TV-ball with c[s'] = wZ_in_min(s') for s' in S_in, 0
    elsewhere. Both solved by TVBallLinearMinLP.

The tree DP for cumulative inside-trajectory probability up to depth j (paper
Eq. 73 with a horizon) is built on these primitives in Phase 6 sub-step 2.

The DeltaKComputer is the inside-mismatch path; LeakageComputer is the
omitted-trajectory path. Phase 6's certificate aggregator combines both.
"""

from __future__ import annotations

import numpy as np

from robust_pomdp import TabularPOMDP
from robust_pomdp.bounds.tv_ball_lp import TVBallLinearMinLP
from robust_pomdp.uncertainty.uncertainty_sets import UncertaintySets


class LeakageComputer:
    """Inside-trajectory primitives for paper Eq. 73. One instance per
    (model, uncertainty, S_in, Z_in)."""

    def __init__(self,
                 model: TabularPOMDP,
                 uncertainty: UncertaintySets,
                 S_in: list[int],
                 Z_in: list[int]) -> None:
        S_in_sorted = sorted(set(int(s) for s in S_in))
        Z_in_sorted = sorted(set(int(z) for z in Z_in))
        if not S_in_sorted or not Z_in_sorted:
            raise ValueError("S_in and Z_in must be non-empty")
        for s in S_in_sorted:
            if not 0 <= s < model.n_states:
                raise ValueError(f"S_in index {s} out of range for n_states={model.n_states}")
        for z in Z_in_sorted:
            if not 0 <= z < model.n_obs:
                raise ValueError(f"Z_in index {z} out of range for n_obs={model.n_obs}")

        self.model = model
        self.uncertainty = uncertainty
        self.S_in = S_in_sorted
        self.Z_in = Z_in_sorted

        # Persistent LP solvers (built once, reused across all queries).
        self._obs_lp = TVBallLinearMinLP(n=model.n_obs)
        self._state_lp = TVBallLinearMinLP(n=model.n_states)

        # Fixed indicator vector for Z_in (used as the wZ_in_min objective).
        self._c_Z_in_indicator = np.zeros(model.n_obs, dtype=np.float64)
        for z in Z_in_sorted:
            self._c_Z_in_indicator[z] = 1.0

        # Lazy caches.
        self._wz_in_min_cache: dict[int, float] = {}
        self._step_cache: dict[tuple[int, int], float] = {}

    def wZ_in_min(self, s_next: int) -> float:
        if s_next not in self._wz_in_min_cache:
            p_nom = self.model.O[s_next, :]
            rho = self.uncertainty.observation_radius(s_next)
            result = self._obs_lp.solve(p_nom, rho, self._c_Z_in_indicator)
            self._wz_in_min_cache[s_next] = max(0.0, min(1.0, result))
        return self._wz_in_min_cache[s_next]

    def robust_inside_prob_step(self, s: int, a: int) -> float:
        key = (s, a)
        if key not in self._step_cache:
            # Outer min over P_T uses c[s'] = wZ_in_min(s') for s' in S_in (else 0).
            c = np.zeros(self.model.n_states, dtype=np.float64)
            for s_p in self.S_in:
                c[s_p] = self.wZ_in_min(s_p)
            p_nom = self.model.T[s, a, :]
            rho = self.uncertainty.transition_radius(s, a)
            result = self._state_lp.solve(p_nom, rho, c)
            self._step_cache[key] = max(0.0, min(1.0, result))
        return self._step_cache[key]

    # -- raw LP wrappers for the inside-trajectory tree DP (Phase 6 sub-step 2) --

    def obs_min_over_TV_ball(self, s_next: int, c: np.ndarray) -> float:
        """min over P_Z(.|s_next) in TV-ball of Sum_z c[z] * P_Z(z).
        Used by the inner step of the inside-trajectory recursion."""
        p_nom = self.model.O[s_next, :]
        rho = self.uncertainty.observation_radius(s_next)
        return self._obs_lp.solve(p_nom, rho, c)

    def state_min_over_TV_ball(self, s: int, a: int, c: np.ndarray) -> float:
        """min over P_T(.|s, a) in TV-ball of Sum_{s'} c[s'] * P_T(s').
        Used by the outer step of the inside-trajectory recursion."""
        p_nom = self.model.T[s, a, :]
        rho = self.uncertainty.transition_radius(s, a)
        return self._state_lp.solve(p_nom, rho, c)
