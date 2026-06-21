"""
LeakageComputer - one-step robust min primitives for the omitted-trajectory
(leakage) certificate.

Holds two persistent full-TV-ball minimizers and exposes them as linear-
objective wrappers:

  obs_min_over_TV_ball(s_next, c) = min over P_Z(.|s_next) in TV-ball of
                                      Sum_z c[z] * P_Z(z | s_next)
  state_min_over_TV_ball(s, a, c) = min over P_T(.|s, a) in TV-ball of
                                      Sum_{s'} c[s'] * P_T(s' | s, a)

Both take an arbitrary coefficient vector c over the FULL space, so the
in-support structure (which observations / next-states count) is encoded
purely by which entries of c are non-zero. The inside-trajectory tree DP
(inside_trajectory.py) builds those coefficients per node from the tree's
per-node supports S_in(h) and Z_in(ha) -- so this class is support-agnostic
(one instance per model, reused across all nodes and supports).
"""

from __future__ import annotations

import numpy as np

from robust_pomdp import TabularPOMDP
from robust_pomdp.bounds.tv_ball_lp import TVBallLinearMinLP
from robust_pomdp.uncertainty.uncertainty_sets import UncertaintySets


class LeakageComputer:
    """Persistent full-TV-ball min primitives for the leakage DP. One instance
    per (model, uncertainty); support is supplied per call via the coefficients."""

    def __init__(self, model: TabularPOMDP, uncertainty: UncertaintySets) -> None:
        self.model = model
        self.uncertainty = uncertainty
        self._obs_lp = TVBallLinearMinLP(n=model.n_obs)
        self._state_lp = TVBallLinearMinLP(n=model.n_states)

    def obs_min_over_TV_ball(self, s_next: int, c: np.ndarray) -> float:
        """min over P_Z(.|s_next) in TV-ball of Sum_z c[z] * P_Z(z).
        Inner step of the inside-trajectory recursion."""
        p_nom = self.model.O[s_next, :]
        rho = self.uncertainty.observation_radius(s_next)
        return self._obs_lp.solve(p_nom, rho, c)

    def state_min_over_TV_ball(self, s: int, a: int, c: np.ndarray) -> float:
        """min over P_T(.|s, a) in TV-ball of Sum_{s'} c[s'] * P_T(s').
        Outer step of the inside-trajectory recursion."""
        p_nom = self.model.T[s, a, :]
        rho = self.uncertainty.transition_radius(s, a)
        return self._state_lp.solve(p_nom, rho, c)