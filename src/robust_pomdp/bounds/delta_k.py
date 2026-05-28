"""
DeltaKComputer - orchestrator for Phase 5's Delta_K^rob_split (paper Eq. 47 + 71).

For a fixed (model, uncertainty, S_in, Z_in), computes the Eq. 47 decoupled
bound:
    Delta_K^rob_split(h, a) = max over s in S_in of [
        transition_part(s, a) + observation_part(s, a)
    ]
The four piece supremums are:
    wZ_max(s')             = sup_O      Sum_{z in Z_in} O(z | s')
    delta_z_inside_max(s') = sup_O,Ohat Sum_{z in Z_in} |O(z|s') - Ohat(z|s')|
    transition_part(s, a)  = sup_T,That Sum_{s' in S_in} wZ_max(s') |T(s'|s,a) - That(s'|s,a)|
    observation_part(s, a) = sup_That   Sum_{s' in S_in} That(s'|s,a) * delta_z_inside_max(s')

This module is the public entry point for Phase 5. All four pieces are lazy-
cached per (model, uncertainty, S_in, Z_in) instance, so sweeping over (s, a)
pairs is fast.
"""

from __future__ import annotations

import numpy as np

from robust_pomdp import TabularPOMDP
from robust_pomdp.bounds.l1_max import ProjectedL1MaxMILP
from robust_pomdp.bounds.wz_max import WZMaxLP
from robust_pomdp.uncertainty.projected_sets import ProjectedTVBall
from robust_pomdp.uncertainty.uncertainty_sets import UncertaintySets


class DeltaKComputer:
    """Orchestrator for Delta_K^rob_split(h, a). One instance per
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

        # Persistent solver instances (built once, reused across all queries).
        self._wz_lp = WZMaxLP(n=model.n_obs)
        self._delta_z_milp = ProjectedL1MaxMILP(n=model.n_obs, support=Z_in_sorted)
        self._trans_milp = ProjectedL1MaxMILP(n=model.n_states, support=S_in_sorted)
        self._obs_lp = ProjectedTVBall(n=len(S_in_sorted))

        # Lazy caches.
        self._wz_max_cache: dict[int, float] = {}
        self._delta_z_cache: dict[int, float] = {}
        self._trans_cache: dict[tuple[int, int], float] = {}
        self._obs_cache: dict[tuple[int, int], float] = {}

    # -- per-s_next quantities --------------------------------------------------

    def wZ_max(self, s_next: int) -> float:
        if s_next not in self._wz_max_cache:
            p_nom = self.model.O[s_next, :]
            rho = self.uncertainty.observation_radius(s_next)
            self._wz_max_cache[s_next] = self._wz_lp.solve(p_nom, rho, self.Z_in)
        return self._wz_max_cache[s_next]

    def delta_z_inside_max(self, s_next: int) -> float:
        if s_next not in self._delta_z_cache:
            p_nom = self.model.O[s_next, :]
            rho = self.uncertainty.observation_radius(s_next)
            self._delta_z_cache[s_next] = self._delta_z_milp.solve(p_nom, rho)  # unit weights
        return self._delta_z_cache[s_next]

    # -- per-(s, a) quantities --------------------------------------------------

    def transition_part(self, s: int, a: int) -> float:
        key = (s, a)
        if key not in self._trans_cache:
            weights = np.array([self.wZ_max(s_p) for s_p in self.S_in], dtype=np.float64)
            p_nom = self.model.T[s, a, :]
            rho = self.uncertainty.transition_radius(s, a)
            self._trans_cache[key] = self._trans_milp.solve(p_nom, rho, weights=weights)
        return self._trans_cache[key]

    def observation_part(self, s: int, a: int) -> float:
        key = (s, a)
        if key not in self._obs_cache:
            # Objective: max over T_hat in projected ball of
            #     Sum_{s' in S_in} T_hat(s' | s, a) * delta_z_inside_max(s').
            # ProjectedTVBall minimizes; negate coefficients, negate result.
            delta_z_table = np.array([self.delta_z_inside_max(s_p) for s_p in self.S_in], dtype=np.float64,)
            p_nom_restricted = np.array([self.model.T[s, a, s_p] for s_p in self.S_in], dtype=np.float64,)
            rho = self.uncertainty.transition_radius(s, a)
            min_val, _ = self._obs_lp.solve(p_nom_restricted, rho, -delta_z_table)
            self._obs_cache[key] = -float(min_val)
        return self._obs_cache[key]

    # -- aggregation ------------------------------------------------------------

    def row(self, s: int, a: int) -> float:
        return self.transition_part(s, a) + self.observation_part(s, a)

    def delta_k_rob_split(self, a: int) -> float:
        return max(self.row(s, a) for s in self.S_in)
