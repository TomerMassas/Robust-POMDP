"""
DeltaKExactOrthant - EXACT robust kernel mismatch (paper Eq. 71), small-scale.

Computes the exact one-step robust kernel mismatch
    Delta_K^rob(s,a) = sup over (P_T, P_Z, P_T^in, P_Z^in) of
                         Sum_{(z,s') in Z_in x S_in} |K(z,s'|s,a) - K_hat(z,s'|s,a)|
    K(z,s'|s,a)     = P_T(s'|s,a) * P_Z(z|s')          (full one-step kernel)
    K_hat(z,s'|s,a) = P_T^in(s'|s,a) * P_Z^in(z|s')    (projected one-step kernel)
and Delta_K^rob(a) = max over s in S_in of Delta_K^rob(s,a).

The sum is over the SHARED SUPPORT Z_in x S_in only -- this is the inside-support
mismatch (paper Eq. 25 / 72). Out-of-support leakage is the SEPARATE omitted-
trajectory term (Eq. 73); including it here would double-count it.

This is the EXACT quantity that DeltaKComputer (delta_k.py) upper-bounds via the
Eq. 47 split. Built as a ground-truth measurement tool to quantify the split's
looseness. NOT for production sweeps -- cost is 2^(|S_in|*|Z_in|) sign patterns.

Method (sign-orthant + rectangular decomposition):
  ||x||_1 = max_sigma <sigma, x> over sign vectors sigma in {+1,-1}. Swapping the
  two maximizations (rigorous: both are maxima; the joint sup over the independent
  (full, projected) pair separates):

    Delta_K^rob(s,a) = max_sigma [ FullSup(sigma) - ProjInf(sigma) ]
    FullSup(sigma) = max_{P_T in ball}     Sum_{s'} P_T(s') * c_{s'}(sigma)
    ProjInf(sigma) = min_{P_T^in in pball} Sum_{s' in S_in} P_T^in(s') * chat_{s'}(sigma)
    c_{s'}(sigma)    = max_{P_Z(.|s') in ball}      <sigma_{.,s'}, P_Z>        (full obs ball)
    chat_{s'}(sigma) = min_{P_Z^in(.|s') in pball}  <sigma_{.,s'}|Z_in, P_Z^in> (projected obs ball)

  Rectangularity lets the obs sup/inf pull inside the transition sum independently
  per s' (valid since P_T(s'), P_T^in(s') >= 0), so once sigma is fixed everything
  is linear-over-a-ball -- no bilinearity. c_{s'}, chat_{s'} depend only on s' and
  that column's signs, so they are precomputed once and reused across all (s,a).

All |S_in| x |Z_in| signs are free (the sum has no out-of-support region). The
obs coefficient is 0 for z not in Z_in, and c_{s'} = 0 for s' not in S_in -- the
full P_T / P_Z may still place mass there, but it carries zero objective weight.

TV convention: ||P-Q||_1 (no 1/2). Handled inside the reused solvers.
"""

from __future__ import annotations

import itertools

import numpy as np

from robust_pomdp import TabularPOMDP
from robust_pomdp.bounds.tv_ball_lp import TVBallLinearMinLP
from robust_pomdp.uncertainty.projected_sets import ProjectedTVBall
from robust_pomdp.uncertainty.uncertainty_sets import UncertaintySets


class DeltaKExactOrthant:
    """Exact Delta_K^rob via sign-orthant enumeration. One instance per
    (model, uncertainty, S_in, Z_in). Small-scale only."""

    def __init__(self,
                 model: TabularPOMDP,
                 uncertainty: UncertaintySets,
                 S_in: list[int],
                 Z_in: list[int],
                 *,
                 max_free_bits: int = 16) -> None:
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

        free_bits = len(S_in_sorted) * len(Z_in_sorted)
        if free_bits > max_free_bits:
            raise ValueError(
                f"|S_in|*|Z_in| = {free_bits} exceeds max_free_bits={max_free_bits} "
                f"(2^{free_bits} = {2**free_bits} sign patterns). This is a small-scale "
                f"ground-truth tool; raise max_free_bits explicitly to override."
            )

        self.model = model
        self.uncertainty = uncertainty
        self.S_in = S_in_sorted
        self.Z_in = Z_in_sorted

        # Persistent solvers (reused across all queries).
        self._obs_full_lp = TVBallLinearMinLP(n=model.n_obs)       # c_{s'} via negation
        self._obs_proj_lp = ProjectedTVBall(n=len(Z_in_sorted))    # chat_{s'}
        self._trans_full_lp = TVBallLinearMinLP(n=model.n_states)  # FullSup via negation
        self._trans_proj_lp = ProjectedTVBall(n=len(S_in_sorted))  # ProjInf

        self._n_patterns = 2 ** len(Z_in_sorted)
        self._build_obs_tables()

        self._exact_cache: dict[tuple[int, int], float] = {}

    def _sigma_columns(self, pattern: int) -> tuple[np.ndarray, np.ndarray]:
        """For a column pattern (a |Z_in|-bit int), return (sigma_full, sigma_zin):
        sigma_full is length n_obs (0 outside Z_in, +/-1 on Z_in per the bits);
        sigma_zin is the length-|Z_in| restriction."""
        sigma_full = np.zeros(self.model.n_obs, dtype=np.float64)
        sigma_zin = np.empty(len(self.Z_in), dtype=np.float64)
        for i, z in enumerate(self.Z_in):
            s = 1.0 if (pattern >> i) & 1 else -1.0
            sigma_full[z] = s
            sigma_zin[i] = s
        return sigma_full, sigma_zin

    def _build_obs_tables(self) -> None:
        """Precompute c_{s'}(pattern) and chat_{s'}(pattern) for each s' in S_in
        and each of the 2^|Z_in| column patterns. Shared across all (s,a)."""
        self._c_table: dict[int, list[float]] = {}      # keyed by s' in S_in
        self._chat_table: dict[int, list[float]] = {}
        Z_in = self.Z_in
        for s_p in self.S_in:
            rho_z = self.uncertainty.observation_radius(s_p)
            o_full = self.model.O[s_p, :]
            o_zin = self.model.O[s_p, Z_in]
            c_list: list[float] = []
            chat_list: list[float] = []
            for pattern in range(self._n_patterns):
                sigma_full, sigma_zin = self._sigma_columns(pattern)
                # c = max <sigma_full, P_Z> = -min <-sigma_full, P_Z> over full ball.
                c_list.append(-self._obs_full_lp.solve(o_full, rho_z, -sigma_full))
                # chat = min <sigma_zin, P_Z^in> over projected ball.
                chat_val, _ = self._obs_proj_lp.solve(o_zin, rho_z, sigma_zin)
                chat_list.append(chat_val)
            self._c_table[s_p] = c_list
            self._chat_table[s_p] = chat_list

    def delta_k_exact(self, s: int, a: int) -> float:
        """Exact Delta_K^rob(s,a): max over joint sign patterns of FullSup - ProjInf."""
        key = (s, a)
        if key in self._exact_cache:
            return self._exact_cache[key]

        rho_t = self.uncertainty.transition_radius(s, a)
        t_full = self.model.T[s, a, :]
        t_proj = self.model.T[s, a, self.S_in]

        best = -np.inf
        # Joint pattern = one column choice per s' in S_in.
        for joint in itertools.product(range(self._n_patterns), repeat=len(self.S_in)):
            c_vec = np.zeros(self.model.n_states, dtype=np.float64)  # 0 for s' not in S_in
            chat_vec = np.empty(len(self.S_in), dtype=np.float64)
            for idx, s_p in enumerate(self.S_in):
                c_vec[s_p] = self._c_table[s_p][joint[idx]]
                chat_vec[idx] = self._chat_table[s_p][joint[idx]]
            # FullSup = max <c, P_T> = -min <-c, P_T> over full transition ball.
            full_sup = -self._trans_full_lp.solve(t_full, rho_t, -c_vec)
            proj_inf, _ = self._trans_proj_lp.solve(t_proj, rho_t, chat_vec)
            bracket = full_sup - proj_inf
            if bracket > best:
                best = bracket

        self._exact_cache[key] = float(best)
        return self._exact_cache[key]

    def delta_k_rob_exact(self, a: int) -> float:
        """Exact Delta_K^rob(a) = max over source s in S_in of delta_k_exact(s,a)."""
        return max(self.delta_k_exact(s, a) for s in self.S_in)