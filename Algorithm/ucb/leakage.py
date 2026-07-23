"""
Leakage certificate core — the continuation-survival δ (one value per state).

`δ` is the paper's Eq. 50 recursion, computed like `robust_q` (O(H) per path).
The final certificate is  eps = R_max·((H-1) - Σ_s b0(s)·δ(s))  (Eq. 49).

For a fixed policy at a node (depth k), δ[s] = worst-case in-support survival
COUNT over the remaining CONTINUATION depths (k+1..H-1), starting from state s:

    δ[s]  = min_{P_T ∈ ProjTVball(T[s,a,·]|S_next, ρ_T(s,a))}
                  Σ_{s'∈S_next} P_T[s'] · w[s']
    w[s'] = min_{P_Z ∈ ProjTVball(O[s',·]|Z_in, ρ_Z(s'))}
                  Σ_{z∈Z_in} P_Z[z] · (1 + δ_child_z[s'])      (0 if s' leaks out of child z)

The `1 +` inside the obs step is the child's own-depth survival (surviving into
depth k+1); δ_child counts deeper. There is NO own-depth +1 on δ[s] itself — it
is a continuation count, so the terminal is δ_{H-1} = 0 (Eq. 51). Both minimizations
use the PROJECTED ball; by Lemma 2 this equals the full-ball leakage (off-support
entries have coefficient 0 → the adversary leaks there for free). Same cached LPs
as `robust_q`.

Base cases (terminal δ = H-1-depth, frontier δ = 0) are handled by `backup`, not
here — this function is only called for interior actions.
"""

from __future__ import annotations

import numpy as np

from Algorithm.core.pomdp import TabularPOMDP
from Algorithm.core.projected_tv_ball import ProjectedTVBall
from Algorithm.core.uncertainty import UncertaintySets


def leakage_delta(model: TabularPOMDP,
                  uncertainty: UncertaintySets,
                  a: int,
                  S_in: list[int],
                  Z_in: list[int],
                  S_next: list[int],
                  child_delta: dict[int, dict[int, float]],
                  lp_cache: dict[int, ProjectedTVBall] | None = None,
                  ) -> dict[int, float]:
    """Interior-action continuation-survival δ[s] for s in S_in.

    Args:
        child_delta: {z: {s': δ}} — continuation-survival dict of the child reached by z.
        lp_cache: shared {n: ProjectedTVBall(n)} (fresh if None).
    Returns {s: δ[s]} — continuation count only (the own-depth +1 lives in the
    parent's obs coefficient, not here).
    """
    if lp_cache is None:
        lp_cache = {}
    n_z, n_sn = len(Z_in), len(S_next)
    full_obs = (n_z == model.n_obs)
    full_trans = (n_sn == model.n_states)
    if n_z not in lp_cache:
        lp_cache[n_z] = ProjectedTVBall(n_z)
    if n_sn not in lp_cache:
        lp_cache[n_sn] = ProjectedTVBall(n_sn)
    lp_obs, lp_trans = lp_cache[n_z], lp_cache[n_sn]

    # Observation step: w[s'] per next-state = worst-case (1 + child continuation),
    # routed through obs z; s' leaking out of child z contributes 0.
    w = np.empty(n_sn, dtype=np.float64)
    for i, s_next in enumerate(S_next):
        coeffs = np.array([(1.0 + child_delta[z][s_next]) if s_next in child_delta[z] else 0.0
                           for z in Z_in], dtype=np.float64)
        p_nom = np.array([model.O[s_next, z] for z in Z_in], dtype=np.float64)
        rho_z = uncertainty.observation_radius(s_next)
        w[i], _ = lp_obs.solve(p_nom, rho_z, coeffs, full_support=full_obs)

    # Transition step: δ[s] = worst-case continuation survival (no own-depth +1).
    delta: dict[int, float] = {}
    for s in S_in:
        p_nom = np.array([model.T[s, a, s_next] for s_next in S_next], dtype=np.float64)
        rho_t = uncertainty.transition_radius(s, a)
        cont, _ = lp_trans.solve(p_nom, rho_t, w, full_support=full_trans)
        delta[int(s)] = cont
    return delta
