"""
Leakage certificate core — the survival scalar `c` (one value per state).

`c` is the tighter shared-adversary leakage term, computed like `robust_q`
(O(H) per path). The final certificate is  eps = R_max * (H - sum_s b0(s)*c(s)).

For a fixed policy at a node (depth k), c[s] = worst-case in-support survival
SUMMED over the remaining reward depths, starting from state s:

    c[s] = 1 + min_{P_T ∈ ProjTVball(T[s,a,·]|S_next, ρ_T(s,a))}
                 Σ_{s'∈S_next} P_T[s'] · cw[s']
    cw[s'] = min_{P_Z ∈ ProjTVball(O[s',·]|Z_in, ρ_Z(s'))}
                 Σ_{z∈Z_in} P_Z[z] · child_c[z].get(s', 0)

The leading `1` is this node's own-depth survival term (survival to depth k
from state s is trivially 1). Both minimizations use the PROJECTED ball; by
Lemma 2 this equals the full-ball leakage (off-support entries have coefficient
0 → the adversary leaks there for free). Same cached LPs as `robust_q`.

Base cases (terminal / frontier) are handled by `backup`, not here — this
function is only called for interior actions.
"""

from __future__ import annotations

import numpy as np

from Algorithm.core.pomdp import TabularPOMDP
from Algorithm.core.projected_tv_ball import ProjectedTVBall
from Algorithm.core.uncertainty import UncertaintySets


def leakage_c(model: TabularPOMDP,
              uncertainty: UncertaintySets,
              a: int,
              S_in: list[int],
              Z_in: list[int],
              S_next: list[int],
              child_c: dict[int, dict[int, float]],
              lp_cache: dict[int, ProjectedTVBall] | None = None,
              ) -> dict[int, float]:
    """Interior-action survival scalars c[s] for s in S_in.

    Args:
        child_c: {z: {s': survival}} — survival dict of the child reached by z.
        lp_cache: shared {n: ProjectedTVBall(n)} (fresh if None).
    Returns {s: c[s]} with the own-depth +1 already included.
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

    # Observation step: cw[s'] per next-state (child survival routed through obs z).
    cw = np.empty(n_sn, dtype=np.float64)
    for i, s_next in enumerate(S_next):
        coeffs = np.array([child_c[z].get(s_next, 0.0) for z in Z_in], dtype=np.float64)
        p_nom = np.array([model.O[s_next, z] for z in Z_in], dtype=np.float64)
        rho_z = uncertainty.observation_radius(s_next)
        cw[i], _ = lp_obs.solve(p_nom, rho_z, coeffs, full_support=full_obs)

    # Transition step: c[s] = 1 + worst-case continuation.
    c: dict[int, float] = {}
    for s in S_in:
        p_nom = np.array([model.T[s, a, s_next] for s_next in S_next], dtype=np.float64)
        rho_t = uncertainty.transition_radius(s, a)
        cont, _ = lp_trans.solve(p_nom, rho_t, cw, full_support=full_trans)
        c[int(s)] = 1.0 + cont
    return c
