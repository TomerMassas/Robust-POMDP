"""
Robust action-value backup via the two-step rectangular LP.

Shared primitive for the full-support ground-truth solver and the UCB planner
(and later the RUDB planner's value term). Computes, for a fixed (belief, action):

    Q_rob = sum_{s in S_in} belief[s] * ( R[s,a] + sigma[s] )
    sigma[s] = min_{P_T in ProjTVball(T[s,a,.]|S_next, rho_T(s,a))}  sum_{s' in S_next} P_T[s'] * w[s']
    w[s']    = min_{P_Z in ProjTVball(O[s',.]|Z_in,  rho_Z(s'))}    sum_{z in Z_in}   P_Z[z]  * V_child[z]

Both minimizations are over the PROJECTED TV ball (sub-probabilities on the
sampled support; leakage is charged on the TV budget). Under rectangularity the
two-step decomposition is exact (obs worst-case per next-state, then transition
worst-case per current state).

DELIBERATE change from src/robust_pomdp compute_robust_q: next-states range over
S_next = union_z child_z.S_in (the sampled next-state support, paper Eq. 8),
NOT the node's own S_in. The certificate-validity test is what confirms this.

Everything is passed explicitly (no tree objects) so this is unit-testable in
isolation. `lp_cache` maps support size n -> a persistent ProjectedTVBall(n),
reused across the whole planning call.
"""

from __future__ import annotations

import numpy as np

from Algorithm.core.pomdp import TabularPOMDP
from Algorithm.core.projected_tv_ball import ProjectedTVBall
from Algorithm.core.uncertainty import UncertaintySets


def robust_q(model: TabularPOMDP,
             uncertainty: UncertaintySets,
             a: int,
             belief: np.ndarray,
             S_in: list[int],
             Z_in: list[int],
             S_next: list[int],
             V_child: dict[int, float],
             lp_cache: dict[int, ProjectedTVBall] | None = None,
             ) -> tuple[float, np.ndarray, np.ndarray]:
    """Robust Q(belief, a) over the projected support.

    Args:
        model, uncertainty: the POMDP and its uncertainty radii.
        a: action index (0-based).
        belief: length-|S_in| vector aligned with S_in (sorted). Need not be
            normalized (root uses the unnormalized restriction of b0).
        S_in: sorted current-state support (belief lives here).
        Z_in: sorted observation support.
        S_next: sorted next-state support (= union of children's S_in).
        V_child: {z: V_robust of the child reached by z}, for z in Z_in.
        lp_cache: {n: ProjectedTVBall(n)} reused across calls (fresh if None).

    Returns:
        (Q, w, sigma) where w is aligned with S_next and sigma with S_in.
        (w/sigma are returned for later caching / dirty-flag reuse; RUDB will
        additionally need the argmin models, one line away via solve().)
    """
    if lp_cache is None:
        lp_cache = {}
    belief = np.asarray(belief, dtype=np.float64)
    assert belief.shape == (len(S_in),), \
        f"belief shape {belief.shape} != ({len(S_in)},)"

    n_z, n_sn = len(Z_in), len(S_next)
    full_obs = (n_z == model.n_obs)
    full_trans = (n_sn == model.n_states)

    if n_z not in lp_cache:
        lp_cache[n_z] = ProjectedTVBall(n_z)
    if n_sn not in lp_cache:
        lp_cache[n_sn] = ProjectedTVBall(n_sn)
    lp_obs = lp_cache[n_z]
    lp_trans = lp_cache[n_sn]

    # Objective coefficients for the observation step (independent of s').
    V_coeff = np.array([V_child.get(z, 0.0) for z in Z_in], dtype=np.float64)

    # Step 1 -- worst-case observation model, one solve per next-state s'.
    w = np.empty(n_sn, dtype=np.float64)
    for i, s_next in enumerate(S_next):
        p_nom = np.array([model.O[s_next, z] for z in Z_in], dtype=np.float64)
        rho_z = uncertainty.observation_radius(s_next)
        w[i], _ = lp_obs.solve(p_nom, rho_z, V_coeff, full_support=full_obs)

    # Step 2 -- worst-case transition model, one solve per current state s.
    sigma = np.empty(len(S_in), dtype=np.float64)
    Q = 0.0
    for i, s in enumerate(S_in):
        p_nom = np.array([model.T[s, a, s_next] for s_next in S_next], dtype=np.float64)
        rho_t = uncertainty.transition_radius(s, a)
        sigma[i], _ = lp_trans.solve(p_nom, rho_t, w, full_support=full_trans)
        Q += belief[i] * (model.reward(s, a) + sigma[i])

    return float(Q), w, sigma
