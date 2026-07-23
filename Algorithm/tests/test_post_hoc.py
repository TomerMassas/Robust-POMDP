"""
Test 5 (headline) — post-hoc guarantee validity, on the synthetic domain, on the
NEW core: |V_full^pi - V_proj^pi| <= eps for a fixed policy pi.

This is the first place S_next + continuation-δ + tighter-bound + our belief are checked
together against real ground truth.

Checks:
  A. rho=0 full-support V_full^pi equals a direct nominal value recursion
     (self-contained ground-truth sanity).
  B. validity |V_full - V_proj| <= eps across rho x support x (full-depth).
  C. frontier validity: a truncated (max_depth) tree stays bounded.
  D. full support -> eps ~ 0 (no projection, tight).

Run:  python -m pytest Algorithm/tests/test_post_hoc.py -q
"""

import numpy as np

from Algorithm.core.uncertainty import uniform_uncertainty
from Algorithm.domains.synthetic import (ABORT, INSPECT, MITIGATE, PROCEED,
                                         make_initial_belief, make_synthetic_pomdp)
from Algorithm.experiments.post_hoc_guarantees.fixed_policy_tree import (
    certificate_eps, eval_fixed_policy, r_max)
from Algorithm.experiments.post_hoc_guarantees.full_solver import solve_full


def _warning_policy(M, warning_low=0.4, warning_high=0.75):
    """Depth-0 -> inspect; else action by last-obs warning level (A1 design rule)."""
    def policy(obs_path):
        if len(obs_path) == 0:
            return INSPECT
        warn = obs_path[-1] / (M - 1)
        if warn < warning_low:
            return PROCEED
        if warn < warning_high:
            return MITIGATE
        return ABORT
    return policy


def _nominal_value(model, policy, b0, H, abort_action):
    """Direct nominal value of the fixed policy (rho=0 ground truth), full support."""
    def _rec(depth, obs_path, belief):
        a = policy(obs_path)
        r = float(sum(belief[s] * model.R[s, a] for s in range(model.n_states)))
        if depth == H - 1 or a == abort_action:
            return r
        cont = 0.0
        nxt = belief @ model.T[:, a, :]                       # Σ_s b[s] T[s,a,·]
        for z in range(model.n_obs):
            joint = nxt * model.O[:, z]                        # P(s', z | b, a)
            pz = float(joint.sum())
            if pz <= 0.0:
                continue
            cont += pz * _rec(depth + 1, obs_path + (z,), joint / pz)
        return r + cont
    return _rec(0, (), np.asarray(b0, dtype=np.float64))


def test_full_value_rho0_matches_nominal():
    N, M, H = 5, 3, 3
    model, b0 = make_synthetic_pomdp(N, M), make_initial_belief(N)
    policy = _warning_policy(M)
    unc = uniform_uncertainty(N, model.n_actions, 0.0, 0.0)
    root = eval_fixed_policy(model, unc, policy, b0, H, list(range(N)), list(range(M)), ABORT, {})
    expected = _nominal_value(model, policy, b0, H, ABORT)
    assert abs(root.V_rob - expected) < 1e-9, (root.V_rob, expected)


def test_h1_exact_root_reward():
    """H=1: root is terminal -> V_proj is the exact immediate reward for ANY projected S_in."""
    N, M, H = 5, 3, 1
    model, b0 = make_synthetic_pomdp(N, M), make_initial_belief(N)
    policy = _warning_policy(M)
    unc = uniform_uncertainty(N, model.n_actions, 0.1, 0.1)
    a = policy(())
    expected = float(np.asarray(b0) @ model.R[:, a])
    for S_in in ([0], [0, 2], [1, 3, 4], list(range(N))):
        root = eval_fixed_policy(model, unc, policy, b0, H, S_in, [0, 1], ABORT, {})
        assert abs(root.V_rob - expected) < 1e-12, (S_in, root.V_rob, expected)


def test_post_hoc_validity_sweep():
    N, M, H = 5, 3, 3
    model, b0 = make_synthetic_pomdp(N, M), make_initial_belief(N)
    policy = _warning_policy(M)
    Rmax = r_max(model)
    S_all, Z_all = list(range(N)), list(range(M))
    for rho in [0.0, 0.05, 0.15]:
        unc = uniform_uncertainty(N, model.n_actions, rho, rho)
        v_full = eval_fixed_policy(model, unc, policy, b0, H, S_all, Z_all, ABORT, {}).V_rob
        for S_in in ([0, 1, 2], [0, 2, 4], [0, 1, 2, 3]):
            for Z_in in ([0, 1], [1, 2], [0, 1, 2]):
                proj = eval_fixed_policy(model, unc, policy, b0, H, S_in, Z_in, ABORT, {})
                eps = certificate_eps(proj, b0, H, Rmax)
                err = abs(v_full - proj.V_rob)
                assert err <= eps + 1e-6, (rho, S_in, Z_in, v_full, proj.V_rob, eps, err)


def test_post_hoc_frontier_validity():
    """Truncated tree (frontier leaves below depth H-1) stays bounded by eps."""
    N, M, H = 5, 3, 4
    model, b0 = make_synthetic_pomdp(N, M), make_initial_belief(N)
    policy = _warning_policy(M)
    Rmax = r_max(model)
    unc = uniform_uncertainty(N, model.n_actions, 0.1, 0.1)
    v_full = eval_fixed_policy(model, unc, policy, b0, H, list(range(N)), list(range(M)), ABORT, {}).V_rob
    for cut in (1, 2):
        proj = eval_fixed_policy(model, unc, policy, b0, H, [0, 1, 2], [0, 1], ABORT, {}, max_depth=cut)
        eps = certificate_eps(proj, b0, H, Rmax)
        assert abs(v_full - proj.V_rob) <= eps + 1e-6, (cut, v_full, proj.V_rob, eps)


def test_post_hoc_full_support_eps_zero():
    """At full support there is no projection -> eps ~ 0 (certificate is tight)."""
    N, M, H = 5, 3, 3
    model, b0 = make_synthetic_pomdp(N, M), make_initial_belief(N)
    policy = _warning_policy(M)
    Rmax = r_max(model)
    unc = uniform_uncertainty(N, model.n_actions, 0.12, 0.12)
    root = eval_fixed_policy(model, unc, policy, b0, H, list(range(N)), list(range(M)), ABORT, {})
    assert certificate_eps(root, b0, H, Rmax) < 1e-6, certificate_eps(root, b0, H, Rmax)


# --- greedy policy (extracted + frozen from the full solve) --------------------

def _nominal_optimal(model, b0, H, abort_action):
    """Direct nominal OPTIMAL value (rho=0 ground truth for the expectimax solve)."""
    def _rec(depth, belief):
        best = -np.inf
        for a in range(model.n_actions):
            r = float(sum(belief[s] * model.R[s, a] for s in range(model.n_states)))
            if depth == H - 1 or a == abort_action:
                val = r
            else:
                nxt = belief @ model.T[:, a, :]
                cont = 0.0
                for z in range(model.n_obs):
                    joint = nxt * model.O[:, z]
                    pz = float(joint.sum())
                    if pz > 0.0:
                        cont += pz * _rec(depth + 1, joint / pz)
                val = r + cont
            best = max(best, val)
        return best
    return _rec(0, np.asarray(b0, dtype=np.float64))


def test_greedy_solve_rho0_matches_nominal_optimal():
    N, M, H = 5, 3, 3
    model, b0 = make_synthetic_pomdp(N, M), make_initial_belief(N)
    unc = uniform_uncertainty(N, model.n_actions, 0.0, 0.0)
    V_full, _, _ = solve_full(model, unc, b0, H, ABORT, {})
    assert abs(V_full - _nominal_optimal(model, b0, H, ABORT)) < 1e-9, V_full


def test_greedy_extraction_consistency():
    """Re-evaluating the frozen greedy pi_a on full support returns q_root[a]."""
    N, M, H = 5, 3, 3
    model, b0 = make_synthetic_pomdp(N, M), make_initial_belief(N)
    unc = uniform_uncertainty(N, model.n_actions, 0.05, 0.05)
    S_all, Z_all = list(range(N)), list(range(M))
    _, q_root, pbra = solve_full(model, unc, b0, H, ABORT, {})
    for a, pi_a in pbra.items():
        root = eval_fixed_policy(model, unc, (lambda op, _p=pi_a: _p[op]),
                                 b0, H, S_all, Z_all, ABORT, {})
        assert abs(root.V_rob - q_root[a]) < 1e-9, (a, root.V_rob, q_root[a])


def test_greedy_post_hoc_validity():
    """Frozen greedy policy: |q_root[a] - V_proj(a)| <= eps(a) on projected supports."""
    N, M, H = 5, 3, 3
    model, b0 = make_synthetic_pomdp(N, M), make_initial_belief(N)
    Rmax = r_max(model)
    for rho in [0.0, 0.05, 0.15]:
        unc = uniform_uncertainty(N, model.n_actions, rho, rho)
        _, q_root, pbra = solve_full(model, unc, b0, H, ABORT, {})
        for a, pi_a in pbra.items():
            for S_in in ([0, 1, 2], [0, 2, 4]):
                for Z_in in ([0, 1], [0, 2]):
                    proj = eval_fixed_policy(model, unc, (lambda op, _p=pi_a: _p[op]),
                                             b0, H, S_in, Z_in, ABORT, {})
                    eps = certificate_eps(proj, b0, H, Rmax)
                    err = abs(q_root[a] - proj.V_rob)
                    assert err <= eps + 1e-6, (rho, a, S_in, Z_in, q_root[a], proj.V_rob, eps, err)
