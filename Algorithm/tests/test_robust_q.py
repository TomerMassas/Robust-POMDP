"""
Test 3 — robust_q (two-step rectangular LP over the projected ball).

Independent checks (do NOT reuse robust_q internals):
  A. rho=0 equals the plain nominal expectation over the projected support.
  B. Q is non-increasing in rho (the adversary can only lower it).
  C. on a 2-state / 2-obs full-support case, the two-step LP equals a brute-force
     joint worst-case min (validates the decomposition + the LP).

Run:  python -m pytest Algorithm/tests/test_robust_q.py -q
"""

import numpy as np

from Algorithm.core.pomdp import TabularPOMDP
from Algorithm.core.uncertainty import uniform_uncertainty
from Algorithm.core.robust_q import robust_q


def _small_pomdp(rng, n_states=3, n_actions=2, n_obs=3):
    T = rng.random((n_states, n_actions, n_states))
    T /= T.sum(axis=2, keepdims=True)
    O = rng.random((n_states, n_obs))
    O /= O.sum(axis=1, keepdims=True)
    R = rng.uniform(-5.0, 5.0, size=(n_states, n_actions))
    return TabularPOMDP.from_matrices(T, O, R)


def _nominal_q(model, a, belief, S_in, Z_in, S_next, V_child):
    """Direct nominal expectation over the projected support (rho=0 reference)."""
    w = {sp: sum(model.O[sp, z] * V_child[z] for z in Z_in) for sp in S_next}
    Q = 0.0
    for i, s in enumerate(S_in):
        sig = sum(model.T[s, a, sp] * w[sp] for sp in S_next)
        Q += belief[i] * (model.R[s, a] + sig)
    return Q


def test_rho_zero_equals_nominal():
    rng = np.random.default_rng(3)
    model = _small_pomdp(rng)
    S_in, Z_in, S_next = [0, 1, 2], [0, 1], [0, 1]  # projected obs (drop z=2) + projected next-state (drop s=2)
    belief = np.array([0.5, 0.3, 0.2])
    V_child = {0: 2.0, 1: -1.0}
    unc = uniform_uncertainty(model.n_states, model.n_actions, 0.0, 0.0)
    Q, _, _ = robust_q(model, unc, 0, belief, S_in, Z_in, S_next, V_child)
    assert abs(Q - _nominal_q(model, 0, belief, S_in, Z_in, S_next, V_child)) < 1e-9


def test_monotone_in_rho():
    rng = np.random.default_rng(4)
    model = _small_pomdp(rng)
    S_in, Z_in, S_next = [0, 1, 2], [0, 1], [0, 1]
    belief = np.array([0.5, 0.3, 0.2])
    V_child = {0: 2.0, 1: -1.0}
    prev = None
    for rho in [0.0, 0.1, 0.3, 0.6]:
        unc = uniform_uncertainty(model.n_states, model.n_actions, rho, rho)
        Q, _, _ = robust_q(model, unc, 0, belief, S_in, Z_in, S_next, V_child)
        if prev is not None:
            assert Q <= prev + 1e-9, (rho, Q, prev)
        prev = Q


def test_two_step_matches_bruteforce_full_ball():
    """2 states, 2 obs, 1 action, full support -> full TV ball vs brute-force joint min."""
    T = np.zeros((2, 1, 2))
    T[0, 0, :] = [0.6, 0.4]
    T[1, 0, :] = [0.3, 0.7]
    O = np.array([[0.7, 0.3], [0.2, 0.8]])
    R = np.array([[1.0], [2.0]])
    model = TabularPOMDP.from_matrices(T, O, R)
    V_child = {0: 3.0, 1: -2.0}
    belief = np.array([0.55, 0.45])
    rho_t, rho_z = 0.5, 0.4
    unc = uniform_uncertainty(2, 1, rho_t, rho_z)

    Q, _, _ = robust_q(model, unc, 0, belief, [0, 1], [0, 1], [0, 1], V_child)

    # Brute force (full TV ball: for 2 outcomes, |p - p_nom| <= rho/2).
    def _min_linear(p_nom, half, v0, v1):
        lo, hi = max(0.0, p_nom - half), min(1.0, p_nom + half)
        grid = np.linspace(lo, hi, 4001)
        return float(np.min(grid * v0 + (1.0 - grid) * v1))

    w0 = _min_linear(model.O[0, 0], rho_z / 2, V_child[0], V_child[1])
    w1 = _min_linear(model.O[1, 0], rho_z / 2, V_child[0], V_child[1])
    sig0 = _min_linear(model.T[0, 0, 0], rho_t / 2, w0, w1)
    sig1 = _min_linear(model.T[1, 0, 0], rho_t / 2, w0, w1)
    Q_bf = belief[0] * (model.R[0, 0] + sig0) + belief[1] * (model.R[1, 0] + sig1)

    assert abs(Q - Q_bf) < 5e-3, (Q, Q_bf)
