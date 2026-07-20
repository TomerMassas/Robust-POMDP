"""
Per-node sampler scheme (scheme="sampler") on the NEW core.

Checks:
  A. validity — |V_full^pi - V_proj^pi| <= eps for the SAMPLED tree, across K,
     seeds, rho (the headline post-hoc guarantee, now under random per-node support).
  B. eps shrinks / support grows with K (convergence, without assuming exact 0 —
     the sparse synthetic domain floors above 0, as in A1).
  C. sample_pick properties — sorted, |result| <= K, deduped, in-range, never a
     zero-probability index; and the mode dominates a peaked distribution.
  D. determinism — the same seed reproduces the same V_proj.

Run:  python -m pytest Algorithm/tests/test_sampler.py -q
"""

import numpy as np

from Algorithm.core.uncertainty import uniform_uncertainty
from Algorithm.domains.synthetic import ABORT, make_initial_belief, make_synthetic_pomdp
from Algorithm.experiments.post_hoc_guarantees.fixed_policy_tree import (
    certificate_eps, r_max)
from Algorithm.experiments.post_hoc_guarantees.full_solver import solve_full
from Algorithm.experiments.post_hoc_guarantees.sampled_policy_tree import (
    avg_support_fraction, eval_sampled_policy)
from Algorithm.experiments.post_hoc_guarantees.support_picker import sample_pick


def _greedy(model, unc, b0, H):
    """Frozen greedy policy + its true robust value (full-support ground truth)."""
    V_full, q_root, pbra = solve_full(model, unc, b0, H, ABORT, {})
    best_a = max(q_root, key=q_root.get)
    return (lambda op, _p=pbra[best_a]: _p[op]), V_full


def test_sampler_validity_sweep():
    N, M, H = 5, 3, 4
    model, b0 = make_synthetic_pomdp(N, M), make_initial_belief(N)
    Rmax = r_max(model)
    for rho in (0.0, 0.05, 0.15):
        unc = uniform_uncertainty(N, model.n_actions, rho, rho)
        policy, V_full = _greedy(model, unc, b0, H)
        for K in (1, 2, 4, 8):
            for seed in range(4):
                rng = np.random.default_rng(seed)
                root = eval_sampled_policy(model, unc, policy, b0, H, K, rng, ABORT, {})
                eps = certificate_eps(root, b0, H, Rmax)
                err = abs(V_full - root.V_rob)
                assert err <= eps + 1e-6, (rho, K, seed, V_full, root.V_rob, eps, err)


def test_sampler_eps_shrinks_with_K():
    N, M, H = 5, 3, 4
    model, b0 = make_synthetic_pomdp(N, M), make_initial_belief(N)
    Rmax = r_max(model)
    unc = uniform_uncertainty(N, model.n_actions, 0.05, 0.05)
    policy, _ = _greedy(model, unc, b0, H)

    def mean_stats(K):
        eps_l, frac_l = [], []
        for seed in range(6):
            rng = np.random.default_rng(500 + seed)
            root = eval_sampled_policy(model, unc, policy, b0, H, K, rng, ABORT, {})
            eps_l.append(certificate_eps(root, b0, H, Rmax))
            s, z = avg_support_fraction(root, N, M)
            frac_l.append(0.5 * (s + z))
        return float(np.mean(eps_l)), float(np.mean(frac_l))

    eps_lo, frac_lo = mean_stats(1)
    eps_hi, frac_hi = mean_stats(64)
    assert eps_hi < eps_lo, (eps_lo, eps_hi)      # more support -> less leakage
    assert frac_hi > frac_lo, (frac_lo, frac_hi)  # bigger K -> bigger support


def test_sample_pick_properties():
    probs = np.array([0.6, 0.3, 0.1, 0.0])
    for K in (1, 2, 3, 5):
        picks = sample_pick(probs, K, np.random.default_rng(K))
        assert picks == sorted(picks)
        assert len(picks) <= K
        assert len(set(picks)) == len(picks)
        assert all(0 <= p < len(probs) for p in picks)
        assert 3 not in picks                     # zero-probability index never drawn
    assert sample_pick(np.zeros(4), 5, np.random.default_rng(0)) == []   # no mass
    hits = sum(0 in sample_pick(probs, 4, np.random.default_rng(s)) for s in range(50))
    assert hits >= 45                             # mode (p=0.6) almost always present


def test_sampler_determinism():
    N, M, H = 5, 3, 4
    model, b0 = make_synthetic_pomdp(N, M), make_initial_belief(N)
    unc = uniform_uncertainty(N, model.n_actions, 0.1, 0.1)
    policy, _ = _greedy(model, unc, b0, H)
    v1 = eval_sampled_policy(model, unc, policy, b0, H, 8,
                             np.random.default_rng(7), ABORT, {}).V_rob
    v2 = eval_sampled_policy(model, unc, policy, b0, H, 8,
                             np.random.default_rng(7), ABORT, {}).V_rob
    assert v1 == v2
