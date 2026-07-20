"""
Test 1 — the duplicated ProjectedTVBall is behaviourally faithful to the src
version (values match on random inputs), plus the rho=0 short-circuit.

Run from repo root:  python -m pytest Algorithm/tests/test_projected_tv_ball.py -q
"""

import numpy as np
import pytest

from Algorithm.core.projected_tv_ball import ProjectedTVBall as NewBall

try:
    from robust_pomdp.uncertainty.projected_sets import ProjectedTVBall as SrcBall
    _HAVE_SRC = True
except Exception:  # editable install missing / import error
    _HAVE_SRC = False


def _rand_subprob(rng, n):
    raw = rng.random(n)
    return raw / raw.sum() * rng.uniform(0.4, 1.0)  # sub-probability (mass in (0,1])


@pytest.mark.skipif(not _HAVE_SRC, reason="src robust_pomdp not importable")
def test_matches_src_projected():
    """Values identical to src on the projected (sub-probability) ball."""
    rng = np.random.default_rng(0)
    for n in [1, 2, 3, 5, 8]:
        new, src = NewBall(n), SrcBall(n)
        for _ in range(50):
            p = _rand_subprob(rng, n)
            rho = rng.uniform(0.0, 1.0)
            c = rng.uniform(-10.0, 10.0, size=n)
            v_new, _ = new.solve(p, rho, c)
            v_src, _ = src.solve(p, rho, c)
            assert abs(v_new - v_src) < 1e-7, (n, v_new, v_src)


@pytest.mark.skipif(not _HAVE_SRC, reason="src robust_pomdp not importable")
def test_matches_src_full_support():
    """Values identical to src on the full TV ball (full_support=True)."""
    rng = np.random.default_rng(1)
    for n in [2, 3, 5]:
        new, src = NewBall(n), SrcBall(n)
        for _ in range(30):
            raw = rng.random(n)
            p = raw / raw.sum()  # full distribution
            rho = rng.uniform(0.0, 1.0)
            c = rng.uniform(-10.0, 10.0, size=n)
            v_new, _ = new.solve(p, rho, c, full_support=True)
            v_src, _ = src.solve(p, rho, c, full_support=True)
            assert abs(v_new - v_src) < 1e-7, (n, v_new, v_src)


def test_rho_zero_returns_nominal():
    """rho=0 short-circuit returns the nominal vector and its plain expectation."""
    b = NewBall(3)
    p = np.array([0.2, 0.3, 0.4])
    c = np.array([1.0, 2.0, 3.0])
    v, pp = b.solve(p, 0.0, c)
    assert np.allclose(pp, p)
    assert abs(v - float(np.dot(p, c))) < 1e-12


def test_full_support_sums_to_one():
    """Under full_support the optimal vector is a full distribution (Sum = 1)."""
    b = NewBall(4)
    raw = np.array([0.1, 0.2, 0.3, 0.4])
    _, pp = b.solve(raw, 0.5, np.array([1.0, -2.0, 3.0, 0.5]), full_support=True)
    assert abs(pp.sum() - 1.0) < 1e-7
    assert (pp >= -1e-9).all()
