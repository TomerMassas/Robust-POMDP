"""
Test 2 — projected-Bayes belief: unnormalized root restriction, renormalized
children (vs a hand Bayes update), and the uniform fallback.

Run:  python -m pytest Algorithm/tests/test_belief.py -q
"""

import numpy as np

from Algorithm.core.pomdp import TabularPOMDP
from Algorithm.ucb.belief import projected_bayes, restrict


def _hand_model():
    T = np.zeros((3, 1, 3))
    T[0, 0, :] = [0.5, 0.5, 0.0]
    T[1, 0, :] = [0.0, 0.6, 0.4]
    T[2, 0, :] = [0.0, 0.0, 1.0]
    O = np.array([[0.9, 0.1], [0.5, 0.5], [0.4, 0.6]])
    R = np.zeros((3, 1))
    return TabularPOMDP.from_matrices(T, O, R)


def test_restrict_is_unnormalized():
    b0 = np.array([0.1, 0.2, 0.3, 0.4])
    b = restrict(b0, [0, 2])
    assert np.allclose(b, [0.1, 0.3])       # keeps original mass (sums to m_in = 0.4)


def test_projected_bayes_matches_hand_and_normalized():
    m = _hand_model()
    belief = np.array([0.4, 0.6])           # over parent support [0, 1]
    b = projected_bayes(belief, m, a=0, z=1, S_in_parent=[0, 1], S_in_child=[1, 2])
    # numerator[s'] = (Σ_s belief[s]·T[s,0,s']) · O[s',1]
    #   s'=1: (0.4·0.5 + 0.6·0.6)·0.5 = 0.56·0.5 = 0.280
    #   s'=2: (0.4·0.0 + 0.6·0.4)·0.6 = 0.24·0.6 = 0.144
    num = np.array([0.280, 0.144])
    assert abs(b.sum() - 1.0) < 1e-12
    assert np.allclose(b, num / num.sum())


def test_projected_bayes_uniform_fallback():
    m = _hand_model()
    # From s=2, action 0 only reaches s=2 (T[2,0,:]=[0,0,1]); child support {0,1}
    # is unreachable -> numerator 0 -> uniform fallback.
    b = projected_bayes(np.array([1.0]), m, a=0, z=0, S_in_parent=[2], S_in_child=[0, 1])
    assert np.allclose(b, [0.5, 0.5])
