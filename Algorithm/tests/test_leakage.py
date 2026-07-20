"""
Test 4 — survival scalar `c` (leakage.py) and the backup that assembles it.

Independent checks:
  A. chain forward-enumeration at rho=0 (c_root == Σ_j nominal in-support survival).
  B. chain at rho>0: worst-case shrinks each in-support prob by rho/2 (hand value).
  C. survival_c interior step at rho=0 vs direct.
  D. full support -> c_root = H (no leakage possible).
  E. base cases: abort action -> c = H-depth; frontier node -> c = 1.
  F. Lemma 2: projected-ball leakage == full-ball leakage (vs src TVBallLinearMinLP).

Run:  python -m pytest Algorithm/tests/test_leakage.py -q
"""

import numpy as np
import pytest

from Algorithm.core.pomdp import TabularPOMDP
from Algorithm.core.projected_tv_ball import ProjectedTVBall
from Algorithm.core.tree import HistoryNode, NodeIdCounter
from Algorithm.core.uncertainty import uniform_uncertainty
from Algorithm.ucb.backup import backup
from Algorithm.ucb.leakage import leakage_c

try:
    from robust_pomdp.bounds.tv_ball_lp import TVBallLinearMinLP
    _HAVE_SRC = True
except Exception:
    _HAVE_SRC = False


def _chain_model():
    """3 states / 2 obs / 1 action. Chain 0 --(z0)--> 1 --(z0)--> 2."""
    T = np.zeros((3, 1, 3))
    T[0, 0, :] = [0.5, 0.5, 0.0]
    T[1, 0, :] = [0.0, 0.4, 0.6]
    T[2, 0, :] = [0.0, 0.0, 1.0]
    O = np.array([[0.9, 0.1], [0.5, 0.5], [0.4, 0.6]])
    R = np.zeros((3, 1))
    return TabularPOMDP.from_matrices(T, O, R)


def _build_and_backup_chain(model, rho, H=3):
    unc = uniform_uncertainty(model.n_states, model.n_actions, rho, rho)
    ctr = NodeIdCounter()
    root = HistoryNode(ctr.next_id(), 0); root.S_in = {0}
    child = HistoryNode(ctr.next_id(), 1); child.S_in = {1}
    gchild = HistoryNode(ctr.next_id(), 2); gchild.S_in = {2}   # depth H-1 -> terminal
    an0 = root.expand_action(0, ctr); an0.Z_in = {0}; an0.S_next = {1}; an0.children[0] = child
    an1 = child.expand_action(0, ctr); an1.Z_in = {0}; an1.S_next = {2}; an1.children[0] = gchild
    lp: dict = {}
    for node in (gchild, child, root):     # post-order (leaves first)
        backup(node, np.array([1.0]), model, unc, H, abort_action=None, lp_cache=lp)
    return root


def test_chain_c_rho0_forward_enum():
    m = _chain_model()
    root = _build_and_backup_chain(m, rho=0.0)
    # c_root = 1 + T01·O1·(1 + T12·O2)   with T01=0.5,O1=O[1,0]=0.5,T12=0.6,O2=O[2,0]=0.4
    expected = 1 + 0.5 * 0.5 * (1 + 0.6 * 0.4)
    assert abs(root.c[0] - expected) < 1e-9, (root.c[0], expected)


def test_chain_c_rho_positive_worstcase():
    m = _chain_model()
    root = _build_and_backup_chain(m, rho=0.2)          # each in-support prob shrinks by 0.1
    a01, o1, a12, o2 = 0.5 - 0.1, 0.5 - 0.1, 0.6 - 0.1, 0.4 - 0.1
    expected = 1 + a01 * o1 * (1 + a12 * o2)
    assert abs(root.c[0] - expected) < 1e-6, (root.c[0], expected)


def test_leakage_c_rho0_direct():
    m = _chain_model()
    unc = uniform_uncertainty(m.n_states, m.n_actions, 0.0, 0.0)
    c = leakage_c(m, unc, a=0, S_in=[0], Z_in=[0], S_next=[1], child_c={0: {1: 1.7}}, lp_cache={})
    # cw[1] = O[1,0]·1.7 = 0.5·1.7 = 0.85 ; c[0] = 1 + T[0,0,1]·0.85 = 1 + 0.5·0.85 = 1.425
    assert abs(c[0] - 1.425) < 1e-9, c


def test_full_support_c_equals_H():
    T = np.zeros((2, 1, 2)); T[0, 0, :] = [0.6, 0.4]; T[1, 0, :] = [0.3, 0.7]
    O = np.array([[0.8, 0.2], [0.1, 0.9]]); R = np.zeros((2, 1))
    m = TabularPOMDP.from_matrices(T, O, R)
    unc = uniform_uncertainty(2, 1, 0.3, 0.3)           # rho irrelevant at full support
    H = 2
    ctr = NodeIdCounter()
    root = HistoryNode(ctr.next_id(), 0); root.S_in = {0, 1}
    c0 = HistoryNode(ctr.next_id(), 1); c0.S_in = {0, 1}
    c1 = HistoryNode(ctr.next_id(), 1); c1.S_in = {0, 1}
    an = root.expand_action(0, ctr); an.Z_in = {0, 1}; an.S_next = {0, 1}
    an.children[0] = c0; an.children[1] = c1
    lp: dict = {}
    for node in (c0, c1, root):
        backup(node, np.array([0.5, 0.5]), m, unc, H, abort_action=None, lp_cache=lp)
    assert abs(root.c[0] - 2.0) < 1e-9 and abs(root.c[1] - 2.0) < 1e-9, root.c


def test_base_case_abort():
    m = _chain_model()
    unc = uniform_uncertainty(m.n_states, m.n_actions, 0.1, 0.1)
    ctr = NodeIdCounter()
    node = HistoryNode(ctr.next_id(), 0); node.S_in = {0, 1}
    node.expand_action(0, ctr)                          # action 0 acts as the terminal "abort"
    backup(node, np.array([0.5, 0.5]), m, unc, H=3, abort_action=0, lp_cache={})
    assert abs(node.c[0] - 3.0) < 1e-12 and abs(node.c[1] - 3.0) < 1e-12, node.c  # H - depth = 3


def test_base_case_frontier():
    m = _chain_model()
    unc = uniform_uncertainty(m.n_states, m.n_actions, 0.1, 0.1)
    ctr = NodeIdCounter()
    node = HistoryNode(ctr.next_id(), 1); node.S_in = {0, 1}   # no expanded actions -> frontier
    backup(node, np.array([0.5, 0.5]), m, unc, H=3, abort_action=None, lp_cache={})
    assert abs(node.c[0] - 1.0) < 1e-12 and abs(node.c[1] - 1.0) < 1e-12, node.c


@pytest.mark.skipif(not _HAVE_SRC, reason="src robust_pomdp not importable")
def test_lemma2_projected_equals_full_ball():
    """Leakage over the projected ball equals leakage over the full ball (Lemma 2)."""
    O_full = np.array([0.3, 0.2, 0.4, 0.1])
    Z_in = [0, 2]
    coeffs_proj = np.array([1.5, -0.5])
    rho = 0.3
    v_proj, _ = ProjectedTVBall(2).solve(O_full[Z_in], rho, coeffs_proj)
    coeffs_full = np.array([1.5, 0.0, -0.5, 0.0])       # off-support obs -> coefficient 0
    v_full = TVBallLinearMinLP(4).solve(O_full, rho, coeffs_full)
    assert abs(v_proj - v_full) < 1e-6, (v_proj, v_full)
