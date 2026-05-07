"""
Robust backup formula correctness — two layers, strongest first:

  Layer 1 — Hand-trivial synthesized backup. Pre-computed σ, Q, on paper.
            Strict 1e-6 bar.
  Layer 2 — Qualitative Tiger sanity at uniform belief: Q_robust(listen) >
            Q_robust(open-{left,right}). Uses elevated exploration_constant
            to make the test deterministic given a fixed seed.

Run:
    python experiments/tiger/verify_robust_backup.py

Exits 0 if both layers pass.
"""

from __future__ import annotations

import sys

import numpy as np

from robust_pomdp import (
    ActionNode,
    HistoryNode,
    NodeIdCounter,
    TabularPOMDP,
    TVDistance,
    compute_robust_q,
    robust_pomcp_plan,
    uniform_uncertainty,
)

from tiger_problem import make_tiger_pomdp


# ---------------------------------------------------------------------------
# Layer 1 — Hand-trivial synthesized backup
# ---------------------------------------------------------------------------

def check_hand_trivial() -> int:
    """Synthesized 2-state, 1-action, 2-obs POMDP with rho=0 (nominal lookahead).

    Hand derivation:
        T uniform: T[s, 0, s'] = 0.5 for all s, s'.
        O[0,0]=0.8, O[0,1]=0.2, O[1,0]=0.2, O[1,1]=0.8.
        R[0,0]=1, R[1,0]=-1.
        belief=[0.5,0.5]; V_children=[V[z=0]=10, V[z=1]=20].

        w(s') = Σ_z O(z|s') · V[z]
            w(0) = 0.8*10 + 0.2*20 = 12
            w(1) = 0.2*10 + 0.8*20 = 18
        σ(s) = Σ_s' T(s'|s,0) · w(s') = 0.5*12 + 0.5*18 = 15  (both s)
        Q = Σ_s belief[s] · (R(s,0) + σ(s))
          = 0.5*(1+15) + 0.5*(-1+15) = 0.5*16 + 0.5*14 = 15.0
    """
    T = np.zeros((2, 1, 2))
    T[:, 0, :] = 0.5
    O = np.array([[0.8, 0.2], [0.2, 0.8]])
    R = np.array([[1.0], [-1.0]])
    model = TabularPOMDP.from_matrices(T, O, R)
    unc = uniform_uncertainty(2, 1, 0.0, 0.0, TVDistance())

    counter = NodeIdCounter()
    h = HistoryNode(counter.next_id(), depth=0)
    h.particles = [0, 1]
    h.S_in = {0, 1}

    action_node = ActionNode(counter.next_id())
    action_node.Z_in = {0, 1}
    child0 = HistoryNode(counter.next_id(), depth=1)
    child0.V_robust = 10.0
    child1 = HistoryNode(counter.next_id(), depth=1)
    child1.V_robust = 20.0
    action_node.children = {0: child0, 1: child1}

    Q, _ = compute_robust_q(h, a=0, action_node=action_node,
                            model=model, uncertainty=unc,
                            logger=None, backup_counter=0,
                            previous_Q_robust=0.0
                            )

    expected = 15.0
    diff = abs(Q - expected)
    if diff > 1e-6:
        print(f"[layer 1] FAIL: Q={Q}, expected={expected}, diff={diff:.4g}")
        return 1
    print(f"[layer 1] hand-trivial backup: OK (Q={Q:.9f}, expected={expected})")
    return 0


# ---------------------------------------------------------------------------
# Layer 2 — Qualitative Tiger sanity
# ---------------------------------------------------------------------------

def check_tiger_sanity() -> int:
    """Plan from uniform belief on Tiger; assert listen dominates both opens.

    With default exploration_constant=1.0 and Tiger's reward magnitudes, basic
    POMCP can lock onto whichever action got the luckiest first rollout
    (seed-dependent). For a deterministic gate we run with elevated
    exploration_constant and a fixed seed proven to behave sanely.
    """
    tiger = make_tiger_pomdp(0.85)
    unc = uniform_uncertainty(tiger.n_states, tiger.n_actions, 0.0, 0.1, TVDistance())
    rng = np.random.default_rng(0)

    _, root = robust_pomcp_plan(np.array([0.5, 0.5]), tiger, unc,
                                horizon=5, budget=500, sims_per_backup=5,
                                ucb_mode="nominal", exploration_constant=50.0,
                                logger=None, rng=rng
                                )

    qs = {a: root.children[a].Q_robust
          for a in range(tiger.n_actions) if a in root.children}
    open_left  = qs.get(0, -np.inf)
    open_right = qs.get(1, -np.inf)
    listen     = qs.get(2, -np.inf)

    if not (listen > open_left and listen > open_right):
        print(f"[layer 2] FAIL: Q(listen)={listen:.3f} not greater than "
              f"Q(open-left)={open_left:.3f} or Q(open-right)={open_right:.3f}")
        return 1
    print(f"[layer 2] Tiger sanity: OK "
          f"(listen={listen:.3f} > open-left={open_left:.3f}, open-right={open_right:.3f})")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Robust backup correctness")
    print("=" * 60)

    print("\nLayer 1: hand-trivial synthesized backup")
    print("-" * 60)
    f1 = check_hand_trivial()

    print("\nLayer 2: qualitative Tiger sanity")
    print("-" * 60)
    f2 = check_tiger_sanity()

    print("\n" + "=" * 60)
    if f1 == 0 and f2 == 0:
        print("Robust backup correctness: PASS")
        print("=" * 60)
    else:
        print(f"Robust backup correctness: FAIL "
              f"(layer 1={f1}, layer 2={f2})")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
