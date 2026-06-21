"""Phase A - full-robust-greedy solver smoke.

Verifies solve_full_greedy:
  1. rho=0 -> V_full equals an INDEPENDENT nominal-optimal expectimax (no
     robustness, computed by a separate recursion).
  2. rho>0 -> V_full^rob <= V_full^nominal (robustness can only lower the value).
  3. pi_full sanity: non-empty, root action present at key (), all actions valid.

Run from anywhere:
    python experiments/a1/smoke_test/phaseA_greedy_solver_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(_SCRIPT.parents[1]))  # experiments/a1/
sys.path.insert(0, str(_SCRIPT.parents[3]))  # repo root

from robust_pomdp.uncertainty.uncertainty_sets import TVDistance, uniform_uncertainty

from synthetic_pomdp import ABORT, make_initial_belief, make_synthetic_pomdp
from tree_evaluator import projected_bayes_update
from greedy_solver import solve_full_greedy


def nominal_optimal(model, belief, depth, H, abort_action):
    """Independent nominal-optimal expectimax (no robustness, full support).
    V(b) = max_a [ Sum_s b(s) R(s,a)  +  (continuing: Sum_z P(z|b,a) V(bayes)) ]."""
    S = list(range(model.n_states))
    Z = list(range(model.n_obs))
    best = -np.inf
    for a in range(model.n_actions):
        imm = float(sum(belief[s] * model.R[s, a] for s in S))
        if depth == H or a == abort_action:
            q = imm
        else:
            fut = 0.0
            for z in Z:
                pz = float(sum(belief[s] * model.T[s, a, sp] * model.O[sp, z]
                               for s in S for sp in S))
                if pz > 1e-15:
                    child = projected_bayes_update(belief, model, a, z, S)
                    fut += pz * nominal_optimal(model, child, depth + 1, H, abort_action)
            q = imm + fut
        best = max(best, q)
    return best


def main():
    N, M, H = 4, 3, 3
    model = make_synthetic_pomdp(N=N, M=M)
    b0 = make_initial_belief(N=N)

    # ---- Test 1: rho=0 -> V_full == nominal optimal ----
    unc0 = uniform_uncertainty(N, model.n_actions, 0.0, 0.0, TVDistance())
    v_full_0, pi_full_0 = solve_full_greedy(model, unc0, b0, H, abort_action=ABORT)
    v_nom = nominal_optimal(model, np.asarray(b0, dtype=np.float64), 0, H, ABORT)
    assert abs(v_full_0 - v_nom) < 1e-7, f"rho=0: V_full={v_full_0:.6f} != nominal opt {v_nom:.6f}"
    print(f"  (1) rho=0: V_full = {v_full_0:.6f}  ==  nominal optimal {v_nom:.6f}   OK")

    # ---- Test 2: rho>0 -> robust <= nominal ----
    unc = uniform_uncertainty(N, model.n_actions, 0.1, 0.1, TVDistance())
    v_full_r, pi_full_r = solve_full_greedy(model, unc, b0, H, abort_action=ABORT)
    assert v_full_r <= v_nom + 1e-7, f"robust {v_full_r:.6f} > nominal {v_nom:.6f}"
    print(f"  (2) rho=0.1: V_full^rob = {v_full_r:.6f}  <=  nominal {v_nom:.6f}   OK")

    # ---- Test 3: pi_full sanity ----
    assert len(pi_full_r) > 0, "pi_full is empty"
    assert () in pi_full_r, "pi_full missing root key ()"
    assert all(0 <= a < model.n_actions for a in pi_full_r.values()), "invalid action in pi_full"
    print(f"  (3) pi_full: {len(pi_full_r)} greedy histories, root action = "
          f"{pi_full_r[()]}, all actions valid   OK")

    print("\nPhase A (full-robust-greedy solver) OK.")


if __name__ == "__main__":
    main()