"""
Robust backup correctness — projected uncertainty test.

A single check that exercises every distinctive piece of `compute_robust_q`:
  - rho_T > 0 AND rho_Z > 0           (LPs are non-trivial)
  - S_in strict subset of S            (transition projection engaged)
  - Z_in strict subset of Z            (observation projection engaged)
  - V_children with mixed signs        (adversary has a real choice between
                                        within-support shift and out-of-support
                                        leakage)

Synthetic 4-state, 2-action, 4-observation POMDP. Hand-built HistoryNode and
ActionNode passed directly to compute_robust_q. The expected Q_robust is
hand-derived in `verify_robust_backup_derivation.md` (next to this file)
to be exactly -5.12.

Run:
    python experiments/tiger/verify_robust_backup.py

Exits 0 if the test passes.
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
    UncertaintySets,
    compute_robust_q,
)


# ---------------------------------------------------------------------------
# Synthetic POMDP setup
# ---------------------------------------------------------------------------

def build_test_pomdp() -> TabularPOMDP:
    """Build the 4-state, 2-action, 4-obs POMDP defined in the derivation md."""
    n_states, n_actions, n_obs = 4, 2, 4

    # T[s, a, s']
    T = np.zeros((n_states, n_actions, n_states))

    # Action a=0 — the action under test. Each row sums to 1.
    T[0, 0, :] = [0.3, 0.4, 0.2, 0.1]
    T[1, 0, :] = [0.5, 0.2, 0.1, 0.2]

    # Action a=1 — placeholder (not used by the test). Uniform.
    T[0, 1, :] = 0.25
    T[1, 1, :] = 0.25

    # s ∈ S_out — never read by the test (LP iterates over s ∈ S_in only).
    # Set to uniform so from_matrices' shape checks pass with valid distributions.
    T[2, :, :] = 0.25
    T[3, :, :] = 0.25

    # O[s', z]
    O = np.zeros((n_states, n_obs))
    O[0, :] = [0.4, 0.3, 0.2, 0.1]
    O[1, :] = [0.2, 0.5, 0.1, 0.2]
    O[2, :] = 0.25  # placeholder
    O[3, :] = 0.25  # placeholder

    # R[s, a]
    R = np.zeros((n_states, n_actions))
    R[0, 0] = 1.0
    R[1, 0] = 2.0
    # R[*, 1] left at 0.0 — not used by the test.

    return TabularPOMDP.from_matrices(T, O, R)


def build_test_uncertainty(n_states: int, n_actions: int) -> UncertaintySets:
    """rho_T = rho_Z = 0.2 uniformly."""
    return UncertaintySets(rho_T=np.full((n_states, n_actions), 0.2),
                           rho_Z=np.full(n_states, 0.2),
                           metric=TVDistance(),
                           )


def build_test_tree() -> tuple[HistoryNode, ActionNode]:
    """Hand-build the (HistoryNode, ActionNode) pair compute_robust_q expects.

    HistoryNode h:
      - particles = [0]*6 + [1]*4  -> empirical belief [0.6, 0.4] over S_in
      - S_in = {0, 1}
    ActionNode for a=0:
      - Z_in = {0, 1}
      - children: z=0 -> child with V_robust = +10, z=1 -> child with V_robust = -20
    """
    counter = NodeIdCounter()

    h = HistoryNode(counter.next_id(), depth=0)
    h.particles = [0] * 6 + [1] * 4
    h.S_in = {0, 1}

    action_node = ActionNode(counter.next_id())
    action_node.Z_in = {0, 1}

    child_z0 = HistoryNode(counter.next_id(), depth=1)
    child_z0.V_robust = 10.0
    child_z1 = HistoryNode(counter.next_id(), depth=1)
    child_z1.V_robust = -20.0
    action_node.children = {0: child_z0, 1: child_z1}

    return h, action_node


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------

EXPECTED_Q_ROBUST = -5.12
TOLERANCE = 1e-6


def check_projected_robust_backup() -> int:
    """Run compute_robust_q on the synthetic problem and assert Q == -5.12."""
    model = build_test_pomdp()
    uncertainty = build_test_uncertainty(model.n_states, model.n_actions)
    h, action_node = build_test_tree()

    Q, _ = compute_robust_q(h, a=0, action_node=action_node,
                            model=model, uncertainty=uncertainty,
                            logger=None, backup_counter=0,
                            previous_Q_robust=0.0,
                            )

    diff = abs(Q - EXPECTED_Q_ROBUST)
    if diff > TOLERANCE:
        print(f"FAIL: Q_robust = {Q:.9f}, expected = {EXPECTED_Q_ROBUST}, "
              f"diff = {diff:.4g}")
        return 1
    print(f"OK: Q_robust = {Q:.9f}, expected = {EXPECTED_Q_ROBUST} "
          f"(diff = {diff:.4g})")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Robust backup correctness — projected uncertainty test")
    print("=" * 60)
    print()
    print("Synthetic 4-state, 2-action, 4-obs POMDP")
    print("  S_in = {0, 1}, S_out = {2, 3}")
    print("  Z_in = {0, 1}, Z_out = {2, 3}")
    print("  rho_T = rho_Z = 0.2")
    print("  belief = [0.6, 0.4], V_children = [+10, -20]")
    print("  See verify_robust_backup_derivation.md for the hand derivation.")
    print()

    f = check_projected_robust_backup()

    print()
    print("=" * 60)
    if f == 0:
        print("Robust backup correctness: PASS")
    else:
        print("Robust backup correctness: FAIL")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
