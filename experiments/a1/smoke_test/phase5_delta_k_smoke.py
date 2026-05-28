"""Phase 5 - DeltaKComputer integration + weighted orthant cross-check.

Builds a tiny synthetic POMDP, constructs DeltaKComputer, exercises all four
piece methods + row + delta_k_rob_split. Crucially: cross-checks
transition_part's MILP against projected_l1_max_orthant with the same
(wZ_max) weights - this is the first verification of the weighted code path
end-to-end.

Plus sanity: rho_T = rho_Z = 0 -> delta_k_rob_split = 0 for all actions.

Run from anywhere:
    python experiments/a1/smoke_test/phase5_delta_k_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from robust_pomdp import TabularPOMDP
from robust_pomdp.bounds.delta_k import DeltaKComputer
from robust_pomdp.bounds.l1_max import projected_l1_max_orthant
from robust_pomdp.uncertainty.uncertainty_sets import TVDistance, uniform_uncertainty


def build_tiny_pomdp() -> TabularPOMDP:
    """4 states, 2 actions, 3 observations. T and O rows sum to 1."""
    T = np.array([
        [[0.70, 0.20, 0.10, 0.00], [0.10, 0.60, 0.20, 0.10]],
        [[0.10, 0.70, 0.20, 0.00], [0.20, 0.40, 0.30, 0.10]],
        [[0.05, 0.15, 0.60, 0.20], [0.10, 0.10, 0.60, 0.20]],
        [[0.00, 0.10, 0.30, 0.60], [0.00, 0.20, 0.40, 0.40]],
    ], dtype=np.float64)
    O = np.array([
        [0.70, 0.20, 0.10],
        [0.40, 0.40, 0.20],
        [0.20, 0.40, 0.40],
        [0.10, 0.30, 0.60],
    ], dtype=np.float64)
    R = np.zeros((4, 2), dtype=np.float64)
    return TabularPOMDP.from_matrices(T, O, R)


def main() -> None:
    model = build_tiny_pomdp()
    N, A, M = model.n_states, model.n_actions, model.n_obs
    S_in = [0, 1]
    Z_in = [0, 1]
    rho_T, rho_Z = 0.2, 0.2

    uncertainty = uniform_uncertainty(N, A, rho_T, rho_Z, TVDistance())
    computer = DeltaKComputer(model, uncertainty, S_in=S_in, Z_in=Z_in)

    print(f"DeltaKComputer: N={N} A={A} M={M} S_in={S_in} Z_in={Z_in} rho_T=rho_Z={rho_T}")
    print()

    # Per-s_next.
    for s_p in S_in:
        wz = computer.wZ_max(s_p)
        dz = computer.delta_z_inside_max(s_p)
        print(f"  s_next={s_p}: wZ_max={wz:.6f}  delta_z_inside_max={dz:.6f}")
    print()

    # Per-(s, a) with transition_part orthant cross-check.
    wZ_max_array = np.array([computer.wZ_max(s_p) for s_p in S_in], dtype=np.float64)
    max_diff = 0.0
    for s in S_in:
        for a in range(A):
            tp = computer.transition_part(s, a)
            op = computer.observation_part(s, a)
            r = computer.row(s, a)
            # Cross-check transition_part MILP vs orthant (weighted).
            tp_orth = projected_l1_max_orthant(model.T[s, a, :], rho_T, S_in, n=N, weights=wZ_max_array,)
            diff = abs(tp - tp_orth)
            max_diff = max(max_diff, diff)
            assert diff < 1e-6, f"(s={s}, a={a}): trans MILP={tp} orthant={tp_orth} diff={diff}"
            print(f"  (s={s}, a={a}): trans={tp:.6f}  obs={op:.6f}  row={r:.6f}  "
                  f"(trans-orth diff={diff:.2e})")
    print(f"\n  max transition_part MILP-vs-orthant diff = {max_diff:.2e}")
    print()

    # delta_k_rob_split per action.
    for a in range(A):
        print(f"  delta_k_rob_split(a={a}) = {computer.delta_k_rob_split(a):.6f}")

    # Sanity: rho_T = rho_Z = 0 -> all pieces 0.
    print()
    uncertainty_zero = uniform_uncertainty(N, A, 0.0, 0.0, TVDistance())
    computer_zero = DeltaKComputer(model, uncertainty_zero, S_in=S_in, Z_in=Z_in)
    for a in range(A):
        dk = computer_zero.delta_k_rob_split(a)
        assert abs(dk) < 1e-9, f"rho=0: expected 0, got {dk}"
        print(f"  rho=0: delta_k_rob_split(a={a}) = {dk:.2e}  (== 0)")

    # Cache-reuse sanity.
    _ = computer.transition_part(0, 0)
    assert (0, 0) in computer._trans_cache, "transition_part cache miss"
    assert 0 in computer._wz_max_cache and 1 in computer._wz_max_cache, "wZ_max cache miss"
    print("\nDeltaKComputer integration OK (weighted orthant cross-check + sanity + caches)")


if __name__ == "__main__":
    main()
