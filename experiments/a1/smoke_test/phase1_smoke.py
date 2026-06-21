"""Phase 1 sanity — verify shape constraints, boundary clipping, belief sum, policy rule.

Run from anywhere:
    python experiments/a1/smoke_test/phase1_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthetic_pomdp import (
    make_synthetic_pomdp,
    make_initial_belief,
    class_sizes,
    INSPECT,
    PROCEED,
    MITIGATE,
    ABORT,
)


def main() -> None:
    N, M = 50, 8
    m = make_synthetic_pomdp(N=N, M=M)

    # Row sums.
    T_sums = m.T.sum(axis=2)
    O_sums = m.O.sum(axis=1)
    assert np.allclose(T_sums, 1.0), f"T row sums off: {T_sums.min()=}, {T_sums.max()=}"
    assert np.allclose(O_sums, 1.0), f"O row sums off: {O_sums.min()=}, {O_sums.max()=}"

    # Boundary clipping.
    assert m.T[0, MITIGATE, 0] == 1.0, "mitigate at s=0 must self-loop"
    assert m.T[N - 1, PROCEED, N - 1] == 1.0, "proceed at s=N-1 must self-loop"
    assert m.T[N - 1, INSPECT, N - 1] == 1.0, "inspect at s=N-1 must self-loop"

    # Abort is self-loop (terminal handled in tree builder).
    for s in range(N):
        assert m.T[s, ABORT, s] == 1.0, f"abort at s={s} must self-loop"

    # Reward sign sanity per class.
    n_safe, n_risky, n_cat = class_sizes(N)
    s_safe = 0
    s_risky = n_safe
    s_cat = N - 1
    assert m.R[s_safe, PROCEED] == 10.0
    assert m.R[s_risky, PROCEED] == -20.0
    assert m.R[s_cat, PROCEED] == -100.0
    assert m.R[s_safe, MITIGATE] == -2.0
    assert m.R[s_risky, MITIGATE] == -8.0
    assert m.R[s_cat, MITIGATE] == -30.0
    assert (m.R[:, INSPECT] == -1.0).all()
    assert (m.R[:, ABORT] == 0.0).all()

    # Initial belief.
    b = make_initial_belief(N=N)
    assert abs(b.sum() - 1.0) < 1e-9, f"b.sum() = {b.sum()}"
    # Intra-class decay (safe class).
    assert b[0] > b[1] > b[n_safe - 1], "safe-class decay broken"
    # Class total masses.
    safe_mass = b[:n_safe].sum()
    risky_mass = b[n_safe:n_safe + n_risky].sum()
    cat_mass = b[n_safe + n_risky:].sum()
    assert abs(safe_mass - 0.85) < 1e-9, f"safe mass {safe_mass}"
    assert abs(risky_mass - 0.14) < 1e-9, f"risky mass {risky_mass}"
    assert abs(cat_mass - 0.01) < 1e-9, f"cat mass {cat_mass}"

    # Observation peak at risk-aligned warning.
    target_safe = round((s_safe / (N - 1)) * (M - 1))
    target_cat = round((s_cat / (N - 1)) * (M - 1))
    assert m.O[s_safe, :].argmax() == target_safe, "safe obs should peak at low warning"
    assert m.O[s_cat, :].argmax() == target_cat, "cat obs should peak at high warning"

    print(f"phase 1 OK | N={N} M={M} | sizes=({n_safe},{n_risky},{n_cat}) | "
          f"b0[:3]={b[:3]} b0[-3:]={b[-3:]}")


if __name__ == "__main__":
    main()
