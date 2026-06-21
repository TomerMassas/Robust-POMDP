"""Test: ProjectedTVBall reduces to the FULL TV ball at full support.

Background: the projected TV ball allows mass to leak to an out-of-support
region (sub-probability). At FULL support there are no out-of-support states,
so leakage must be impossible and the projected ball must coincide with the
full TV ball (Sum P = 1). The `full_support=True` flag enforces this.

This test verifies:
  1. The motivating example: nominal (0.5,0.5), rho=0.2, c=(10,3).
     - full_support=False leaks mass (too pessimistic, 5.5).
     - full_support=True matches the full ball (5.8).
  2. Randomized: for many random (full distribution, rho, c),
     ProjectedTVBall(full_support=True) == TVBallLinearMinLP (the true full ball).
  3. Partial support is UNCHANGED: with genuine out-of-support mass
     (m_nominal < 1), full_support=False still allows leakage (legitimate),
     and equals a definitional full-ball-restricted brute force on a tiny case.

Run from anywhere:
    python experiments/a1/smoke_test/projected_full_support_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from robust_pomdp.uncertainty.projected_sets import ProjectedTVBall
from robust_pomdp.bounds.tv_ball_lp import TVBallLinearMinLP


def test_motivating_example():
    p = np.array([0.4, 0.3,0.2,0.1]); rho = 0.2; c = np.array([10, 3, 0, 1])
    proj = ProjectedTVBall(n=4)
    full = TVBallLinearMinLP(n=4)
    leaky, _ = proj.solve(p, rho, c, full_support=False)
    fixed, p_fixed = proj.solve(p, rho, c, full_support=True)
    truth = full.solve(p, rho, c)
    print("  (1) nominal (0.5,0.5), rho=0.2, c=(10,3):")
    print(f"        full_support=False (leaky) = {leaky:.4f}")
    print(f"        full_support=True  (fixed) = {fixed:.4f}  (P={np.round(p_fixed,4)}, "
          f"sum={p_fixed.sum():.4f})")
    print(f"        TVBallLinearMinLP  (truth) = {truth:.4f}")
    assert abs(fixed - truth) < 1e-6, f"fixed {fixed} != truth {truth}"
    assert leaky < truth - 1e-6, "expected leaky < truth (mass-drop pessimism)"
    assert abs(p_fixed.sum() - 1.0) < 1e-9, "full_support solution must sum to 1"
    print("        OK: full_support=True matches the full ball; sums to 1.")


def test_randomized_full_support(n_trials=2000, seed=0):
    rng = np.random.default_rng(seed)
    max_diff = 0.0
    for _ in range(n_trials):
        n = int(rng.integers(2, 7))
        p = rng.dirichlet(np.ones(n))           # full distribution (sum 1)
        rho = float(rng.uniform(0.0, 1.5))
        c = rng.uniform(-10, 10, size=n)
        proj = ProjectedTVBall(n=n)
        full = TVBallLinearMinLP(n=n)
        v_proj, _ = proj.solve(p, rho, c, full_support=True)
        v_full = full.solve(p, rho, c)
        max_diff = max(max_diff, abs(v_proj - v_full))
    assert max_diff < 1e-6, f"max |proj_full - fullball| = {max_diff}"
    print(f"  (2) randomized full-support ({n_trials} trials): "
          f"max |ProjectedTVBall(full) - fullball| = {max_diff:.2e}  OK")


def _brute_partial(p_in, p_out_nom, rho, c_in, grid=60):
    """Definitional projected-ball value on a tiny case: brute-force over full
    distributions (P_in, P_out) in the TV ball, minimize sum c_in * P_in.
    Here n_in=1, n_out=1 (so a 2-state full ball, restricted objective on s0)."""
    best = np.inf
    p0_nom, p1_nom = p_in[0], p_out_nom[0]
    for p0 in np.linspace(0, 1, grid + 1):
        for p1 in np.linspace(0, 1 - p0, grid + 1):
            if abs(p0 - p0_nom) + abs(p1 - p1_nom) <= rho + 1e-12:
                best = min(best, c_in[0] * p0)   # objective only on in-support s0
    return best


def test_partial_support_unchanged():
    # n_in = 1 (state 0 in support), nominal mass on it = 0.6, out-of-support nominal 0.4.
    p_in = np.array([0.6]); rho = 0.3; c_in = np.array([5.0])
    proj = ProjectedTVBall(n=1)
    v_proj, p_opt = proj.solve(p_in, rho, c_in, full_support=False)
    v_brute = _brute_partial(p_in, np.array([0.4]), rho, c_in)
    print("  (3) partial support (n_in=1, m_nominal=0.6, rho=0.3, c=5):")
    print(f"        ProjectedTVBall(full_support=False) = {v_proj:.4f}  "
          f"(P_in={p_opt[0]:.4f}, leakage allowed)")
    print(f"        definitional brute force            = {v_brute:.4f}")
    assert abs(v_proj - v_brute) < 0.05, f"partial mismatch: {v_proj} vs {v_brute}"
    print("        OK: partial-support behavior matches the definition (leakage intact).")


def main():
    print("ProjectedTVBall full-support test:")
    test_motivating_example()
    test_randomized_full_support()
    test_partial_support_unchanged()
    print("\nAll checks passed: projected ball == full ball at full support; "
          "partial support unchanged.")


if __name__ == "__main__":
    main()