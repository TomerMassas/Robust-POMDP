"""
Phase 2 verification — §1.3 #2: LP correctness.

Three layers of checking (per plan §1.3 #2 and §5 Phase 2), strongest first:

  Layer 1 — Hand-checkable trivials. Fixed analytical answers, 1e-6 bar.
  Layer 2 — Internal consistency on a grid of inputs. Solver-agnostic
            structural properties, 1e-9 bar.
  Layer 3 — Soft Julia cross-check (optional). Loaded from lp_reference.json
            if present. optimal_value diff > 1e-3 fails; optimal_p divergences
            on entries flagged degenerate are reported, not failed.

Run:
    python experiments/tiger/verify_lp_correctness.py

Exits 0 if the §1.3 #2 gate passes (layer 1 + layer 2 + layer 3 value-diffs).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from robust_pomdp.uncertainty.projected_sets import ProjectedTVBall


REF_PATH = Path(__file__).parent / "lp_reference.json"


# ---------------------------------------------------------------------------
# Layer 1 — hand-checkable trivials
# ---------------------------------------------------------------------------

def check_hand_trivials() -> int:
    """Hand-derived analytical answers. Strict 1e-6.

    rho = 0  -> optimal_p = p_nominal, optimal_value = sum(p_nominal * c).
    n = 1, rho = 0.5, p_o = [0.5], c = [1.0] -> p = [0.25], value = 0.25.
        Reasoning: TV constraint becomes 2|p - p_o| <= rho, so |p - p_o| <= 0.25.
        Min of p over [0.25, 0.75] is 0.25.
    n = 2, rho = 0.2, p_o = [0.5, 0.5], c = [1, -1] -> p = [0.4, 0.6], value = -0.2.
        Reasoning: shift 0.1 mass from idx 0 to idx 1; TV cost is 0.1+0.1+0 = 0.2 = rho.
    """
    failures = 0

    lp2 = ProjectedTVBall(n=2)

    val, p = lp2.solve([0.5, 0.5], rho=0.0, objective_coeffs=[1.0, -1.0])
    if not (abs(val - 0.0) < 1e-6 and np.allclose(p, [0.5, 0.5], atol=1e-6)):
        print(f"[layer 1] FAIL rho=0: val={val}, p={p}")
        failures += 1
    else:
        print("[layer 1] rho=0 trivial: OK")

    lp1 = ProjectedTVBall(n=1)
    val, p = lp1.solve([0.5], rho=0.5, objective_coeffs=[1.0])
    if not (abs(val - 0.25) < 1e-6 and abs(p[0] - 0.25) < 1e-6):
        print(f"[layer 1] FAIL n=1: val={val}, p={p}")
        failures += 1
    else:
        print("[layer 1] n=1, rho=0.5, p_o=[0.5], c=[1]: OK")

    val, p = lp2.solve([0.5, 0.5], rho=0.2, objective_coeffs=[1.0, -1.0])
    if not (abs(val - (-0.2)) < 1e-6 and np.allclose(p, [0.4, 0.6], atol=1e-6)):
        print(f"[layer 1] FAIL n=2: val={val}, p={p}")
        failures += 1
    else:
        print("[layer 1] n=2, rho=0.2, p_o=[0.5,0.5], c=[1,-1]: OK")

    return failures


# ---------------------------------------------------------------------------
# Layer 2 — internal consistency on a grid
# ---------------------------------------------------------------------------

GRID_N2: list[tuple[list[float], float, list[float]]] = [
    ([0.5, 0.5], 0.05, [1.0, -1.0]),
    ([0.5, 0.5], 0.10, [1.0, -1.0]),
    ([0.5, 0.5], 0.20, [10.0, -10.0]),
    ([0.85, 0.15], 0.05, [1.0, -1.0]),
    ([0.85, 0.15], 0.20, [-2.0, 1.0]),
    ([0.30, 0.70], 0.10, [5.0, 5.0]),
    ([0.30, 0.70], 0.30, [-1.0, 0.5]),
]

GRID_N3: list[tuple[list[float], float, list[float]]] = [
    ([0.4, 0.4, 0.2], 0.05, [1.0, -1.0, 0.5]),
    ([0.4, 0.4, 0.2], 0.20, [-2.0, 1.0, 1.0]),
    ([0.7, 0.2, 0.1], 0.05, [1.0, -1.0, 0.5]),
    ([0.7, 0.2, 0.1], 0.20, [-2.0, 1.0, 1.0]),
]


def check_internal_consistency() -> int:
    """Properties any correct solve must satisfy, regardless of solver.

    For every (p_nom, rho, c):
      optimal_p[i] >= -1e-9 (non-negativity, with tiny solver tolerance)
      sum(optimal_p) <= 1 + 1e-9 (sub-probability)
      |sum(optimal_p * c) - optimal_value| < 1e-9 (objective consistency)
    """
    failures = 0
    for n, grid in [(2, GRID_N2), (3, GRID_N3)]:
        lp = ProjectedTVBall(n=n)
        for p_nom, rho, c in grid:
            val, p_star = lp.solve(p_nom, rho, c)
            if np.any(p_star < -1e-9):
                print(f"[layer 2] FAIL non-neg: p_nom={p_nom}, rho={rho}, c={c}, p={p_star}")
                failures += 1
            if np.sum(p_star) > 1 + 1e-9:
                print(f"[layer 2] FAIL sub-prob: p_nom={p_nom}, rho={rho}, c={c}, "
                      f"sum={np.sum(p_star)}"
                      )
                failures += 1
            recomputed = float(np.dot(p_star, c))
            if abs(recomputed - val) > 1e-9:
                print(
                    f"[layer 2] FAIL obj: p_nom={p_nom}, rho={rho}, c={c}, "
                    f"val={val}, recomputed={recomputed}"
                )
                failures += 1

    n_total = len(GRID_N2) + len(GRID_N3)
    if failures == 0:
        print(f"[layer 2] internal consistency on {n_total} inputs: OK")
    return failures


# ---------------------------------------------------------------------------
# Layer 3 — soft Julia cross-check
# ---------------------------------------------------------------------------

def check_julia_cross() -> tuple[int, int]:
    """Compare against Julia outputs from lp_reference.json (if present).

    Returns (value_failures, p_warnings_on_nondegenerate).
    Only value_failures count toward the §1.3 #2 gate.
    """
    if not REF_PATH.exists():
        print(
            f"[layer 3] {REF_PATH.name} not found — skipping. "
            "Run experiments/tiger/_capture_lp_reference.jl on the Julia side first."
        )
        return 0, 0

    with REF_PATH.open() as f:
        reference = json.load(f)

    val_failures = 0
    p_warn_nondegenerate = 0
    p_warn_degenerate = 0
    by_n: dict[int, ProjectedTVBall] = {}

    for entry in reference:
        p_nom = entry["p_nominal"]
        rho = entry["rho"]
        c = entry["c"]
        julia_val = entry["optimal_value"]
        julia_p = np.asarray(entry["optimal_p"], dtype=np.float64)
        is_degenerate = bool(entry.get("degenerate", False))

        n = len(p_nom)
        if n not in by_n:
            by_n[n] = ProjectedTVBall(n=n)
        py_val, py_p = by_n[n].solve(p_nom, rho, c)

        val_diff = abs(py_val - julia_val)
        if val_diff > 1e-3:
            print(
                f"[layer 3] FAIL value: p_nom={p_nom}, rho={rho}, c={c} "
                f"-> py={py_val:.9g}, julia={julia_val:.9g}, diff={val_diff:.6g}"
            )
            val_failures += 1
            continue

        p_diff = float(np.max(np.abs(py_p - julia_p)))
        if p_diff > 1e-3:
            if is_degenerate:
                p_warn_degenerate += 1
            else:
                p_warn_nondegenerate += 1
                print(
                    f"[layer 3] WARN non-deg p-diff: p_nom={p_nom}, rho={rho}, "
                    f"c={c}, py_p={py_p}, julia_p={julia_p}, max_diff={p_diff:.6g}"
                )

    n_total = len(reference)
    print(
        f"[layer 3] cross-check on {n_total} entries: "
        f"value mismatches={val_failures}, "
        f"p-warns on non-degenerate={p_warn_nondegenerate}, "
        f"p-warns on degenerate={p_warn_degenerate} (expected, not failures)"
    )
    return val_failures, p_warn_nondegenerate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Phase 2 verification - §1.3 #2: LP correctness")
    print("=" * 60)

    print("\nLayer 1: hand-checkable trivials")
    print("-" * 60)
    f1 = check_hand_trivials()

    print("\nLayer 2: internal consistency")
    print("-" * 60)
    f2 = check_internal_consistency()

    print("\nLayer 3: soft Julia cross-check")
    print("-" * 60)
    f3_val, _ = check_julia_cross()

    print("\n" + "=" * 60)
    if f1 == 0 and f2 == 0 and f3_val == 0:
        print("Phase 2 GATE §1.3 #2: PASS")
        print("=" * 60)
    else:
        print(
            f"Phase 2 GATE §1.3 #2: FAIL "
            f"(layer 1={f1}, layer 2={f2}, layer 3 value mismatches={f3_val})"
        )
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
