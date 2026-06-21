"""Phase B - projected-tree-follows-pi_full consistency smoke.

Verifies the wiring of the greedy policy through the projected evaluator:
  1. At FULL support, the projected tree following pi_full evaluates to exactly
     V_full (from solve_full_greedy) -- this is the m=1 endpoint where the
     certificate must close. Relies on the ProjectedTVBall full-support fix.
  2. The leakage-only certificate at full support is 0.
  3. At partial support, certificate >= true_error (validity).

Run from anywhere:
    python experiments/a1/smoke_test/phaseB_consistency_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(_SCRIPT.parents[1]))  # experiments/a1/
sys.path.insert(0, str(_SCRIPT.parents[3]))  # repo root

from robust_pomdp.bounds.certificate import compute_certificate
from robust_pomdp.uncertainty.uncertainty_sets import TVDistance, uniform_uncertainty

from synthetic_pomdp import ABORT, make_initial_belief, make_synthetic_pomdp
from tree_evaluator import build_policy_tree_by_path, evaluate_robust_value
from greedy_solver import solve_full_greedy


def main():
    N, M, H = 4, 3, 3
    model = make_synthetic_pomdp(N=N, M=M)
    b0 = make_initial_belief(N=N)
    unc = uniform_uncertainty(N, model.n_actions, 0.1, 0.1, TVDistance())

    V_full, pi_full = solve_full_greedy(model, unc, b0, H, abort_action=ABORT)
    policy = lambda obs_path: pi_full[obs_path]
    S_full, Z_full = list(range(N)), list(range(M))

    # ---- (1) full-support projected eval following pi_full == V_full ----
    root_f, nd_f = build_policy_tree_by_path(model, policy, b0, H,
                                             S_in=S_full, Z_in=Z_full, abort_action=ABORT)
    v_proj_full = evaluate_robust_value(root_f, nd_f, model, unc)
    assert abs(v_proj_full - V_full) < 1e-7, (
        f"full-support V_proj={v_proj_full:.6f} != V_full={V_full:.6f} "
        f"(diff {abs(v_proj_full - V_full):.2e})"
    )
    print(f"  (1) full support: V_proj(follow pi_full) = {v_proj_full:.6f}  ==  "
          f"V_full {V_full:.6f}   OK")

    # ---- (2) leakage certificate at full support == 0 ----
    cert_full = compute_certificate(root_f, nd_f, model, unc, b0, H)
    assert abs(cert_full["certificate"]) < 1e-9, (
        f"full-support cert = {cert_full['certificate']}, expected 0"
    )
    print(f"  (2) full support: leakage certificate = {cert_full['certificate']:.2e}  (== 0)   OK")

    # ---- (3) partial support: certificate >= true_error ----
    S_in, Z_in = [0, 1], [0, 1]
    root_p, nd_p = build_policy_tree_by_path(model, policy, b0, H,
                                             S_in=S_in, Z_in=Z_in, abort_action=ABORT)
    v_proj_p = evaluate_robust_value(root_p, nd_p, model, unc)
    cert_p = compute_certificate(root_p, nd_p, model, unc, b0, H)
    true_err = abs(V_full - v_proj_p)
    assert true_err <= cert_p["certificate"] + 1e-6, (
        f"partial: true_error {true_err:.4f} > cert {cert_p['certificate']:.4f}"
    )
    print(f"  (3) partial (S_in={S_in}, Z_in={Z_in}): V_proj={v_proj_p:.4f}  "
          f"true_error={true_err:.4f}  cert={cert_p['certificate']:.4f}  (valid)   OK")

    print("\nPhase B (projected tree follows pi_full) OK.")


if __name__ == "__main__":
    main()