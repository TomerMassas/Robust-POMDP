"""Phase 6 sub-step 3 - certificate validity integration smoke.

Builds full and projected fixed-policy trees, computes V_full and V_proj via
the existing robust evaluator, then verifies Eq. 74 certificate validity:

1. Full support + rho=0: certificate = 0, true_error = 0.
2. Partial support + rho=0: certificate >= true_error (validity).
3. Partial support + rho > 0: certificate >= true_error (validity).

Run from anywhere:
    python experiments/a1/smoke_test/phase6_certificate_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(_SCRIPT.parents[1]))  # experiments/a1/
sys.path.insert(0, str(_SCRIPT.parents[3]))  # repo root

from robust_pomdp.bounds.certificate import compute_certificate
from robust_pomdp.uncertainty.uncertainty_sets import TVDistance, uniform_uncertainty

from experiments.a1.synthetic_pomdp import ABORT, make_initial_belief, make_synthetic_pomdp
from experiments.a1.fixed_policy import warning_threshold_policy
from experiments.a1.tree_evaluator import build_fixed_policy_tree, evaluate_robust_value


def _build(model, b0, H, S_in, Z_in):
    return build_fixed_policy_tree(
        model, warning_threshold_policy, b0, H, S_in=S_in, Z_in=Z_in,
        abort_action=ABORT,
    )


def main() -> None:
    N, M, H = 4, 3, 3
    model = make_synthetic_pomdp(N=N, M=M)
    b0 = make_initial_belief(N=N)
    A = model.n_actions
    S_full = list(range(N))
    Z_full = list(range(M))
    S_in_proj = [0, 1]
    Z_in_proj = [0, 1]

    # Trees: full once, projected once. Reuse across rho variants.
    root_full, nd_full = _build(model, b0, H, S_full, Z_full)
    root_proj, nd_proj = _build(model, b0, H, S_in_proj, Z_in_proj)

    # ---- Test 1: rho=0, full support: certificate = 0, true_error = 0 ----
    unc_zero = uniform_uncertainty(N, A, 0.0, 0.0, TVDistance())
    V_full_z = evaluate_robust_value(root_full, nd_full, model, unc_zero)
    V_proj_z_fullsup = evaluate_robust_value(root_full, nd_full, model, unc_zero)
    cert_fullsup = compute_certificate(
        root_full, nd_full, model, unc_zero, S_full, Z_full, b0, H,
    )
    assert abs(cert_fullsup["certificate"]) < 1e-9, (
        f"full + rho=0: certificate = {cert_fullsup['certificate']}, expected 0"
    )
    err_fullsup = abs(V_full_z - V_proj_z_fullsup)
    assert err_fullsup < 1e-9, f"full + rho=0: true_error = {err_fullsup}, expected 0"
    print(f"  rho=0, full support: V_full={V_full_z:.4f}  cert=0  true_error=0  OK")

    # ---- Test 2: rho=0, partial support: certificate >= true_error ----
    V_proj_z = evaluate_robust_value(root_proj, nd_proj, model, unc_zero)
    cert_p_z = compute_certificate(
        root_proj, nd_proj, model, unc_zero, S_in_proj, Z_in_proj, b0, H,
    )
    err_p_z = abs(V_full_z - V_proj_z)
    assert err_p_z <= cert_p_z["certificate"] + 1e-6, (
        f"partial + rho=0: true_error {err_p_z} > cert {cert_p_z['certificate']}"
    )
    print(f"  rho=0, S_in=[0,1], Z_in=[0,1]:")
    print(f"        V_full={V_full_z:.4f}  V_proj={V_proj_z:.4f}  true_error={err_p_z:.4f}")
    print(f"        certificate={cert_p_z['certificate']:.4f}  (>= true_error)  OK")
    print(f"        delta_b={cert_p_z['delta_b']:.4f}  R_max={cert_p_z['R_max']:.1f}  "
          f"m_in={cert_p_z['m_in']:.4f}")
    print(f"        phi:                {[f'{p:.4f}' for p in cert_p_z['phi']]}")
    print(f"        max_path_sum_dK:    {[f'{d:.4f}' for d in cert_p_z['max_path_sum_delta_K']]}")
    print(f"        per_depth_bound:    {[f'{b:.4f}' for b in cert_p_z['per_depth_bound']]}")

    # ---- Test 3: rho > 0, partial support: certificate >= true_error ----
    rho = 0.1
    unc_mid = uniform_uncertainty(N, A, rho, rho, TVDistance())
    V_full_r = evaluate_robust_value(root_full, nd_full, model, unc_mid)
    V_proj_r = evaluate_robust_value(root_proj, nd_proj, model, unc_mid)
    cert_p_r = compute_certificate(
        root_proj, nd_proj, model, unc_mid, S_in_proj, Z_in_proj, b0, H,
    )
    err_p_r = abs(V_full_r - V_proj_r)
    assert err_p_r <= cert_p_r["certificate"] + 1e-6, (
        f"partial + rho={rho}: true_error {err_p_r} > cert {cert_p_r['certificate']}"
    )
    print(f"  rho={rho}, S_in=[0,1], Z_in=[0,1]:")
    print(f"        V_full={V_full_r:.4f}  V_proj={V_proj_r:.4f}  true_error={err_p_r:.4f}")
    print(f"        certificate={cert_p_r['certificate']:.4f}  (>= true_error)  OK")
    print(f"        delta_b={cert_p_r['delta_b']:.4f}  R_max={cert_p_r['R_max']:.1f}")
    print(f"        phi:                {[f'{p:.4f}' for p in cert_p_r['phi']]}")
    print(f"        max_path_sum_dK:    {[f'{d:.4f}' for d in cert_p_r['max_path_sum_delta_K']]}")
    print(f"        per_depth_bound:    {[f'{b:.4f}' for b in cert_p_r['per_depth_bound']]}")

    print("\nCertificate validity OK (full=0 + partial-rho-0 + partial-rho-pos)")


if __name__ == "__main__":
    main()
