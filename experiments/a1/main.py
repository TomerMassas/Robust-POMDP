"""
A1 single-run driver. Edit experiments/a1/config.py and run:

    python experiments/a1/main.py

or run from PyCharm (working directory = repo root).

Builds full + projected fixed-policy trees, computes V_full / V_proj /
certificate, prints the breakdown, and renders the trees as standalone HTML
under config.out_dir.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(_SCRIPT.parents[0]))  # experiments/a1/
sys.path.insert(0, str(_SCRIPT.parents[2]))  # repo root

from robust_pomdp.bounds.certificate import compute_certificate
from robust_pomdp.bounds.support_picker import exponential_decay_pick, uniform_pick
from robust_pomdp.uncertainty.uncertainty_sets import TVDistance, uniform_uncertainty

from synthetic_pomdp import ABORT, ACTION_NAMES, make_initial_belief, make_synthetic_pomdp
from fixed_policy import warning_threshold_policy
from tree_evaluator import build_fixed_policy_tree, evaluate_robust_value
from config import A1Config

from viz.tree_viz import render_tree_to_html


def _resolve_support(explicit, m, total, scheme, decay, n_bins):
    """Explicit list overrides; otherwise pick via the named scheme."""
    if explicit is not None:
        return list(explicit)
    if scheme == "uniform":
        return uniform_pick(m, total)
    if scheme == "exp_decay":
        return exponential_decay_pick(m, total, n_bins=n_bins, decay=decay)
    raise ValueError(f"unknown scheme: {scheme!r}")


def main(cfg: A1Config) -> dict:
    # Build the POMDP.
    pomdp_kwargs = dict(
                        inspect_drift=cfg.inspect_drift,
                        proceed_drift_1=cfg.proceed_drift_1,
                        proceed_drift_2=cfg.proceed_drift_2,
                        mitigate_drift=cfg.mitigate_drift,
                        obs_noise=cfg.obs_noise,
                        warning_low=cfg.warning_low,
                        warning_high=cfg.warning_high,
                        rewards=cfg.rewards,
                    )
    if cfg.risk_fractions is not None:
        pomdp_kwargs["risk_fractions"] = cfg.risk_fractions
    model = make_synthetic_pomdp(N=cfg.N, M=cfg.M, **pomdp_kwargs)

    # Initial belief.
    belief_kwargs = dict(class_mass=cfg.class_mass, lambdas=cfg.lambdas)
    if cfg.risk_fractions is not None:
        belief_kwargs["risk_fractions"] = cfg.risk_fractions
    b0 = make_initial_belief(N=cfg.N, **belief_kwargs)

    # Policy closure (binds the threshold kwargs to the tree-builder's 3-arg signature).
    def policy(depth, last_obs, M):
        return warning_threshold_policy(depth, last_obs, M, low=cfg.warning_low, high=cfg.warning_high,)

    # Uncertainty.
    uncertainty = uniform_uncertainty(cfg.N, model.n_actions, cfg.rho_T, cfg.rho_Z, TVDistance(),)

    # Resolve projection support (explicit override or picker).
    S_in = _resolve_support(cfg.S_in, cfg.m_S, cfg.N, cfg.scheme, cfg.decay, cfg.n_bins)
    Z_in = _resolve_support(cfg.Z_in, cfg.m_Z, cfg.M, cfg.scheme, cfg.decay, cfg.n_bins)

    # Trees: full once, projected once.
    S_full = list(range(cfg.N))
    Z_full = list(range(cfg.M))
    root_full, nd_full = build_fixed_policy_tree(model, policy, b0, cfg.H, S_in=S_full, Z_in=Z_full, abort_action=ABORT,)
    V_full = evaluate_robust_value(root_full, nd_full, model, uncertainty)

    root_proj, nd_proj = build_fixed_policy_tree(model, policy, b0, cfg.H, S_in=S_in, Z_in=Z_in, abort_action=ABORT,)
    V_proj = evaluate_robust_value(root_proj, nd_proj, model, uncertainty)

    # Certificate.
    cert = compute_certificate(root_proj, nd_proj, model, uncertainty, S_in, Z_in, b0, cfg.H,
                               delta_k_mode=cfg.delta_k_mode,
                               delta_k_max_free_bits=cfg.delta_k_max_free_bits)
    true_error = abs(V_full - V_proj)
    valid = true_error <= cert["certificate"] + 1e-6

    # Output paths -- relative paths resolve against experiments/a1/.
    out_dir = Path(cfg.out_dir)
    if not out_dir.is_absolute():
        out_dir = _SCRIPT.parents[0] / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    full_html = out_dir / "tree_full.html"
    proj_html = out_dir / "tree_projected.html"
    if cfg.write_tree_html:
        render_tree_to_html(
                            root_full, nd_full, ACTION_NAMES, full_html,
                            title=f"A1 full tree (N={cfg.N}, M={cfg.M}, H={cfg.H})",
                        )
        render_tree_to_html(
                            root_proj, nd_proj, ACTION_NAMES, proj_html,
                            title=f"A1 projected tree (|S_in|={len(S_in)}, |Z_in|={len(Z_in)})",
                        )

    # Summary print.
    print(f"\nA1 single run | N={cfg.N} M={cfg.M} H={cfg.H}  "
          f"S_in={S_in} Z_in={Z_in}  rho_T={cfg.rho_T} rho_Z={cfg.rho_Z}  "
          f"delta_K={cfg.delta_k_mode}")
    print(f"  V_full       = {V_full:.4f}")
    print(f"  V_proj       = {V_proj:.4f}")
    print(f"  true_error   = {true_error:.4f}")
    print(f"  certificate  = {cert['certificate']:.4f}  "
          f"({'VALID' if valid else 'INVALID'})")
    c_tot = cert['certificate'] or 1.0   # avoid div-by-zero at full support
    print(f"  breakdown    : delta_b={cert['delta_b_contribution']:.2f} "
          f"({100*cert['delta_b_contribution']/c_tot:.0f}%)  "
          f"delta_K={cert['delta_K_contribution']:.2f} "
          f"({100*cert['delta_K_contribution']/c_tot:.0f}%)  "
          f"leakage={cert['leakage_contribution']:.2f} "
          f"({100*cert['leakage_contribution']/c_tot:.0f}%)")

    if cfg.verbose:
        print(f"  delta_b={cert['delta_b']:.4f}  R_max={cert['R_max']:.2f}  m_in={cert['m_in']:.4f}")
        print(f"  phi                : {[f'{p:.4f}' for p in cert['phi']]}")
        print(f"  max_path_sum_dK    : {[f'{d:.4f}' for d in cert['max_path_sum_delta_K']]}")
        print(f"  per_depth_bound    : {[f'{b:.4f}' for b in cert['per_depth_bound']]}")
        if cfg.write_tree_html:
            print(f"  trees              : {full_html}")
            print(f"                       {proj_html}")

    return {
        "V_full": V_full, "V_proj": V_proj, "true_error": true_error, "valid": valid,
        **cert,
    }


if __name__ == "__main__":
    main(A1Config())

