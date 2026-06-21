"""
A1 single-run driver. Edit experiments/a1/config.py and run:

    python experiments/a1/main.py

or run from PyCharm (working directory = repo root).

Solves the full-robust-greedy policy pi_full once (solve_full_greedy), then
evaluates the projected tree that FOLLOWS pi_full over the sampled support,
computes V_proj and the leakage certificate, prints the breakdown, and renders
the trees as standalone HTML under config.out_dir.

build_reference (support-independent) and evaluate_projection (per-support) are
factored out so a sweep can solve the greedy policy once and reuse it.
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
from tree_evaluator import build_policy_tree_by_path, evaluate_robust_value
from greedy_solver import solve_full_greedy
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


def build_reference(cfg: A1Config) -> dict:
    """Support-INDEPENDENT setup: POMDP, belief, uncertainty, and the full-
    robust-greedy policy/value (solved ONCE). Reused across a support sweep."""
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

    belief_kwargs = dict(class_mass=cfg.class_mass, lambdas=cfg.lambdas)
    if cfg.risk_fractions is not None:
        belief_kwargs["risk_fractions"] = cfg.risk_fractions
    b0 = make_initial_belief(N=cfg.N, **belief_kwargs)

    uncertainty = uniform_uncertainty(cfg.N, model.n_actions, cfg.rho_T, cfg.rho_Z, TVDistance(),)

    v_full, pi_full = solve_full_greedy(model, uncertainty, b0, cfg.H, abort_action=ABORT)
    r_max = float(max(abs(model.R.min()), abs(model.R.max())))
    return dict(model=model, b0=b0, uncertainty=uncertainty,
                V_full=v_full, pi_full=pi_full, R_max=r_max, H=cfg.H)


def evaluate_projection(ref: dict, S_in: list[int], Z_in: list[int]) -> dict:
    """Per-SUPPORT evaluation: build the projected tree following pi_full,
    compute V_proj and the leakage certificate. Returns scalars + the tree (for viz)."""
    model, b0, unc, H = ref["model"], ref["b0"], ref["uncertainty"], ref["H"]
    policy = lambda obs_path: ref["pi_full"][obs_path]
    root, nd = build_policy_tree_by_path(model, policy, b0, H, S_in=S_in, Z_in=Z_in, abort_action=ABORT,)
    v_proj = evaluate_robust_value(root, nd, model, unc)
    cert = compute_certificate(root, nd, model, unc, b0, H)
    true_error = abs(ref["V_full"] - v_proj)
    valid = true_error <= cert["certificate"] + 1e-6
    return dict(V_proj=v_proj, true_error=true_error, valid=valid,
                root=root, node_data=nd, **cert)


def main(cfg: A1Config) -> dict:
    ref = build_reference(cfg)
    model, b0 = ref["model"], ref["b0"]
    V_full, pi_full = ref["V_full"], ref["pi_full"]
    policy = lambda obs_path: pi_full[obs_path]

    S_in = _resolve_support(cfg.S_in, cfg.m_S, cfg.N, cfg.scheme, cfg.decay, cfg.n_bins)
    Z_in = _resolve_support(cfg.Z_in, cfg.m_Z, cfg.M, cfg.scheme, cfg.decay, cfg.n_bins)

    res = evaluate_projection(ref, S_in, Z_in)
    V_proj, true_error, valid = res["V_proj"], res["true_error"], res["valid"]

    # Output paths -- relative paths resolve against experiments/a1/.
    out_dir = Path(cfg.out_dir)
    if not out_dir.is_absolute():
        out_dir = _SCRIPT.parents[0] / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    full_html = out_dir / "tree_full.html"
    proj_html = out_dir / "tree_projected.html"
    if cfg.write_tree_html:
        S_full, Z_full = list(range(cfg.N)), list(range(cfg.M))
        root_full, nd_full = build_policy_tree_by_path(model, policy, b0, cfg.H,
                                                       S_in=S_full, Z_in=Z_full, abort_action=ABORT,)
        render_tree_to_html(root_full, nd_full, ACTION_NAMES, full_html,
                            title=f"A1 full greedy tree (N={cfg.N}, M={cfg.M}, H={cfg.H})")
        render_tree_to_html(res["root"], res["node_data"], ACTION_NAMES, proj_html,
                            title=f"A1 projected tree (|S_in|={len(S_in)}, |Z_in|={len(Z_in)})")

    # Summary print.
    print(f"\nA1 single run | N={cfg.N} M={cfg.M} H={cfg.H}  "
          f"S_in={S_in} Z_in={Z_in}  rho_T={cfg.rho_T} rho_Z={cfg.rho_Z}")
    print(f"  V_full       = {V_full:.4f}")
    print(f"  V_proj       = {V_proj:.4f}")
    print(f"  true_error   = {true_error:.4f}")
    print(f"  certificate  = {res['certificate']:.4f}  ({'VALID' if valid else 'INVALID'})")
    c_tot = res['certificate'] or 1.0
    print(f"  breakdown    : delta_b={res['delta_b_contribution']:.2f} "
          f"({100*res['delta_b_contribution']/c_tot:.0f}%)  "
          f"leakage={res['leakage_contribution']:.2f} "
          f"({100*res['leakage_contribution']/c_tot:.0f}%)")

    if cfg.verbose:
        print(f"  R_max={ref['R_max']:.2f}  m_in={res['m_in']:.4f}")
        print(f"  phi                : {[f'{p:.4f}' for p in res['phi']]}")
        print(f"  per_depth_bound    : {[f'{b:.4f}' for b in res['per_depth_bound']]}")
        if cfg.write_tree_html:
            print(f"  trees              : {full_html}")
            print(f"                       {proj_html}")

    return {"V_full": V_full, "V_proj": V_proj, "true_error": true_error, "valid": valid,
            "certificate": res["certificate"], "R_max": ref["R_max"], "m_in": res["m_in"],
            "delta_b_contribution": res["delta_b_contribution"],
            "leakage_contribution": res["leakage_contribution"]}


if __name__ == "__main__":
    main(A1Config())