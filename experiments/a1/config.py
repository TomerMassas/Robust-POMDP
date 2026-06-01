"""
A1 single-run configuration.

Edit values here and run main.py. All knobs default to the A1 design PDF
+ today's session decisions (mitigate_drift=0.75, uniform mitigate cost,
class-aligned obs targets).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from synthetic_pomdp import RewardSpec


@dataclass
class A1Config:
    # --- Sizes / horizon ---
    N: int = 30               # full state space size
    M: int = 10                # full obs space size
    H: int = 5                # planning horizon

    # --- Projection support ---
    # Provide explicit S_in / Z_in to override, OR leave None and let the
    # picker fill them from (m_S, m_Z, scheme, decay, n_bins).
    S_in: list[int] | None = None
    Z_in: list[int] | None = None
    m_S: float = 0.5                   # fraction of state space kept
    m_Z: float = 0.5                   # fraction of obs space kept
    scheme: str = "exp_decay"           # "uniform" or "exp_decay"
    decay: float = 0.1                  # exp_decay parameter (ignored for uniform)
    n_bins: int = 10                    # exp_decay parameter (ignored for uniform)

    # --- Warning policy thresholds (also drive the obs model's class-aligned targets) ---
    warning_low: float = 0.4
    warning_high: float = 0.75

    # --- Risk fractions: None -> falls back to make_synthetic_pomdp default (0.70, 0.29, 0.01) ---
    risk_fractions: tuple[float, float, float] | None = (0.6, 0.3, 0.1) #None

    # --- Action drifts ---
    inspect_drift: float = 0.10
    proceed_drift_1: float = 0.20
    proceed_drift_2: float = 0.05
    mitigate_drift: float = 0.75

    # --- Observation noise (exp-decay scale around target) ---
    obs_noise: float = 0.5

    # --- Reward spec (full RewardSpec dataclass; edit fields here to vary) ---
    rewards: RewardSpec = field(default_factory=lambda: RewardSpec())

    # --- Initial belief shape ---
    class_mass: tuple[float, float, float] = (0.85, 0.14, 0.01)
    lambdas: tuple[float, float, float] = (0.03, 0.05, 0.10)

    # --- Uncertainty radii (TV-ball, codebase uses ||P-Q||_1 -- rho is 2x textbook TV) ---
    rho_T: float = 0.1
    rho_Z: float = 0.1

    # --- Delta_K mode: "split" (Eq.47 upper bound, scalable) or "exact"
    #     (Eq.71 sign-orthant, small support only: |S_in|*|Z_in| <= max_free_bits) ---
    delta_k_mode: str = "exact" # exact , split
    delta_k_max_free_bits: int = 16

    # --- Output controls ---
    out_dir: str = "results/single_run"   # relative to experiments/a1/
    write_tree_html: bool = True
    verbose: bool = True                  # print per-depth breakdown
