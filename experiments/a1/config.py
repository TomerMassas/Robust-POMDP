"""
A1 configuration.

Edit values here and run main.py (uniform-projection single run) or a sweep
driver under runs/. Parameters are grouped by role:

  - SIZES + SYNTHETIC POMDP MODEL + INITIAL BELIEF + UNCERTAINTY
        shared by both projection methods (both run on the same synthetic POMDP;
        they differ only in HOW the per-node support is chosen).
  - PROJECTION (UNIFORM masking)   -- the synthetic sweep / single run.
  - PROJECTION (PER-NODE sampling) -- the MCTS-style sampler (runs/sweep_sampler).
  - OUTPUT.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from synthetic_pomdp import RewardSpec


@dataclass
class A1Config:
    # ================= Sizes (shared) =================
    N: int = 10               # full state space size
    M: int = 10               # full observation space size
    H: int = 3                # planning horizon

    # ============ Synthetic POMDP model (shared) ============
    # None -> make_synthetic_pomdp default (0.70, 0.29, 0.01).
    risk_fractions: tuple[float, float, float] | None = (0.6, 0.3, 0.1)
    # Action transition drifts.
    inspect_drift: float = 0.10
    proceed_drift_1: float = 0.20
    proceed_drift_2: float = 0.05
    mitigate_drift: float = 0.75
    # Observation model: exp-decay noise scale + class boundaries on the obs axis
    # (safe -> [0, low_z], risky -> [low_z+1, high_z], cat -> [high_z+1, M-1]).
    # NOTE: warning_low/high are obs-model class boundaries, not a policy.
    obs_noise: float = 0.5
    warning_low: float = 0.4
    warning_high: float = 0.75
    # Rewards (full RewardSpec; edit fields via RewardSpec(...) to vary).
    rewards: RewardSpec = field(default_factory=lambda: RewardSpec())

    # ============ Initial belief (shared) ============
    class_mass: tuple[float, float, float] = (0.85, 0.14, 0.01)
    lambdas: tuple[float, float, float] = (0.03, 0.05, 0.10)

    # ============ Uncertainty (shared) ============
    # TV-ball; codebase uses ||P-Q||_1 (no 1/2), so rho is 2x textbook TV.
    rho_T: float = 0.05
    rho_Z: float = 0.05

    # ====== Projection method: UNIFORM masking (synthetic) ======
    # Same support at every node. Provide explicit S_in/Z_in to override, OR
    # leave None and let the picker fill them from (m_S, m_Z, scheme, decay, n_bins).
    S_in: list[int] | None = None
    Z_in: list[int] | None = None
    m_S: float = 0.5                    # fraction of state space kept
    m_Z: float = 0.5                    # fraction of obs space kept
    scheme: str = "exp_decay"           # "uniform" or "exp_decay"
    decay: float = 0.1                  # exp_decay parameter (ignored for uniform)
    n_bins: int = 10                    # exp_decay parameter (ignored for uniform)

    # ====== Projection method: PER-NODE sampling (MCTS-style) ======
    # K states/observations sampled per node from the projected belief (with
    # replacement, deduped -> |support| <= K). n_seeds: number of random seeds
    # the sampler sweep averages/scatters over; seed: base RNG seed.
    K: int = 3
    n_seeds: int = 10
    seed: int = 0

    # ================= Output =================
    out_dir: str = "results/single_run"   # relative to experiments/a1/
    write_tree_html: bool = True
    verbose: bool = True                  # print per-depth breakdown