"""
Central experiment configuration — one place to set every knob.

Edit the defaults here (or construct `Config(...)` with overrides) and pass it to
an experiment's `run(cfg)`. `Config.build()` returns the (model, b0, uncertainty)
triple so every experiment builds the domain the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from Algorithm.core.uncertainty import UncertaintySets, uniform_uncertainty
from Algorithm.domains.synthetic import (RewardSpec, make_initial_belief,
                                         make_synthetic_pomdp)
from Algorithm.core.pomdp import TabularPOMDP


@dataclass
class Config:
    # --- domain: synthetic ordered-risk POMDP ---
    N: int = 10                                       # states
    M: int = 10                                        # observations
    H: int = 5                                        # reward depths (0..H-1)
    risk_fractions: tuple[float, float, float] = (0.70, 0.29, 0.01)
    inspect_drift: float = 0.10
    proceed_drift_1: float = 0.20
    proceed_drift_2: float = 0.05
    mitigate_drift: float = 0.75
    obs_noise: float = 0.5
    warning_low: float = 0.4                          # shared by obs model AND warning policy
    warning_high: float = 0.75
    rewards: RewardSpec = field(default_factory=RewardSpec)

    # --- initial belief ---
    class_mass: tuple[float, float, float] = (0.85, 0.14, 0.01)
    lambdas: tuple[float, float, float] = (0.03, 0.05, 0.10)

    # --- uncertainty (TV radii; L1 convention, 2x textbook) ---
    rho_T: float = 0.1
    rho_Z: float = 0.1

    # --- post-hoc experiment: which fixed policy to certify ---
    policy: str = "greedy"                            # "warning" | "greedy"

    # --- support sweep (fraction schemes) ---
    fractions: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)
    scheme: str = "exp_decay"                         # "uniform" | "exp_decay" | "sampler"
    decay: float = 0.5                                # exp_decay only
    n_bins: int = 10                                  # exp_decay only

    # --- per-node sampler (scheme="sampler"): K draws/node from the belief ---
    K_grid: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 2048)
    n_seeds: int = 5                                  # random seeds per K (scatter)
    seed: int = 0                                     # base RNG seed

    # --- output ---
    out_dir: str = "results"                          # relative -> resolved next to the run script
    write_plot: bool = True

    def build(self) -> tuple[TabularPOMDP, np.ndarray, UncertaintySets]:
        """Construct (model, b0, uncertainty) from the config."""
        model = make_synthetic_pomdp(
            self.N, self.M,
            risk_fractions=self.risk_fractions,
            inspect_drift=self.inspect_drift,
            proceed_drift_1=self.proceed_drift_1,
            proceed_drift_2=self.proceed_drift_2,
            mitigate_drift=self.mitigate_drift,
            obs_noise=self.obs_noise,
            warning_low=self.warning_low,
            warning_high=self.warning_high,
            rewards=self.rewards,
        )
        b0 = make_initial_belief(
            self.N,
            risk_fractions=self.risk_fractions,
            class_mass=self.class_mass,
            lambdas=self.lambdas,
        )
        unc = uniform_uncertainty(self.N, model.n_actions, self.rho_T, self.rho_Z)
        return model, b0, unc
