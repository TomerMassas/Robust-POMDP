"""
Synthetic ordered-risk POMDP.

DUPLICATED from experiments/a1/synthetic_pomdp.py (logic unchanged; only the
TabularPOMDP import points at Algorithm.core) so Algorithm/ is self-contained.

Hidden state space S = {0, ..., N-1} ordered by hidden risk (safe / risky /
catastrophic). Actions A = {INSPECT, PROCEED, MITIGATE, ABORT}. Observations
Z = {0, ..., M-1} ordered by warning severity.

ABORT self-loops with 0 reward (T[s,ABORT,s]=1, R[s,ABORT]=0); the planners /
evaluators treat it as terminal, which is value-equivalent and leak-free
because its continuation value is 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from Algorithm.core.pomdp import TabularPOMDP


# 0-based action indices.
INSPECT = 0
PROCEED = 1
MITIGATE = 2
ABORT = 3
N_ACTIONS = 4
ACTION_NAMES = ("inspect", "proceed", "mitigate", "abort")


@dataclass(frozen=True)
class RewardSpec:
    """Reward defaults per the A1 design PDF (costs stored positive)."""
    inspect_cost: float        = 1.0

    proceed_safe: float        = 5.0
    proceed_risky_cost: float  = 5.0
    proceed_cat_cost: float    = 10.0

    mitigate_safe_cost: float  = 2.0
    mitigate_risky_cost: float = 2.0
    mitigate_cat_cost: float   = 5.0

    abort: float               = 0.0


def class_sizes(N: int,
                risk_fractions: tuple[float, float, float] = (0.70, 0.29, 0.01)
                ) -> tuple[int, int, int]:
    """Partition N into (n_safe, n_risky, n_cat). Catastrophic clamped to >= 1."""
    safe_frac, _risky_frac, cat_frac = risk_fractions
    n_safe = int(safe_frac * N)
    n_cat = max(1, int(cat_frac * N))
    n_risky = N - n_safe - n_cat
    if n_risky < 0:
        raise ValueError(
            f"Risk fractions {risk_fractions} produce n_risky={n_risky} < 0 for N={N}"
        )
    return n_safe, n_risky, n_cat


def make_synthetic_pomdp(N: int,
                         M: int,
                         *,
                         risk_fractions: tuple[float, float, float] = (0.70, 0.29, 0.01),
                         inspect_drift: float = 0.10,
                         proceed_drift_1: float = 0.20,
                         proceed_drift_2: float = 0.05,
                         mitigate_drift: float = 0.75,
                         obs_noise: float = 0.5,
                         warning_low: float = 0.4,
                         warning_high: float = 0.75,
                         rewards: RewardSpec | None = None
                         ) -> TabularPOMDP:
    """Build the synthetic ordered-risk TabularPOMDP per the A1 design PDF."""
    if N < 2:
        raise ValueError(f"N must be >= 2, got {N}")
    if M < 2:
        raise ValueError(f"M must be >= 2, got {M}")

    rewards = rewards or RewardSpec()
    n_safe, n_risky, n_cat = class_sizes(N, risk_fractions)

    T = np.zeros((N, N_ACTIONS, N), dtype=np.float64)
    for s in range(N):
        T[s, INSPECT, s] += 1.0 - inspect_drift
        T[s, INSPECT, min(s + 1, N - 1)] += inspect_drift

        T[s, PROCEED, s] += 1.0 - proceed_drift_1 - proceed_drift_2
        T[s, PROCEED, min(s + 1, N - 1)] += proceed_drift_1
        T[s, PROCEED, min(s + 2, N - 1)] += proceed_drift_2

        T[s, MITIGATE, s] += 1.0 - mitigate_drift
        T[s, MITIGATE, max(s - 1, 0)] += mitigate_drift

        T[s, ABORT, s] = 1.0

    O = np.zeros((N, M), dtype=np.float64)
    low_z = int(np.floor(warning_low * (M - 1)))
    high_z = int(np.floor(warning_high * (M - 1)))
    safe_tgt_hi = low_z + 0.5
    risky_tgt_lo, risky_tgt_hi = low_z + 0.5, high_z + 0.5
    cat_tgt_lo = high_z + 0.5
    for s in range(N):
        if s < n_safe:
            target = (s / n_safe) * safe_tgt_hi
        elif s < n_safe + n_risky:
            p = s - n_safe
            target = risky_tgt_lo + ((p + 0.5) / n_risky) * (risky_tgt_hi - risky_tgt_lo)
        else:
            p = s - n_safe - n_risky
            target = cat_tgt_lo + ((p + 1) / n_cat) * (M - 1 - cat_tgt_lo)
        raw = np.exp(-np.abs(np.arange(M) - target) / obs_noise)
        O[s, :] = raw / raw.sum()

    R = np.zeros((N, N_ACTIONS), dtype=np.float64)
    for s in range(N):
        R[s, INSPECT] = -rewards.inspect_cost
        R[s, ABORT] = rewards.abort
        if s < n_safe:
            R[s, PROCEED] = rewards.proceed_safe
            R[s, MITIGATE] = -rewards.mitigate_safe_cost
        elif s < n_safe + n_risky:
            R[s, PROCEED] = -rewards.proceed_risky_cost
            R[s, MITIGATE] = -rewards.mitigate_risky_cost
        else:
            R[s, PROCEED] = -rewards.proceed_cat_cost
            R[s, MITIGATE] = -rewards.mitigate_cat_cost

    return TabularPOMDP.from_matrices(T, O, R)


def make_initial_belief(N: int,
                        *,
                        risk_fractions: tuple[float, float, float] = (0.70, 0.29, 0.01),
                        class_mass: tuple[float, float, float] = (0.85, 0.14, 0.01),
                        lambdas: tuple[float, float, float] = (0.03, 0.05, 0.10)
                        ) -> np.ndarray:
    """Initial belief: per-class total mass + intra-class exponential decay."""
    if abs(sum(class_mass) - 1.0) > 1e-9:
        raise ValueError(f"class_mass must sum to 1, got {sum(class_mass)}")
    n_safe, n_risky, n_cat = class_sizes(N, risk_fractions)

    b0 = np.zeros(N, dtype=np.float64)
    starts = (0, n_safe, n_safe + n_risky)
    sizes = (n_safe, n_risky, n_cat)
    for start, size, total_mass, decay in zip(starts, sizes, class_mass, lambdas):
        if size == 0:
            continue
        weights = np.exp(-decay * np.arange(size))
        b0[start:start + size] = total_mass * weights / weights.sum()

    return b0
