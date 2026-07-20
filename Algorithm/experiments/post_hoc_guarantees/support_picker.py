"""
Support pickers for the post-hoc-guarantees experiment — they choose WHICH support
the fixed policy is projected onto. Experiment tooling, not a core math primitive:
the planner grows its support by per-sim sampling, never by these pickers.

`uniform_pick` / `exponential_decay_pick` (dup'd from src/robust_pomdp/bounds/):
deterministic index masks over an ordered state/obs space, sized by a target
fraction; return a sorted list[int]. uniform = evenly spaced; exp_decay = dense at
the low (safe) end, sparse at the high (catastrophic) end.

`sample_pick`: the per-node sampler scheme — draw K indices from a belief-predictive
(with replacement, deduped), concentrating support on high-probability states/obs.
"""

from __future__ import annotations

import numpy as np


def uniform_pick(target_fraction: float, N: int) -> list[int]:
    """Evenly spaced K = round(target_fraction · N) indices in [0, N).

    `target_fraction <= 0` returns []; `>= 1` returns the full range.
    """
    if target_fraction <= 0.0:
        return []
    if target_fraction >= 1.0:
        return list(range(N))
    K = max(1, int(round(target_fraction * N)))
    if K >= N:
        return list(range(N))
    return np.linspace(0, N, K, endpoint=False).astype(int).tolist()


def exponential_decay_pick(target_fraction: float,
                           N: int,
                           *,
                           n_bins: int = 10,
                           decay: float = 0.5
                           ) -> list[int]:
    """Spread pick: dense at the low end, sparse at the high end.

    Bin [0, N) into `n_bins` equal-size bins (last absorbs the remainder). Per-bin
    allocation ∝ exp(-decay · b) · bin_size_b. Total kept = round(target_fraction ·
    N), distributed via floor + largest-remainder; bins saturate at capacity and
    overflow water-fills the next-leftmost unsaturated bin. Within a kept bin,
    takes the first `keep_b` indices.
    """
    if target_fraction <= 0.0:
        return []
    if target_fraction >= 1.0:
        return list(range(N))
    K_total = max(1, int(round(target_fraction * N)))
    if K_total >= N:
        return list(range(N))
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if n_bins > N:
        n_bins = N

    base_bin = N // n_bins
    bin_sizes = np.array([base_bin] * (n_bins - 1) + [N - base_bin * (n_bins - 1)],
                         dtype=int)
    bin_starts = np.concatenate(([0], np.cumsum(bin_sizes)[:-1])).astype(int)

    weights = np.exp(-decay * np.arange(n_bins))
    raw = K_total * weights * bin_sizes / (weights * bin_sizes).sum()

    keep_counts = np.minimum(np.floor(raw).astype(int), bin_sizes)
    leftover = K_total - int(keep_counts.sum())
    remainders = raw - np.floor(raw)
    for b in np.argsort(-remainders):
        if leftover == 0:
            break
        if keep_counts[b] < bin_sizes[b]:
            keep_counts[b] += 1
            leftover -= 1
    while leftover > 0:
        unsat = np.where(keep_counts < bin_sizes)[0]
        if len(unsat) == 0:
            break
        keep_counts[unsat[0]] += 1
        leftover -= 1

    picks: list[int] = []
    for b in range(n_bins):
        for k in range(int(keep_counts[b])):
            picks.append(int(bin_starts[b] + k))
    return picks


def sample_pick(probs: np.ndarray,
                K: int,
                rng: np.random.Generator,
                ) -> list[int]:
    """Belief-weighted support: draw K indices ~ `probs` WITH replacement, dedup,
    sort. |result| <= K, and peaked distributions yield fewer unique indices — the
    POMCP-style focusing on high-probability ("promising") states/observations.

    Unlike uniform/exp_decay (which pick from an ordered index range by a target
    fraction), this samples from a probability vector — so it is a PER-NODE picker:
    each node draws its own support from its own belief-predictive.

    `probs` need not be normalized (it is normalized internally). Returns [] when
    the total mass is <= 0 (e.g. an impossible observation branch).
    """
    probs = np.asarray(probs, dtype=np.float64)
    total = float(probs.sum())
    if total <= 0.0:
        return []
    draws = rng.choice(len(probs), size=K, replace=True, p=probs / total)
    return sorted(set(int(d) for d in draws))
