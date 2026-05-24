"""
Deterministic support pickers for A1.

`uniform_pick`: evenly spaced indices.
`exponential_decay_pick`: bin the index range; per-bin retention decays with
bin index. Dense at the low end (safe side), sparse at the high end
(catastrophic). Calibrated so total kept ≈ round(target_fraction · N).

Both pickers return a sorted `list[int]` of kept indices over an ordered
state/observation space.
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
    K = int(round(target_fraction * N))
    if K <= 0:
        return []
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

    Bin [0, N) into `n_bins` equal-size bins (last bin absorbs the remainder).
    Per-bin allocation is proportional to exp(-decay · b) · bin_size_b. Total
    kept = round(target_fraction · N), distributed via floor + largest-remainder;
    bins saturate at their capacity and overflow water-fills the next-leftmost
    unsaturated bin. Within each kept bin, takes the first `keep_b` indices.
    """
    if target_fraction <= 0.0:
        return []
    if target_fraction >= 1.0:
        return list(range(N))
    K_total = int(round(target_fraction * N))
    if K_total <= 0:
        return []
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
