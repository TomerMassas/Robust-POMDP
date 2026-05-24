"""Phase 2 sanity — verify support picker shapes, counts, and decay-direction.

Run from anywhere:
    python experiments/a1/smoke_test/phase2_smoke.py
"""

from __future__ import annotations

from robust_pomdp.bounds.support_picker import uniform_pick, exponential_decay_pick


def main() -> None:
    N = 100
    n_bins = 10
    decay = 0.5
    bin_size = N // n_bins  # 10

    # Edge cases.
    assert uniform_pick(0.0, N) == []
    assert uniform_pick(1.0, N) == list(range(N))
    assert exponential_decay_pick(0.0, N, n_bins=n_bins, decay=decay) == []
    assert exponential_decay_pick(1.0, N, n_bins=n_bins, decay=decay) == list(range(N))

    # Basic invariants across a fraction sweep.
    for frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for picker_name, picks in [
            ("uniform", uniform_pick(frac, N)),
            ("exp_decay", exponential_decay_pick(frac, N, n_bins=n_bins, decay=decay)),
        ]:
            expected_K = round(frac * N)
            assert len(picks) == expected_K, \
                f"{picker_name}({frac}) K={len(picks)} != {expected_K}"
            assert picks == sorted(picks), f"{picker_name}({frac}) not sorted"
            assert len(set(picks)) == len(picks), f"{picker_name}({frac}) has duplicates"
            assert min(picks) >= 0 and max(picks) < N, \
                f"{picker_name}({frac}) out of range"

    # Decay direction: at moderate fraction, bin 0 keeps more than the last bin.
    picks_30 = exponential_decay_pick(0.30, N, n_bins=n_bins, decay=decay)
    per_bin = [
        sum(1 for p in picks_30 if b * bin_size <= p < (b + 1) * bin_size)
        for b in range(n_bins)
    ]
    assert per_bin[0] > per_bin[-1], \
        f"bin 0 keep {per_bin[0]} should exceed bin {n_bins - 1} keep {per_bin[-1]}"
    assert per_bin[0] > per_bin[n_bins // 2], \
        f"bin 0 keep {per_bin[0]} should exceed bin {n_bins // 2} keep {per_bin[n_bins // 2]}"

    # High-fraction saturation: front bins must saturate but total still = K.
    picks_90 = exponential_decay_pick(0.90, N, n_bins=n_bins, decay=decay)
    per_bin_90 = [
        sum(1 for p in picks_90 if b * bin_size <= p < (b + 1) * bin_size)
        for b in range(n_bins)
    ]
    assert sum(per_bin_90) == 90
    assert per_bin_90[0] == bin_size, f"bin 0 should saturate at 0.9, got {per_bin_90[0]}"

    print(f"phase 2 OK | N={N} n_bins={n_bins} decay={decay}")
    print(f"  uniform(0.3, {N}): {uniform_pick(0.30, N)}")
    print(f"  exp_decay(0.3, {N}): {picks_30}")
    print(f"    per-bin keep counts: {per_bin}")
    print(f"  exp_decay(0.9, {N}) per-bin: {per_bin_90}")


if __name__ == "__main__":
    main()
