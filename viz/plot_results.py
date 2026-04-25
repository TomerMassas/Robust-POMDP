"""
Plot experiment results from CSV files.

Pure CSV consumer — no Julia dependency. Reads CSVs from
`experiments/results/` and writes PNG figures to
`experiments/results/figures/`.

Usage:
    cd viz
    python plot_results.py                       # all figures we know how to make
    python plot_results.py --experiments E2,E3   # only the listed experiments

Adding a new plot:
- Add a function `plot_eK(...)` below
- Register it in PLOTTERS at the bottom

The CSV schema (per the experiment plan):
    experiment_id, date, description, planner_label, rho_T, rho_Z, ucb_mode,
    budget, sims_per_backup, horizon, eta_world_obs, eta_world_trans,
    n_trials, seed_base,
    mean_return, std_return, sem_return, mean_runtime_s,
    n_correct_open, n_wrong_open, n_listen_actions
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


# ---------------------------------------------------------------------------
# CSV discovery
# ---------------------------------------------------------------------------

def find_csvs(experiment_id: str) -> list[Path]:
    """Return all CSVs whose filename starts with `{experiment_id}_`."""
    return sorted(RESULTS_DIR.glob(f"{experiment_id}_*.csv"))


def read_latest(experiment_id: str, suffix_filter: str | None = None) -> pd.DataFrame | None:
    """Pick the most recent CSV for an experiment_id (optionally filtered by description suffix)."""
    candidates = find_csvs(experiment_id)
    if suffix_filter is not None:
        candidates = [c for c in candidates if suffix_filter in c.name]
    if not candidates:
        return None
    # Filenames sort lexically; ISO date in the name puts the newest last.
    return pd.read_csv(candidates[-1])


# ---------------------------------------------------------------------------
# Plot recipes
# ---------------------------------------------------------------------------

def plot_e1():
    """E1 — Baseline equivalence: BasicPOMCP vs robust(ρ=0). Bar chart with SEM."""
    df = read_latest("E1")
    if df is None:
        print("E1: no CSV found, skipping.")
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = df["planner_label"].tolist()
    means = df["mean_return"].tolist()
    sems = df["sem_return"].tolist()
    ax.bar(range(len(labels)), means, yerr=sems, capsize=6,
           color=["#4c72b0", "#dd8452"])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean episodic return")
    ax.set_title("E1 — Baseline equivalence (nominal Tiger)")
    ax.axhline(0, color="black", lw=0.5)
    fig.tight_layout()
    out = FIGURES_DIR / "E1_baseline_equivalence.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out.relative_to(REPO_ROOT)}")


def plot_e2_e3():
    """E2/E3 — ρ_Z sweep, nominal vs perturbed worlds, side by side."""
    e2 = read_latest("E2")
    e3 = read_latest("E3")
    if e2 is None and e3 is None:
        print("E2/E3: no CSVs found, skipping.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, df, title in zip(
        axes,
        [e2, e3],
        ["E2 — Nominal world", "E3 — Perturbed world (η=0.20)"],
    ):
        if df is None:
            ax.set_visible(False)
            continue
        df = df.sort_values("rho_Z")
        ax.errorbar(df["rho_Z"], df["mean_return"], yerr=df["sem_return"],
                    marker="o", capsize=4, color="#4c72b0")
        ax.set_xlabel("ρ_Z")
        ax.set_title(title)
        ax.axhline(0, color="black", lw=0.5)
    axes[0].set_ylabel("Mean episodic return")
    fig.tight_layout()
    out = FIGURES_DIR / "E2_E3_rho_z_sweep.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out.relative_to(REPO_ROOT)}")


def plot_e4():
    """E4 — η sweep, vanilla(ρ=0) vs robust(ρ_Z=0.2). Two lines."""
    df = read_latest("E4")
    if df is None:
        print("E4: no CSV found, skipping.")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for rho_Z, color in [(0.0, "#dd8452"), (0.2, "#4c72b0")]:
        sub = df[df["rho_Z"] == rho_Z].sort_values("eta_world_obs")
        if sub.empty:
            continue
        label = "vanilla (ρ_Z=0)" if rho_Z == 0.0 else f"robust (ρ_Z={rho_Z})"
        ax.errorbar(sub["eta_world_obs"], sub["mean_return"], yerr=sub["sem_return"],
                    marker="o", capsize=4, color=color, label=label)
    ax.axvline(0.2, color="grey", linestyle="--", alpha=0.5,
               label="ρ_Z calibration (0.20)")
    ax.set_xlabel("World perturbation η")
    ax.set_ylabel("Mean episodic return")
    ax.set_title("E4 — Vanilla vs robust under increasing model misspecification")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend()
    fig.tight_layout()
    out = FIGURES_DIR / "E4_eta_sweep_paired.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out.relative_to(REPO_ROOT)}")


def plot_e5():
    """E5 — sims_per_backup sweep. Dual-axis (return + runtime). Both worlds."""
    nominal = read_latest("E5", suffix_filter="nominal")
    perturbed = read_latest("E5", suffix_filter="perturbed")
    if nominal is None and perturbed is None:
        print("E5: no CSVs found, skipping.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, df, title in zip(
        axes,
        [nominal, perturbed],
        ["Nominal world", "Perturbed world (η=0.20)"],
    ):
        if df is None:
            ax.set_visible(False)
            continue
        df = df.sort_values("sims_per_backup")
        x = df["sims_per_backup"]
        ax.errorbar(x, df["mean_return"], yerr=df["sem_return"],
                    marker="o", capsize=4, color="#4c72b0", label="return")
        ax.set_xlabel("sims_per_backup")
        ax.set_ylabel("Mean return", color="#4c72b0")
        ax.set_xscale("log")
        ax.set_title(f"E5 — {title}")
        ax2 = ax.twinx()
        ax2.plot(x, df["mean_runtime_s"], marker="s", color="#dd8452",
                 label="runtime [s]")
        ax2.set_ylabel("Mean planner runtime [s]", color="#dd8452")
    fig.tight_layout()
    out = FIGURES_DIR / "E5_simsperbackup_sweep.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out.relative_to(REPO_ROOT)}")


def plot_e6():
    """E6 — UCB mode comparison (paired bar chart, both worlds)."""
    nominal = read_latest("E6", suffix_filter="nominal")
    perturbed = read_latest("E6", suffix_filter="perturbed")
    if nominal is None and perturbed is None:
        print("E6: no CSVs found, skipping.")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    modes = [":nominal", ":robust"]
    x = list(range(len(modes)))
    for i, (df, label, color) in enumerate([
        (nominal, "Nominal world", "#4c72b0"),
        (perturbed, "Perturbed (η=0.20)", "#dd8452"),
    ]):
        if df is None:
            continue
        df = df.set_index("ucb_mode")
        means = [df.loc[m, "mean_return"] if m in df.index else 0.0 for m in modes]
        sems = [df.loc[m, "sem_return"] if m in df.index else 0.0 for m in modes]
        ax.bar([xi + (i - 0.5) * width for xi in x], means, width,
               yerr=sems, capsize=4, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("Mean episodic return")
    ax.set_title("E6 — UCB mode comparison")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend()
    fig.tight_layout()
    out = FIGURES_DIR / "E6_ucbmode_compare.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out.relative_to(REPO_ROOT)}")


def plot_e7():
    """E7 — Horizon sweep, nominal vs perturbed, two lines on one axis."""
    nominal = read_latest("E7", suffix_filter="nominal")
    perturbed = read_latest("E7", suffix_filter="perturbed")
    if nominal is None and perturbed is None:
        print("E7: no CSVs found, skipping.")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for df, label, color in [
        (nominal, "Nominal world", "#4c72b0"),
        (perturbed, "Perturbed (η=0.20)", "#dd8452"),
    ]:
        if df is None:
            continue
        df = df.sort_values("horizon")
        ax.errorbar(df["horizon"], df["mean_return"], yerr=df["sem_return"],
                    marker="o", capsize=4, color=color, label=label)
    ax.set_xlabel("horizon")
    ax.set_ylabel("Mean episodic return")
    ax.set_title("E7 — Return vs horizon")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend()
    fig.tight_layout()
    out = FIGURES_DIR / "E7_horizon_sweep.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

PLOTTERS = {
    "E1": plot_e1,
    "E2": plot_e2_e3,
    "E3": plot_e2_e3,  # same plot, runs once below via dedup
    "E4": plot_e4,
    "E5": plot_e5,
    "E6": plot_e6,
    "E7": plot_e7,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=str, default=None,
                        help="Comma-separated experiment ids (e.g. E1,E3). Default: all.")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if args.experiments:
        wanted = {e.strip() for e in args.experiments.split(",")}
    else:
        wanted = set(PLOTTERS.keys())

    # Dedup recipes (E2 and E3 share one plot function)
    fns_already_run: set = set()
    for eid in sorted(wanted):
        fn = PLOTTERS.get(eid)
        if fn is None:
            print(f"No plotter registered for {eid}, skipping.")
            continue
        if fn in fns_already_run:
            continue
        print(f"== {eid} ==")
        fn()
        fns_already_run.add(fn)


if __name__ == "__main__":
    main()
