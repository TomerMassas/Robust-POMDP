"""
Plot experiment results from CSV files.

Pure CSV consumer. Reads CSVs from `experiments/tiger/results/<Ek>/csv/`
and writes PNG figures to `experiments/tiger/results/<Ek>/figures/`.

Usage:
    python viz/plot_results.py                          # all figures we know how to make
    python viz/plot_results.py --experiments E1         # just E1
    python viz/plot_results.py --experiments E1,E3      # E1 and E3

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
RESULTS_DIR = REPO_ROOT / "experiments" / "tiger" / "results"


# ---------------------------------------------------------------------------
# CSV discovery
# ---------------------------------------------------------------------------

def find_csvs(experiment_id: str) -> list[Path]:
    """Return all CSVs in `results/<experiment_id>/csv/`."""
    csv_dir = RESULTS_DIR / experiment_id / "csv"
    if not csv_dir.is_dir():
        return []
    return sorted(csv_dir.glob("*.csv"))


def read_latest(experiment_id: str,
                suffix_filter: str | None = None,
                raw: bool = False
                ) -> pd.DataFrame | None:
    """Pick the most recent summary or raw CSV for an experiment_id.

    Summary CSVs end in ``.csv``; raw CSVs end in ``_raw.csv``. Disambiguate
    so callers asking for the summary don't accidentally pick up a raw file.
    """
    candidates = find_csvs(experiment_id)
    if raw:
        candidates = [c for c in candidates if c.name.endswith("_raw.csv")]
    else:
        candidates = [c for c in candidates if not c.name.endswith("_raw.csv")]
    if suffix_filter is not None:
        candidates = [c for c in candidates if suffix_filter in c.name]
    if not candidates:
        return None
    # Filenames sort lexically; ISO date in the name puts the newest last.
    return pd.read_csv(candidates[-1])


# ---------------------------------------------------------------------------
# Plot recipes
# ---------------------------------------------------------------------------

def plot_e1_violin():
    """E1 — Distribution of per-episode returns as a violin per planner."""
    raw = read_latest("E1", raw=True)
    if raw is None:
        print("E1 violin: no raw CSV found, skipping.")
        return
    planners = ["BasicPOMCP", "robust_pomcp"]
    data = [raw.loc[raw["planner_label"] == p, "episode_return"].to_numpy()
            for p in planners]

    fig, ax = plt.subplots(figsize=(6, 5))
    parts = ax.violinplot(data,
                          positions=range(len(planners)),
                          showmeans=False,
                          showmedians=False,
                          showextrema=False
                          )
    for body, color in zip(parts["bodies"], ["#4c72b0", "#dd8452"]):
        body.set_facecolor(color)
        body.set_alpha(0.6)
        body.set_edgecolor("black")
    bp = ax.boxplot(data,
                    positions=range(len(planners)),
                    widths=0.15,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False
                    )
    for box in bp["boxes"]:
        box.set_facecolor("white")
        box.set_edgecolor("black")
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(2)

    ax.set_xticks(range(len(planners)))
    ax.set_xticklabels(planners)
    ax.set_ylabel("Total episode return")
    n = int(raw.groupby("planner_label").size().min())
    ax.set_title(f"E1 — Distribution of episode returns (n={n}, nominal Tiger)")
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()

    out_dir = RESULTS_DIR / "E1" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "violin_returns.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out.relative_to(REPO_ROOT)}")


def plot_e1_actions():
    """E1 — Action breakdown: stacked bar of correct / wrong / listen per planner."""
    df = read_latest("E1")
    if df is None:
        print("E1 actions: no summary CSV found, skipping.")
        return
    df = df.set_index("planner_label").loc[["BasicPOMCP", "robust_pomcp"]]
    correct = df["n_correct_open"].to_numpy()
    wrong = df["n_wrong_open"].to_numpy()
    listen = df["n_listen_actions"].to_numpy()
    total_actions = int((df["n_trials"] * df["horizon"]).iloc[0])

    fig, ax = plt.subplots(figsize=(6, 5))
    x = range(len(df))
    width = 0.55
    b1 = ax.bar(x, correct, width, label="correct open", color="#55a868")
    b2 = ax.bar(x, wrong, width, bottom=correct, label="wrong open", color="#c44e52")
    b3 = ax.bar(x, listen, width, bottom=correct + wrong, label="listen", color="#8172b3")

    for bars, values, base in [
        (b1, correct, [0] * len(correct)),
        (b2, wrong, correct),
        (b3, listen, correct + wrong),
    ]:
        for bar, v, b in zip(bars, values, base):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        b + v / 2,
                        str(int(v)),
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=10,
                        fontweight="bold"
                        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(df.index.tolist())
    ax.set_ylabel("Total action count")
    ax.set_title(f"E1 — Action breakdown ({total_actions} total actions per planner)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()

    out_dir = RESULTS_DIR / "E1" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "action_breakdown.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out.relative_to(REPO_ROOT)}")


def plot_e1():
    plot_e1_violin()
    plot_e1_actions()


def plot_e2_line():
    """E2 — Mean return ± SEM vs ρ_Z, on the nominal world."""
    df = read_latest("E2")
    if df is None:
        print("E2 line: no summary CSV found, skipping.")
        return
    df = df.sort_values("rho_Z")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(df["rho_Z"],
                df["mean_return"],
                yerr=df["sem_return"],
                marker="o",
                capsize=4,
                color="#4c72b0",
                linewidth=1.5
                )
    ax.set_xlabel("ρ_Z (planner's observation uncertainty radius)")
    ax.set_ylabel("Mean episodic return")
    n = int(df["n_trials"].iloc[0])
    ax.set_title(f"E2 — Pessimism cost on nominal world (n={n} per ρ_Z)")
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()

    out_dir = RESULTS_DIR / "E2" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "line_return_vs_rho.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out.relative_to(REPO_ROOT)}")


def plot_e2_violins():
    """E2 — One violin per ρ_Z showing the per-episode return distribution."""
    raw = read_latest("E2", raw=True)
    if raw is None:
        print("E2 violins: no raw CSV found, skipping.")
        return
    rho_values = sorted(raw["rho_Z"].unique())
    data = [raw.loc[raw["rho_Z"] == r, "episode_return"].to_numpy() for r in rho_values]

    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot(data,
                          positions=range(len(rho_values)),
                          showmeans=False,
                          showmedians=False,
                          showextrema=False
                          )
    cmap = plt.get_cmap("viridis")
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(cmap(i / max(1, len(rho_values) - 1)))
        body.set_alpha(0.6)
        body.set_edgecolor("black")
    bp = ax.boxplot(data,
                    positions=range(len(rho_values)),
                    widths=0.15,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False
                    )
    for box in bp["boxes"]:
        box.set_facecolor("white")
        box.set_edgecolor("black")
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(2)

    ax.set_xticks(range(len(rho_values)))
    ax.set_xticklabels([f"{r:g}" for r in rho_values])
    ax.set_xlabel("ρ_Z")
    ax.set_ylabel("Total episode return")
    n = int(raw.groupby("rho_Z").size().min())
    ax.set_title(f"E2 — Return distribution per ρ_Z (n={n} per violin, nominal world)")
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()

    out_dir = RESULTS_DIR / "E2" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "violin_returns.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out.relative_to(REPO_ROOT)}")


def plot_e2_actions():
    """E2 — Stacked action breakdown per ρ_Z (correct / wrong / listen)."""
    df = read_latest("E2")
    if df is None:
        print("E2 actions: no summary CSV found, skipping.")
        return
    df = df.sort_values("rho_Z").reset_index(drop=True)
    rho_values = df["rho_Z"].tolist()
    correct = df["n_correct_open"].to_numpy()
    wrong = df["n_wrong_open"].to_numpy()
    listen = df["n_listen_actions"].to_numpy()
    total_actions = int((df["n_trials"] * df["horizon"]).iloc[0])

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(df))
    width = 0.6
    b1 = ax.bar(x, correct, width, label="correct open", color="#55a868")
    b2 = ax.bar(x, wrong, width, bottom=correct, label="wrong open", color="#c44e52")
    b3 = ax.bar(x, listen, width, bottom=correct + wrong, label="listen", color="#8172b3")

    for bars, values, base in [
        (b1, correct, [0] * len(correct)),
        (b2, wrong, correct),
        (b3, listen, correct + wrong),
    ]:
        for bar, v, b in zip(bars, values, base):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        b + v / 2,
                        str(int(v)),
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=10,
                        fontweight="bold"
                        )

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{r:g}" for r in rho_values])
    ax.set_xlabel("ρ_Z")
    ax.set_ylabel("Total action count")
    ax.set_title(f"E2 — Action breakdown per ρ_Z ({total_actions} actions per bar)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()

    out_dir = RESULTS_DIR / "E2" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "action_breakdown.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out.relative_to(REPO_ROOT)}")


def plot_e2():
    plot_e2_line()
    plot_e2_violins()
    plot_e2_actions()


def plot_e3_line():
    """E3 — Fan of curves: mean return vs eta, per rho_Z + vanilla baseline.

    Reads `results/E3/csv/E3_*.csv` produced by the 2D rho_Z x eta sweep.
    Each robust_pomcp curve only spans its own eta range (eta in [0, rho_Z/2]),
    so the curves fan out — wider rho_Z means longer eta range. The vanilla
    BasicPOMCP curve spans the union of all etas as a single dashed line.
    """
    df = read_latest("E3")
    if df is None:
        print("E3 line: no summary CSV found, skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    robust_df = df[df["planner_label"] == "robust_pomcp"]
    rho_z_values = sorted(robust_df["rho_Z"].unique())
    cmap = plt.get_cmap("viridis")
    n_rho = max(1, len(rho_z_values))
    for i, rho_z in enumerate(rho_z_values):
        sub = robust_df[robust_df["rho_Z"] == rho_z].sort_values("eta_world_obs")
        if sub.empty:
            continue
        color = cmap(i / max(1, n_rho - 1)) if n_rho > 1 else cmap(0.5)
        ax.errorbar(sub["eta_world_obs"],
                    sub["mean_return"],
                    yerr=sub["sem_return"],
                    marker="o",
                    capsize=4,
                    color=color,
                    label=f"robust ρ_Z={rho_z:g}",
                    linewidth=1.5,
                    linestyle="-" if len(sub) > 1 else "None"
                    )

    vanilla_df = df[df["planner_label"] == "BasicPOMCP"].sort_values("eta_world_obs")
    if not vanilla_df.empty:
        ax.errorbar(vanilla_df["eta_world_obs"],
                    vanilla_df["mean_return"],
                    yerr=vanilla_df["sem_return"],
                    marker="s",
                    capsize=4,
                    color="black",
                    linestyle="--",
                    label="BasicPOMCP (vanilla)",
                    linewidth=1.5
                    )

    ax.set_xlabel("η (world perturbation magnitude)")
    ax.set_ylabel("Mean episodic return")
    n = int(df["n_trials"].iloc[0])
    ax.set_title(f"E3 — Robust (per ρ_Z) vs vanilla on perturbed Tiger (n={n} per config)")
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    out_dir = RESULTS_DIR / "E3" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "line_return_vs_eta_per_rho.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out.relative_to(REPO_ROOT)}")


def plot_e3():
    plot_e3_line()


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
        ax.errorbar(sub["eta_world_obs"],
                    sub["mean_return"],
                    yerr=sub["sem_return"],
                    marker="o",
                    capsize=4,
                    color=color,
                    label=label
                    )
    ax.axvline(0.2, color="grey", linestyle="--", alpha=0.5, label="ρ_Z calibration (0.20)")
    ax.set_xlabel("World perturbation η")
    ax.set_ylabel("Mean episodic return")
    ax.set_title("E4 — Vanilla vs robust under increasing model misspecification")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend()
    fig.tight_layout()
    out_dir = RESULTS_DIR / "E4" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "E4_eta_sweep_paired.png"
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
    for ax, df, title in zip(axes,
                              [nominal, perturbed],
                              ["Nominal world", "Perturbed world (η=0.20)"]
                              ):
        if df is None:
            ax.set_visible(False)
            continue
        df = df.sort_values("sims_per_backup")
        x = df["sims_per_backup"]
        ax.errorbar(x,
                    df["mean_return"],
                    yerr=df["sem_return"],
                    marker="o",
                    capsize=4,
                    color="#4c72b0",
                    label="return"
                    )
        ax.set_xlabel("sims_per_backup")
        ax.set_ylabel("Mean return", color="#4c72b0")
        ax.set_xscale("log")
        ax.set_title(f"E5 — {title}")
        ax2 = ax.twinx()
        ax2.plot(x, df["mean_runtime_s"], marker="s", color="#dd8452", label="runtime [s]")
        ax2.set_ylabel("Mean planner runtime [s]", color="#dd8452")
    fig.tight_layout()
    out_dir = RESULTS_DIR / "E5" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "E5_simsperbackup_sweep.png"
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
    for i, (df, label, color) in enumerate([(nominal, "Nominal world", "#4c72b0"),
                                             (perturbed, "Perturbed (η=0.20)", "#dd8452")]
                                            ):
        if df is None:
            continue
        df = df.set_index("ucb_mode")
        means = [df.loc[m, "mean_return"] if m in df.index else 0.0 for m in modes]
        sems = [df.loc[m, "sem_return"] if m in df.index else 0.0 for m in modes]
        ax.bar([xi + (i - 0.5) * width for xi in x],
               means,
               width,
               yerr=sems,
               capsize=4,
               color=color,
               label=label
               )
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("Mean episodic return")
    ax.set_title("E6 — UCB mode comparison")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend()
    fig.tight_layout()
    out_dir = RESULTS_DIR / "E6" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "E6_ucbmode_compare.png"
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
        ax.errorbar(df["horizon"],
                    df["mean_return"],
                    yerr=df["sem_return"],
                    marker="o",
                    capsize=4,
                    color=color,
                    label=label
                    )
    ax.set_xlabel("horizon")
    ax.set_ylabel("Mean episodic return")
    ax.set_title("E7 — Return vs horizon")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend()
    fig.tight_layout()
    out_dir = RESULTS_DIR / "E7" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "E7_horizon_sweep.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

PLOTTERS = {
    "E1": plot_e1,
    "E2": plot_e2,
    "E3": plot_e3,
    "E4": plot_e4,
    "E5": plot_e5,
    "E6": plot_e6,
    "E7": plot_e7,
}


def main():
    global RESULTS_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=str, default=None,
                        help="Comma-separated experiment ids (e.g. E1,E3). Default: all."
                        )
    parser.add_argument("--results-dir", type=Path, default=None,
                        help=f"Override the directory CSVs are read from "
                             f"(default: {RESULTS_DIR})."
                        )
    args = parser.parse_args()

    if args.results_dir is not None:
        RESULTS_DIR = args.results_dir.resolve()

    if args.experiments:
        wanted = {e.strip() for e in args.experiments.split(",")}
    else:
        wanted = set(PLOTTERS.keys())

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
