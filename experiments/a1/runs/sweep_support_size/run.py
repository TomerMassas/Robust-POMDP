"""
A1 sweep over m_S (projected state fraction). Holds everything else at the
current A1Config defaults. Produces:
  - data.csv     : per-m_S V_full, V_proj, certificate, true_error.
  - fan_plot.png : V_full horizontal + V_proj curve + V_proj +/- certificate envelope.

Run from anywhere:
    python experiments/a1/runs/sweep_m_S/run.py
"""

from __future__ import annotations

import csv
import re
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(_SCRIPT.parents[2]))  # experiments/a1/
sys.path.insert(0, str(_SCRIPT.parents[4]))  # repo root

from config import A1Config
from main import main as run_a1


def run_sweep(support_size, base_cfg):
    results = []
    for supp in support_size:
        cfg = replace(base_cfg, m_S=float(supp), m_Z=float(supp),write_tree_html=False, verbose=False)
        r = run_a1(cfg)
        results.append({
            "Support": float(supp),
            "V_full": r["V_full"],
            "V_proj": r["V_proj"],
            "certificate": r["certificate"],
            "true_error": r["true_error"],
            "delta_b_contribution": r["delta_b_contribution"],
            "delta_K_contribution": r["delta_K_contribution"],
            "leakage_contribution": r["leakage_contribution"],
        })
    return results


def _next_version(out_dir: Path) -> int:
    """Next shared version index: max existing V across data_V*.csv and
    fan_plot_V*.png, + 1. Starts at 1."""
    versions = [0]
    for f in out_dir.glob("data_V*.csv"):
        m = re.search(r"_V(\d+)\.csv$", f.name)
        if m:
            versions.append(int(m.group(1)))
    for f in out_dir.glob("fan_plot_V*.png"):
        m = re.search(r"_V(\d+)\.png$", f.name)
        if m:
            versions.append(int(m.group(1)))
    return max(versions) + 1


def write_csv(results, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def read_csv(csv_path):
    """Read a saved data_V*.csv back into the results list-of-dicts (all numeric)."""
    with open(csv_path, newline="") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def plot_fan(results, out_path, title):
    m_S = [r["Support"] for r in results]
    V_full = results[0]["V_full"]   # constant across m_S
    V_proj = [r["V_proj"] for r in results]
    cert = [r["certificate"] for r in results]
    upper = [v + c for v, c in zip(V_proj, cert)]
    lower = [v - c for v, c in zip(V_proj, cert)]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    # ax.fill_between(m_S, lower, upper, alpha=0.2, color="tab:blue",label="V_proj +/- certificate")
    ax.plot(m_S, V_proj, marker="o", color="tab:blue", label="V_proj")
    ax.axhline(V_full, color="black", linestyle="--",label=f"V_full = {V_full:.3f}")
    ax.set_xlabel("m_S (projected state fraction)")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    base_cfg = A1Config()
    support_size = np.linspace(0.1, 1.0, 5)

    out_dir = _SCRIPT.parents[0]   # the sweep_m_S/ folder itself
    out_dir.mkdir(parents=True, exist_ok=True)

    version = _next_version(out_dir)
    csv_path = out_dir / f"data_V{version}.csv"
    plot_path = out_dir / f"fan_plot_V{version}.png"

    results = run_sweep(support_size, base_cfg)
    write_csv(results, csv_path)
    plot_fan(
        results, plot_path,
        title=(f"A1 m_S sweep | N={base_cfg.N}, M={base_cfg.M}, H={base_cfg.H}, "
               f"rho_T={base_cfg.rho_T}, rho_Z={base_cfg.rho_Z}, "
               f"scheme={base_cfg.scheme} decay={base_cfg.decay}, "
               f"m_Z={base_cfg.m_Z}"),
    )

    print(f"Wrote {csv_path} and {plot_path}\n")
    for r in results:
        print(f"  m_S={r['Support']:.2f}  V_full={r['V_full']:.3f}  "
              f"V_proj={r['V_proj']:.3f}  cert={r['certificate']:.3f}  "
              f"true_err={r['true_error']:.3f}")


def plot_only(version: int):
    """Re-plot from an already-saved data_V{version}.csv WITHOUT re-running the
    sweep. Writes fan_plot_V{version}.png (overwrites). Title uses the current
    A1Config defaults -- adjust if the config changed since the run."""
    out_dir = _SCRIPT.parents[0]
    csv_path = out_dir / f"data_V{version}.csv"
    plot_path = out_dir / f"fan_plot_V{version}.png"
    base_cfg = A1Config()
    results = read_csv(csv_path)
    plot_fan(
        results, plot_path,
        title=(f"A1 m_S sweep | N={base_cfg.N}, M={base_cfg.M}, H={base_cfg.H}, "
               f"rho_T={base_cfg.rho_T}, rho_Z={base_cfg.rho_Z}, "
               f"scheme={base_cfg.scheme} decay={base_cfg.decay}, "
               f"m_Z={base_cfg.m_Z}"),
    )
    print(f"Re-plotted {plot_path} from {csv_path}")


if __name__ == "__main__":
    main()
    # plot_only(3)   # re-plot from data_V3.csv only (comment out main() above)
