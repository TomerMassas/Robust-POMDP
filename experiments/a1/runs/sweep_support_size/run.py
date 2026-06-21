"""
A1 support-size sweep (greedy policy pi_full, leakage certificate).

Solves the full-robust-greedy policy ONCE (support-independent), then sweeps the
joint support fraction m (= m_S = m_Z). For each m: pick (S_in, Z_in) via the
configured scheme, build the projected tree FOLLOWING pi_full, compute V_proj
and the leakage-only certificate.

Fan plot (option a): V_full (constant), V_proj curve, the certificate envelope
V_proj +/- cert CLIPPED to [-V_max, +V_max], and the +/- V_max reference lines
(V_max = R_max * (H+1) = the trivial reward-range bound). The certificate is
INFORMATIVE where the envelope detaches from the +/- V_max band.

Versioned outputs: data_V{n}.csv, fan_plot_V{n}.png. plot_only(n) re-plots a
saved CSV without re-running.

Run from anywhere:
    python experiments/a1/runs/sweep_support_size/run.py
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
from main import build_reference, evaluate_projection, _resolve_support


def run_sweep(m_grid, base_cfg):
    """Greedy solve once, then evaluate each support level."""
    ref = build_reference(base_cfg)        # solves pi_full / V_full ONCE
    V_full = ref["V_full"]
    R_max = ref["R_max"]
    V_max = R_max * (base_cfg.H + 1)
    results = []
    for m in m_grid:
        S_in = _resolve_support(None, float(m), base_cfg.N,
                                base_cfg.scheme, base_cfg.decay, base_cfg.n_bins)
        Z_in = _resolve_support(None, float(m), base_cfg.M,
                                base_cfg.scheme, base_cfg.decay, base_cfg.n_bins)
        res = evaluate_projection(ref, S_in, Z_in)
        results.append({
            "m": float(m),
            "n_S_in": len(S_in),
            "n_Z_in": len(Z_in),
            "V_full": V_full,
            "V_proj": res["V_proj"],
            "certificate": res["certificate"],
            "true_error": res["true_error"],
            "valid": int(res["valid"]),
        })
    return results, V_full, V_max


def _next_version(out_dir: Path) -> int:
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
    with open(csv_path, newline="") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def plot_fan(results, V_full, V_max, out_path, title):
    m = [r["m"] for r in results]
    V_proj = [r["V_proj"] for r in results]
    cert = [r["certificate"] for r in results]
    # Certificate interval, clipped to the trivial [-V_max, +V_max] knowledge.
    upper = [min(v + c, V_max) for v, c in zip(V_proj, cert)]
    lower = [max(v - c, -V_max) for v, c in zip(V_proj, cert)]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(V_max, color="tab:red", ls=":", lw=1.2, label=f"+/- V_max = {V_max:.1f}")
    ax.axhline(-V_max, color="tab:red", ls=":", lw=1.2)
    ax.fill_between(m, lower, upper, alpha=0.18, color="tab:blue",
                    label="V_proj +/- certificate")
    ax.plot(m, V_proj, marker="o", color="tab:blue", label="V_proj")
    ax.axhline(V_full, color="black", ls="--", lw=1.5, label=f"V_full = {V_full:.3f}")
    ax.set_xlabel("support fraction  m  (m = m_S = m_Z) at all nodes")
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    base_cfg = replace(A1Config(), write_tree_html=False, verbose=False)
    m_grid = np.linspace(0.1, 1.0, 10)

    out_dir = _SCRIPT.parents[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    version = _next_version(out_dir)
    csv_path = out_dir / f"data_V{version}.csv"
    plot_path = out_dir / f"fan_plot_V{version}.png"

    results, V_full, V_max = run_sweep(m_grid, base_cfg)
    write_csv(results, csv_path)
    plot_fan(results, V_full, V_max, plot_path,
             title=(f"support sweep| Ns={base_cfg.N}, Nz={base_cfg.M}, "
                    f"H={base_cfg.H}, rho_T={base_cfg.rho_T}, rho_Z={base_cfg.rho_Z}"))

    print(f"V_full = {V_full:.4f}   V_max = {V_max:.4f}\n")
    for r in results:
        flag = "" if r["valid"] else "  !! INVALID"
        print(f"  m={r['m']:.2f}  |S_in|={int(r['n_S_in'])} |Z_in|={int(r['n_Z_in'])}  "
              f"V_proj={r['V_proj']:.3f}  cert={r['certificate']:.3f}  "
              f"true_err={r['true_error']:.3f}{flag}")
    print(f"\nWrote {csv_path.name} and {plot_path.name}")


def plot_only(version: int):
    """Re-plot from a saved data_V{version}.csv without re-running the sweep."""
    out_dir = _SCRIPT.parents[0]
    results = read_csv(out_dir / f"data_V{version}.csv")
    base_cfg = A1Config()
    V_full = results[0]["V_full"]
    V_max = results[0]["R_max"] * (base_cfg.H + 1) if "R_max" in results[0] \
        else max(r["V_proj"] + r["certificate"] for r in results)
    plot_fan(results, V_full, V_max, out_dir / f"fan_plot_V{version}.png",
             title=f"A1 support sweep (greedy) | replot V{version}")
    print(f"Re-plotted fan_plot_V{version}.png")


if __name__ == "__main__":
    main()
    # plot_only(1)   # re-plot a saved CSV (comment out main() above)