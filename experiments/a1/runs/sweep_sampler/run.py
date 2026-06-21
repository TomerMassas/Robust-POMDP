"""
A1 per-node sampler sweep (MCTS-style projection, leakage certificate).

Solves the full-robust-greedy policy pi_full ONCE, then for each (K, seed) builds
a per-node sampled-projection tree (S_in(h) / Z_in(ha) drawn from the projected
belief, with replacement, deduped), and computes V_proj + the leakage-only
certificate. Because support varies per node and is random, each run is a point
(avg_support_fraction, V_proj, cert); the x-axis is the AVERAGE support fraction
over the tree.

Plot: scatter of V_proj vs avg support fraction (colored by K), with the
certificate envelope V_proj +/- cert (clipped to [-V_max, +V_max]) as error bars,
the V_full reference, and the +/- V_max lines. The certificate shrinks as K (hence
avg support) grows but FLOORS above 0 -- the sparse synthetic POMDP always lets
the robust adversary leak to unsampled states.

Versioned outputs: data_V{n}.csv, sampler_plot_V{n}.png. plot_only(n) re-plots.

Run from anywhere:
    python experiments/a1/runs/sweep_sampler/run.py
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

from robust_pomdp.bounds.certificate import compute_certificate
from experiments.a1.config import A1Config
from experiments.a1.main import build_reference
from experiments.a1.synthetic_pomdp import ABORT
from experiments.a1.tree_evaluator import evaluate_robust_value
from experiments.a1.sampled_projection import build_sampled_projection_tree


def _support_fractions(root, node_data, N, M):
    """(mean |S_in(h)|/N over history nodes, mean |Z_in(ha)|/M over action nodes)."""
    s_fracs, z_fracs = [], []

    def _walk(node):
        data = node_data[node.id]
        s_fracs.append(len(node.S_in) / N)
        if data.is_terminal:
            return
        an = node.children[data.policy_action]
        z_fracs.append(len(an.Z_in) / M)
        for child in an.children.values():
            _walk(child)

    _walk(root)
    return float(np.mean(s_fracs)), float(np.mean(z_fracs))


def run_sweep(K_grid, n_seeds, base_cfg):
    """Greedy solve once; then a sampled tree per (K, seed)."""
    ref = build_reference(base_cfg)
    model, b0, unc = ref["model"], ref["b0"], ref["uncertainty"]
    V_full, pi_full, H = ref["V_full"], ref["pi_full"], ref["H"]
    V_max = ref["R_max"] * (H + 1)
    results = []
    for K in K_grid:
        for seed in range(n_seeds):
            root, nd = build_sampled_projection_tree(
                model, pi_full, b0, H, K=K, rng=np.random.default_rng(base_cfg.seed + seed),
                abort_action=ABORT)
            v_proj = evaluate_robust_value(root, nd, model, unc)
            cert = compute_certificate(root, nd, model, unc, b0, H)["certificate"]
            s_frac, z_frac = _support_fractions(root, nd, base_cfg.N, base_cfg.M)
            avg_frac = 0.5 * (s_frac + z_frac)
            results.append({
                "K": int(K), "seed": int(seed),
                "avg_state_frac": s_frac, "avg_obs_frac": z_frac, "avg_frac": avg_frac,
                "V_full": V_full, "V_max": V_max, "V_proj": v_proj, "certificate": cert,
                "true_error": abs(V_full - v_proj),
                "valid": int(abs(V_full - v_proj) <= cert + 1e-6),
            })
    return results, V_full, V_max


def _next_version(out_dir: Path) -> int:
    versions = [0]
    for f in out_dir.glob("data_V*.csv"):
        m = re.search(r"_V(\d+)\.csv$", f.name)
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


def plot_sampler(results, V_full, V_max, out_path, title):
    Ks = sorted({int(r["K"]) for r in results})
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(V_max, color="tab:red", ls=":", lw=1.0, label=f"+/- V_max = {V_max:.0f}")
    ax.axhline(-V_max, color="tab:red", ls=":", lw=1.0)
    ax.axhline(V_full, color="black", ls="--", lw=1.5, label=f"V_full = {V_full:.3f}")
    for i, K in enumerate(Ks):
        rows = [r for r in results if int(r["K"]) == K]
        x = [r["avg_frac"] for r in rows]
        vp = [r["V_proj"] for r in rows]
        cert = [r["certificate"] for r in rows]
        up = [min(v + c, V_max) - v for v, c in zip(vp, cert)]
        dn = [v - max(v - c, -V_max) for v, c in zip(vp, cert)]
        color = cmap(i / max(1, len(Ks) - 1))
        ax.errorbar(x, vp, yerr=[dn, up], fmt="o", ms=5, color=color, ecolor=color,
                    elinewidth=1.0, alpha=0.6, capsize=2, label=f"K={K}")
    ax.set_xlabel("average support fraction  (mean over nodes of |S_in|/N, |Z_in|/M)")
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    base_cfg = replace(A1Config(), write_tree_html=False, verbose=False)
    K_grid = [1, 2, 4, 8, 16, 32, 64, 128, 256, 2048]
    n_seeds = base_cfg.n_seeds

    out_dir = _SCRIPT.parents[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    version = _next_version(out_dir)
    csv_path = out_dir / f"data_V{version}.csv"
    plot_path = out_dir / f"sampler_plot_V{version}.png"

    results, V_full, V_max = run_sweep(K_grid, n_seeds, base_cfg)
    write_csv(results, csv_path)
    plot_sampler(results, V_full, V_max, plot_path,
                 title=(f"A1 sampler sweep | N={base_cfg.N}, M={base_cfg.M}, H={base_cfg.H}, "
                        f"rho_T={base_cfg.rho_T}, rho_Z={base_cfg.rho_Z}, n_seeds={n_seeds}"))

    print(f"V_full = {V_full:.4f}   V_max = {V_max:.4f}\n")
    for K in K_grid:
        rows = [r for r in results if r["K"] == K]
        af = float(np.mean([r["avg_frac"] for r in rows]))
        vp = float(np.mean([r["V_proj"] for r in rows]))
        ce = float(np.mean([r["certificate"] for r in rows]))
        allvalid = all(r["valid"] for r in rows)
        print(f"  K={K}: avg_frac={af:.3f}  mean V_proj={vp:.3f}  mean cert={ce:.3f}  "
              f"all_valid={bool(allvalid)}")
    print(f"\nWrote {csv_path.name} and {plot_path.name}")


def plot_only(version: int):
    out_dir = _SCRIPT.parents[0]
    results = read_csv(out_dir / f"data_V{version}.csv")
    V_full = results[0]["V_full"]
    V_max = results[0].get("V_max") or max(r["V_proj"] + r["certificate"] for r in results)
    plot_sampler(results, V_full, V_max, out_dir / f"sampler_plot_V{version}.png",
                 title=f"A1 sampler sweep | replot V{version}")
    print(f"Re-plotted sampler_plot_V{version}.png")


if __name__ == "__main__":
    main()
    # plot_only(1)