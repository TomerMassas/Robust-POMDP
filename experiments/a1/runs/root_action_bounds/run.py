"""
A1 per-root-action Q-certificate bands vs sampling effort K.

Question: as we sample more (larger K), do the certified intervals around each
root action's robust Q-value shrink enough to SEPARATE the optimal action from
the rest? Full separation -> we can certify the best root action without ever
refining the support further -> "we can stop planning."

For each root action a we hold a FIXED policy pi_a = "take a at the root, then
act optimally" (extracted once from the full-robust-greedy expectimax). Its true
robust value is Q_full(h0, a) = q_root[a]. For each (K, seed) we build the per-
node sampled-projection tree FOLLOWING pi_a, and read off:
    Q_proj(a) = evaluate_robust_value(tree)
    eps(a)    = compute_certificate(tree)             (leakage-only, Theorem 1)
Theorem 1 (applied to the fixed policy pi_a) guarantees
    Q_full(h0, a)  in  [ Q_proj(a) - eps(a),  Q_proj(a) + eps(a) ]
for every K and seed -- so every band must contain its dashed Q_full line.

Plot: x = K (log2). Per action: mean-over-seeds Q_proj line + shaded +/- eps band
(clipped to [-V_max, V_max]), and a dashed horizontal Q_full(a) reference. A
vertical "can stop planning" marker is drawn at the smallest K where the best
guaranteed lower bound clears every other action's upper bound:
    max_a [Q_proj(a) - eps(a)]  >=  max_{a' != argmax} [Q_proj(a') + eps(a')].

CAVEAT (abort): pi_abort aborts at the root, so its tree is depth-0 only. The
leakage certificate charges the initial out-of-support mass (1 - m_in) at all
H+1 depths, so eps(abort) = (H+1)*R_max*(1 - m_in) -- valid but ~(H+1)x looser
than the true depth-0-only error R_max*(1 - m_in). It still -> 0 as K -> inf
(m_in -> 1). Flagged on the plot.

Versioned outputs: data_V{n}.csv, bands_plot_V{n}.png. plot_only(n) re-plots.

Run from anywhere:
    python experiments/a1/runs/root_action_bounds/run.py
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
from experiments.a1.synthetic_pomdp import ABORT, ACTION_NAMES
from experiments.a1.tree_evaluator import evaluate_robust_value
from experiments.a1.sampled_projection import build_sampled_projection_tree

# Stable color per action index (inspect, proceed, mitigate, abort).
_ACTION_COLORS = {0: "tab:blue", 1: "tab:green", 2: "tab:orange", 3: "tab:gray"}


def _avg_support_fraction(root, node_data, N, M):
    """Mean over history nodes of |S_in|/N and over action nodes of |Z_in|/M,
    averaged together. A depth-0 (abort) tree has no action nodes -> the state
    fraction stands in for the obs fraction."""
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
    s = float(np.mean(s_fracs))
    z = float(np.mean(z_fracs)) if z_fracs else s
    return 0.5 * (s + z)


def _frac_by_K(rows, K_grid):
    """Shared x-coordinate per K: mean avg_frac over the CONTINUING (non-abort)
    root-action trees -- the abort tree is depth-0 and degenerate, so it does not
    define 'the support fraction of the planning tree'."""
    out = {}
    for K in K_grid:
        cont = [r["avg_frac"] for r in rows
                if int(r["K"]) == int(K) and int(r["action"]) != ABORT]
        same_K = [r["avg_frac"] for r in rows if int(r["K"]) == int(K)]
        out[int(K)] = float(np.mean(cont)) if cont else float(np.mean(same_K))
    return out


def run_experiment(K_grid, n_seeds, base_cfg):
    """Solve the full expectimax ONCE (q_root + per-root-action policies), then
    for each (K, seed, root action a) build the sampled tree following pi_a and
    record Q_proj(a) and eps(a). Returns per-(K, a) rows aggregated over seeds."""
    ref = build_reference(base_cfg)
    model, b0, unc, H = ref["model"], ref["b0"], ref["uncertainty"], ref["H"]
    q_root = ref["q_root"]
    pi_by_root_action = ref["pi_by_root_action"]
    V_full, R_max = ref["V_full"], ref["R_max"]
    V_max = R_max * (H + 1)
    N, M = base_cfg.N, base_cfg.M
    actions = sorted(q_root.keys())

    rows = []
    for K in K_grid:
        for a in actions:
            qps, eps, fracs = [], [], []
            worst_margin = -np.inf
            for seed in range(n_seeds):
                root, nd = build_sampled_projection_tree(
                    model, pi_by_root_action[a], b0, H, K=int(K),
                    rng=np.random.default_rng(base_cfg.seed + seed), abort_action=ABORT)
                q_proj = evaluate_robust_value(root, nd, model, unc)
                cert = compute_certificate(root, nd, model, unc, b0, H)["certificate"]
                qps.append(q_proj)
                eps.append(cert)
                fracs.append(_avg_support_fraction(root, nd, N, M))
                # Validity (Theorem 1 for the fixed policy pi_a): per seed.
                worst_margin = max(worst_margin, abs(q_root[a] - q_proj) - cert)
            rows.append({
                "K": int(K), "action": int(a),
                "avg_frac": float(np.mean(fracs)),
                "Q_full": float(q_root[a]),
                "Q_proj_mean": float(np.mean(qps)),
                "Q_proj_std": float(np.std(qps)),
                "eps_mean": float(np.mean(eps)),
                "eps_min": float(np.min(eps)),
                "eps_max": float(np.max(eps)),
                "valid": int(worst_margin <= 1e-6),
                "worst_margin": float(worst_margin),
            })
    return rows, V_full, V_max, actions


def _separation_K(rows, actions, K_grid):
    """Smallest K (on mean Q_proj/eps) where the best guaranteed lower bound
    clears every other action's upper bound. None if never separated."""
    for K in K_grid:
        sub = {r["action"]: r for r in rows if r["K"] == int(K)}
        lows = {a: sub[a]["Q_proj_mean"] - sub[a]["eps_mean"] for a in actions}
        highs = {a: sub[a]["Q_proj_mean"] + sub[a]["eps_mean"] for a in actions}
        a_hat = max(actions, key=lambda a: lows[a])
        other_high = max(highs[a] for a in actions if a != a_hat)
        if lows[a_hat] >= other_high:
            return int(K), a_hat
    return None, None


def plot_bands(rows, V_full, V_max, actions, K_grid, out_path, title):
    # x-axis = average support fraction (mean over nodes of |S_in|/N, |Z_in|/M),
    # shared per K across actions; K is kept on a secondary top axis.
    frac_by_K = _frac_by_K(rows, K_grid)
    Ks = [int(k) for k in K_grid]
    xs_K = [frac_by_K[k] for k in Ks]

    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.axhline(V_max, color="tab:red", ls=":", lw=1.0, label=f"+/- V_max = {V_max:.0f}")
    ax.axhline(-V_max, color="tab:red", ls=":", lw=1.0)

    for a in actions:
        sub = sorted([r for r in rows if r["action"] == a], key=lambda r: r["K"])
        xs = [frac_by_K[int(r["K"])] for r in sub]
        qp = np.array([r["Q_proj_mean"] for r in sub])
        ep = np.array([r["eps_mean"] for r in sub])
        lower = np.clip(qp - ep, -V_max, V_max)
        upper = np.clip(qp + ep, -V_max, V_max)
        color = _ACTION_COLORS.get(a, None)
        ax.fill_between(xs, lower, upper, color=color, alpha=0.15)
        ax.plot(xs, qp, marker="o", ms=4, color=color, lw=1.6,
                label=f"{ACTION_NAMES[a]} (Q_proj +/- cert)")
        ax.plot(xs, [r["Q_full"] for r in sub], ls="--", lw=1.3, color=color, alpha=0.9)

    sep_K, a_hat = _separation_K(rows, actions, K_grid)
    if sep_K is not None:
        x_sep = frac_by_K[sep_K]
        ax.axvline(x_sep, color="black", lw=1.4, alpha=0.7)
        ax.annotate(f"separated at frac={x_sep:.2f} (K={sep_K})\n"
                    f"-> can stop ({ACTION_NAMES[a_hat]})",
                    xy=(x_sep, -V_max * 0.85), xytext=(-6, 0), textcoords="offset points",
                    fontsize=9, va="center", ha="right",
                    bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.85))

    ax.set_xlabel("average support fraction  (mean over nodes of |S_in|/N, |Z_in|/M)")
    ax.set_ylabel("root robust Q-value")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Secondary top axis: which K produced each support fraction.
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(xs_K)
    ax2.set_xticklabels([str(k) for k in Ks], fontsize=7)
    ax2.set_xlabel("sampling effort K", fontsize=8)

    fig.text(0.99, 0.01, "dashed = true Q_full(h0,a); band must contain it",
             ha="right", va="bottom", fontsize=7, style="italic", color="gray")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _next_version(out_dir: Path) -> int:
    versions = [0]
    for f in out_dir.glob("data_V*.csv"):
        m = re.search(r"_V(\d+)\.csv$", f.name)
        if m:
            versions.append(int(m.group(1)))
    return max(versions) + 1


def write_csv(rows, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(csv_path):
    with open(csv_path, newline="") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def main():
    base_cfg = replace(A1Config(), write_tree_html=False, verbose=False)
    K_grid = [16, 32, 256, 512, 1024, 2048]
    n_seeds = 1# base_cfg.n_seeds

    out_dir = _SCRIPT.parents[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    version = _next_version(out_dir)
    csv_path = out_dir / f"data_V{version}.csv"
    plot_path = out_dir / f"bands_plot_V{version}.png"

    rows, V_full, V_max, actions = run_experiment(K_grid, n_seeds, base_cfg)
    write_csv(rows, csv_path)
    plot_bands(rows, V_full, V_max, actions, K_grid, plot_path,
               title=(f"A1 root-action Q-certificates vs support fraction | "
                      f"N={base_cfg.N}, M={base_cfg.M}, H={base_cfg.H}, "
                      f"rho_T={base_cfg.rho_T}, rho_Z={base_cfg.rho_Z}, n_seeds={n_seeds}"))

    print(f"V_full = {V_full:.4f}   V_max = {V_max:.4f}")
    print(f"Q_full per root action: "
          + "  ".join(f"{ACTION_NAMES[a]}={rows_a['Q_full']:.3f}"
                      for a in actions
                      for rows_a in [next(r for r in rows if r['action'] == a)]))
    all_valid = all(r["valid"] for r in rows)
    print(f"validity (Q_full in every band, all K/seeds): {all_valid}"
          + ("" if all_valid else "  !! INVALID"))
    print()
    frac_by_K = _frac_by_K(rows, K_grid)
    for K in K_grid:
        parts = []
        for a in actions:
            r = next(x for x in rows if x["K"] == K and x["action"] == a)
            parts.append(f"{ACTION_NAMES[a][:3]}={r['Q_proj_mean']:5.2f}+/-{r['eps_mean']:5.2f}")
        print(f"  K={K:5d} (frac={frac_by_K[K]:.2f}):  " + "  ".join(parts))
    sep_K, a_hat = _separation_K(rows, actions, K_grid)
    if sep_K is not None:
        print(f"\nSEPARATED at K={sep_K}: '{ACTION_NAMES[a_hat]}' bound clears all others "
              f"-> can stop planning.")
    else:
        print("\nNot separated within K_grid (optimal action's band still overlaps others).")
    print(f"\nWrote {csv_path.name} and {plot_path.name}")


def plot_only(version: int):
    out_dir = _SCRIPT.parents[0]
    rows = read_csv(out_dir / f"data_V{version}.csv")
    for r in rows:
        r["K"], r["action"] = int(r["K"]), int(r["action"])
    actions = sorted({r["action"] for r in rows})
    K_grid = sorted({r["K"] for r in rows})
    V_full = rows[0]["Q_full"]  # placeholder; not drawn directly
    V_max = max(r["Q_proj_mean"] + r["eps_mean"] for r in rows)
    plot_bands(rows, V_full, V_max, actions, K_grid,
               out_dir / f"bands_plot_V{version}.png", title=f"A1 root-action bands | replot V{version}")
    print(f"Re-plotted bands_plot_V{version}.png")


if __name__ == "__main__":
    main()
    # plot_only(1)