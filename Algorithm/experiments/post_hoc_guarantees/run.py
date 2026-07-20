"""
Post-hoc guarantees experiment (config-driven, A1-style output).

Demonstrates: for a FIXED policy pi (any origin), the certificate deterministically
bounds its true robust value — |V_full^pi - V_proj^pi| <= eps — and eps tightens
toward 0 as the sampled support grows toward the full space.

The policy is chosen in the config (cfg.policy): the hand-coded "warning" rule, or
the frozen "greedy" (optimal) policy extracted from the full robust solve.

The support SCHEME is chosen in the config (cfg.scheme):
  - "uniform" / "exp_decay": deterministic mask (same support at every node),
    swept over cfg.fractions -> data_V{n}.csv + posthoc_V{n}.png (band plot);
  - "sampler": per-node K-sampling from the belief (POMCP-style focusing),
    swept over cfg.K_grid x cfg.n_seeds -> sampler_data_V{n}.csv +
    sampler_V{n}.png (scatter, A1-style).

Ergonomics (like the A1 runs):
  - all knobs live in Algorithm/experiments/config.py (Config);
  - run(cfg) dispatches on cfg.scheme, prints a table, and writes versioned
    CSV + PNG into cfg.out_dir;
  - plot_only(n) re-plots from a saved CSV (dispatches on cfg.scheme too);
  - run() raises if validity is ever violated (self-checking).

Run from the repo root:
    python -m Algorithm.experiments.post_hoc_guarantees.run
Play with parameters by editing config.py, or:
    from Algorithm.experiments.post_hoc_guarantees.run import run
    from Algorithm.experiments.config import Config
    run(Config(N=20, M=8, rho_T=0.1, scheme="exp_decay"))
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from Algorithm.experiments.post_hoc_guarantees.support_picker import (
    exponential_decay_pick, uniform_pick)
from Algorithm.domains.synthetic import ABORT, INSPECT, MITIGATE, PROCEED
from Algorithm.experiments.config import Config
from Algorithm.experiments.post_hoc_guarantees.fixed_policy_tree import (
    certificate_eps, eval_fixed_policy, r_max)
from Algorithm.experiments.post_hoc_guarantees.full_solver import solve_full
from Algorithm.experiments.post_hoc_guarantees.sampled_policy_tree import (
    avg_support_fraction, eval_sampled_policy)

_SCRIPT = Path(__file__).resolve()


def warning_policy(M: int, warning_low: float = 0.4, warning_high: float = 0.75):
    """Depth-0 -> inspect; else action by last-obs warning level (memoryless)."""
    def policy(obs_path):
        if not obs_path:
            return INSPECT
        warn = obs_path[-1] / (M - 1)
        if warn < warning_low:
            return PROCEED
        if warn < warning_high:
            return MITIGATE
        return ABORT
    return policy


def _build_policy(cfg: Config, model, unc, b0, S_all, Z_all):
    """Return (policy, V_full) for cfg.policy. 'greedy' is extracted + frozen from the
    full robust solve; 'warning' is the hand-coded rule (V_full = full-support eval)."""
    if cfg.policy == "greedy":
        V_full, q_root, pi_by_root_action = solve_full(model, unc, b0, cfg.H, ABORT, {})
        best_a = max(q_root, key=q_root.get)
        return (lambda op, _p=pi_by_root_action[best_a]: _p[op]), V_full
    if cfg.policy == "warning":
        policy = warning_policy(cfg.M, cfg.warning_low, cfg.warning_high)
        V_full = eval_fixed_policy(model, unc, policy, b0, cfg.H,
                                   S_all, Z_all, ABORT, {}).V_rob
        return policy, V_full
    raise ValueError(f"unknown cfg.policy {cfg.policy!r}; use 'warning' or 'greedy'")


def sweep(cfg: Config) -> list[dict]:
    """Sweep the support fraction for the configured policy; one row per frac."""
    model, b0, unc = cfg.build()
    R_max = r_max(model)
    V_max = R_max * cfg.H
    S_all, Z_all = list(range(cfg.N)), list(range(cfg.M))
    policy, V_full = _build_policy(cfg, model, unc, b0, S_all, Z_all)

    def pick(f, n):
        if cfg.scheme == "uniform":
            return uniform_pick(f, n)
        return exponential_decay_pick(f, n, n_bins=cfg.n_bins, decay=cfg.decay)

    rows: list[dict] = []
    for f in cfg.fractions:
        S_in, Z_in = pick(f, cfg.N), pick(f, cfg.M)
        proj = eval_fixed_policy(model, unc, policy, b0, cfg.H, S_in, Z_in, ABORT, {})
        eps = certificate_eps(proj, b0, cfg.H, R_max)
        err = abs(V_full - proj.V_rob)
        rows.append({
            "policy": cfg.policy,
            "frac": 0.5 * (len(S_in) / cfg.N + len(Z_in) / cfg.M),
            "S": len(S_in), "Z": len(Z_in),
            "V_full": V_full, "V_proj": proj.V_rob, "eps": eps, "err": err,
            "V_max": V_max, "valid": int(err <= eps + 1e-6),
        })
    return rows


# --- output helpers -----------------------------------------------------------

def _out_dir(cfg: Config) -> Path:
    d = Path(cfg.out_dir)
    if not d.is_absolute():
        d = _SCRIPT.parent / d
    d.mkdir(parents=True, exist_ok=True)
    return d


def _next_version(out_dir: Path, stem: str = "data") -> int:
    versions = [0]
    for f in out_dir.glob(f"{stem}_V*.csv"):
        m = re.search(r"_V(\d+)\.csv$", f.name)
        if m:
            versions.append(int(m.group(1)))
    return max(versions) + 1


def _write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return [{k: (v if k == "policy" else float(v)) for k, v in row.items()}
                for row in csv.DictReader(f)]


def _print_table(rows: list[dict]) -> None:
    for name in dict.fromkeys(r["policy"] for r in rows):    # unique, in insertion order
        sub = sorted((r for r in rows if r["policy"] == name), key=lambda r: r["frac"])
        if not sub:
            continue
        print(f"[{name}]  V_full = {sub[0]['V_full']:.4f}")
        print(f"  {'frac':>5} {'|S|':>4} {'|Z|':>4} {'V_proj':>9} {'eps':>8} {'|err|':>8} {'valid':>6}")
        for r in sub:
            print(f"  {r['frac']:5.2f} {int(r['S']):4d} {int(r['Z']):4d} "
                  f"{r['V_proj']:9.4f} {r['eps']:8.3f} {r['err']:8.3f} "
                  f"{str(bool(r['valid'])):>6}")
        print()


def plot_posthoc(rows: list[dict], out_path: Path, title: str) -> bool:
    """Per-policy: V_proj + [V_proj +/- eps] band (clipped to +/-V_max) vs support
    fraction, with the dashed V_full reference. Returns False if matplotlib absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:  # matplotlib not installed
        print(f"  (matplotlib unavailable: {e}; skipping plot)")
        return False

    policies = list(dict.fromkeys(r["policy"] for r in rows))   # whatever's present
    fig, axes = plt.subplots(1, len(policies), figsize=(max(9, 6 * len(policies)), 5.5),
                             sharey=True, squeeze=False)
    for ax, name in zip(axes[0], policies):
        sub = sorted((r for r in rows if r["policy"] == name), key=lambda r: r["frac"])
        x = [r["frac"] for r in sub]
        vp = np.array([r["V_proj"] for r in sub])
        eps = np.array([r["eps"] for r in sub])
        vmax, vfull = sub[0]["V_max"], sub[0]["V_full"]
        ax.fill_between(x, np.clip(vp - eps, -vmax, vmax), np.clip(vp + eps, -vmax, vmax),
                        alpha=0.2, color="tab:blue", label="V_proj +/- eps")
        ax.plot(x, vp, "o-", color="tab:blue", label="V_proj")
        ax.axhline(vfull, ls="--", color="black", label=f"V_full = {vfull:.2f}")
        ax.axhline(vmax, ls=":", color="tab:red", lw=0.8)
        ax.axhline(-vmax, ls=":", color="tab:red", lw=0.8, label=f"+/-V_max = {vmax:.0f}")
        ax.set_title(name)
        ax.set_xlabel("support fraction")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    axes[0][0].set_ylabel("robust value")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


# --- per-node sampler (scheme="sampler") ----------------------------------------

def sweep_sampler(cfg: Config) -> list[dict]:
    """Per-node K-sampling sweep: one point per (K, seed). Same frozen policy as the
    fraction sweep; support drawn from each node's belief (POMCP-style focusing)."""
    model, b0, unc = cfg.build()
    R_max = r_max(model)
    V_max = R_max * cfg.H
    S_all, Z_all = list(range(cfg.N)), list(range(cfg.M))
    policy, V_full = _build_policy(cfg, model, unc, b0, S_all, Z_all)
    lp_cache: dict = {}                       # shared persistent LPs across all trees

    rows: list[dict] = []
    for K in cfg.K_grid:
        for seed in range(cfg.n_seeds):
            rng = np.random.default_rng(cfg.seed + seed)
            root = eval_sampled_policy(model, unc, policy, b0, cfg.H, int(K), rng,
                                       ABORT, lp_cache)
            eps = certificate_eps(root, b0, cfg.H, R_max)
            err = abs(V_full - root.V_rob)
            s_frac, z_frac = avg_support_fraction(root, cfg.N, cfg.M)
            rows.append({
                "K": int(K), "seed": int(seed),
                "avg_state_frac": s_frac, "avg_obs_frac": z_frac,
                "avg_frac": 0.5 * (s_frac + z_frac),
                "V_full": V_full, "V_proj": root.V_rob, "eps": eps, "err": err,
                "V_max": V_max, "valid": int(err <= eps + 1e-6),
            })
    return rows


def _print_sampler_table(rows: list[dict]) -> None:
    print(f"  {'K':>5} {'avg_frac':>9} {'V_proj':>9} {'eps':>9} {'|err|':>8} {'valid':>6}")
    for K in sorted({int(r["K"]) for r in rows}):
        sub = [r for r in rows if int(r["K"]) == K]
        n = len(sub)
        af = sum(r["avg_frac"] for r in sub) / n
        vp = sum(r["V_proj"] for r in sub) / n
        ce = sum(r["eps"] for r in sub) / n
        er = sum(r["err"] for r in sub) / n
        ok = all(r["valid"] for r in sub)
        print(f"  {K:5d} {af:9.3f} {vp:9.3f} {ce:9.3f} {er:8.3f} {str(bool(ok)):>6}")
    print()


def plot_sampler(rows: list[dict], out_path: Path, title: str) -> bool:
    """V_proj vs average support fraction, colored by K, with the [V_proj +/- eps]
    certificate as error bars (clipped to +/-V_max) and the V_full / +/-V_max
    references. Mirrors the A1 sampler plot. False if matplotlib is absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # matplotlib not installed
        print(f"  (matplotlib unavailable: {e}; skipping plot)")
        return False

    Ks = sorted({int(r["K"]) for r in rows})
    cmap = plt.get_cmap("viridis")
    vmax, vfull = rows[0]["V_max"], rows[0]["V_full"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(vmax, color="tab:red", ls=":", lw=1.0, label=f"+/- V_max = {vmax:.0f}")
    ax.axhline(-vmax, color="tab:red", ls=":", lw=1.0)
    ax.axhline(vfull, color="black", ls="--", lw=1.5, label=f"V_full = {vfull:.3f}")
    for i, K in enumerate(Ks):
        sub = [r for r in rows if int(r["K"]) == K]
        x = [r["avg_frac"] for r in sub]
        vp = np.array([r["V_proj"] for r in sub])
        eps = np.array([r["eps"] for r in sub])
        up = np.clip(vp + eps, -vmax, vmax) - vp
        dn = vp - np.clip(vp - eps, -vmax, vmax)
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
    return True


def run_sampler(cfg: Config) -> bool:
    rows = sweep_sampler(cfg)
    print(f"Sampler sweep | N={cfg.N} M={cfg.M} H={cfg.H} "
          f"rho_T={cfg.rho_T} rho_Z={cfg.rho_Z} n_seeds={cfg.n_seeds}")
    print(f"  V_full = {rows[0]['V_full']:.4f}  V_max = {rows[0]['V_max']:.2f}\n")
    _print_sampler_table(rows)

    out = _out_dir(cfg)
    v = _next_version(out, "sampler_data")
    _write_csv(rows, out / f"sampler_data_V{v}.csv")
    wrote_plot = False
    if cfg.write_plot:
        title = (f"Sampler sweep | policy={cfg.policy} | N={cfg.N} M={cfg.M} H={cfg.H} "
                 f"rho_T={cfg.rho_T} rho_Z={cfg.rho_Z} n_seeds={cfg.n_seeds}")
        wrote_plot = plot_sampler(rows, out / f"sampler_V{v}.png", title)

    all_valid = all(r["valid"] for r in rows)
    print(f"Wrote sampler_data_V{v}.csv" + (f" + sampler_V{v}.png" if wrote_plot else ""))
    print(f"ALL VALID: {all_valid}   (eps floors above 0 where sampling misses states)")
    if not all_valid:
        raise SystemExit("SAMPLER POST-HOC VALIDITY VIOLATED")
    return all_valid


def run(cfg: Config | None = None) -> bool:
    cfg = cfg or Config()
    if cfg.scheme == "sampler":
        return run_sampler(cfg)
    rows = sweep(cfg)
    print(f"Post-hoc guarantees | N={cfg.N} M={cfg.M} H={cfg.H} "
          f"rho_T={cfg.rho_T} rho_Z={cfg.rho_Z} scheme={cfg.scheme}")
    print(f"  V_max = {rows[0]['V_max']:.2f}\n")
    _print_table(rows)

    out = _out_dir(cfg)
    v = _next_version(out)
    _write_csv(rows, out / f"data_V{v}.csv")
    wrote_plot = False
    if cfg.write_plot:
        title = (f"Post-hoc guarantees | policy={cfg.policy} | N={cfg.N} M={cfg.M} H={cfg.H} "
                 f"rho_T={cfg.rho_T} rho_Z={cfg.rho_Z} scheme={cfg.scheme}")
        wrote_plot = plot_posthoc(rows, out / f"posthoc_V{v}.png", title)

    all_valid = all(r["valid"] for r in rows)
    print(f"Wrote data_V{v}.csv" + (f" + posthoc_V{v}.png" if wrote_plot else ""))
    print(f"ALL VALID: {all_valid}   (eps -> 0 and |err| -> 0 as frac -> 1)")
    if not all_valid:
        raise SystemExit("POST-HOC VALIDITY VIOLATED")
    return all_valid


def plot_only(version: int, cfg: Config | None = None) -> None:
    cfg = cfg or Config()
    out = _out_dir(cfg)
    if cfg.scheme == "sampler":
        rows = _read_csv(out / f"sampler_data_V{version}.csv")
        plot_sampler(rows, out / f"sampler_V{version}.png",
                     title=f"Sampler sweep | policy={cfg.policy} | replot V{version}")
        print(f"Re-plotted sampler_V{version}.png")
        return
    rows = _read_csv(out / f"data_V{version}.csv")
    plot_posthoc(rows, out / f"posthoc_V{version}.png", title=f"Post-hoc guarantees | policy={cfg.policy} | replot V{version}")
    print(f"Re-plotted posthoc_V{version}.png")


if __name__ == "__main__":
    run()
