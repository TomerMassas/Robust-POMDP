"""
CR-UCT planner demo — ONE planning step, config-driven (like the post-hoc runs).

Runs plan(cfg) on the synthetic domain and prints, per root action: Q_rob, the
certificate eps, the bracket [L, U], and the full-solver q_root (the OPTIMAL
continuation, ground truth). Reports the chosen action, whether it was certified
(anytime stop), and how many simulations were used.

Print-only, single planning step (no episode loop, no CSV/plot) — the runnable
"flow" for playing with ucb_mode / budget / c_ucb from config.py.

Run:
    python -m Algorithm.experiments.planner_demo.run
or:
    from Algorithm.experiments.planner_demo.run import run
    from Algorithm.experiments.config import Config
    run(Config(ucb_mode="robust", budget=2000))
"""

from __future__ import annotations

import numpy as np

from Algorithm.domains.synthetic import ABORT, ACTION_NAMES
from Algorithm.experiments.config import Config
from Algorithm.experiments.post_hoc_guarantees.full_solver import solve_full
from Algorithm.ucb.certificate import action_bracket, r_max
from Algorithm.ucb.planner import plan


def run(cfg: Config | None = None) -> None:
    cfg = cfg or Config()
    model, b0, unc = cfg.build()
    R_max = r_max(model)
    rng = np.random.default_rng(cfg.seed)

    res = plan(model, unc, b0, cfg.H,
               budget=cfg.budget, ucb_mode=cfg.ucb_mode, c_ucb=cfg.c_ucb,
               use_rollouts=cfg.use_rollouts, early_stop=cfg.early_stop,
               abort_action=ABORT, rng=rng)

    _, q_root, _ = solve_full(model, unc, b0, cfg.H, ABORT, {})     # ground truth
    best_a = max(q_root, key=q_root.get)

    print(f"CR-UCT | ucb_mode={cfg.ucb_mode} budget={cfg.budget} c_ucb={cfg.c_ucb:.3f} "
          f"early_stop={cfg.early_stop} | N={cfg.N} M={cfg.M} H={cfg.H} "
          f"rho_T={cfg.rho_T} rho_Z={cfg.rho_Z}")
    print(f"  R_max={R_max:.2f}   |S_in(root)|={len(res.root.S_in)}/{cfg.N}\n")
    print(f"  {'action':>9} {'Q_rob':>9} {'eps':>9} {'L':>9} {'U':>9} {'q_root*':>9}")
    for a in sorted(res.root.children):
        an = res.root.children[a]
        L, U, eps = action_bracket(an, b0, sorted(res.root.S_in), cfg.H, R_max)
        name = ACTION_NAMES[a] if a < len(ACTION_NAMES) else str(a)
        print(f"  {name:>9} {an.Q_rob:9.3f} {eps:9.3f} {L:9.3f} {U:9.3f} {q_root[a]:9.3f}")

    verdict = "CERTIFIED" if res.certified else "budget-exhausted"
    match = "MATCHES" if res.action == best_a else "differs from"
    print(f"\n  q_root* = full-solver Q (OPTIMAL continuation, ground truth); "
          f"best = {ACTION_NAMES[best_a]}")
    print(f"  chosen  = {ACTION_NAMES[res.action]}  ({verdict}, sims={res.sims_used})  "
          f"-> {match} full-solver best")


if __name__ == "__main__":
    run()
