"""
Record one FrozenLake episode for viz/episode_app.py.

Runs a single episode, capturing per-step planner state (action, observation,
reward, root Q_robust, full planner event stream) into one JSON file. The
viewer scrubs through this JSON.

Inline episode loop here (not run_episode) so we can grab `root` and the
SolverLog from each plan() call without touching evaluation/episode.py.
That refactor is deferred to v2+.

Usage:
    python experiments/frozenlake/record_episode.py
    python experiments/frozenlake/record_episode.py --robust
    python experiments/frozenlake/record_episode.py --robust \
        --eta-obs 0.05 --eta-trans 0.10 --seed 7 --horizon 20 --budget 50
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from robust_pomdp import (
    SolverLog,
    bayes_update,
    robust_pomcp_plan,
)

# Allow running this script directly (without installing experiments as a pkg).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from frozenlake_problem import (  # noqa: E402
    N_ACTIONS,
    N_STATES,
    START_INDEX,
    TERMINAL_STATE,
    make_frozenlake_scenario,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--horizon",         type=int,   default=30)
    p.add_argument("--plan-horizon",    type=int,   default=None,help="Cap planner lookahead per step. Default: --horizon - t (plan to end of episode).")
    p.add_argument("--budget",          type=int,   default=100)
    p.add_argument("--sims-per-backup", type=int,   default=5)
    p.add_argument("--seed",            type=int,   default=0)
    p.add_argument("--p",               type=float, default=0.6,help="P(intended direction). Default 0.6 = slippery; 1.0 = deterministic moves.")
    p.add_argument("--epsilon",         type=float, default=0.05,help="Per-bit sensor flip probability (nominal). Higher = noisier observation.")
    p.add_argument("--robust",          action="store_true",help="Set rho_T=rho_Z=0.2 (rho_T localized to HOLE_ADJACENT).")
    p.add_argument("--rho-t",           type=float, default=None,help="Override rho_T (localized). Implies robust.")
    p.add_argument("--rho-z",           type=float, default=None,help="Override rho_Z (uniform). Implies robust.")
    p.add_argument("--eta-obs",         type=float, default=0.0,help="World obs perturbation (added to nominal epsilon).")
    p.add_argument("--eta-trans",       type=float, default=0.0,help="World transition perturbation, localized.")
    p.add_argument("--ucb-mode",        type=str,   default="nominal",choices=["nominal", "robust"])
    p.add_argument("--out",             type=Path,  default=None,help="Override output path. Default: results/episodes/<ts>_<flavor>.json")
    p.add_argument("--strip-events",    action="store_true",help="Drop planner_events from the JSON (cuts size ~100x). Re-record with events when v2/v3 layers need them.")
    return p.parse_args()


def resolve_rho(args: argparse.Namespace) -> tuple[float, float]:
    """Resolve (rho_T, rho_Z) from --robust / --rho-t / --rho-z flags."""
    if args.rho_t is not None or args.rho_z is not None:
        rho_T = args.rho_t if args.rho_t is not None else 0.0
        rho_Z = args.rho_z if args.rho_z is not None else 0.0
    elif args.robust:
        rho_T, rho_Z = 0.2, 0.2
    else:
        rho_T, rho_Z = 0.0, 0.0
    return rho_T, rho_Z


def flavor_tag(rho_T: float, rho_Z: float,
               eta_obs: float, eta_trans: float
               ) -> str:
    """Short tag for the filename: 'nominal', 'robust', 'robust_perturbed', etc."""
    is_robust    = (rho_T > 0.0) or (rho_Z > 0.0)
    is_perturbed = (eta_obs > 0.0) or (eta_trans > 0.0)
    if is_robust and is_perturbed:
        return "robust_perturbed"
    if is_robust:
        return "robust"
    if is_perturbed:
        return "vanilla_perturbed"
    return "nominal"


def main() -> None:
    args = parse_args()
    rho_T, rho_Z = resolve_rho(args)

    scenario = make_frozenlake_scenario(p=args.p, epsilon=args.epsilon)
    model_nominal = scenario.nominal_model
    model_true    = scenario.perturb_world(model_nominal, args.eta_obs, args.eta_trans)
    uncertainty   = scenario.uncertainty_factory(rho_T_val=rho_T, rho_Z_val=rho_Z)

    # Reproducibility: one seed -> two independent streams (world, planner).
    rng_world, rng_planner = np.random.default_rng(args.seed).spawn(2)

    # Initial belief: concentrated at start cell.
    belief = np.zeros(N_STATES)
    belief[START_INDEX] = 1.0
    state = START_INDEX

    records: list[dict] = []
    total_reward = 0.0

    for t in range(args.horizon):
        logger = SolverLog()
        plan_h = min(
            args.plan_horizon if args.plan_horizon is not None else args.horizon - t,
            args.horizon - t,
        )
        action, root = robust_pomcp_plan(belief,
                                         model_nominal,
                                         uncertainty,
                                         horizon=plan_h,
                                         budget=args.budget,
                                         sims_per_backup=args.sims_per_backup,
                                         ucb_mode=args.ucb_mode,
                                         logger=logger,
                                         rng=rng_planner
                                         )

        q_robust: list[float | None] = [
            root.children[a].Q_robust if a in root.children else None
            for a in range(N_ACTIONS)
        ]

        reward     = model_true.reward(state, action)
        next_state = model_true.sample_transition(state, action, rng_world)
        obs        = model_true.sample_observation(next_state, rng_world)

        records.append({
            "t":              t,
            "state":          int(state),
            "belief":         belief.tolist(),
            "action":         int(action),
            "obs":            int(obs),
            "reward":         float(reward),
            "next_state":     int(next_state),
            "q_robust":       q_robust,
            "planner_events": ([] if args.strip_events
                               else [{"type": e.type, **e.data} for e in logger.events]),
        })

        total_reward += reward
        state  = next_state
        if state == TERMINAL_STATE:
            break
        belief = bayes_update(belief, model_nominal, action, obs)

    metadata = {
        "scenario":         "frozenlake",
        "horizon":          args.horizon,
        "plan_horizon":     args.plan_horizon,
        "budget":           args.budget,
        "sims_per_backup":  args.sims_per_backup,
        "ucb_mode":         args.ucb_mode,
        "p":                args.p,
        "epsilon":          args.epsilon,
        "rho_T":            rho_T,
        "rho_Z":            rho_Z,
        "eta_obs":          args.eta_obs,
        "eta_trans":        args.eta_trans,
        "seed":             args.seed,
        "timestamp":        datetime.now().isoformat(timespec="seconds"),
        "total_reward":     float(total_reward),
    }

    if args.out is None:
        ts     = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        flavor = flavor_tag(rho_T, rho_Z, args.eta_obs, args.eta_trans)
        out    = scenario.results_dir / "episodes" / f"{ts}_{flavor}.json"
    else:
        out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as f:
        json.dump({"metadata": metadata, "episode": records}, f)

    n_events_total = sum(len(r["planner_events"]) for r in records)
    print(f"Wrote {out}")
    print(f"  steps={len(records)}, total_reward={total_reward:.4f}, "
          f"events={n_events_total}, size={out.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()

"""

# Record (produces JSON in experiments/frozenlake/results/episodes/)
python experiments/frozenlake/record_episode.py --horizon 50 --plan-horizon 20 --budget 200 --p 0.95 --strip-events

# View
python viz/episode_app.py --episode experiments/frozenlake/results/episodes/<timestamp>_<flavor>.json

"""