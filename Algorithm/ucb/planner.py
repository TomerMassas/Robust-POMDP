"""
CR-UCT — Certified Robust UCT (UCB variant of R-MCTS). One planning step.

Implements PLAN / SIMULATE / SELECT_ACTION / ROLLOUT from `pseudocode.md`. The
robust value + survival backup (BACKUP), the projected-Bayes beliefs, and the tree
data structures are REUSED from the tested core (`ucb.backup`, `ucb.belief`,
`core.tree`); this module only adds the sampling-driven tree growth, UCB action
selection, optional rollout, and the anytime certified stop.

Conventions (match the core): H reward depths 0..H-1, terminal at depth H-1;
`abort_action` is a generic optional terminal (None when a domain has none) and is
special-cased only inside BACKUP, never here. Beliefs are option-B projected-Bayes,
arrays aligned with SORTED support (so they line up with backup's sorted(node.S_in)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from Algorithm.core.pomdp import TabularPOMDP
from Algorithm.core.tree import HistoryNode, NodeIdCounter
from Algorithm.core.uncertainty import UncertaintySets
from Algorithm.ucb.backup import backup
from Algorithm.ucb.belief import projected_bayes, restrict
from Algorithm.ucb.certificate import certified_best_action, r_max


@dataclass
class PlanResult:
    action: int                 # chosen root action
    certified: bool             # True if returned via the anytime certificate
    sims_used: int              # simulations actually run
    root: HistoryNode           # the search tree (per-action Q_rob / c live here)


def plan(model: TabularPOMDP,
         uncertainty: UncertaintySets,
         b0: np.ndarray,
         H: int,
         *,
         budget: int,
         ucb_mode: str = "robust",
         c_ucb: float = 2 ** 0.5,
         use_rollouts: bool = False,
         early_stop: bool = True,
         abort_action: int | None = None,
         rng: np.random.Generator | None = None,
         rollout_policy: Callable[[int], int] | None = None,
         lp_cache: dict | None = None,
         ) -> PlanResult:
    """Run one CR-UCT planning step from belief b0; return the chosen root action.

    ucb_mode drives exploration only ("nominal" -> Q_nom sample-mean, with optional
    rollouts; "robust" -> Q_rob). The root decision and certificate use Q_rob / c in
    both modes.
    """
    if ucb_mode not in ("nominal", "robust"):
        raise ValueError(f"unknown ucb_mode {ucb_mode!r}; use 'nominal' or 'robust'")
    if rng is None:
        rng = np.random.default_rng()
    if lp_cache is None:
        lp_cache = {}
    b0 = np.asarray(b0, dtype=np.float64)
    n_actions = model.n_actions
    ctr = NodeIdCounter()
    R_max = r_max(model)
    V_max = R_max * H                              # return bound; UCB1 assumes values in [0,1]

    def _sample(probs: np.ndarray) -> int:
        p = np.asarray(probs, dtype=np.float64)
        p = p / p.sum()                                # defensive: exact sum-to-1 for rng
        return int(rng.choice(len(p), p=p))

    def _rollout(s: int, d: int) -> float:
        """Nominal random sim to H-1; undiscounted reward sum. Feeds Q_nom only."""
        G, depth = 0.0, d
        while True:
            a = rollout_policy(s) if rollout_policy is not None else int(rng.integers(n_actions))
            G += float(model.R[s, a])
            if depth == H - 1:
                return G
            s = _sample(model.T[s, a, :])
            depth += 1

    def _select_action(node: HistoryNode) -> int:
        unexpanded = node.unexpanded_actions(n_actions)
        if unexpanded:                                 # incremental expansion, one per visit
            a = unexpanded[0]
            node.expand_action(a, ctr)
            return a
        log_n = math.log(node.N)                       # all expanded -> UCB (each an.N >= 1)

        def _score(a: int) -> float:
            an = node.children[a]
            q = an.Q_nom if ucb_mode == "nominal" else an.Q_rob
            q_norm = 0.5 if V_max <= 0 else min(1.0, max(0.0, (q + V_max) / (2 * V_max)))
            return q_norm + c_ucb * math.sqrt(log_n / an.N)

        return max(node.children, key=_score)

    def _simulate(s: int, node: HistoryNode, b: np.ndarray) -> float:
        node.N += 1
        a = _select_action(node)
        an = node.children[a]
        an.N += 1
        r = float(model.R[s, a])
        if node.depth == H - 1:                        # terminal (last reward step)
            backup(node, b, model, uncertainty, H, abort_action, lp_cache)
            if ucb_mode == "nominal":
                an.Q_nom += (r - an.Q_nom) / an.N
            return r

        s_next = _sample(model.T[s, a, :])
        z = _sample(model.O[s_next, :])
        an.S_next.add(s_next)
        child, is_new = an.get_or_create_child(z, node.depth + 1, ctr)
        child.S_in.add(s_next)
        b_child = projected_bayes(b, model, a, z, sorted(node.S_in), sorted(child.S_in))

        if is_new:
            backup(child, b_child, model, uncertainty, H, abort_action, lp_cache)
            G = _rollout(s_next, child.depth) if (ucb_mode == "nominal" and use_rollouts) else 0.0
        else:
            G = _simulate(s_next, child, b_child)

        backup(node, b, model, uncertainty, H, abort_action, lp_cache)
        if ucb_mode == "nominal":
            an.Q_nom += ((r + G) - an.Q_nom) / an.N
        return r + G

    root = HistoryNode(ctr.next_id(), depth=0)
    a_star: int | None = None
    sims_used = 0
    for sim in range(1, budget + 1):
        s = _sample(b0)
        root.S_in.add(s)
        _simulate(s, root, restrict(b0, sorted(root.S_in)))
        sims_used = sim
        if early_stop:
            a_cert = certified_best_action(root, b0, H, R_max, n_actions)
            if a_cert is not None:
                a_star = a_cert
                break

    certified = a_star is not None
    if a_star is None:
        a_star = max(root.children, key=lambda a: root.children[a].Q_rob)
    return PlanResult(action=a_star, certified=certified, sims_used=sims_used, root=root)
