"""
Build + robustly evaluate a FIXED policy on a chosen support (full or projected).

Built entirely on Algorithm.core + Algorithm.ucb (our robust_q / leakage / backup
/ belief), so it uses the NEW conventions — NOT a copy of a1/tree_evaluator:
  - H reward depths at 0..H-1, terminal at depth H-1;
  - ABORT is terminal (0 continuation);
  - next-states over S_next = children's S_in;
  - scalar-c certificate  eps = R_max * (H - sum_s b0[s]*c[s]).

Uniform-mask projection: every history node uses the same (S_in, Z_in). A
`max_depth` truncation leaves frontier leaves (unexpanded, non-terminal) so the
leakage-tail path is exercised.

Evaluating the SAME policy at full support gives the exact rectangular robust
value V_full^pi (ground truth); at a projected support it gives V_proj^pi with
its certificate eps.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from Algorithm.core.pomdp import TabularPOMDP
from Algorithm.core.tree import HistoryNode, NodeIdCounter
from Algorithm.core.uncertainty import UncertaintySets
from Algorithm.ucb.backup import backup
from Algorithm.ucb.belief import projected_bayes, restrict

Policy = Callable[[tuple], int]   # obs_path (z1, ..., zk) -> action


def eval_fixed_policy(model: TabularPOMDP,
                      uncertainty: UncertaintySets,
                      policy: Policy,
                      b0: np.ndarray,
                      H: int,
                      S_in: list[int],
                      Z_in: list[int],
                      abort_action: int | None = None,
                      lp_cache: dict | None = None,
                      max_depth: int | None = None,
                      ) -> HistoryNode:
    """Build the fixed-policy tree over (S_in, Z_in) and back it up. Returns the root.

    `max_depth` (optional): stop expanding below this depth, leaving frontier
    leaves (used to exercise the leakage tail). None = full depth (to H-1/abort).
    """
    if lp_cache is None:
        lp_cache = {}
    S_in = sorted(S_in)
    Z_in = sorted(Z_in)
    ctr = NodeIdCounter()

    def _rec(depth: int, obs_path: tuple, belief: np.ndarray) -> HistoryNode:
        node = HistoryNode(ctr.next_id(), depth)
        node.S_in = set(S_in)
        a = policy(obs_path)
        terminal = (depth == H - 1) or (abort_action is not None and a == abort_action)
        truncate = (max_depth is not None) and (depth >= max_depth) and (not terminal)
        if not truncate:
            an = node.expand_action(a, ctr)
            if not terminal:
                an.Z_in = set(Z_in)
                an.S_next = set(S_in)           # uniform mask: children share this support
                for z in Z_in:
                    child_belief = projected_bayes(belief, model, a, z, S_in, S_in)
                    an.children[z] = _rec(depth + 1, obs_path + (z,), child_belief)
        backup(node, belief, model, uncertainty, H, abort_action, lp_cache)
        return node

    return _rec(0, (), restrict(b0, S_in))


def r_max(model: TabularPOMDP) -> float:
    """R_max = max(|R.min|, |R.max|), per the paper."""
    return float(max(abs(model.R.min()), abs(model.R.max())))


def certificate_eps(root: HistoryNode, b0: np.ndarray, H: int, R_max: float) -> float:
    """Theorem-1 certificate  eps = R_max * (H - sum_{s in S_in(root)} b0[s]*c[s])."""
    b0 = np.asarray(b0, dtype=np.float64)
    survived = sum(b0[s] * root.c[s] for s in sorted(root.S_in))
    return R_max * (H - survived)
