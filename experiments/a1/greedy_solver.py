"""
Full-robust-greedy solver for A1.

Builds a FULL-support expectimax tree (every history branches all actions, all
observations, full models), does a robust backup with max-over-actions, and
exposes:
  - V_full: optimal robust value at the root.
  - pi_full: the optimal robust policy, as a dict {obs_path -> action} along the
    greedy trajectory (solve_full_greedy).
  - q_root: {a: Q_full^rob(h0, a)} for every root action, and pi_by_root_action:
    {a: pi*_a} where pi*_a is the "force a at the root, then act optimally"
    policy keyed by obs_path (solve_with_root_actions). These drive the per-root-
    action Q-certificate experiment.

Robustness reuses compute_robust_q per (history, action). With full support it
uses the corrected full-TV-ball backup. Cost: ~(|A_cont|*|Z|)^H histories.
One-time solve.
"""

from __future__ import annotations

import numpy as np

from robust_pomdp import TabularPOMDP
from robust_pomdp.core.tree import HistoryNode, ActionNode, NodeIdCounter
from robust_pomdp.solvers.robust_pomcp import compute_robust_q
from robust_pomdp.uncertainty.uncertainty_sets import UncertaintySets

from tree_evaluator import project_belief, projected_bayes_update


def _build_expectimax(model, uncertainty, b0_full, H, abort_action, lp_cache):
    """Build the full-support expectimax tree + robust backup with max over
    actions. Returns (root, argmax_action, q_root) where:
      - argmax_action[node.id] = optimal robust action at each history,
      - q_root = {a: Q_full^rob(h0, a)} at the root (per-action robust Q).
    """
    S_full = list(range(model.n_states))
    Z_full = list(range(model.n_obs))
    id_counter = NodeIdCounter()
    argmax_action: dict[int, int] = {}

    def _solve(depth: int, belief: np.ndarray) -> tuple[dict[int, float], HistoryNode]:
        node = HistoryNode(id_counter.next_id(), depth)
        node.S_in = set(S_full)
        q: dict[int, float] = {}
        for a in range(model.n_actions):
            if depth == H or a == abort_action:
                q[a] = float(sum(belief[s] * model.R[s, a] for s in S_full))
            else:
                action_node = ActionNode(id_counter.next_id())
                action_node.Z_in = set(Z_full)
                for z in Z_full:
                    child_belief = projected_bayes_update(belief, model, a, z, S_full)
                    q_child, child = _solve(depth + 1, child_belief)
                    child.V_robust = max(q_child.values())
                    action_node.children[z] = child
                node.children[a] = action_node
                q_val, _ = compute_robust_q(h=node, a=a, action_node=action_node,
                                            model=model, uncertainty=uncertainty,
                                            logger=None, backup_counter=0,
                                            previous_Q_robust=0.0, lp_cache=lp_cache,
                                            belief_override=np.asarray(belief, dtype=np.float64))
                q[a] = q_val
        argmax_action[node.id] = max(q, key=q.get)
        node.V_robust = q[argmax_action[node.id]]
        return q, node

    b0_in = project_belief(np.asarray(b0_full, dtype=np.float64), S_full)
    q_root, root = _solve(0, b0_in)
    return root, argmax_action, q_root


def _extract_policy(root, a, argmax_action) -> dict[tuple[int, ...], int]:
    """The 'force action a at the root, then act optimally' policy, keyed by
    obs_path. pi[()] = a; deeper entries follow argmax along a's subtree. For a
    terminal root action (abort), pi = {(): a} (no continuation)."""
    pi_a: dict[tuple[int, ...], int] = {(): a}
    if a in root.children:
        def _walk(node, obs_path):
            best = argmax_action[node.id]
            pi_a[obs_path] = best
            if best in node.children:
                for z, child in node.children[best].children.items():
                    _walk(child, obs_path + (z,))
        for z, child in root.children[a].children.items():
            _walk(child, (z,))
    return pi_a


def solve_full_greedy(model: TabularPOMDP,
                      uncertainty: UncertaintySets,
                      b0_full: np.ndarray,
                      H: int,
                      *,
                      abort_action: int | None = None,
                      lp_cache: dict | None = None,
                      ) -> tuple[float, dict[tuple[int, ...], int]]:
    """Return (V_full, pi_full): optimal robust value + greedy policy by obs-path."""
    if lp_cache is None:
        lp_cache = {}
    root, argmax_action, q_root = _build_expectimax(model, uncertainty, b0_full, H,
                                                    abort_action, lp_cache)
    best_a = max(q_root, key=q_root.get)
    return q_root[best_a], _extract_policy(root, best_a, argmax_action)


def solve_with_root_actions(model: TabularPOMDP,
                            uncertainty: UncertaintySets,
                            b0_full: np.ndarray,
                            H: int,
                            *,
                            abort_action: int | None = None,
                            lp_cache: dict | None = None,
                            ) -> tuple[float, dict[int, float], dict[int, dict]]:
    """Return (V_full, q_root, pi_by_root_action):
      - V_full = max_a Q_full^rob(h0, a),
      - q_root = {a: Q_full^rob(h0, a)} (true robust Q per root action),
      - pi_by_root_action = {a: pi*_a} -- the 'force a then optimal' policy per a.
    """
    if lp_cache is None:
        lp_cache = {}
    root, argmax_action, q_root = _build_expectimax(model, uncertainty, b0_full, H,
                                                    abort_action, lp_cache)
    v_full = max(q_root.values())
    pi_by_root_action = {a: _extract_policy(root, a, argmax_action) for a in q_root}
    return v_full, q_root, pi_by_root_action