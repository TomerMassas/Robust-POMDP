"""
CR-UCT planner (M4) on the NEW core.

Checks:
  A. runs in both ucb_modes, returns a valid action, expands the root.
  B. determinism — same seed -> same action and same per-action Q_rob.
  C. bracket VALIDITY (the theorem, rigorously): for each expanded root action a,
     extract the tree-greedy policy "take a then greedy", extend it to a total
     policy, evaluate on FULL support (eval_fixed_policy), and assert the value
     lands inside [Q_rob(a) - eps_a, Q_rob(a) + eps_a]. Holds for ANY total
     extension, so a fixed fallback is fine.
  D. eps of the best action shrinks as the budget grows.
  E. high budget + small rho -> returns the full-solver's best action.
  F. certified_best_action logic (unit): separation -> a*, overlap/partial -> None.

Run:  python -m pytest Algorithm/tests/test_planner.py -q
"""

import numpy as np

from Algorithm.core.tree import ActionNode, HistoryNode
from Algorithm.core.uncertainty import uniform_uncertainty
from Algorithm.domains.synthetic import ABORT, make_initial_belief, make_synthetic_pomdp
from Algorithm.experiments.post_hoc_guarantees.fixed_policy_tree import eval_fixed_policy
from Algorithm.experiments.post_hoc_guarantees.full_solver import solve_full
from Algorithm.ucb.certificate import action_bracket, certified_best_action, r_max
from Algorithm.ucb.planner import plan


def _tree_greedy_policy(root: HistoryNode, root_action: int) -> dict:
    """{obs_path -> action} for 'take root_action at root, then argmax Q_rob below'."""
    pi = {(): root_action}
    an = root.children.get(root_action)
    if an is not None:
        def walk(node, obs_path):
            if not node.children:                      # frontier/terminal: no expanded action
                return
            best = max(node.children, key=lambda a: node.children[a].Q_rob)
            pi[obs_path] = best
            for z, child in node.children[best].children.items():
                walk(child, obs_path + (z,))
        for z, child in an.children.items():
            walk(child, (z,))
    return pi


def test_planner_runs_both_modes():
    N, M, H = 5, 3, 3
    model, b0 = make_synthetic_pomdp(N, M), make_initial_belief(N)
    unc = uniform_uncertainty(N, model.n_actions, 0.05, 0.05)
    for mode in ("robust", "nominal"):
        res = plan(model, unc, b0, H, budget=120, ucb_mode=mode, use_rollouts=(mode == "nominal"),
                   abort_action=ABORT, rng=np.random.default_rng(0))
        assert 0 <= res.action < model.n_actions
        assert len(res.root.children) == model.n_actions        # all root actions expanded
        assert res.sims_used == 120 or res.certified


def test_planner_determinism():
    N, M, H = 5, 3, 3
    model, b0 = make_synthetic_pomdp(N, M), make_initial_belief(N)
    unc = uniform_uncertainty(N, model.n_actions, 0.1, 0.1)
    r1 = plan(model, unc, b0, H, budget=150, abort_action=ABORT, rng=np.random.default_rng(3))
    r2 = plan(model, unc, b0, H, budget=150, abort_action=ABORT, rng=np.random.default_rng(3))
    assert r1.action == r2.action and r1.sims_used == r2.sims_used
    for a in r1.root.children:
        assert abs(r1.root.children[a].Q_rob - r2.root.children[a].Q_rob) < 1e-12


def test_bracket_validity():
    """[Q_rob(a) +/- eps_a] must contain the tree-policy's true full-support value."""
    N, M, H = 5, 3, 3
    model, b0 = make_synthetic_pomdp(N, M), make_initial_belief(N)
    Rmax = r_max(model)
    S_all, Z_all = list(range(N)), list(range(M))
    for rho in (0.0, 0.05, 0.15):
        unc = uniform_uncertainty(N, model.n_actions, rho, rho)
        res = plan(model, unc, b0, H, budget=250, early_stop=False,
                   abort_action=ABORT, rng=np.random.default_rng(1))
        for a, an in res.root.children.items():
            L, U, _ = action_bracket(an, b0, sorted(res.root.S_in), H, Rmax)
            pi = _tree_greedy_policy(res.root, a)
            policy = (lambda op, _p=pi: _p.get(op, 0))           # fallback = INSPECT(0)
            v = eval_fixed_policy(model, unc, policy, b0, H, S_all, Z_all, ABORT, {}).V_rob
            assert L - 1e-6 <= v <= U + 1e-6, (rho, a, L, v, U)


def test_eps_shrinks_with_budget():
    N, M, H = 5, 3, 3
    model, b0 = make_synthetic_pomdp(N, M), make_initial_belief(N)
    Rmax = r_max(model)

    def mean_best_eps(budget):
        eps_l = []
        for seed in range(5):
            res = plan(model, unc, b0, H, budget=budget, early_stop=False,
                       abort_action=ABORT, rng=np.random.default_rng(100 + seed))
            best = max(res.root.children, key=lambda a: res.root.children[a].Q_rob)
            eps_l.append(action_bracket(res.root.children[best], b0, sorted(res.root.S_in),
                                        H, Rmax)[2])
        return float(np.mean(eps_l))

    unc = uniform_uncertainty(N, model.n_actions, 0.05, 0.05)
    assert mean_best_eps(400) < mean_best_eps(40)


def test_returns_full_solver_best_at_high_budget():
    N, M, H = 5, 3, 3
    model, b0 = make_synthetic_pomdp(N, M), make_initial_belief(N)
    unc = uniform_uncertainty(N, model.n_actions, 0.05, 0.05)
    _, q_root, _ = solve_full(model, unc, b0, H, ABORT, {})
    best_a = max(q_root, key=q_root.get)
    res = plan(model, unc, b0, H, budget=3000, early_stop=False,
               abort_action=ABORT, rng=np.random.default_rng(0))
    assert res.action == best_a, (res.action, best_a, q_root)


def test_certified_best_action_unit():
    root = HistoryNode(0, 0)
    root.S_in = {0}
    b0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    an0, an1 = ActionNode(1), ActionNode(2)
    root.children = {0: an0, 1: an1}
    # c = H = 3 -> eps = R_max*(3 - 1*3) = 0; tight brackets, clear separation.
    an0.Q_rob, an0.c = 10.0, {0: 3.0}
    an1.Q_rob, an1.c = 5.0, {0: 3.0}
    assert certified_best_action(root, b0, 3, 10.0, 2) == 0
    assert certified_best_action(root, b0, 3, 10.0, 3) is None      # not all expanded
    # c = 2 -> eps = 10*(3-2) = 10; brackets overlap -> not certifiable.
    an0.c, an1.c = {0: 2.0}, {0: 2.0}
    assert certified_best_action(root, b0, 3, 10.0, 2) is None
