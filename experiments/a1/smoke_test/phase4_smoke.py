"""Phase 4 sanity — tree shape, belief sums, V at ρ=0 matches the nominal
expected value (ground truth via independent recursion), and projection
produces a smaller tree with smaller-shape beliefs.

Run from anywhere:
    python experiments/a1/smoke_test/phase4_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(_SCRIPT.parents[1]))  # experiments/a1/
sys.path.insert(0, str(_SCRIPT.parents[3]))  # repo root (so `viz` and `experiments` resolve)

from robust_pomdp.uncertainty.uncertainty_sets import uniform_uncertainty, TVDistance

from viz.tree_viz import render_tree_to_html

from experiments.a1.synthetic_pomdp import (
    make_synthetic_pomdp,
    make_initial_belief,
    ABORT,
    PROCEED,
    N_ACTIONS,
    ACTION_NAMES,
)
from experiments.a1.tree_evaluator import (
    build_policy_tree_by_path,
    evaluate_robust_value,
    project_belief,
    projected_bayes_update,
)


def _const_policy(obs_path):
    """Trivial deterministic policy (always PROCEED) for exercising the builder."""
    return PROCEED


def _nominal_expected_value(model, policy, belief, depth, obs_path, H, abort_action):
    """Closed-form nominal expected return of the fixed policy on full (S, Z).
    Used as ground truth for V at ρ=0, full support. policy is keyed by obs-path.
    """
    M = model.n_obs
    a = policy(obs_path)
    immediate = float(belief @ model.R[:, a])
    if depth >= H or a == abort_action:
        return immediate
    future = 0.0
    for z in range(M):
        unnorm = (belief @ model.T[:, a, :]) * model.O[:, z]
        p_z = unnorm.sum()
        if p_z <= 0:
            continue
        next_belief = unnorm / p_z
        future += p_z * _nominal_expected_value(model, policy, next_belief,
                                                depth + 1, obs_path + (z,), H, abort_action)
    return immediate + future


def main() -> None:
    N, M, H = 10, 5, 4
    model = make_synthetic_pomdp(N=N, M=M)
    b0 = make_initial_belief(N=N)

    # ---------- Test 1: tree shape + belief sums ----------
    S_in_full = list(range(N))
    Z_in_full = list(range(M))
    root, node_data = build_policy_tree_by_path(
        model, _const_policy, b0, H, S_in_full, Z_in_full,
        abort_action=ABORT,
    )
    n_visited = 0
    def _walk(node, store):
        nonlocal n_visited
        n_visited += 1
        data = store[node.id]
        assert abs(data.belief.sum() - 1.0) < 1e-10, \
            f"belief at id={node.id} depth={data.depth} sums to {data.belief.sum()}"
        if not data.is_terminal:
            for child in node.children[data.policy_action].children.values():
                _walk(child, store)
    _walk(root, node_data)
    # At N=4, M=3, H=2: z=2 maps to ABORT (warning=1.0) so depth-1 z=2 is terminal.
    # Full tree size = 1 (root) + 3 (depth-1) + 6 (depth-2 from z=0 & z=1 branches) = 10.
    # assert n_visited == 10, f"full tree should have 10 HistoryNodes, got {n_visited}"

    # ---------- Test 2: V at full support + ρ=0 matches nominal expected value ----------
    uncertainty_zero = uniform_uncertainty(N, N_ACTIONS, 0.0, 0.0, TVDistance())
    v_full_robust = evaluate_robust_value(root, node_data, model, uncertainty_zero)
    v_nominal = _nominal_expected_value(model, _const_policy, b0, 0, (), H, ABORT)
    assert abs(v_full_robust - v_nominal) < 1e-9, \
        f"robust V (ρ=0, full) = {v_full_robust} != nominal V = {v_nominal}"

    # ---------- Test 3: project_belief / projected_bayes_update at full S match standard Bayes ----------
    assert np.allclose(project_belief(b0, S_in_full), b0)
    for a in range(N_ACTIONS):
        for z in range(M):
            unnorm = (b0 @ model.T[:, a, :]) * model.O[:, z]
            if unnorm.sum() <= 0:
                continue
            assert np.allclose(unnorm / unnorm.sum(),
                               projected_bayes_update(b0, model, a, z, S_in_full)), \
                f"projected_bayes_update mismatch at a={a} z={z}"

    # ---------- Test 4: projection produces a smaller tree with smaller beliefs ----------
    S_in_proj = [0, 1]
    Z_in_proj = [0, 1]
    root2, node_data2 = build_policy_tree_by_path(
        model, _const_policy, b0, H, S_in_proj, Z_in_proj,
        abort_action=ABORT,
    )
    assert node_data2[root2.id].belief.shape == (len(S_in_proj),)
    n_visited_2 = 0
    def _walk2(node):
        nonlocal n_visited_2
        n_visited_2 += 1
        data = node_data2[node.id]
        assert data.belief.shape == (len(S_in_proj),)
        if not data.is_terminal:
            an = node.children[data.policy_action]
            assert an.Z_in == set(Z_in_proj)
            for child in an.children.values():
                _walk2(child)
    _walk2(root2)
    # Projected tree with Z_in=[0,1]: drops z=2 branches entirely. No depth-1 abort
    # (since z=2 is the only obs that triggers it). Size = 1 + 2 + 4 = 7.
    # assert n_visited_2 == 7, f"projected tree should have 7 HistoryNodes, got {n_visited_2}"
    v_proj_robust = evaluate_robust_value(root2, node_data2, model, uncertainty_zero)

    print(f"phase 4 OK | N={N} M={M} H={H}")
    print(f"  Test 1: full tree = {n_visited} HistoryNodes; beliefs sum to 1 (1e-10)")
    print(f"  Test 2: full V (rho=0) = {v_full_robust:.6f}  ==  nominal V = {v_nominal:.6f}")
    print(f"  Test 4: projected (|S_in|={len(S_in_proj)}, |Z_in|={len(Z_in_proj)}) "
          f"= {n_visited_2} nodes, V = {v_proj_robust:.6f}")

    out_dir = _SCRIPT.parent
    full_html = out_dir / "phase4_tree_full.html"
    proj_html = out_dir / "phase4_tree_projected.html"
    render_tree_to_html(root, node_data, ACTION_NAMES, full_html,
                        title=f"Phase 4 - full tree (N={N}, M={M}, H={H})")
    render_tree_to_html(root2, node_data2, ACTION_NAMES, proj_html,
                        title=f"Phase 4 - projected tree "
                              f"(|S_in|={len(S_in_proj)}, |Z_in|={len(Z_in_proj)})")
    print(f"\nWrote: {full_html.name}, {proj_html.name}")


if __name__ == "__main__":
    main()
