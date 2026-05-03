"""
Phase 1 smoke test: types from core/ and uncertainty/ instantiate and basic
methods behave correctly.

Run:
    python experiments/tiger/_smoke_phase1.py

Should print a final "[smoke] Phase 1: all checks pass" line and exit 0.

This is not a §1.3 gate — Phase 1 has no formal gate. It's a sanity check
that the foundation types are wired correctly before the LP layer (Phase 2)
builds on them.
"""

from __future__ import annotations

import numpy as np

from robust_pomdp import (
    ActionNode,
    HistoryNode,
    NodeIdCounter,
    TabularPOMDP,
    TVDistance,
    uniform_uncertainty,
)


def main() -> None:
    # --- TabularPOMDP ---
    n_states, n_actions, n_obs = 2, 3, 2
    T = np.zeros((n_states, n_actions, n_states))
    T[:, 0, :] = 0.5  # action 0: uniform reset
    T[:, 1, :] = 0.5  # action 1: uniform reset
    T[0, 2, 0] = 1.0  # action 2: stay (state 0 -> 0)
    T[1, 2, 1] = 1.0  # action 2: stay (state 1 -> 1)
    O = np.array([[0.85, 0.15], [0.15, 0.85]])
    R = np.array([[-100.0, 10.0, -1.0], [10.0, -100.0, -1.0]])

    m = TabularPOMDP.from_matrices(T, O, R)
    assert m.n_states == n_states
    assert m.n_actions == n_actions
    assert m.n_obs == n_obs
    assert m.transition_prob(0, 2, 0) == 1.0
    assert m.transition_prob(0, 2, 1) == 0.0
    assert m.observation_prob(0, 0) == 0.85
    assert m.reward(0, 0) == -100.0
    assert m.transition_dist(0, 2).shape == (n_states,)
    assert m.observation_dist(0).shape == (n_obs,)

    rng = np.random.default_rng(42)
    s_next = m.sample_transition(0, 2, rng)
    assert s_next == 0  # action 2 from state 0 is deterministic
    z = m.sample_observation(0, rng)
    assert z in (0, 1)
    print("[smoke] TabularPOMDP OK")

    # --- UncertaintySets ---
    u = uniform_uncertainty(n_states, n_actions, rho_T=0.0, rho_Z=0.1)
    assert u.rho_T.shape == (n_states, n_actions)
    assert u.rho_Z.shape == (n_states,)
    assert u.transition_radius(0, 0) == 0.0
    assert u.observation_radius(1) == 0.1
    assert isinstance(u.metric, TVDistance)
    print("[smoke] UncertaintySets OK")

    # --- Tree ---
    counter = NodeIdCounter()
    assert counter.next_id() == 1
    assert counter.next_id() == 2

    root = HistoryNode(id=counter.next_id(), depth=0)
    assert root.id == 3
    assert root.depth == 0
    assert root.is_leaf()
    assert root.dirty
    assert not root.has_rollout

    root.add_particle(0)
    root.add_particle(1)
    root.add_particle(0)
    assert len(root.particles) == 3
    assert root.S_in == {0, 1}
    assert root.N == 3
    assert root.dirty

    created_ids, created_actions = root.expand(n_actions=3, id_counter=counter)
    assert created_actions == [0, 1, 2]
    assert len(created_ids) == 3
    assert not root.is_leaf()

    a0 = root.children[0]
    assert isinstance(a0, ActionNode)
    assert a0.N == 0
    assert a0.dirty

    child, was_created = a0.get_or_create_child(z=0, child_depth=1, id_counter=counter)
    assert was_created
    assert isinstance(child, HistoryNode)
    assert child.depth == 1

    child2, was_created2 = a0.get_or_create_child(z=0, child_depth=1, id_counter=counter)
    assert not was_created2
    assert child2 is child

    # update_robust_value / update_nominal_value: leaf-with-rollout fallback
    leaf = HistoryNode(id=counter.next_id(), depth=2)
    leaf.has_rollout = True
    leaf.V_rollout = -5.0
    leaf.update_robust_value()
    assert leaf.V_robust == -5.0
    leaf.update_nominal_value()
    assert leaf.V_nominal == -5.0

    # update_robust_value: visited children
    parent = HistoryNode(id=counter.next_id(), depth=1)
    parent.expand(n_actions=2, id_counter=counter)
    parent.children[0].N = 5
    parent.children[0].Q_robust = -10.0
    parent.children[0].Q_nominal = -8.0
    parent.children[1].N = 3
    parent.children[1].Q_robust = -2.0
    parent.children[1].Q_nominal = -1.0
    parent.update_robust_value()
    parent.update_nominal_value()
    assert parent.V_robust == -2.0  # max over visited
    assert parent.V_nominal == -1.0
    print("[smoke] tree (NodeIdCounter, HistoryNode, ActionNode) OK")

    print("[smoke] Phase 1: all checks pass")


if __name__ == "__main__":
    main()
