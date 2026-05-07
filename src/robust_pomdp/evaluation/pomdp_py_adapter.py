"""
Adapter from our `TabularPOMDP` to pomdp_py's POMDP interface.

Lets us run pomdp_py's POMCP (under the "BasicPOMCP" planner_label, kept for
CSV schema continuity) on the same tabular model the robust solver consumes —
keeping the comparison apples-to-apples without re-encoding the problem.

States, actions, observations are 0-based integers wrapped in tiny hashable
classes. The adapter pre-builds the four pomdp_py models once; the planner
factory composes them into a fresh Agent + POMCP per call.
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import pomdp_py

from robust_pomdp.core.pomdp_types import TabularPOMDP


# ---------------------------------------------------------------------------
# Hashable int wrappers for state / action / observation
# ---------------------------------------------------------------------------

class IntState(pomdp_py.State):
    def __init__(self, idx: int) -> None:
        self.idx = idx

    def __hash__(self) -> int:
        return hash(("S", self.idx))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, IntState) and self.idx == other.idx

    def __repr__(self) -> str:
        return f"IntState({self.idx})"


class IntAction(pomdp_py.Action):
    def __init__(self, idx: int) -> None:
        self.idx = idx

    def __hash__(self) -> int:
        return hash(("A", self.idx))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, IntAction) and self.idx == other.idx

    def __repr__(self) -> str:
        return f"IntAction({self.idx})"


class IntObservation(pomdp_py.Observation):
    def __init__(self, idx: int) -> None:
        self.idx = idx

    def __hash__(self) -> int:
        return hash(("Z", self.idx))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, IntObservation) and self.idx == other.idx

    def __repr__(self) -> str:
        return f"IntObservation({self.idx})"


# ---------------------------------------------------------------------------
# pomdp_py model implementations backed by a TabularPOMDP
# ---------------------------------------------------------------------------

class _TransitionModel(pomdp_py.TransitionModel):
    def __init__(self, model: TabularPOMDP) -> None:
        self.m = model
        self._all_states = [IntState(s) for s in range(model.n_states)]

    def probability(self, next_state, state, action) -> float:
        return self.m.transition_prob(state.idx, action.idx, next_state.idx)

    def sample(self, state, action):
        probs = self.m.transition_dist(state.idx, action.idx).tolist()
        s_next = random.choices(range(self.m.n_states), weights=probs, k=1)[0]
        return IntState(s_next)

    def get_all_states(self):
        return self._all_states


class _ObservationModel(pomdp_py.ObservationModel):
    def __init__(self, model: TabularPOMDP) -> None:
        self.m = model
        self._all_obs = [IntObservation(z) for z in range(model.n_obs)]

    def probability(self, observation, next_state, action) -> float:
        return self.m.observation_prob(next_state.idx, observation.idx)

    def sample(self, next_state, action):
        probs = self.m.observation_dist(next_state.idx).tolist()
        z = random.choices(range(self.m.n_obs), weights=probs, k=1)[0]
        return IntObservation(z)

    def get_all_observations(self):
        return self._all_obs


class _RewardModel(pomdp_py.RewardModel):
    def __init__(self, model: TabularPOMDP) -> None:
        self.m = model

    def sample(self, state, action, next_state) -> float:
        return self.m.reward(state.idx, action.idx)


class _PolicyModel(pomdp_py.RolloutPolicy):
    def __init__(self, n_actions: int) -> None:
        self.n_actions = n_actions
        self._all_actions = [IntAction(a) for a in range(n_actions)]

    def get_all_actions(self, state=None, history=None):
        return self._all_actions

    def sample(self, state):
        return random.choice(self._all_actions)

    def rollout(self, state, history=None):
        return self.sample(state)


# ---------------------------------------------------------------------------
# Wrapper bundle
# ---------------------------------------------------------------------------

class TabularPOMDPWrapper:
    """Bundle of pomdp_py models constructed once from a TabularPOMDP.

    Reusable across many planner constructions; all models are read-only.
    """

    def __init__(self, model: TabularPOMDP) -> None:
        self.inner = model
        self.transition = _TransitionModel(model)
        self.observation = _ObservationModel(model)
        self.reward = _RewardModel(model)
        self.policy = _PolicyModel(model.n_actions)


# ---------------------------------------------------------------------------
# Planner factory
# ---------------------------------------------------------------------------

def make_basic_pomcp_planner(model_planner: TabularPOMDP,
                             cfg,
                             *,
                             exploration_constant: float = 1.0,
                             n_particles: int | None = None
                             ) -> Callable[[np.ndarray], int]:
    """Closure for pomdp_py's POMCP under the "BasicPOMCP" label.

    pomdp_py.POMCP requires a particle belief (it raises TypeError on
    Histogram). Each call samples `n_particles` states from the input belief
    vector and feeds them to a fresh Agent + POMCP — matching Julia's
    `POMDPs.action(planner, b)` semantics with a fresh belief.

    Args:
        n_particles: number of particles to draw per call. Defaults to
            max(cfg.budget, 200) so the particle distribution is at least as
            fine-grained as the simulation count.
    """
    wrapper = TabularPOMDPWrapper(model_planner)
    n_states = model_planner.n_states
    if n_particles is None:
        n_particles = max(cfg.budget, 200)

    def _plan(belief: np.ndarray) -> int:
        sampled_states = random.choices(
            range(n_states), weights=belief.tolist(), k=n_particles,
        )
        particles = pomdp_py.Particles([IntState(s) for s in sampled_states])
        agent = pomdp_py.Agent(
            particles,
            wrapper.policy,
            wrapper.transition,
            wrapper.observation,
            wrapper.reward,
        )
        planner = pomdp_py.POMCP(
            max_depth=cfg.horizon,
            num_sims=cfg.budget,
            discount_factor=1.0,
            exploration_const=exploration_constant,
            rollout_policy=wrapper.policy,
        )
        action = planner.plan(agent)
        return action.idx

    return _plan


def make_basicpomcp_belief_updater(model_planner: TabularPOMDP
                                   ) -> Callable[[np.ndarray, int, int], np.ndarray]:
    """Reuse our discrete Bayes filter for the BasicPOMCP path.

    Mathematically equivalent to pomdp_py's update on a Histogram and avoids
    constructing Histogram objects on the hot path.
    """
    from robust_pomdp.evaluation.belief_update import bayes_update
    return lambda b, a, o: bayes_update(b, model_planner, a, o)
