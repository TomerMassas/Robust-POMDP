"""
Core POMDP types for tabular (discrete, finite) POMDPs.

DUPLICATED verbatim from src/robust_pomdp/core/pomdp_types.py (logic unchanged)
so the Algorithm/ codebase is self-contained for later comparison.

A TabularPOMDP stores explicit transition, observation, and reward matrices.
The robust solver needs explicit model parameters (not just a generative
sampler) to optimize over uncertainty sets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TabularPOMDP:
    """A fully specified tabular POMDP with discrete states, actions, observations.

    Attributes:
        n_states: number of states.
        n_actions: number of actions.
        n_obs: number of observations.
        T: transition matrix, T[s, a, s_next] = P_T^o(s_next | s, a). Shape (n_states, n_actions, n_states).
        O: observation matrix, O[s_next, z] = P_Z^o(z | s_next). Shape (n_states, n_obs).
        R: reward matrix, R[s, a]. Shape (n_states, n_actions).

    State, action, and observation ids are 0-based throughout.
    """

    n_states: int
    n_actions: int
    n_obs: int
    T: np.ndarray
    O: np.ndarray
    R: np.ndarray

    @classmethod
    def from_matrices(cls, T: np.ndarray, O: np.ndarray, R: np.ndarray) -> TabularPOMDP:
        """Construct from matrices, inferring dimensions and validating shape consistency."""
        n_states, n_actions, n_states2 = T.shape
        n_states3, n_obs = O.shape
        n_states4, n_actions2 = R.shape

        assert n_states == n_states2 == n_states3 == n_states4, "State dimensions must match"
        assert n_actions == n_actions2, "Action dimensions must match"

        return cls(n_states=n_states, n_actions=n_actions, n_obs=n_obs, T=T, O=O, R=R)

    # --- Query functions ---

    def transition_prob(self, s: int, a: int, s_next: int) -> float:
        """Return P_T^o(s_next | s, a)."""
        return float(self.T[s, a, s_next])

    def transition_dist(self, s: int, a: int) -> np.ndarray:
        """Return P_T^o(. | s, a) as a vector over states."""
        return self.T[s, a, :]

    def observation_prob(self, s_next: int, z: int) -> float:
        """Return P_Z^o(z | s_next)."""
        return float(self.O[s_next, z])

    def observation_dist(self, s_next: int) -> np.ndarray:
        """Return P_Z^o(. | s_next) as a vector over observations."""
        return self.O[s_next, :]

    def reward(self, s: int, a: int) -> float:
        """Return R(s, a)."""
        return float(self.R[s, a])

    # --- Sampling (for MCTS simulation under the nominal model) ---

    def sample_transition(self, s: int, a: int, rng: np.random.Generator) -> int:
        """Sample next state: s_next ~ P_T^o(. | s, a)."""
        return int(rng.choice(self.n_states, p=self.transition_dist(s, a)))

    def sample_observation(self, s_next: int, rng: np.random.Generator) -> int:
        """Sample observation: z ~ P_Z^o(. | s_next)."""
        return int(rng.choice(self.n_obs, p=self.observation_dist(s_next)))
