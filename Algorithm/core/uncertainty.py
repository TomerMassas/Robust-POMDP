"""
Uncertainty set types for robust POMDPs.

DUPLICATED verbatim from src/robust_pomdp/uncertainty/uncertainty_sets.py
(logic unchanged) so Algorithm/ is self-contained.

  P_T(s,a) = {P_T(.|s,a) : D(P_T || P_T^o) <= rho_T(s,a)}
  P_Z(s)   = {P_Z(.|s)   : D(P_Z || P_Z^o) <= rho_Z(s)}

TV here is L1 (no 1/2 factor): D(P, Q) = sum |P - Q|, so `rho` is 2x the
textbook TV radius (see project_context.md).
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

import numpy as np


class DistanceMetric(ABC):
    """Abstract base for distance metrics defining uncertainty sets."""


class TVDistance(DistanceMetric):
    """Total Variation distance: D(P || Q) = ||P - Q||_1 = sum |P(i) - Q(i)|."""


@dataclass
class UncertaintySets:
    """Uncertainty around a nominal POMDP model.

    Attributes:
        rho_T: transition uncertainty radii, shape (n_states, n_actions).
        rho_Z: observation uncertainty radii, shape (n_states,).
        metric: distance metric defining the shape of the uncertainty set.
    """

    rho_T: np.ndarray
    rho_Z: np.ndarray
    metric: DistanceMetric

    def transition_radius(self, s: int, a: int) -> float:
        """Return rho_T(s, a)."""
        return float(self.rho_T[s, a])

    def observation_radius(self, s_next: int) -> float:
        """Return rho_Z(s_next)."""
        return float(self.rho_Z[s_next])


def uniform_uncertainty(
    n_states: int,
    n_actions: int,
    rho_T: float,
    rho_Z: float,
    metric: DistanceMetric | None = None,
) -> UncertaintySets:
    """Create UncertaintySets with uniform radii (same rho_T for all (s,a), same rho_Z for all s)."""
    if metric is None:
        metric = TVDistance()
    return UncertaintySets(
        rho_T=np.full((n_states, n_actions), rho_T),
        rho_Z=np.full(n_states, rho_Z),
        metric=metric,
    )
