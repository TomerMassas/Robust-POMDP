"""
Fixed history-conditional policy for A1.

Depth 0 → mandatory `inspect`. Otherwise the action is keyed on the
last-observation warning level (warning = last_obs / (M-1)):
  warning <  low  → proceed
  warning <  high → mitigate
  else            → abort

Defaults (low=0.4, high=0.75) match the A1 design PDF.
"""

from __future__ import annotations

from synthetic_pomdp import INSPECT, PROCEED, MITIGATE, ABORT


def warning_threshold_policy(depth: int,
                             last_obs: int | None,
                             M: int,
                             *,
                             low: float = 0.4,
                             high: float = 0.75
                             ) -> int:
    """PDF rule: depth-0 inspect, then by warning level of `last_obs`."""
    if depth == 0:
        return INSPECT
    if last_obs is None:
        raise ValueError("last_obs must be provided when depth > 0")
    warning = last_obs / (M - 1)
    if warning < low:
        return PROCEED
    if warning < high:
        return MITIGATE
    return ABORT
