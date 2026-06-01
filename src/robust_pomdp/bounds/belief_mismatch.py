"""
Belief-mismatch term Delta_b for the projected value bound.

Delta_b is the IN-SUPPORT belief mismatch between the full belief and the
projected belief:
    Delta_b = Sum_{s in S_in} |b_full(s) - b_proj(s)|.

It is computed explicitly against the projected belief the tree actually used
(not hardcoded), so it stays correct under any projected-belief definition:
  - unnormalized restriction (current): b_proj(s) = b_full(s) on S_in  ->  0.
  - renormalized:                        b_proj(s) = b_full(s) / m_in  ->  (1 - m_in)
        (the in-support renormalization gap), where m_in = Sum_{s in S_in} b_full(s).

The out-of-support mass (1 - m_in) is NOT counted here -- it is belief leakage,
carried by the omitted-trajectory term at depth j=0 (1 - phi_0 = 1 - m_in).
This in-support / out-of-support split composes correctly for either definition
without double-counting: Delta_b handles disagreement on the shared support, the
leakage term handles all mass leaving it.

TV convention: codebase uses ||P-Q||_1 (no 1/2 factor).
"""

from __future__ import annotations

import numpy as np


def delta_b(b0_full: np.ndarray, b0_proj: np.ndarray, S_in: list[int]) -> float:
    """In-support belief mismatch  Sum_{s in S_in} |b_full(s) - b_proj(s)|.

    Args:
        b0_full: full-S belief vector (length n_states), indexed by state.
        b0_proj: projected belief on the support, length |S_in|, aligned with
            sorted(S_in) (e.g. node_data[root.id].belief from the tree builder).
        S_in: in-support state indices.
    """
    b = np.asarray(b0_full, dtype=np.float64)
    bp = np.asarray(b0_proj, dtype=np.float64)
    S_in_sorted = sorted(int(s) for s in S_in)
    if bp.shape != (len(S_in_sorted),):
        raise ValueError(f"b0_proj shape {bp.shape} != ({len(S_in_sorted)},)")
    return float(sum(abs(b[s] - bp[i]) for i, s in enumerate(S_in_sorted)))