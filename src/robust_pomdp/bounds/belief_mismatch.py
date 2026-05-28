"""
Belief-mismatch term Delta_b for paper Eq. 74 (Phase 6 sub-step 3).

Closed form:
    Delta_b = ||b_full - b_proj_padded||_1 = 2 * (1 - m_in)

where:
    m_in = Sum_{s in S_in} b_full(s)
    b_proj_padded(s) = b_full(s) / m_in   for s in S_in    (renormalized)
    b_proj_padded(s) = 0                   for s not in S_in

Derivation (two independent pieces, each equal to 1 - m_in):
  Piece 1 -- renormalization gap inside S_in.
    For s in S_in:  |b_full(s) - b_full(s)/m_in| = b_full(s) * (1 - m_in)/m_in
    Sum over S_in:  m_in * (1 - m_in)/m_in = (1 - m_in)
  Piece 2 -- dropped mass outside S_in.
    For s not in S_in:  |b_full(s) - 0| = b_full(s)
    Sum over (S \ S_in):  1 - m_in

Total: 2 * (1 - m_in). The "2" comes from the projection contributing leakage
in BOTH directions: it inflates the kept states (piece 1) AND drops the unkept
ones (piece 2).

TV convention: codebase uses ||P-Q||_1 (no 1/2 factor), so this matches DeltaK
and the leakage path -- no half factor.
"""

from __future__ import annotations

import numpy as np


def delta_b(b0_full: np.ndarray, S_in: list[int]) -> float:
    """Closed-form Eq. 74 belief mismatch: 2 * (1 - m_in)."""
    b = np.asarray(b0_full, dtype=np.float64)
    m_in = float(sum(b[s] for s in S_in))
    return 2.0 * (1.0 - m_in)
