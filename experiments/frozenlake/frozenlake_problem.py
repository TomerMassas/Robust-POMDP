"""
FrozenLake POMDP problem definition (8x8, gym-standard layout).

Adapts the classic FrozenLake MDP into a partially-observed setting:

  - State: cell on an 8x8 grid (row-major idx 0-63), plus one absorbing
    terminal state (idx 64). |S| = 65.
  - Action: 4 cardinal moves {LEFT=0, DOWN=1, RIGHT=2, UP=3}, gym convention.
  - Observation: 4-bit hole-adjacency reading (one bit per direction
    N/S/E/W: "is the cell in that direction a hole?"). |Z| = 16. Bits flip
    independently with probability `epsilon`. Off-grid edges read as 0
    (wall != hole). Terminal state observation is uniform over the 16.
  - Reward (Tamir-style shaped):
        frozen cell s: r(s) = 1 / (d(s)+1)^3 where d(s) = Manhattan(s, goal)
        goal cell:     r = 1   (earned the step the agent IS in goal)
        hole cell:     r = 0
        terminal:      r = 0
  - Transition: slip mechanic with parameter p — with prob p move intended
    direction, (1-p)/2 each perpendicular, 0 backward. Off-grid moves
    collapse onto staying in place. Holes and goal transition to terminal
    deterministically (any action). Terminal self-loops.

Uncertainty layout:
  - Z (observation): uniform across all cells. World epsilon is
    (epsilon_nominal + eta_obs); planner hedges with rho_Z (uniform).
  - T (transition): localized to HOLE_ADJACENT cells (frozen cells that
    have at least one hole as a 4-neighbor). Outside HOLE_ADJACENT, world
    matches planner exactly. Inside HOLE_ADJACENT, world uses
    (p_nominal - eta_trans) — more slippery near holes. Planner hedges
    with rho_T uniformly within HOLE_ADJACENT, zero elsewhere.

See `make_frozenlake_scenario` for the bundled Scenario.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from robust_pomdp import TabularPOMDP

from robust_pomdp.evaluation.episode import EpisodeStep
from robust_pomdp.evaluation.scenario import Scenario, default_results_dir
from robust_pomdp.uncertainty.uncertainty_sets import (
    TVDistance,
    UncertaintySets,
)


# ---------------------------------------------------------------------------
# Grid layout
# ---------------------------------------------------------------------------

# Gym FrozenLake-v1 standard 8x8 layout.
GRID_LAYOUT = (
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
)

N_ROWS = 8
N_COLS = 8
N_GRID_CELLS = N_ROWS * N_COLS  # 64
TERMINAL_STATE = N_GRID_CELLS    # 64
N_STATES = N_GRID_CELLS + 1      # 65
N_ACTIONS = 4
N_OBS = 16  # 2^4 hole-adjacency bits


# ---------------------------------------------------------------------------
# Actions (gym convention)
# ---------------------------------------------------------------------------

ACTION_LEFT = 0
ACTION_DOWN = 1
ACTION_RIGHT = 2
ACTION_UP = 3

# (drow, dcol) per action
ACTION_DELTAS = [
    (0, -1),  # LEFT
    (1, 0),   # DOWN
    (0, 1),   # RIGHT
    (-1, 0),  # UP
]

# For each action, the two perpendicular actions used for slip.
PERPENDICULARS = {
    ACTION_LEFT:  (ACTION_UP,    ACTION_DOWN),
    ACTION_DOWN:  (ACTION_LEFT,  ACTION_RIGHT),
    ACTION_RIGHT: (ACTION_UP,    ACTION_DOWN),
    ACTION_UP:    (ACTION_LEFT,  ACTION_RIGHT),
}

# Direction order for the 4 obs bits (N, S, E, W).
OBS_DIRECTIONS = [
    (-1, 0),  # N
    (1, 0),   # S
    (0, 1),   # E
    (0, -1),  # W
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cell_to_idx(r: int, c: int) -> int:
    return r * N_COLS + c


def idx_to_cell(idx: int) -> tuple[int, int]:
    return divmod(idx, N_COLS)


def is_in_grid(r: int, c: int) -> bool:
    return 0 <= r < N_ROWS and 0 <= c < N_COLS


def manhattan_to_goal(r: int, c: int) -> int:
    gr, gc = GOAL_CELL
    return abs(r - gr) + abs(c - gc)


def apply_action(r: int, c: int, a: int) -> tuple[int, int]:
    """Move from (r, c) in direction a. Off-grid → stay in place."""
    dr, dc = ACTION_DELTAS[a]
    new_r, new_c = r + dr, c + dc
    if not is_in_grid(new_r, new_c):
        return r, c
    return new_r, new_c


def compute_true_obs_bits(r: int, c: int,
                          hole_set: set[tuple[int, int]] | frozenset[tuple[int, int]],
                          ) -> tuple[int, int, int, int]:
    """Return the 4-tuple of true (N, S, E, W) bits at cell (r, c).

    bit_d = 1 iff the cell in direction d is a hole. Off-grid neighbors → 0.
    """
    bits = []
    for dr, dc in OBS_DIRECTIONS:
        nr, nc = r + dr, c + dc
        if is_in_grid(nr, nc) and (nr, nc) in hole_set:
            bits.append(1)
        else:
            bits.append(0)
    return tuple(bits)  # type: ignore[return-value]


def bits_to_obs_idx(bits: tuple[int, int, int, int]) -> int:
    """Encode (N, S, E, W) bits into integer 0-15."""
    n, s, e, w = bits
    return n + 2 * s + 4 * e + 8 * w


def obs_idx_to_bits(z: int) -> tuple[int, int, int, int]:
    """Decode integer 0-15 into (N, S, E, W) bits."""
    return (z & 1, (z >> 1) & 1, (z >> 2) & 1, (z >> 3) & 1)


# ---------------------------------------------------------------------------
# Layout-derived constants
# ---------------------------------------------------------------------------

def _parse_layout() -> tuple[frozenset[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    """Find holes, start, goal in GRID_LAYOUT. Returns (holes, start, goal)."""
    holes: set[tuple[int, int]] = set()
    start: tuple[int, int] | None = None
    goal: tuple[int, int] | None = None
    for r, row in enumerate(GRID_LAYOUT):
        for c, ch in enumerate(row):
            if ch == "H":
                holes.add((r, c))
            elif ch == "S":
                start = (r, c)
            elif ch == "G":
                goal = (r, c)
    assert start is not None, "No start cell found in GRID_LAYOUT"
    assert goal is not None, "No goal cell found in GRID_LAYOUT"
    return frozenset(holes), start, goal


HOLES, START_CELL, GOAL_CELL = _parse_layout()
HOLE_INDICES: frozenset[int] = frozenset(cell_to_idx(r, c) for r, c in HOLES)
GOAL_INDEX = cell_to_idx(*GOAL_CELL)
START_INDEX = cell_to_idx(*START_CELL)


def _compute_hole_adjacent() -> frozenset[int]:
    """Frozen cells with at least one hole as a 4-neighbor."""
    adj: set[int] = set()
    for r in range(N_ROWS):
        for c in range(N_COLS):
            cell_idx = cell_to_idx(r, c)
            if cell_idx in HOLE_INDICES or cell_idx == GOAL_INDEX:
                continue  # not "frozen cells"
            for dr, dc in OBS_DIRECTIONS:
                nr, nc = r + dr, c + dc
                if is_in_grid(nr, nc) and (nr, nc) in HOLES:
                    adj.add(cell_idx)
                    break
    return frozenset(adj)


HOLE_ADJACENT_INDICES = _compute_hole_adjacent()


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_T(p_at_cell: Callable[[int], float]) -> np.ndarray:
    """Build T[s, a, s'] = P(s' | s, a).

    For frozen cells: slip mechanic with parameter p_at_cell(idx). For holes
    and goal: transition deterministically to terminal. Terminal: self-loop.
    """
    T = np.zeros((N_STATES, N_ACTIONS, N_STATES))

    # Holes and goal → terminal
    for h_idx in HOLE_INDICES:
        T[h_idx, :, TERMINAL_STATE] = 1.0
    T[GOAL_INDEX, :, TERMINAL_STATE] = 1.0

    # Terminal self-loop
    T[TERMINAL_STATE, :, TERMINAL_STATE] = 1.0

    # Frozen cells: slip mechanic
    for r in range(N_ROWS):
        for c in range(N_COLS):
            cell_idx = cell_to_idx(r, c)
            if cell_idx in HOLE_INDICES or cell_idx == GOAL_INDEX:
                continue

            p = p_at_cell(cell_idx)
            slip = (1.0 - p) / 2.0

            for a in range(N_ACTIONS):
                perp_a, perp_b = PERPENDICULARS[a]
                next_intended = apply_action(r, c, a)
                next_perp_a = apply_action(r, c, perp_a)
                next_perp_b = apply_action(r, c, perp_b)

                T[cell_idx, a, cell_to_idx(*next_intended)] += p
                T[cell_idx, a, cell_to_idx(*next_perp_a)] += slip
                T[cell_idx, a, cell_to_idx(*next_perp_b)] += slip

    return T


def build_O(epsilon: float) -> np.ndarray:
    """Build O[s, z] = P(z | s).

    For grid cells: 4 independent Bernoulli bit-flips with parameter epsilon.
    For terminal: uniform over all 16 observations.
    """
    O = np.zeros((N_STATES, N_OBS))

    for r in range(N_ROWS):
        for c in range(N_COLS):
            cell_idx = cell_to_idx(r, c)
            true_bits = compute_true_obs_bits(r, c, HOLES)
            for z in range(N_OBS):
                obs_bits = obs_idx_to_bits(z)
                prob = 1.0
                for tb, ob in zip(true_bits, obs_bits):
                    prob *= (1.0 - epsilon) if ob == tb else epsilon
                O[cell_idx, z] = prob

    O[TERMINAL_STATE, :] = 1.0 / N_OBS

    return O


def build_R() -> np.ndarray:
    """Build R[s, a]. Tamir-style shaped reward.

    Frozen cell s: r(s) = 1 / (d(s)+1)^3 where d(s) = Manhattan(s, goal).
    Goal: r = 1. Hole: r = 0. Terminal: r = 0.
    Reward is per-(s, a) but doesn't depend on a here.
    """
    R = np.zeros((N_STATES, N_ACTIONS))

    for r in range(N_ROWS):
        for c in range(N_COLS):
            cell_idx = cell_to_idx(r, c)
            if cell_idx in HOLE_INDICES:
                R[cell_idx, :] = 0.0
            elif cell_idx == GOAL_INDEX:
                R[cell_idx, :] = 1.0
            else:
                d = manhattan_to_goal(r, c)
                R[cell_idx, :] = 1.0 / (d + 1) ** 3

    R[TERMINAL_STATE, :] = 0.0

    return R


def make_frozenlake_pomdp(epsilon: float = 0.05, p: float = 0.6) -> TabularPOMDP:
    """Build the FrozenLake POMDP as a TabularPOMDP.

    Args:
        epsilon: per-bit sensor flip probability (default 0.05).
        p: probability of moving in the intended direction (default 0.6).

    Returns:
        TabularPOMDP with n_states=65, n_actions=4, n_obs=16.
    """
    T = build_T(lambda idx: p)
    O = build_O(epsilon)
    R = build_R()
    return TabularPOMDP.from_matrices(T, O, R)


# ---------------------------------------------------------------------------
# Scenario plumbing — perturbation, per-trial metrics, scenario factory
# ---------------------------------------------------------------------------

def make_frozenlake_scenario(epsilon: float = 0.05, p: float = 0.6) -> Scenario:
    """Bundle the FrozenLake plumbing into a Scenario for `run_sweep`."""
    nominal = make_frozenlake_pomdp(epsilon=epsilon, p=p)

    def perturb_world(model_nominal_arg: TabularPOMDP,
                      eta_obs: float,
                      eta_trans: float
                      ) -> TabularPOMDP:
        """Apply Z (uniform) and/or T (localized to HOLE_ADJACENT) perturbations."""
        if eta_obs == 0.0 and eta_trans == 0.0:
            return model_nominal_arg

        if eta_obs != 0.0:
            O_pert = build_O(epsilon + eta_obs)
        else:
            O_pert = model_nominal_arg.O

        if eta_trans != 0.0:
            def p_at_cell(idx: int) -> float:
                return (p - eta_trans) if idx in HOLE_ADJACENT_INDICES else p
            T_pert = build_T(p_at_cell)
        else:
            T_pert = model_nominal_arg.T

        return TabularPOMDP.from_matrices(T_pert, O_pert, model_nominal_arg.R)

    def metrics_fn(trajectory: list[EpisodeStep]) -> dict[str, float]:
        """Per-trial diagnostics: did agent fall in a hole / reach goal, # active steps."""
        fell = int(any(s.next_state in HOLE_INDICES for s in trajectory))
        reached = int(any(s.next_state == GOAL_INDEX for s in trajectory))
        n_active = sum(1 for s in trajectory if s.state != TERMINAL_STATE)
        return {"n_fell_in_hole":   fell,
                "n_reached_goal":   reached,
                "n_steps_active":   n_active}

    def uncertainty_factory(rho_T_val: float, rho_Z_val: float) -> UncertaintySets:
        """Localized rho_T over HOLE_ADJACENT; uniform rho_Z."""
        rho_T_arr = np.zeros((N_STATES, N_ACTIONS))
        for idx in HOLE_ADJACENT_INDICES:
            rho_T_arr[idx, :] = rho_T_val
        rho_Z_arr = np.full(N_STATES, rho_Z_val)
        return UncertaintySets(rho_T=rho_T_arr, rho_Z=rho_Z_arr, metric=TVDistance())

    return Scenario(name="frozenlake",
                    nominal_model=nominal,
                    perturb_world=perturb_world,
                    metrics_fn=metrics_fn,
                    metric_columns=["n_fell_in_hole", "n_reached_goal", "n_steps_active"],
                    results_dir=default_results_dir("frozenlake"),
                    uncertainty_factory=uncertainty_factory
                    )
