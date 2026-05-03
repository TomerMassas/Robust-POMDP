"""
Event-based solver log.

The solver emits events in chronological order, each event carrying all the
information needed to describe what happened, update the UI's view of the tree,
and highlight the active node at this moment.

Event types are dispatched via a string `type` field (serialized cleanly to JSON).
All node references use string ids of the form "h{id}" or "a{id}".

Zero overhead when the log is None.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Node ID helpers
# ---------------------------------------------------------------------------

def history_id(n: int) -> str:
    return f"h{n}"


def action_id(n: int) -> str:
    return f"a{n}"


# ---------------------------------------------------------------------------
# LP result records (used inside robust_backup events)
# ---------------------------------------------------------------------------

@dataclass
class ObsLPResult:
    s_next: int
    nominal_p: list[float]
    radius: float
    optimal_p: list[float]
    objective_value: float


@dataclass
class TransLPResult:
    state: int
    nominal_p: list[float]
    radius: float
    optimal_p: list[float]
    objective_value: float


# ---------------------------------------------------------------------------
# Event record
# ---------------------------------------------------------------------------

@dataclass
class Event:
    type: str
    data: dict[str, Any]


# ---------------------------------------------------------------------------
# SolverLog
# ---------------------------------------------------------------------------

class SolverLog:
    """Flat, chronological list of events.

    Create with SolverLog() and pass to robust_pomcp_plan(..., logger=log).
    After the run, log.events is the full trace in execution order.
    """

    def __init__(self) -> None:
        self.events: list[Event] = []

    # --- Emission helpers ---

    def emit_sim_start(self, sim_index: int, initial_state: int, root_node_id: int) -> None:
        self.events.append(Event("sim_start", {
            "sim_index":     sim_index,
            "initial_state": initial_state,
            "active_node":   history_id(root_node_id),
        }))

    def emit_sim_step(self, *, sim_index: int, step_in_sim: int,
                      active_history_id: int, depth: int, state: int,
                      ucb_qs: list[float], ucb_ns: list[int],
                      action_chosen: int, action_node_id: int,
                      s_next: int, obs: int, reward: float,
                      next_history_id: int,
                      created_action_nodes: list[int],
                      created_history_nodes: list[int]
                      ) -> None:
        self.events.append(Event("sim_step", {
            "sim_index":             sim_index,
            "step_in_sim":           step_in_sim,
            "active_node":           history_id(active_history_id),
            "depth":                 depth,
            "state":                 state,
            "ucb_qs":                ucb_qs,
            "ucb_ns":                ucb_ns,
            "action_chosen":         action_chosen,
            "action_node_id":        action_id(action_node_id),
            "s_next":                s_next,
            "obs":                   obs,
            "reward":                reward,
            "next_history_id":       history_id(next_history_id),
            "created_action_nodes":  [action_id(i) for i in created_action_nodes],
            "created_history_nodes": [history_id(i) for i in created_history_nodes],
        }))

    def emit_expand(self, *, sim_index: int, node_id: int, depth: int,
                    created_action_nodes: list[int],
                    created_action_indices: list[int]
                    ) -> None:
        self.events.append(Event("expand", {
            "sim_index":              sim_index,
            "node_id":                history_id(node_id),
            "depth":                  depth,
            "created_action_nodes":   [action_id(i) for i in created_action_nodes],
            "created_action_indices": created_action_indices,
        }))

    def emit_rollout(self, *, sim_index: int, leaf_node_id: int,
                     start_state: int, start_depth: int,
                     rollout_steps: list[dict[str, Any]],
                     total_return: float
                     ) -> None:
        self.events.append(Event("rollout", {
            "sim_index":     sim_index,
            "leaf_node_id":  history_id(leaf_node_id),
            "start_state":   start_state,
            "start_depth":   start_depth,
            "rollout_steps": rollout_steps,
            "total_return":  total_return,
        }))

    def emit_backup_nominal_step(self, *, sim_index: int,
                                 action_node_id: int,
                                 previous_Q_nominal: float,
                                 new_Q_nominal: float,
                                 previous_N: int, new_N: int,
                                 total_return_from_here: float
                                 ) -> None:
        self.events.append(Event("backup_nominal_step", {
            "sim_index":              sim_index,
            "action_node_id":         action_id(action_node_id),
            "previous_Q_nominal":     previous_Q_nominal,
            "new_Q_nominal":          new_Q_nominal,
            "previous_N":             previous_N,
            "new_N":                  new_N,
            "total_return_from_here": total_return_from_here,
        }))

    def emit_sim_end(self, *, sim_index: int, total_return: float) -> None:
        self.events.append(Event("sim_end", {
            "sim_index":    sim_index,
            "total_return": total_return,
        }))

    def emit_robust_backup(self, *, backup_index: int,
                           history_node_id: int, action_node_id: int,
                           node_depth: int, action: int,
                           S_in: list[int], Z_in: list[int],
                           belief: list[float],
                           V_children: list[float],
                           obs_lp_results: list[ObsLPResult],
                           trans_lp_results: list[TransLPResult],
                           previous_Q_robust: float,
                           new_Q_robust: float
                           ) -> None:
        self.events.append(Event("robust_backup", {
            "backup_index":      backup_index,
            "history_node_id":   history_id(history_node_id),
            "action_node_id":    action_id(action_node_id),
            "node_depth":        node_depth,
            "action":            action,
            "S_in":              S_in,
            "Z_in":              Z_in,
            "belief":            belief,
            "V_children":        V_children,
            "obs_lp_results":    [_lp_obs_dict(r) for r in obs_lp_results],
            "trans_lp_results":  [_lp_trans_dict(r) for r in trans_lp_results],
            "previous_Q_robust": previous_Q_robust,
            "new_Q_robust":      new_Q_robust,
        }))


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def num_events(log: SolverLog) -> int:
    return len(log.events)


def _lp_obs_dict(r: ObsLPResult) -> dict[str, Any]:
    return {
        "s_next":          r.s_next,
        "nominal_p":       r.nominal_p,
        "radius":          r.radius,
        "optimal_p":       r.optimal_p,
        "objective_value": r.objective_value,
    }


def _lp_trans_dict(r: TransLPResult) -> dict[str, Any]:
    return {
        "state":           r.state,
        "nominal_p":       r.nominal_p,
        "radius":          r.radius,
        "optimal_p":       r.optimal_p,
        "objective_value": r.objective_value,
    }
