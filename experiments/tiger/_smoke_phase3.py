# experiments/tiger/_smoke_phase3.py
#
# Phase 3 sanity check.
# Synthesizes a SolverLog with one event of each major type, exports it to JSON,
# and verifies the JSON structure is correct.
#
# Run:
#   python experiments/tiger/_smoke_phase3.py

from __future__ import annotations

import json
import sys
from pathlib import Path

from robust_pomdp.debug.logger import (
    ObsLPResult,
    SolverLog,
    TransLPResult,
    action_id,
    history_id,
    num_events,
)
from robust_pomdp.debug.json_export import export_log_to_json

OUT = Path(__file__).parent / "_smoke_phase3_output.json"

failures = 0


def check(cond: bool, msg: str) -> None:
    global failures
    if not cond:
        print(f"FAIL: {msg}")
        failures += 1


# ---------------------------------------------------------------------------
# 1. Build a log with one event of each type
# ---------------------------------------------------------------------------

log = SolverLog()

log.emit_sim_start(sim_index=0, initial_state=0, root_node_id=1)
log.emit_sim_step(sim_index=0, step_in_sim=0,
                  active_history_id=1, depth=0, state=0,
                  ucb_qs=[-50.0, -30.0, -10.0], ucb_ns=[0, 0, 0],
                  action_chosen=2, action_node_id=2,
                  s_next=0, obs=0, reward=-1.0,
                  next_history_id=3,
                  created_action_nodes=[2, 4, 6],
                  created_history_nodes=[3]
                  )
log.emit_expand(sim_index=0, node_id=3, depth=1,
                created_action_nodes=[7, 8, 9],
                created_action_indices=[0, 1, 2]
                )
log.emit_rollout(sim_index=0, leaf_node_id=3,
                 start_state=0, start_depth=1,
                 rollout_steps=[{"action": 2, "state": 0, "reward": -1.0}],
                 total_return=-5.0
                 )
log.emit_backup_nominal_step(sim_index=0, action_node_id=2,
                             previous_Q_nominal=0.0, new_Q_nominal=-6.0,
                             previous_N=0, new_N=1,
                             total_return_from_here=-6.0
                             )
log.emit_sim_end(sim_index=0, total_return=-6.0)
log.emit_robust_backup(backup_index=0,
                       history_node_id=1, action_node_id=2,
                       node_depth=0, action=2,
                       S_in=[0, 1], Z_in=[0, 1],
                       belief=[0.5, 0.5],
                       V_children=[-5.0, -5.0],
                       obs_lp_results=[
                           ObsLPResult(s_next=0, nominal_p=[0.85, 0.15], radius=0.1,
                                       optimal_p=[0.75, 0.25], objective_value=-4.0
                                       ),
                       ],
                       trans_lp_results=[
                           TransLPResult(state=0, nominal_p=[1.0], radius=0.0,
                                         optimal_p=[1.0], objective_value=-5.0
                                         ),
                       ],
                       previous_Q_robust=0.0,
                       new_Q_robust=-97.0
                       )

check(num_events(log) == 7, f"expected 7 events, got {num_events(log)}")

# ---------------------------------------------------------------------------
# 2. Export to JSON
# ---------------------------------------------------------------------------

export_log_to_json(log, OUT, metadata={"problem": "tiger_smoke", "budget": 1})

check(OUT.exists(), "JSON file was not created")

with OUT.open() as f:
    data = json.load(f)

# Structural checks
check("metadata" in data, "missing 'metadata' key")
check("events" in data, "missing 'events' key")
check(data["metadata"]["n_events"] == 7, f"metadata.n_events wrong: {data['metadata'].get('n_events')}")
check("timestamp" in data["metadata"], "missing metadata.timestamp")
check(data["metadata"]["problem"] == "tiger_smoke", "metadata.problem not passed through")

# Check each event type is present with correct fields
by_type = {ev["type"]: ev for ev in data["events"]}

check("sim_start" in by_type, "missing sim_start event")
check(by_type["sim_start"]["active_node"] == "h1", "sim_start.active_node wrong")
check(by_type["sim_start"]["initial_state"] == 0, "sim_start.initial_state wrong")

check("sim_step" in by_type, "missing sim_step event")
ss = by_type["sim_step"]
check(ss["action_chosen"] == 2, "sim_step.action_chosen wrong")
check(ss["action_node_id"] == "a2", "sim_step.action_node_id wrong")
check(ss["next_history_id"] == "h3", "sim_step.next_history_id wrong")
check(ss["created_action_nodes"] == ["a2", "a4", "a6"], "sim_step.created_action_nodes wrong")
check(ss["created_history_nodes"] == ["h3"], "sim_step.created_history_nodes wrong")

check("expand" in by_type, "missing expand event")
ex = by_type["expand"]
check(ex["node_id"] == "h3", "expand.node_id wrong")
check(ex["created_action_indices"] == [0, 1, 2], "expand.created_action_indices wrong")

check("rollout" in by_type, "missing rollout event")
check(by_type["rollout"]["leaf_node_id"] == "h3", "rollout.leaf_node_id wrong")

check("backup_nominal_step" in by_type, "missing backup_nominal_step event")
bns = by_type["backup_nominal_step"]
check(bns["action_node_id"] == "a2", "backup_nominal_step.action_node_id wrong")
check(abs(bns["new_Q_nominal"] - (-6.0)) < 1e-9, "backup_nominal_step.new_Q_nominal wrong")

check("sim_end" in by_type, "missing sim_end event")

check("robust_backup" in by_type, "missing robust_backup event")
rb = by_type["robust_backup"]
check(rb["history_node_id"] == "h1", "robust_backup.history_node_id wrong")
check(rb["action_node_id"] == "a2", "robust_backup.action_node_id wrong")
check(len(rb["obs_lp_results"]) == 1, "robust_backup.obs_lp_results length wrong")
check("s_next" in rb["obs_lp_results"][0], "obs_lp_result missing s_next")
check("state" in rb["trans_lp_results"][0], "trans_lp_result missing state")
check(abs(rb["new_Q_robust"] - (-97.0)) < 1e-9, "robust_backup.new_Q_robust wrong")

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

if failures == 0:
    print(f"Phase 3 smoke: PASS  ({num_events(log)} events, JSON written to {OUT})")
    OUT.unlink()  # clean up
else:
    print(f"Phase 3 smoke: FAIL  ({failures} check(s) failed)")
    sys.exit(1)
