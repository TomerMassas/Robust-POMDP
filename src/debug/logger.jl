"""
Event-based solver log.

The solver emits events in chronological order, each event carrying all the
information needed to:
  1. Describe what happened (for human-readable display)
  2. Update the UI's view of the tree (which nodes exist, their current fields)
  3. Highlight the active node at this moment

Event types are dispatched via a String `type` field (serialized cleanly to JSON).
All node references use string ids of the form "h{id}" or "a{id}".

Zero overhead when logger is `nothing`.
"""

# --- Node ID helpers ---

"Format a history node id for references in events: h{id}."
history_id(n::Int) = "h$n"

"Format an action node id for references in events: a{id}."
action_id(n::Int) = "a$n"

# --- LP result records (used inside robust_backup events) ---

struct ObsLPResult
    s_next::Int
    nominal_p::Vector{Float64}
    radius::Float64
    optimal_p::Vector{Float64}
    objective_value::Float64
end

struct TransLPResult
    state::Int
    nominal_p::Vector{Float64}
    radius::Float64
    optimal_p::Vector{Float64}
    objective_value::Float64
end

# --- Event record ---

"""
A single event in the solver log. Fields:
  type: one of :sim_start, :sim_step, :expand, :rollout,
        :backup_nominal_step, :sim_end, :robust_backup
  data: Dict of event-specific fields
"""
struct Event
    type::Symbol
    data::Dict{String, Any}
end

# --- SolverLog ---

"""
A SolverLog is a flat, chronological list of events.

Create with `SolverLog()` and pass to `robust_pomcp_plan(..., logger=log)`.
After the run, `log.events` is the full trace in execution order.
"""
mutable struct SolverLog
    events::Vector{Event}
end

SolverLog() = SolverLog(Event[])

num_events(log::SolverLog) = length(log.events)

# --- Emission helpers (called by the solver; one per event type) ---

function emit_sim_start!(log::SolverLog, sim_index::Int, initial_state::Int,
                         root_node_id::Int)
    push!(log.events, Event(:sim_start, Dict{String, Any}(
        "sim_index"     => sim_index,
        "initial_state" => initial_state,
        "active_node"   => history_id(root_node_id),
    )))
end

function emit_sim_step!(log::SolverLog; sim_index::Int, step_in_sim::Int,
                        active_history_id::Int, depth::Int, state::Int,
                        ucb_qs::Vector{Float64}, ucb_ns::Vector{Int},
                        action_chosen::Int, action_node_id::Int,
                        s_next::Int, obs::Int, reward::Float64,
                        next_history_id::Int,
                        created_action_nodes::Vector{Int},
                        created_history_nodes::Vector{Int})
    push!(log.events, Event(:sim_step, Dict{String, Any}(
        "sim_index"            => sim_index,
        "step_in_sim"          => step_in_sim,
        "active_node"          => history_id(active_history_id),
        "depth"                => depth,
        "state"                => state,
        "ucb_qs"               => ucb_qs,
        "ucb_ns"               => ucb_ns,
        "action_chosen"        => action_chosen,
        "action_node_id"       => action_id(action_node_id),
        "s_next"               => s_next,
        "obs"                  => obs,
        "reward"               => reward,
        "next_history_id"      => history_id(next_history_id),
        "created_action_nodes" => [action_id(i) for i in created_action_nodes],
        "created_history_nodes"=> [history_id(i) for i in created_history_nodes],
    )))
end

function emit_expand!(log::SolverLog; sim_index::Int, node_id::Int, depth::Int,
                      created_action_nodes::Vector{Int},
                      created_action_indices::Vector{Int})
    # created_action_indices: for each created action node, its action index (1..n_actions)
    push!(log.events, Event(:expand, Dict{String, Any}(
        "sim_index"              => sim_index,
        "node_id"                => history_id(node_id),
        "depth"                  => depth,
        "created_action_nodes"   => [action_id(i) for i in created_action_nodes],
        "created_action_indices" => created_action_indices,
    )))
end

function emit_rollout!(log::SolverLog; sim_index::Int, leaf_node_id::Int,
                       start_state::Int, start_depth::Int,
                       rollout_steps::Vector{Dict{String, Any}},
                       total_return::Float64)
    push!(log.events, Event(:rollout, Dict{String, Any}(
        "sim_index"     => sim_index,
        "leaf_node_id"  => history_id(leaf_node_id),
        "start_state"   => start_state,
        "start_depth"   => start_depth,
        "rollout_steps" => rollout_steps,
        "total_return"  => total_return,
    )))
end

function emit_backup_nominal_step!(log::SolverLog; sim_index::Int,
                                   action_node_id::Int,
                                   previous_Q_nominal::Float64,
                                   new_Q_nominal::Float64,
                                   previous_N::Int, new_N::Int,
                                   total_return_from_here::Float64)
    push!(log.events, Event(:backup_nominal_step, Dict{String, Any}(
        "sim_index"              => sim_index,
        "action_node_id"         => action_id(action_node_id),
        "previous_Q_nominal"     => previous_Q_nominal,
        "new_Q_nominal"          => new_Q_nominal,
        "previous_N"             => previous_N,
        "new_N"                  => new_N,
        "total_return_from_here" => total_return_from_here,
    )))
end

function emit_sim_end!(log::SolverLog; sim_index::Int, total_return::Float64)
    push!(log.events, Event(:sim_end, Dict{String, Any}(
        "sim_index"    => sim_index,
        "total_return" => total_return,
    )))
end

function emit_robust_backup!(log::SolverLog; backup_index::Int,
                             history_node_id::Int, action_node_id::Int,
                             node_depth::Int, action::Int,
                             S_in::Vector{Int}, Z_in::Vector{Int},
                             belief::Vector{Float64},
                             V_children::Vector{Float64},
                             obs_lp_results::Vector{ObsLPResult},
                             trans_lp_results::Vector{TransLPResult},
                             previous_Q_robust::Float64,
                             new_Q_robust::Float64)
    push!(log.events, Event(:robust_backup, Dict{String, Any}(
        "backup_index"       => backup_index,
        "history_node_id"    => history_id(history_node_id),
        "action_node_id"     => action_id(action_node_id),
        "node_depth"         => node_depth,
        "action"             => action,
        "S_in"               => S_in,
        "Z_in"               => Z_in,
        "belief"             => belief,
        "V_children"         => V_children,
        "obs_lp_results"     => [_lp_obs_dict(r) for r in obs_lp_results],
        "trans_lp_results"   => [_lp_trans_dict(r) for r in trans_lp_results],
        "previous_Q_robust"  => previous_Q_robust,
        "new_Q_robust"       => new_Q_robust,
    )))
end

function _lp_obs_dict(r::ObsLPResult)
    return Dict{String, Any}(
        "s_next"          => r.s_next,
        "nominal_p"       => r.nominal_p,
        "radius"          => r.radius,
        "optimal_p"       => r.optimal_p,
        "objective_value" => r.objective_value,
    )
end

function _lp_trans_dict(r::TransLPResult)
    return Dict{String, Any}(
        "state"           => r.state,
        "nominal_p"       => r.nominal_p,
        "radius"          => r.radius,
        "optimal_p"       => r.optimal_p,
        "objective_value" => r.objective_value,
    )
end
