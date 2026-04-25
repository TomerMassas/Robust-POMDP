"""
Solver A: Robust-Value POMCP

Builds a search tree via Monte Carlo simulations (nominal model),
then computes robust values via worst-case Bellman backups using
projected uncertainty sets.

Tunable parameters:
  sims_per_backup: number of simulations between successive robust backups.
    sims_per_backup == budget is two-phase; sims_per_backup == 1 is fully interleaved.
  ucb_mode: :nominal (optimistic) or :robust (pessimistic) exploration

Observability:
  Pass a SolverLog to `logger` and the solver emits fine-grained events
  (sim_start, sim_step, expand, rollout, backup_nominal_step, sim_end,
  robust_backup) in chronological order for later replay.
"""

# --- Top-level planner ---

"""
    robust_pomcp_plan(belief, model, uncertainty, horizon, budget, sims_per_backup;
                      ucb_mode, exploration_constant, logger)

Plan a single action using Robust-Value POMCP.

`sims_per_backup` controls how many simulations run between successive robust
backups. `sims_per_backup == budget` is two-phase; `sims_per_backup == 1` is
fully interleaved.

Returns: (best_action, root)
"""
function robust_pomcp_plan(belief::Vector{Float64}, model::TabularPOMDP,
                           uncertainty::UncertaintySets, horizon::Int,
                           budget::Int, sims_per_backup::Int;
                           ucb_mode::Symbol=:nominal,
                           exploration_constant::Float64=1.0,
                           logger::Union{SolverLog, Nothing}=nothing)

    id_counter = NodeIdCounter()
    root_id = next_node_id!(id_counter)
    root = HistoryNode(root_id, 0)

    # State machinery for robust_backup indexing across the run
    backup_counter = Ref(0)

    for batch in 1:ceil(Int, budget / sims_per_backup)
        n_sims_this_batch = min(sims_per_backup, budget - (batch - 1) * sims_per_backup)

        for _ in 1:n_sims_this_batch
            sim_index = _next_sim_index(logger)
            s = sample_from_belief(belief)

            if logger !== nothing
                emit_sim_start!(logger, sim_index, s, root.id)
            end

            step_counter = Ref(0)
            total_ret = simulate!(s, root, horizon, model, ucb_mode,
                                  exploration_constant, id_counter,
                                  logger, sim_index, step_counter)

            if logger !== nothing
                emit_sim_end!(logger; sim_index=sim_index, total_return=total_ret)
            end
        end

        robust_backup!(root, model, uncertainty, logger, backup_counter)
    end

    # Pick action with highest Q_robust
    best_action = 0
    best_value = -Inf
    for (a, action_node) in root.children
        if action_node.Q_robust > best_value
            best_value = action_node.Q_robust
            best_action = a
        end
    end

    return (best_action, root)
end

"""Return the next sim_index by counting sim_start events already emitted."""
function _next_sim_index(logger::Union{SolverLog, Nothing})
    logger === nothing && return 0
    count = 0
    for ev in logger.events
        ev.type === :sim_start && (count += 1)
    end
    return count + 1
end

# --- Simulation ---

"""
Run one POMCP simulation from state s at the given history node.

If a `logger` is present, emits sim_step, expand, rollout, and backup_nominal_step
events in execution order.

Returns: total return from this point downward.
"""
function simulate!(s::Int, 
                   node::HistoryNode, 
                   depth::Int, 
                   model::TabularPOMDP,
                   ucb_mode::Symbol, 
                   c::Float64,
                   id_counter::NodeIdCounter,
                   logger::Union{SolverLog, Nothing},
                   sim_index::Int,
                   step_counter::Ref{Int})
    if depth == 0
        return 0.0
    end

    add_particle!(node, s)

    # Leaf node: expand and rollout
    if is_leaf(node)
        created_ids, created_actions = expand!(node, model.n_actions, id_counter)
        if logger !== nothing
            emit_expand!(logger;
                sim_index=sim_index, node_id=node.id, depth=node.depth,
                created_action_nodes=created_ids,
                created_action_indices=created_actions)
        end

        # Rollout
        rollout_return, rollout_steps = rollout_with_log(s, depth, model)
        node.V_rollout = rollout_return
        node.has_rollout = true

        if logger !== nothing
            emit_rollout!(logger;
                sim_index=sim_index,
                leaf_node_id=node.id,
                start_state=s,
                start_depth=node.depth,
                rollout_steps=rollout_steps,
                total_return=rollout_return)
        end

        return rollout_return
    end

    # Select action via UCB — capture the UCB values we actually saw
    a, ucb_qs, ucb_ns = ucb_select_with_info(node, ucb_mode, c, model.n_actions)

    # Sample next state, observation, reward (under nominal model)
    s_next = sample_transition(model, s, a)
    z = sample_observation(model, s_next)
    r = reward(model, s, a)

    # Get or create action/obs children; track newly-created ids
    action_node = node.children[a]
    child, child_was_created = get_or_create_child!(action_node, z,
                                                    node.depth + 1, id_counter)

    if logger !== nothing
        step_counter[] += 1
        emit_sim_step!(logger;
            sim_index=sim_index,
            step_in_sim=step_counter[],
            active_history_id=node.id,
            depth=node.depth,
            state=s,
            ucb_qs=ucb_qs,
            ucb_ns=ucb_ns,
            action_chosen=a,
            action_node_id=action_node.id,
            s_next=s_next,
            obs=z,
            reward=r,
            next_history_id=child.id,
            created_action_nodes=Int[],
            created_history_nodes=child_was_created ? [child.id] : Int[])
    end

    # Recurse
    future_return = simulate!(s_next, child, depth - 1, model, ucb_mode, c,
                              id_counter, logger, sim_index, step_counter)
    total_return = r + future_return

    # Update Q_nominal (incremental mean) — emit a backup_nominal_step event
    prev_Q = action_node.Q_nominal
    prev_N = action_node.N
    action_node.N += 1
    action_node.Q_nominal += (total_return - action_node.Q_nominal) / action_node.N
    action_node.dirty = true

    if logger !== nothing
        emit_backup_nominal_step!(logger;
            sim_index=sim_index,
            action_node_id=action_node.id,
            previous_Q_nominal=prev_Q,
            new_Q_nominal=action_node.Q_nominal,
            previous_N=prev_N,
            new_N=action_node.N,
            total_return_from_here=total_return)
    end

    update_nominal_value!(node)
    return total_return
end

# --- Rollout ---

"""Random rollout. Returns (total_return, steps) where steps is a list of dicts."""
function rollout_with_log(s::Int, depth::Int, model::TabularPOMDP)
    total_return = 0.0
    current_s = s
    steps = Dict{String, Any}[]

    for _ in 1:depth
        a = rand(1:model.n_actions)
        r = reward(model, current_s, a)
        s_next = sample_transition(model, current_s, a)
        push!(steps, Dict{String, Any}(
            "state"  => current_s,
            "action" => a,
            "reward" => r,
            "s_next" => s_next,
        ))
        total_return += r
        current_s = s_next
    end

    return (total_return, steps)
end

# --- UCB Action Selection ---

"""
UCB selection that also returns the Q values and visit counts used for the
decision (for logging). Action order is 1..n_actions.
"""
function ucb_select_with_info(node::HistoryNode, ucb_mode::Symbol, c::Float64,
                              n_actions::Int)
    ucb_qs = Float64[]
    ucb_ns = Int[]

    for a in 1:n_actions
        an = node.children[a]
        q = (ucb_mode == :robust) ? an.Q_robust : an.Q_nominal
        push!(ucb_qs, q)
        push!(ucb_ns, an.N)
    end

    # Visit unvisited actions first
    for a in 1:n_actions
        if node.children[a].N == 0
            return (a, ucb_qs, ucb_ns)
        end
    end

    # Standard UCB
    log_N = log(node.N)
    best_action = 0
    best_ucb = -Inf
    for a in 1:n_actions
        an = node.children[a]
        q = (ucb_mode == :robust) ? an.Q_robust : an.Q_nominal
        ucb_value = q + c * sqrt(log_N / an.N)
        if ucb_value > best_ucb
            best_ucb = ucb_value
            best_action = a
        end
    end

    return (best_action, ucb_qs, ucb_ns)
end

# --- Robust Backup ---

"""
Bottom-up robust value computation. Only recomputes dirty nodes.

Invariant exploited here: if a node is not dirty, NONE of its descendants are
either. So we short-circuit the entire subtree at the first clean node hit.
This saves significant work when K < N (interleaved backups on large trees).
"""
function robust_backup!(node::HistoryNode, model::TabularPOMDP,
                        uncertainty::UncertaintySets,
                        logger::Union{SolverLog, Nothing},
                        backup_counter::Ref{Int})
    # Short-circuit: clean subtree means no work below this node.
    if !node.dirty
        return
    end

    for (a, action_node) in node.children
        # Skip entire action subtree if the action node itself is clean —
        # its Q_robust is still correct from a previous backup, and its
        # descendants can't be dirty (see invariant above).
        if !action_node.dirty
            continue
        end

        # Recurse into observation children (bottom-up). Individual clean
        # subtrees bail out at their entry via the node.dirty check above.
        for (_, child) in action_node.children
            robust_backup!(child::HistoryNode, model, uncertainty, logger, backup_counter)
        end

        prev_Qr = action_node.Q_robust
        if isempty(action_node.children)
            action_node.Q_robust = action_node.Q_nominal
        else
            action_node.Q_robust = compute_robust_q(node, a, action_node, model,
                                                   uncertainty, logger,
                                                   backup_counter, prev_Qr)
        end
        action_node.dirty = false
    end

    update_robust_value!(node)
    node.dirty = false
end

# --- Core: Robust Q Computation (two-step LP decomposition) ---

function compute_robust_q(h::HistoryNode, a::Int, action_node::ActionNode,
                          model::TabularPOMDP, uncertainty::UncertaintySets,
                          logger::Union{SolverLog, Nothing},
                          backup_counter::Ref{Int},
                          previous_Q_robust::Float64)
    S_in = collect(h.S_in)
    Z_in = collect(action_node.Z_in)

    belief = compute_belief(h.particles, S_in)

    V_children = Float64[
        haskey(action_node.children, z) ?
            (action_node.children[z]::HistoryNode).V_robust : 0.0
        for z in Z_in
    ]

    obs_lp_results = ObsLPResult[]
    trans_lp_results = TransLPResult[]

    # Step 1: observation LP per s'
    w = Vector{Float64}(undef, length(S_in))
    for (i, s_next) in enumerate(S_in)
        p_nominal_obs = Float64[observation_prob(model, s_next, z) for z in Z_in]
        ρ_z = observation_radius(uncertainty, s_next)
        w_val, optimal_p = minimize_over_projected_tv_ball(p_nominal_obs, ρ_z, V_children)
        w[i] = w_val
        if logger !== nothing
            push!(obs_lp_results, ObsLPResult(s_next, p_nominal_obs, ρ_z, optimal_p, w_val))
        end
    end

    # Step 2: transition LP per s
    Q_val = 0.0
    for (i, s) in enumerate(S_in)
        p_nominal_trans = Float64[transition_prob(model, s, a, s_next) for s_next in S_in]
        ρ_t = transition_radius(uncertainty, s, a)
        σ, optimal_p = minimize_over_projected_tv_ball(p_nominal_trans, ρ_t, w)
        Q_val += belief[i] * (reward(model, s, a) + σ)
        if logger !== nothing
            push!(trans_lp_results, TransLPResult(s, p_nominal_trans, ρ_t, optimal_p, σ))
        end
    end

    if logger !== nothing
        backup_counter[] += 1
        emit_robust_backup!(logger;
            backup_index=backup_counter[],
            history_node_id=h.id,
            action_node_id=action_node.id,
            node_depth=h.depth,
            action=a,
            S_in=S_in,
            Z_in=Z_in,
            belief=belief,
            V_children=V_children,
            obs_lp_results=obs_lp_results,
            trans_lp_results=trans_lp_results,
            previous_Q_robust=previous_Q_robust,
            new_Q_robust=Q_val)
    end

    return Q_val
end

# --- Helpers ---

"""Compute empirical belief from particles over S_in. Returns vector aligned with S_in."""
function compute_belief(particles::Vector{Int}, S_in::Vector{Int})
    counts = zeros(Int, length(S_in))
    state_to_idx = Dict(s => i for (i, s) in enumerate(S_in))
    for s in particles
        if haskey(state_to_idx, s)
            counts[state_to_idx[s]] += 1
        end
    end
    total = sum(counts)
    return counts ./ total
end

"""Sample a state from a belief vector."""
function sample_from_belief(belief::Vector{Float64})
    return rand(Categorical(belief))
end

# --- Diagnostics ---

"""Count HistoryNodes and ActionNodes in a tree."""
function tree_size(root::HistoryNode)
    n_history = 1
    n_action = 0
    for (_, action_node) in root.children
        n_action += 1
        for (_, child) in action_node.children
            nh, na = tree_size(child::HistoryNode)
            n_history += nh
            n_action += na
        end
    end
    return (n_history, n_action)
end
