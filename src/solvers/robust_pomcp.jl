"""
Solver A: Robust-Value POMCP

Builds a search tree via Monte Carlo simulations (nominal model),
then computes robust values via worst-case Bellman backups using
projected uncertainty sets.

Tunable parameters:
  K: robust backup frequency (K=N is two-phase, K=1 is fully interleaved)
  ucb_mode: "nominal" (optimistic) or "robust" (pessimistic) exploration
"""

# --- Top-level planner ---

"""
    robust_pomcp_plan(belief, model, uncertainty, horizon, budget, batch_size;
                      ucb_mode, exploration_constant)

Plan a single action from the current belief using Robust-Value POMCP.

Args:
    belief: probability vector over states (length n_states, sums to 1)
    model: TabularPOMDP (nominal model)
    uncertainty: UncertaintySets
    horizon: planning depth H
    budget: total number of simulations N
    batch_size: K — simulations between robust backups

Keyword Args:
    ucb_mode: :nominal or :robust (which Q to use for UCB)
    exploration_constant: c in UCB formula

Returns:
    best_action: the action with highest Q_robust at root
"""
function robust_pomcp_plan(belief::Vector{Float64}, model::TabularPOMDP,
                           uncertainty::UncertaintySets, horizon::Int,
                           budget::Int, batch_size::Int;
                           ucb_mode::Symbol=:nominal,
                           exploration_constant::Float64=1.0)

    root = HistoryNode(0)

    for batch in 1:ceil(Int, budget / batch_size)
        n_sims = min(batch_size, budget - (batch - 1) * batch_size)

        for _ in 1:n_sims
            s = sample_from_belief(belief)
            simulate!(s, root, horizon, model, ucb_mode, exploration_constant)
        end

        robust_backup!(root, model, uncertainty)
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

    return best_action
end

# --- Simulation ---

"""
    simulate!(s, node, depth, model, ucb_mode, c)

Run one POMCP simulation from state s at history node.
Builds tree by sampling from the nominal model.
Updates Q_nominal via incremental mean.

Returns: total return from this point.
"""
function simulate!(s::Int, node::HistoryNode, depth::Int, model::TabularPOMDP,
                   ucb_mode::Symbol, c::Float64)
    if depth == 0
        return 0.0
    end

    add_particle!(node, s)

    # Leaf node: expand and rollout
    if is_leaf(node)
        expand!(node, model.n_actions)
        return rollout(s, depth, model)
    end

    # Select action
    a = ucb_select(node, ucb_mode, c)

    # Sample from nominal model
    s_next = sample_transition(model, s, a)
    z = sample_observation(model, s_next)
    r = reward(model, s, a)

    # Get or create child for this observation
    action_node = node.children[a]
    child = get_or_create_child!(action_node, z, node.depth + 1)

    # Recurse
    future_return = simulate!(s_next, child, depth - 1, model, ucb_mode, c)
    total_return = r + future_return

    # Update Q_nominal (incremental mean)
    action_node.N += 1
    action_node.Q_nominal += (total_return - action_node.Q_nominal) / action_node.N
    action_node.dirty = true

    # Update cached V_nominal on the history node
    update_nominal_value!(node)

    return total_return
end

# --- Rollout ---

"""Random rollout from state s for remaining depth steps under nominal model."""
function rollout(s::Int, depth::Int, model::TabularPOMDP)
    total_return = 0.0
    current_s = s

    for _ in 1:depth
        a = rand(1:model.n_actions)
        total_return += reward(model, current_s, a)
        current_s = sample_transition(model, current_s, a)
    end

    return total_return
end

# --- UCB Action Selection ---

"""
    ucb_select(node, ucb_mode, c)

Select an action using UCB. Unvisited actions are selected first.

ucb_mode:
  :nominal → UCB uses Q_nominal (optimistic exploration)
  :robust  → UCB uses Q_robust when available (pessimistic exploration)
"""
function ucb_select(node::HistoryNode, ucb_mode::Symbol, c::Float64)
    # Visit unvisited actions first
    for (a, action_node) in node.children
        if action_node.N == 0
            return a
        end
    end

    # UCB selection
    log_N = log(node.N)
    best_action = 0
    best_ucb = -Inf

    for (a, action_node) in node.children
        q = (ucb_mode == :robust) ? action_node.Q_robust : action_node.Q_nominal
        ucb_value = q + c * sqrt(log_N / action_node.N)

        if ucb_value > best_ucb
            best_ucb = ucb_value
            best_action = a
        end
    end

    return best_action
end

# --- Robust Backup ---

"""
    robust_backup!(node, model, uncertainty)

Bottom-up robust value computation. Only recomputes dirty nodes.
"""
function robust_backup!(node::HistoryNode, model::TabularPOMDP,
                        uncertainty::UncertaintySets)
    for (a, action_node) in node.children
        # First recurse into children (bottom-up)
        for (z, child) in action_node.children
            robust_backup!(child::HistoryNode, model, uncertainty)
        end

        # Compute Q_robust for this action node if dirty
        if action_node.dirty
            if isempty(action_node.children)
                # Leaf action node: use nominal value from rollouts
                action_node.Q_robust = action_node.Q_nominal
            else
                action_node.Q_robust = compute_robust_q(node, a, action_node, model, uncertainty)
            end
            action_node.dirty = false
        end
    end

    # Update cached V_robust
    update_robust_value!(node)
    node.dirty = false
end

# --- Core: Robust Q Computation (two-step LP decomposition) ---

"""
    compute_robust_q(history_node, action, action_node, model, uncertainty)

Compute Q_robust(h, a) using the two-step LP decomposition:
  Step 1: for each s' ∈ S_in, solve obs LP → w(s')
  Step 2: for each s in belief, solve transition LP using w(s') → σ(s)
  Q_robust = Σ_s b(s) * [R(s,a) + σ(s)]
"""
function compute_robust_q(h::HistoryNode, a::Int, action_node::ActionNode,
                          model::TabularPOMDP, uncertainty::UncertaintySets)
    S_in = collect(h.S_in)
    Z_in = collect(action_node.Z_in)

    # Compute belief from particles
    belief = compute_belief(h.particles, S_in)

    # Collect children robust values: V(z) for each z ∈ Z_in
    V_children = Float64[
        haskey(action_node.children, z) ?
            (action_node.children[z]::HistoryNode).V_robust : 0.0
        for z in Z_in
    ]

    # --- Step 1: for each s' ∈ S_in, solve observation LP → w(s') ---
    w = Vector{Float64}(undef, length(S_in))
    for (i, s_next) in enumerate(S_in)
        # Nominal obs probs restricted to Z_in
        p_nominal_obs = Float64[observation_prob(model, s_next, z) for z in Z_in]
        ρ_z = observation_radius(uncertainty, s_next)

        w[i], _ = minimize_over_projected_tv_ball(p_nominal_obs, ρ_z, V_children)
    end

    # --- Step 2: for each s in belief, solve transition LP → σ(s) ---
    Q_val = 0.0
    for (i, s) in enumerate(S_in)
        # Nominal transition probs restricted to S_in
        p_nominal_trans = Float64[transition_prob(model, s, a, s_next) for s_next in S_in]
        ρ_t = transition_radius(uncertainty, s, a)

        σ, _ = minimize_over_projected_tv_ball(p_nominal_trans, ρ_t, w)

        Q_val += belief[i] * (reward(model, s, a) + σ)
    end

    return Q_val
end

# --- Helper ---

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

"""Sample a state from a belief vector (probability distribution over 1:n_states)."""
function sample_from_belief(belief::Vector{Float64})
    return rand(Categorical(belief))
end
