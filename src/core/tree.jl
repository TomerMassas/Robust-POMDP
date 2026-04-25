"""
Search tree data structures for online POMDP planning.

The tree alternates between HistoryNode (belief nodes) and ActionNode (action edges):
  HistoryNode → (pick action) → ActionNode → (observe) → HistoryNode → ...

Each HistoryNode stores particles defining the belief and S_in (sampled state support).
Each ActionNode stores Q-values and Z_in (sampled observation support).

Every node gets a unique integer `id` at creation time (via `next_node_id!`).
Events emitted during simulation reference nodes by these ids.
"""

# --- Node ID allocator ---

"""Mutable counter producing unique sequential node ids. Reset per planner call."""
mutable struct NodeIdCounter
    next::Int
end

NodeIdCounter() = NodeIdCounter(0)

"""Return a fresh id and advance the counter."""
function next_node_id!(counter::NodeIdCounter)
    counter.next += 1
    return counter.next
end

# --- ActionNode ---

"""
    ActionNode

Represents an action edge in the search tree.

Fields:
- `id`: unique integer id, assigned at creation
- `children`: observation → child HistoryNode
- `Z_in`: set of observations seen (sampled observation support)
- `N`: visit count
- `Q_nominal`: average return from simulations (standard POMCP value)
- `Q_robust`: robust value from worst-case backup (computed by RobustBackup)
- `dirty`: whether this node needs robust value recomputation
"""
mutable struct ActionNode
    id::Int
    children::Dict{Int, Any}    # obs → HistoryNode (Any to handle forward reference)
    Z_in::Set{Int}
    N::Int
    Q_nominal::Float64
    Q_robust::Float64
    dirty::Bool
end

function ActionNode(id::Int)
    return ActionNode(id, Dict{Int, Any}(), Set{Int}(), 0, 0.0, 0.0, true)
end

# --- HistoryNode ---

"""
    HistoryNode

Represents a belief node in the search tree.

Fields:
- `particles`: list of states sampled at this node (may contain duplicates)
- `S_in`: set of unique states (sampled state support)
- `children`: action → ActionNode
- `N`: visit count
- `depth`: depth in tree (root = 0)
- `dirty`: whether this node needs robust value recomputation
"""
mutable struct HistoryNode
    id::Int
    particles::Vector{Int}
    S_in::Set{Int}
    children::Dict{Int, ActionNode}
    N::Int
    depth::Int
    dirty::Bool
    V_robust::Float64       # cached max_a Q_robust(a), updated during robust backup
    V_nominal::Float64      # cached max_a Q_nominal(a), updated during simulation
    V_rollout::Float64      # rollout return saved when this node is first expanded
    has_rollout::Bool       # whether V_rollout has been set
end

function HistoryNode(id::Int, depth::Int)
    return HistoryNode(id, Int[], Set{Int}(), Dict{Int, ActionNode}(), 0, depth, true, 0.0, 0.0, 0.0, false)
end

# Fix the forward reference: ActionNode children actually hold HistoryNodes
# (Julia handles this fine since we used Any in ActionNode.children)

# --- Mutating helpers ---

"""Add a particle (sampled state) to a history node. Marks node as dirty."""
function add_particle!(node::HistoryNode, s::Int)
    push!(node.particles, s)
    push!(node.S_in, s)
    node.N += 1
    node.dirty = true
end

"""
Record a new observation at an action node. Creates child if needed. Marks as dirty.
Returns (child_node, was_created) so callers can log which nodes were new.
"""
function get_or_create_child!(action_node::ActionNode, z::Int, child_depth::Int,
                              id_counter::NodeIdCounter)
    push!(action_node.Z_in, z)
    was_created = false
    if !haskey(action_node.children, z)
        new_id = next_node_id!(id_counter)
        action_node.children[z] = HistoryNode(new_id, child_depth)
        action_node.dirty = true
        was_created = true
    end
    return (action_node.children[z]::HistoryNode, was_created)
end

"""
Expand a history node: create ActionNodes for all actions.
Returns (created_ids, created_actions) where both vectors align:
  created_ids[i] is the new ActionNode's id
  created_actions[i] is the action index (1..n_actions) that node corresponds to
"""
function expand!(node::HistoryNode, n_actions::Int, id_counter::NodeIdCounter)
    created_ids = Int[]
    created_actions = Int[]
    for a in 1:n_actions
        if !haskey(node.children, a)
            new_id = next_node_id!(id_counter)
            node.children[a] = ActionNode(new_id)
            push!(created_ids, new_id)
            push!(created_actions, a)
        end
    end
    return (created_ids, created_actions)
end

"""Check if a history node is a leaf (not yet expanded)."""
is_leaf(node::HistoryNode) = isempty(node.children)

"""
Update the cached V_robust from children's Q_robust values.

Only considers actions that have been visited (N > 0). If no action has been
visited, falls back to V_rollout (the sampled rollout return at this node).
If V_rollout hasn't been set either, falls back to 0.
"""
function update_robust_value!(node::HistoryNode)
    if is_leaf(node)
        node.V_robust = node.has_rollout ? node.V_rollout : 0.0
        return
    end

    visited_qs = [an.Q_robust for (_, an) in node.children if an.N > 0]

    if isempty(visited_qs)
        node.V_robust = node.has_rollout ? node.V_rollout : 0.0
    else
        node.V_robust = maximum(visited_qs)
    end
end

"""
Update the cached V_nominal from children's Q_nominal values.

Same logic as update_robust_value!: only visited actions count; V_rollout is the fallback.
"""
function update_nominal_value!(node::HistoryNode)
    if is_leaf(node)
        node.V_nominal = node.has_rollout ? node.V_rollout : 0.0
        return
    end

    visited_qs = [an.Q_nominal for (_, an) in node.children if an.N > 0]

    if isempty(visited_qs)
        node.V_nominal = node.has_rollout ? node.V_rollout : 0.0
    else
        node.V_nominal = maximum(visited_qs)
    end
end
