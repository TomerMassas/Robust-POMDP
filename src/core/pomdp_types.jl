"""
Core POMDP types for tabular (discrete, finite) POMDPs.

A TabularPOMDP stores explicit transition, observation, and reward matrices.
This is what the robust solver operates on — it needs explicit model parameters
(not just a generative sampler) to optimize over uncertainty sets.
"""

using Distributions: Categorical

"""
    TabularPOMDP

A fully specified tabular POMDP with discrete states, actions, and observations.

Fields:
- `n_states`: number of states
- `n_actions`: number of actions
- `n_obs`: number of observations
- `T`: transition matrix, T[s, a, s_next] = P_T^o(s_next | s, a)
- `O`: observation matrix, O[s_next, z] = P_Z^o(z | s_next)
- `R`: reward matrix, R[s, a]
"""
struct TabularPOMDP
    n_states::Int
    n_actions::Int
    n_obs::Int
    T::Array{Float64, 3}   # (n_states, n_actions, n_states)
    O::Array{Float64, 2}   # (n_states, n_obs)
    R::Array{Float64, 2}   # (n_states, n_actions)
end

function TabularPOMDP(T::Array{Float64, 3}, O::Array{Float64, 2}, R::Array{Float64, 2})
    n_states, n_actions, n_states2 = size(T)
    n_states3, n_obs = size(O)
    n_states4, n_actions2 = size(R)

    @assert n_states == n_states2 == n_states3 == n_states4 "State dimensions must match"
    @assert n_actions == n_actions2 "Action dimensions must match"

    return TabularPOMDP(n_states, n_actions, n_obs, T, O, R)
end

# --- Query functions ---

"""Return P_T^o(s_next | s, a)."""
transition_prob(m::TabularPOMDP, s::Int, a::Int, s_next::Int) = m.T[s, a, s_next]

"""Return the full transition distribution P_T^o(· | s, a) as a vector over states."""
transition_dist(m::TabularPOMDP, s::Int, a::Int) = @view m.T[s, a, :]

"""Return P_Z^o(z | s_next)."""
observation_prob(m::TabularPOMDP, s_next::Int, z::Int) = m.O[s_next, z]

"""Return the full observation distribution P_Z^o(· | s_next) as a vector over observations."""
observation_dist(m::TabularPOMDP, s_next::Int) = @view m.O[s_next, :]

"""Return R(s, a)."""
reward(m::TabularPOMDP, s::Int, a::Int) = m.R[s, a]

# --- Sampling (for POMCP simulation under nominal model) ---

"""Sample next state: s_next ~ P_T^o(· | s, a)."""
function sample_transition(m::TabularPOMDP, s::Int, a::Int)
    return rand(Categorical(transition_dist(m, s, a)))
end

"""Sample observation: z ~ P_Z^o(· | s_next)."""
function sample_observation(m::TabularPOMDP, s_next::Int)
    return rand(Categorical(observation_dist(m, s_next)))
end
