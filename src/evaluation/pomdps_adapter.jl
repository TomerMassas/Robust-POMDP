"""
Adapter from our `TabularPOMDP` to the JuliaPOMDP `POMDPs.POMDP` interface.

This lets us run BasicPOMCP (and POMDPTools' DiscreteUpdater) on the same Tiger
model that our robust solver consumes — keeping the comparison apples-to-apples
without re-encoding the problem.

States, actions, observations are integers 1..N. The wrapper carries the
discount factor and an initial belief (used only when POMDPs.jl asks for
`initialstate`; the experiment harness supplies its own belief at solve-time).
"""

using POMDPs
using POMDPTools

struct TabularPOMDPWrapper <: POMDP{Int, Int, Int}
    inner::TabularPOMDP
    discount_factor::Float64
    initial_belief::Vector{Float64}
end

# --- POMDPs.jl required methods ---

POMDPs.states(m::TabularPOMDPWrapper) = 1:m.inner.n_states
POMDPs.actions(m::TabularPOMDPWrapper) = 1:m.inner.n_actions
POMDPs.observations(m::TabularPOMDPWrapper) = 1:m.inner.n_obs

POMDPs.stateindex(m::TabularPOMDPWrapper, s::Int) = s
POMDPs.actionindex(m::TabularPOMDPWrapper, a::Int) = a
POMDPs.obsindex(m::TabularPOMDPWrapper, o::Int) = o

POMDPs.discount(m::TabularPOMDPWrapper) = m.discount_factor
POMDPs.isterminal(m::TabularPOMDPWrapper, s::Int) = false  # Tiger is horizon-bounded

function POMDPs.transition(m::TabularPOMDPWrapper, s::Int, a::Int)
    return SparseCat(1:m.inner.n_states, m.inner.T[s, a, :])
end

function POMDPs.observation(m::TabularPOMDPWrapper, a::Int, s_next::Int)
    return SparseCat(1:m.inner.n_obs, m.inner.O[s_next, :])
end

POMDPs.reward(m::TabularPOMDPWrapper, s::Int, a::Int) = m.inner.R[s, a]

function POMDPs.initialstate(m::TabularPOMDPWrapper)
    return SparseCat(1:m.inner.n_states, m.initial_belief)
end
