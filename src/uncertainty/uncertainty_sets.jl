"""
Uncertainty set types for robust POMDPs.

Defines the uncertainty sets P_T(s,a) and P_Z(s) from the paper (Eq. 1-2):
  P_T(s,a) = {P_T(·|s,a) : D(P_T || P_T^o) ≤ ρ_T(s,a)}
  P_Z(s)   = {P_Z(·|s)   : D(P_Z || P_Z^o) ≤ ρ_Z(s)}

The distance metric D is pluggable (generic interface).
"""

# --- Distance Metrics ---

"""Abstract type for distance metrics used to define uncertainty sets."""
abstract type DistanceMetric end

"""
    TVDistance

Total Variation distance: D(P || Q) = ‖P - Q‖₁ = Σ |P(i) - Q(i)|.
Leads to LP-based optimization problems.
"""
struct TVDistance <: DistanceMetric end

# --- Uncertainty Sets ---

"""
    UncertaintySets

Defines the uncertainty around a nominal POMDP model.

Fields:
- `ρ_T`: transition uncertainty radii, ρ_T[s, a] for each (state, action)
- `ρ_Z`: observation uncertainty radii, ρ_Z[s_next] for each state
- `metric`: distance metric defining the shape of the uncertainty set
"""
struct UncertaintySets
    ρ_T::Array{Float64, 2}   # (n_states, n_actions)
    ρ_Z::Vector{Float64}     # (n_states,)
    metric::DistanceMetric
end

"""Return the transition uncertainty radius ρ_T(s, a)."""
transition_radius(u::UncertaintySets, s::Int, a::Int) = u.ρ_T[s, a]

"""Return the observation uncertainty radius ρ_Z(s_next)."""
observation_radius(u::UncertaintySets, s_next::Int) = u.ρ_Z[s_next]

"""
    uniform_uncertainty(n_states, n_actions, ρ_T, ρ_Z, metric)

Create uncertainty sets with uniform radii (same ρ_T for all (s,a), same ρ_Z for all s).
Convenient for experiments where uncertainty is homogeneous.
"""
function uniform_uncertainty(n_states::Int, n_actions::Int, ρ_T::Float64, ρ_Z::Float64, metric::DistanceMetric=TVDistance())
    return UncertaintySets(
        fill(ρ_T, n_states, n_actions),
        fill(ρ_Z, n_states),
        metric
    )
end
