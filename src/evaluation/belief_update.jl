"""
Discrete Bayes belief update for tabular POMDPs.

Used by the episode harness to refresh the agent's belief between successive
planner calls, given the action taken and the observation seen.

Formula:
  b'(s') ∝ O(s', z) · Σ_s T(s, a, s') · b(s)

If the denominator is zero (z is impossible under (b, a)), we fall back to the
prior over s' weighted by the transition kernel — this should not happen with
any well-formed POMDP but guards against numerical edge cases.
"""

"""
    bayes_update(belief, model, action, obs) -> new_belief

Bayes filter step. `belief` is a probability vector over states; returns a new
probability vector over states.
"""
function bayes_update(belief::Vector{Float64}, model::TabularPOMDP,
                      action::Int, obs::Int)
    n = model.n_states
    new_belief = Vector{Float64}(undef, n)
    @inbounds for s_next in 1:n
        pred = 0.0
        for s in 1:n
            pred += model.T[s, action, s_next] * belief[s]
        end
        new_belief[s_next] = model.O[s_next, obs] * pred
    end
    Z = sum(new_belief)
    if Z > 0.0
        new_belief ./= Z
    else
        # Pathological: observation impossible under (b, a). Fall back to predicted prior.
        @inbounds for s_next in 1:n
            pred = 0.0
            for s in 1:n
                pred += model.T[s, action, s_next] * belief[s]
            end
            new_belief[s_next] = pred
        end
        Zp = sum(new_belief)
        new_belief ./= (Zp > 0 ? Zp : 1.0)
    end
    return new_belief
end
