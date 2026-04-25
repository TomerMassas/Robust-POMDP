"""
World perturbation utilities for robustness experiments.

For Tiger we perturb the observation model by reducing the listen accuracy:
the *true* world's observation distribution is shifted, while the planner
keeps believing the nominal accuracy. This creates the model misspecification
that the robust planner is supposed to handle gracefully.

Transition perturbations are stubbed (Tiger uses ρ_T = 0 in v1 experiments).
"""

"""
    perturb_observation_model(O, η; mode=:reduce_accuracy) -> O_perturbed

Return a new observation matrix with the listen accuracy reduced by `η`.

Tiger convention: `O[1, 1] = O[2, 2] = α` (listen accuracy), and rows sum to 1.
After perturbation, `O[1,1] = O[2,2] = α − η`, and the off-diagonals fill in
to preserve row-sums of 1.

Currently implemented modes:
- `:reduce_accuracy` — symmetric reduction in diagonal entries.
"""
function perturb_observation_model(O::Array{Float64, 2}, η::Float64;
                                   mode::Symbol=:reduce_accuracy)
    if mode === :reduce_accuracy
        @assert size(O) == (2, 2) "reduce_accuracy mode is Tiger-specific (2 states, 2 obs)"
        O_new = copy(O)
        O_new[1, 1] -= η
        O_new[1, 2] += η
        O_new[2, 1] += η
        O_new[2, 2] -= η
        @assert all(0.0 .<= O_new .<= 1.0) "Perturbation η=$η drove probabilities out of [0,1]"
        return O_new
    else
        error("Unsupported perturbation mode: $mode")
    end
end

"""
    perturb_transition_model(T, η; mode=:identity) -> T_perturbed

Stub. Returns `T` unchanged when `mode == :identity`. Implement other modes
when transition uncertainty experiments need them.
"""
function perturb_transition_model(T::Array{Float64, 3}, η::Float64;
                                  mode::Symbol=:identity)
    if mode === :identity || η == 0.0
        return T
    else
        error("Unsupported transition perturbation mode: $mode (only :identity supported)")
    end
end

"""
    build_perturbed_tiger(model_nominal, η_obs; η_trans=0.0)
        -> TabularPOMDP

Convenience: produce a perturbed Tiger world with reduced listen accuracy.
The reward matrix is left untouched.
"""
function build_perturbed_tiger(model_nominal::TabularPOMDP, η_obs::Float64;
                               η_trans::Float64=0.0)
    O_pert = η_obs == 0.0 ? model_nominal.O :
             perturb_observation_model(model_nominal.O, η_obs)
    T_pert = η_trans == 0.0 ? model_nominal.T :
             perturb_transition_model(model_nominal.T, η_trans)
    return TabularPOMDP(T_pert, O_pert, model_nominal.R)
end
