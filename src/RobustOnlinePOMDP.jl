module RobustOnlinePOMDP

# Core POMDP types and tree
include("core/pomdp_types.jl")
include("core/tree.jl")

# Uncertainty sets and projected set LP
include("uncertainty/uncertainty_sets.jl")
include("uncertainty/projected_sets.jl")

# Debug / observability (must be included before solvers so they can use it)
include("debug/logger.jl")

# Solvers
include("solvers/robust_pomcp.jl")

# JSON export (included last — it serializes everything above)
include("debug/json_export.jl")

# Evaluation / experiment harness (depends on solver + POMDPs.jl ecosystem)
include("evaluation/belief_update.jl")
include("evaluation/perturbation.jl")
include("evaluation/episode.jl")
include("evaluation/pomdps_adapter.jl")
include("evaluation/sweep.jl")

# Public API
export TabularPOMDP, UncertaintySets, TVDistance, DistanceMetric
export uniform_uncertainty
export robust_pomcp_plan
export SolverLog, num_events
export ObsLPResult, TransLPResult
export export_log_to_json

# Evaluation / experiments
export bayes_update
export EpisodeStep, run_episode
export perturb_observation_model, perturb_transition_model, build_perturbed_tiger
export TabularPOMDPWrapper
export ExperimentConfig, run_config, run_sweep, sweep_filename

end # module
