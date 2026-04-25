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

# Public API
export TabularPOMDP, UncertaintySets, TVDistance, DistanceMetric
export uniform_uncertainty
export robust_pomcp_plan
export SolverLog, num_events
export ObsLPResult, TransLPResult
export export_log_to_json

end # module
