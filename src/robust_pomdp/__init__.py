"""Public API for the robust_pomdp package.

Re-exports the public surface from subpackages so callers can write:
    from robust_pomdp import TabularPOMDP, robust_pomcp_plan, ...
"""

from robust_pomdp.core.pomdp_types import TabularPOMDP
from robust_pomdp.core.tree import ActionNode, HistoryNode, NodeIdCounter
from robust_pomdp.uncertainty.uncertainty_sets import (
    DistanceMetric,
    TVDistance,
    UncertaintySets,
    uniform_uncertainty,
)
from robust_pomdp.debug.logger import ObsLPResult, SolverLog, TransLPResult, num_events
from robust_pomdp.debug.json_export import export_log_to_json
from robust_pomdp.solvers.robust_pomcp import (
    compute_belief,
    compute_robust_q,
    robust_pomcp_plan,
    sample_from_belief,
    tree_size,
)
from robust_pomdp.evaluation.belief_update import bayes_update
from robust_pomdp.evaluation.episode import EpisodeStep, run_episode
from robust_pomdp.evaluation.perturbation import (
    build_perturbed_tiger,
    perturb_observation_model,
    perturb_transition_model,
)
from robust_pomdp.evaluation.pomdp_py_adapter import (
    TabularPOMDPWrapper,
    make_basic_pomcp_planner,
)
from robust_pomdp.evaluation.sweep import (
    ExperimentConfig,
    run_config,
    run_sweep,
    sweep_filename,
)

__all__ = [
    # core
    "TabularPOMDP",
    "ActionNode",
    "HistoryNode",
    "NodeIdCounter",
    # uncertainty
    "DistanceMetric",
    "TVDistance",
    "UncertaintySets",
    "uniform_uncertainty",
    # debug
    "ObsLPResult",
    "SolverLog",
    "TransLPResult",
    "num_events",
    "export_log_to_json",
    # solver
    "robust_pomcp_plan",
    "compute_robust_q",
    "compute_belief",
    "sample_from_belief",
    "tree_size",
    # evaluation
    "bayes_update",
    "EpisodeStep",
    "run_episode",
    "perturb_observation_model",
    "perturb_transition_model",
    "build_perturbed_tiger",
    "TabularPOMDPWrapper",
    "make_basic_pomcp_planner",
    "ExperimentConfig",
    "run_config",
    "run_sweep",
    "sweep_filename",
]
