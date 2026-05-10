"""
Scenario abstraction — bundles per-toy plumbing for the experiment harness.

A Scenario is everything that's specific to a benchmark problem:
  - the nominal model the planner uses
  - how to perturb the model to build the true world (eta_obs, eta_trans)
  - which per-trial diagnostics to extract from each trajectory
  - where to write results

Pulling these out of `sweep.py` lets that file stay scenario-agnostic, so
adding a new benchmark only needs a new `make_<name>_scenario()` factory
in `experiments/<name>/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from robust_pomdp.core.pomdp_types import TabularPOMDP
from robust_pomdp.evaluation.episode import EpisodeStep
from robust_pomdp.uncertainty.uncertainty_sets import UncertaintySets


PerturbWorldFn = Callable[[TabularPOMDP, float, float], TabularPOMDP]
MetricsFn = Callable[[list[EpisodeStep]], dict[str, float]]
UncertaintyFactoryFn = Callable[[float, float], UncertaintySets]


# Repo root, derived from this file's location:
# scenario.py lives at <repo>/src/robust_pomdp/evaluation/scenario.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def default_results_dir(name: str) -> Path:
    """Standard layout: <repo>/experiments/<name>/results/."""
    return _REPO_ROOT / "experiments" / name / "results"


@dataclass
class Scenario:
    """Per-toy bundle consumed by `run_sweep` / `run_config`.

    Attributes:
        name: short identifier (e.g. "tiger", "frozenlake"). Used for the default
            results directory by `default_results_dir(name)`.
        nominal_model: the planner's view of the world.
        perturb_world: (model_nominal, eta_obs, eta_trans) -> model_true.
            Returning the nominal model unchanged is valid when both etas are 0.
        metrics_fn: trajectory -> dict of per-trial metric values. Keys must
            match `metric_columns`.
        metric_columns: column names appended to the CSV schema, in order.
        results_dir: where this scenario's CSVs and figures live.
        uncertainty_factory: (rho_T, rho_Z) -> UncertaintySets. Builds the
            planner's uncertainty set per config. For uniform-rho scenarios
            this wraps `uniform_uncertainty`; scenarios with localized
            uncertainty (e.g., Tamir-style hole-adjacent T radii in FrozenLake)
            provide a custom factory.
    """
    name: str
    nominal_model: TabularPOMDP
    perturb_world: PerturbWorldFn
    metrics_fn: MetricsFn
    metric_columns: list[str]
    results_dir: Path
    uncertainty_factory: UncertaintyFactoryFn
