"""
Experiment sweep infrastructure.

`run_config` runs `n_trials` episodes for one configuration and returns summary
statistics. `run_sweep` runs many configs, writes one CSV per sweep, and
appends a description line to `experiments/results/INDEX.md`.

Two planner backends are supported via the `planner_label` field:
  - "BasicPOMCP" — uses BasicPOMCP.jl + POMDPTools' DiscreteUpdater
  - "robust_pomcp" — uses our `robust_pomcp_plan` + the discrete `bayes_update`
"""

using Random
using Statistics
using Dates
using CSV
using DataFrames
using POMDPs
using POMDPTools
using BasicPOMCP

# --- Config struct ---

"""
    ExperimentConfig

One row in a sweep. The same struct describes both BasicPOMCP and robust runs;
fields not relevant to a backend (e.g., ρ_T for BasicPOMCP) are simply ignored.
"""
Base.@kwdef struct ExperimentConfig
    experiment_id::String       # e.g., "E1", "E3"
    date::String                 # ISO date "2026-04-25"
    description::String          # short, e.g., "rho_z_sweep_perturbed"
    planner_label::String        # "BasicPOMCP" | "robust_pomcp"
    ρ_T::Float64 = 0.0
    ρ_Z::Float64 = 0.0
    ucb_mode::Symbol = :nominal
    budget::Int = 500
    sims_per_backup::Int = 5
    horizon::Int = 5
    η_world_obs::Float64 = 0.0   # obs perturbation magnitude on the TRUE world
    η_world_trans::Float64 = 0.0
    n_trials::Int = 100
    seed_base::Int = 1
    listen_accuracy::Float64 = 0.85
end

# --- Planner factories ---
# Each factory returns a closure: (belief::Vector{Float64}) -> action::Int

"""
Build a `planner_act` closure for our robust solver.
"""
function make_robust_planner(model_planner::TabularPOMDP,
                             uncertainty::UncertaintySets,
                             cfg::ExperimentConfig)
    return belief -> begin
        action, _ = robust_pomcp_plan(belief, model_planner, uncertainty,
                                      cfg.horizon, cfg.budget, cfg.sims_per_backup;
                                      ucb_mode=cfg.ucb_mode,
                                      logger=nothing)
        return action
    end
end

"""
Build a `planner_act` closure for BasicPOMCP via POMDPs.jl.

We construct a POMCPSolver with the same budget/horizon/discount as our robust
planner, plant the wrapper POMDP, and convert the belief vector to a SparseCat
each time it's called.
"""
function make_basic_pomcp_planner(model_planner::TabularPOMDP,
                                  cfg::ExperimentConfig)
    pomdp = TabularPOMDPWrapper(model_planner, 1.0,
                                fill(1.0/model_planner.n_states, model_planner.n_states))
    solver = POMCPSolver(tree_queries=cfg.budget,
                         max_depth=cfg.horizon,
                         c=1.0)
    planner = POMDPs.solve(solver, pomdp)
    return belief -> begin
        b = SparseCat(1:model_planner.n_states, belief)
        return POMDPs.action(planner, b)
    end
end

# --- Belief updaters ---

function make_robust_belief_updater(model_planner::TabularPOMDP)
    return (b, a, o) -> bayes_update(b, model_planner, a, o)
end

function make_basicpomcp_belief_updater(model_planner::TabularPOMDP)
    # Reuse our discrete Bayes filter — it's mathematically equivalent to
    # POMDPTools' DiscreteUpdater and avoids constructing DiscreteBelief objects
    # in the hot path. The planner consumes a Vector{Float64} either way.
    return (b, a, o) -> bayes_update(b, model_planner, a, o)
end

# --- Per-trial trajectory diagnostics ---

"""
Extract action diagnostics from a single trajectory (Tiger-specific bookkeeping).

Returns (n_correct_open, n_wrong_open, n_listen_actions). For Tiger:
- "open-left" = 1, correct iff state was tiger-right (s=2)
- "open-right" = 2, correct iff state was tiger-left (s=1)
- "listen" = 3
"""
function tiger_action_counts(trajectory)
    n_correct = 0
    n_wrong = 0
    n_listen = 0
    for step in trajectory
        if step.action == 1
            (step.state == 2) ? (n_correct += 1) : (n_wrong += 1)
        elseif step.action == 2
            (step.state == 1) ? (n_correct += 1) : (n_wrong += 1)
        elseif step.action == 3
            n_listen += 1
        end
    end
    return (n_correct, n_wrong, n_listen)
end

# --- run_config ---

"""
    run_config(cfg, model_nominal) -> NamedTuple

Run `cfg.n_trials` Tiger episodes for one experiment configuration. Returns
summary statistics suitable for a single CSV row.
"""
function run_config(cfg::ExperimentConfig, model_nominal::TabularPOMDP)
    # Build the *true* world (may be perturbed).
    model_true = build_perturbed_tiger(model_nominal, cfg.η_world_obs;
                                       η_trans=cfg.η_world_trans)

    # Build the planner's uncertainty set (only used by robust planner).
    uncertainty = uniform_uncertainty(
        model_nominal.n_states, model_nominal.n_actions,
        cfg.ρ_T, cfg.ρ_Z, TVDistance()
    )

    # Build the planner_act closure + belief updater.
    if cfg.planner_label == "robust_pomcp"
        planner_act = make_robust_planner(model_nominal, uncertainty, cfg)
        belief_update_fn = make_robust_belief_updater(model_nominal)
    elseif cfg.planner_label == "BasicPOMCP"
        planner_act = make_basic_pomcp_planner(model_nominal, cfg)
        belief_update_fn = make_basicpomcp_belief_updater(model_nominal)
    else
        error("Unknown planner_label: $(cfg.planner_label)")
    end

    belief_init = fill(1.0/model_nominal.n_states, model_nominal.n_states)

    returns = Float64[]
    runtimes = Float64[]
    n_correct_total = 0
    n_wrong_total = 0
    n_listen_total = 0

    for trial in 1:cfg.n_trials
        rng = MersenneTwister(cfg.seed_base + trial - 1)
        t0 = time()
        total_reward, trajectory = run_episode(
            planner_act, belief_update_fn, model_true,
            belief_init, cfg.horizon, rng
        )
        elapsed = time() - t0

        push!(returns, total_reward)
        push!(runtimes, elapsed)
        nc, nw, nl = tiger_action_counts(trajectory)
        n_correct_total += nc
        n_wrong_total += nw
        n_listen_total += nl
    end

    n = length(returns)
    mean_r = mean(returns)
    std_r = n > 1 ? std(returns) : 0.0
    sem_r = n > 1 ? std_r / sqrt(n) : 0.0

    return (
        experiment_id   = cfg.experiment_id,
        date            = cfg.date,
        description     = cfg.description,
        planner_label   = cfg.planner_label,
        rho_T           = cfg.ρ_T,
        rho_Z           = cfg.ρ_Z,
        ucb_mode        = String(cfg.ucb_mode),
        budget          = cfg.budget,
        sims_per_backup = cfg.sims_per_backup,
        horizon         = cfg.horizon,
        eta_world_obs   = cfg.η_world_obs,
        eta_world_trans = cfg.η_world_trans,
        n_trials        = cfg.n_trials,
        seed_base       = cfg.seed_base,
        mean_return     = mean_r,
        std_return      = std_r,
        sem_return      = sem_r,
        mean_runtime_s  = mean(runtimes),
        n_correct_open  = n_correct_total,
        n_wrong_open    = n_wrong_total,
        n_listen_actions = n_listen_total,
    )
end

# --- run_sweep + CSV/INDEX writing ---

const RESULTS_DIR = joinpath(@__DIR__, "..", "..", "experiments", "results")
const INDEX_PATH = joinpath(RESULTS_DIR, "INDEX.md")

"""
Compute the canonical CSV filename for a sweep:
  {Ek}_{date}_{description}.csv
"""
function sweep_filename(experiment_id::String, date::String, description::String)
    return "$(experiment_id)_$(date)_$(description).csv"
end

"""
Append a one-line description for this sweep to INDEX.md (creating the file
if needed). Idempotent: skips if the same filename is already listed.
"""
function append_to_index!(filename::String, oneliner::String)
    mkpath(RESULTS_DIR)
    if !isfile(INDEX_PATH)
        open(INDEX_PATH, "w") do io
            println(io, "# Experiment results index")
            println(io)
        end
    end
    existing = read(INDEX_PATH, String)
    if occursin(filename, existing)
        return  # already indexed
    end
    open(INDEX_PATH, "a") do io
        println(io, "- `$filename` — $oneliner")
    end
end

"""
    run_sweep(experiment_id, date, description, configs, model_nominal;
              index_oneliner=description) -> DataFrame

Run a list of `ExperimentConfig`s, each contributing one row, write the result
to a CSV under `experiments/results/`, and append an INDEX.md entry.
"""
function run_sweep(experiment_id::String, date::String, description::String,
                   configs::Vector{ExperimentConfig},
                   model_nominal::TabularPOMDP;
                   index_oneliner::Union{String, Nothing}=nothing)
    rows = NamedTuple[]
    for (i, cfg) in enumerate(configs)
        @info "[$experiment_id] config $i / $(length(configs)): $(cfg.planner_label) ρ_Z=$(cfg.ρ_Z) η_obs=$(cfg.η_world_obs)"
        push!(rows, run_config(cfg, model_nominal))
    end
    df = DataFrame(rows)

    mkpath(RESULTS_DIR)
    filename = sweep_filename(experiment_id, date, description)
    csv_path = joinpath(RESULTS_DIR, filename)
    CSV.write(csv_path, df)

    oneliner = index_oneliner === nothing ? description : index_oneliner
    append_to_index!(filename, oneliner)

    @info "Sweep written: $csv_path"
    return df
end
