"""
Main experiment runner.

Usage:
    julia --project=. experiments/run_experiments.jl                # all experiments
    julia --project=. experiments/run_experiments.jl --only E1      # just E1
    julia --project=. experiments/run_experiments.jl --only E1,E3   # E1 and E3

Each experiment writes one CSV under `experiments/results/` and appends a line
to `experiments/results/INDEX.md`. Filename convention:
    {Ek}_{ISO_date}_{short_description}.csv

See `docs/discussion/01b_solver_A_pseudocode.md` and the plan file for context.
"""

using RobustOnlinePOMDP
using Dates

include("tiger_problem.jl")

# ---------------------------------------------------------------------------
# CLI: parse --only Ek,...
# ---------------------------------------------------------------------------

function parse_only(args::Vector{String})
    for (i, a) in enumerate(args)
        if a == "--only" && i + 1 <= length(args)
            return Set(strip.(split(args[i+1], ",")))
        end
    end
    return nothing  # nothing means "run all"
end

const SELECTED = parse_only(ARGS)
should_run(eid) = SELECTED === nothing || eid in SELECTED

# ---------------------------------------------------------------------------
# Common setup
# ---------------------------------------------------------------------------

const TODAY = string(Dates.today())  # ISO yyyy-mm-dd
const TIGER = make_tiger_pomdp(listen_accuracy=0.85)

const N_TRIALS_DEFAULT = 100
const BUDGET_DEFAULT   = 500
const HORIZON_DEFAULT  = 5

# ---------------------------------------------------------------------------
# E1 — Baseline equivalence (sanity)
#   BasicPOMCP vs robust_pomcp_plan(ρ=0) on the nominal Tiger world.
# ---------------------------------------------------------------------------

function experiment_E1()
    cfgs = ExperimentConfig[
        ExperimentConfig(
            experiment_id = "E1",
            date = TODAY,
            description = "baseline_basicpomcp_vs_rho0",
            planner_label = "BasicPOMCP",
            n_trials = N_TRIALS_DEFAULT,
            budget = BUDGET_DEFAULT,
            horizon = HORIZON_DEFAULT,
        ),
        ExperimentConfig(
            experiment_id = "E1",
            date = TODAY,
            description = "baseline_basicpomcp_vs_rho0",
            planner_label = "robust_pomcp",
            ρ_T = 0.0, ρ_Z = 0.0,
            ucb_mode = :nominal,
            n_trials = N_TRIALS_DEFAULT,
            budget = BUDGET_DEFAULT,
            sims_per_backup = 5,
            horizon = HORIZON_DEFAULT,
        ),
    ]
    run_sweep("E1", TODAY, "baseline_basicpomcp_vs_rho0", cfgs, TIGER;
              index_oneliner = "Baseline equivalence: BasicPOMCP vs robust_pomcp(ρ=0) on nominal Tiger")
end

# ---------------------------------------------------------------------------
# E2 — ρ_Z sweep, nominal world
# ---------------------------------------------------------------------------

function experiment_E2()
    cfgs = ExperimentConfig[]
    for ρ_Z in (0.0, 0.05, 0.1, 0.2, 0.3)
        push!(cfgs, ExperimentConfig(
            experiment_id = "E2",
            date = TODAY,
            description = "rho_z_sweep_nominal",
            planner_label = "robust_pomcp",
            ρ_Z = ρ_Z,
            ucb_mode = :nominal,
            n_trials = N_TRIALS_DEFAULT,
            budget = BUDGET_DEFAULT,
            sims_per_backup = 5,
            horizon = HORIZON_DEFAULT,
        ))
    end
    run_sweep("E2", TODAY, "rho_z_sweep_nominal", cfgs, TIGER;
              index_oneliner = "ρ_Z sweep (0..0.3) on nominal Tiger world")
end

# ---------------------------------------------------------------------------
# E3 — ρ_Z sweep, perturbed world (η=0.20)
# ---------------------------------------------------------------------------

function experiment_E3()
    cfgs = ExperimentConfig[]
    for ρ_Z in (0.0, 0.05, 0.1, 0.2, 0.3)
        push!(cfgs, ExperimentConfig(
            experiment_id = "E3",
            date = TODAY,
            description = "rho_z_sweep_perturbed_eta0.20",
            planner_label = "robust_pomcp",
            ρ_Z = ρ_Z,
            ucb_mode = :nominal,
            η_world_obs = 0.20,
            n_trials = N_TRIALS_DEFAULT,
            budget = BUDGET_DEFAULT,
            sims_per_backup = 5,
            horizon = HORIZON_DEFAULT,
        ))
    end
    run_sweep("E3", TODAY, "rho_z_sweep_perturbed_eta0.20", cfgs, TIGER;
              index_oneliner = "ρ_Z sweep (0..0.3) on η=0.20 perturbed Tiger (headline)")
end

# ---------------------------------------------------------------------------
# E4 — Adversarial η sweep, fixed planner pair (vanilla vs robust ρ_Z=0.2)
# ---------------------------------------------------------------------------

function experiment_E4()
    cfgs = ExperimentConfig[]
    for η in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
        for ρ_Z in (0.0, 0.2)
            push!(cfgs, ExperimentConfig(
                experiment_id = "E4",
                date = TODAY,
                description = "eta_sweep_paired",
                planner_label = "robust_pomcp",   # both use our solver; differ in ρ_Z
                ρ_Z = ρ_Z,
                ucb_mode = :nominal,
                η_world_obs = η,
                n_trials = N_TRIALS_DEFAULT,
                budget = BUDGET_DEFAULT,
                sims_per_backup = 5,
                horizon = HORIZON_DEFAULT,
            ))
        end
    end
    run_sweep("E4", TODAY, "eta_sweep_paired", cfgs, TIGER;
              index_oneliner = "η sweep on world; vanilla(ρ=0) vs robust(ρ_Z=0.2)")
end

# ---------------------------------------------------------------------------
# E5 — sims_per_backup sweep (nominal + perturbed → 2 CSVs)
# ---------------------------------------------------------------------------

function experiment_E5()
    for (suffix, η_obs, oneliner) in (
        ("nominal", 0.0, "sims_per_backup sweep on nominal world"),
        ("perturbed_eta0.20", 0.20, "sims_per_backup sweep on η=0.20 perturbed world"),
    )
        cfgs = ExperimentConfig[]
        for K in (1, 5, 25, 100, 500)
            push!(cfgs, ExperimentConfig(
                experiment_id = "E5",
                date = TODAY,
                description = "simsperbackup_sweep_$suffix",
                planner_label = "robust_pomcp",
                ρ_Z = 0.1,
                ucb_mode = :nominal,
                η_world_obs = η_obs,
                n_trials = N_TRIALS_DEFAULT,
                budget = BUDGET_DEFAULT,
                sims_per_backup = K,
                horizon = HORIZON_DEFAULT,
            ))
        end
        run_sweep("E5", TODAY, "simsperbackup_sweep_$suffix", cfgs, TIGER;
                  index_oneliner = oneliner)
    end
end

# ---------------------------------------------------------------------------
# E6 — UCB mode comparison (:nominal vs :robust at ρ_Z=0.1)
# ---------------------------------------------------------------------------

function experiment_E6()
    for (suffix, η_obs, oneliner) in (
        ("nominal", 0.0, "ucb_mode compare on nominal world"),
        ("perturbed_eta0.20", 0.20, "ucb_mode compare on η=0.20 perturbed world"),
    )
        cfgs = ExperimentConfig[]
        for mode in (:nominal, :robust)
            push!(cfgs, ExperimentConfig(
                experiment_id = "E6",
                date = TODAY,
                description = "ucbmode_compare_$suffix",
                planner_label = "robust_pomcp",
                ρ_Z = 0.1,
                ucb_mode = mode,
                η_world_obs = η_obs,
                n_trials = N_TRIALS_DEFAULT,
                budget = BUDGET_DEFAULT,
                sims_per_backup = 5,
                horizon = HORIZON_DEFAULT,
            ))
        end
        run_sweep("E6", TODAY, "ucbmode_compare_$suffix", cfgs, TIGER;
                  index_oneliner = oneliner)
    end
end

# ---------------------------------------------------------------------------
# E7 — Horizon sweep (optional)
# ---------------------------------------------------------------------------

function experiment_E7()
    for (suffix, η_obs, oneliner) in (
        ("nominal", 0.0, "horizon sweep on nominal world"),
        ("perturbed_eta0.20", 0.20, "horizon sweep on η=0.20 perturbed world"),
    )
        cfgs = ExperimentConfig[]
        for h in (3, 5, 7, 10)
            push!(cfgs, ExperimentConfig(
                experiment_id = "E7",
                date = TODAY,
                description = "horizon_sweep_$suffix",
                planner_label = "robust_pomcp",
                ρ_Z = 0.1,
                ucb_mode = :nominal,
                η_world_obs = η_obs,
                n_trials = N_TRIALS_DEFAULT,
                budget = BUDGET_DEFAULT,
                sims_per_backup = 5,
                horizon = h,
            ))
        end
        run_sweep("E7", TODAY, "horizon_sweep_$suffix", cfgs, TIGER;
                  index_oneliner = oneliner)
    end
end

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

const EXPERIMENTS = [
    ("E1", experiment_E1),
    ("E2", experiment_E2),
    ("E3", experiment_E3),
    ("E4", experiment_E4),
    ("E5", experiment_E5),
    ("E6", experiment_E6),
    ("E7", experiment_E7),
]

for (id, fn) in EXPERIMENTS
    if should_run(id)
        @info "==== Running $id ===="
        fn()
    else
        @info "Skipping $id"
    end
end

@info "All requested experiments complete."
