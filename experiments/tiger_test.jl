"""
Tiger POMDP — Solver A end-to-end test script.

Runs the solver with a logger enabled, reports summary at the root,
and exports the entire run (tree + simulations + backups) to a JSON file
at `experiments/tiger_run.json`.

The JSON file is consumed by the Python Dash UI in `viz/app.py`.

Parameters (edit these for different runs):
  - budget: number of simulations (use 50 for debugging, 500 for full run)
  - horizon: planning depth
  - ρ_T, ρ_Z: uncertainty radii
"""

using RobustOnlinePOMDP
include("tiger_problem.jl")

println("=" ^ 70)
println("Tiger POMDP — Robust POMCP")
println("=" ^ 70)

# Build problem
tiger = make_tiger_pomdp(listen_accuracy=0.85)

# Uncertainty: small for debugging
ρ_T_val = 0.0
ρ_Z_val = 0.1
uncertainty = uniform_uncertainty(tiger.n_states, 
                                    tiger.n_actions,
                                    ρ_T_val, 
                                    ρ_Z_val,
                                    TVDistance()
                                    )

# Parameters
belief = [0.5, 0.5]
horizon = 5
budget = 500          # small for debugging; raise to 500 for a full run
sims_per_backup = 5   # robust backup every 5 sims (interleaved)

println("Parameters:")
println("  belief           = $belief")
println("  horizon          = $horizon")
println("  budget           = $budget")
println("  sims_per_backup  = $sims_per_backup")
println("  ρ_T         = $ρ_T_val")
println("  ρ_Z         = $ρ_Z_val")
println()

# Run solver with logger enabled
println("Running robust_pomcp_plan...")
log = SolverLog()

action, root = robust_pomcp_plan(belief,
                                 tiger,
                                 uncertainty,
                                 horizon,
                                 budget,
                                 sims_per_backup;
                                 ucb_mode=:nominal,
                                 logger=log
                                )

action_names = ["open-left", "open-right", "listen"]
println("Solver returned action $action ($(action_names[action]))")
println()

# Q values at root
println("Q values at root:")
println("  Action           Q_nominal     Q_robust      N_visits")
println("  " * "-" ^ 55)
for a in 1:tiger.n_actions
    if haskey(root.children, a)
        an = root.children[a]
        q_n = round(an.Q_nominal, digits=3)
        q_r = round(an.Q_robust, digits=3)
        println("  $(rpad(action_names[a], 16)) $(rpad(q_n, 13)) $(rpad(q_r, 13)) $(an.N)")
    else
        println("  $(rpad(action_names[a], 16)) (not expanded)")
    end
end
println()

# Tree size
n_h, n_a = RobustOnlinePOMDP.tree_size(root)
println("Tree: $n_h HistoryNodes, $n_a ActionNodes")
println("Log:  $(num_events(log)) events")
println()

# Export the event log to JSON for the Python UI
output_path = joinpath(@__DIR__, "tiger_run.json")
export_log_to_json(
    log, output_path;
    metadata = Dict(
        "problem"    => "Tiger",
        "n_states"   => tiger.n_states,
        "n_actions"  => tiger.n_actions,
        "n_obs"      => tiger.n_obs,
        "horizon"    => horizon,
        "budget"     => budget,
        "sims_per_backup" => sims_per_backup,
        "ucb_mode"   => "nominal",
        "rho_T"      => ρ_T_val,
        "rho_Z"      => ρ_Z_val,
        "belief"     => belief,
        "action_names" => action_names,
        "state_names"  => ["tiger-left", "tiger-right"],
        "obs_names"    => ["hear-left", "hear-right"],
    )
)

println("JSON exported: $output_path")
println()
println("To explore: `cd viz && python app.py`")
