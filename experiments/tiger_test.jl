"""
Tiger POMDP — Solver A end-to-end test script.

This is a notebook-style script: run top-to-bottom, comment out sections as needed.

Sections:
  T1: Pure end-to-end run (sanity: does it run?)
  T2: Uniform belief → should pick listen
  T3: Skewed belief → should pick the correct opening action
  T4: Q-value bar chart (Q_nominal vs Q_robust)
  T5: Convergence plot (Q vs budget N)
  T6: ρ sweep (Q vs uncertainty radius)
  T7: K sweep
  T8: UCB mode sweep
"""

using RobustOnlinePOMDP
include("tiger_problem.jl")

# ==============================================================================
# T1: Pure end-to-end run
# ==============================================================================

println("=" ^ 70)
println("T1: End-to-end run")
println("=" ^ 70)

# Build problem
tiger = make_tiger_pomdp(listen_accuracy=0.85)

# Uncertainty: no transition uncertainty, moderate observation uncertainty
uncertainty = uniform_uncertainty(
    tiger.n_states, tiger.n_actions,
    0.0,    # ρ_T
    0.1,    # ρ_Z
    TVDistance()
)

# Parameters
belief = [0.5, 0.5]
horizon = 5
budget = 500
batch_size = 500   # K = N → two-phase

# Run solver
println("Running robust_pomcp_plan...")
println("  belief = $belief")
println("  horizon = $horizon, budget = $budget, K = $batch_size")
println("  ρ_T = 0.0, ρ_Z = 0.1")
println()

elapsed = @elapsed (action, root) = robust_pomcp_plan(
    belief, tiger, uncertainty,
    horizon, budget, batch_size;
    ucb_mode=:nominal
)

# Report
action_names = ["open-left", "open-right", "listen"]
println("Solver returned action $action ($(action_names[action]))")
println("Elapsed time: $(round(elapsed, digits=3)) s")
println()

# Q values at root for every action
println("Q values at root:")
println("  Action            Q_nominal        Q_robust       N_visits")
println("  " * "-" ^ 60)
for a in 1:tiger.n_actions
    if haskey(root.children, a)
        an = root.children[a]
        q_n = round(an.Q_nominal, digits=3)
        q_r = round(an.Q_robust, digits=3)
        println("  $(rpad(action_names[a], 17)) $(rpad(q_n, 16)) $(rpad(q_r, 15)) $(an.N)")
    else
        println("  $(rpad(action_names[a], 17)) (not expanded)")
    end
end
println()

# Tree size
n_h, n_a = RobustOnlinePOMDP.tree_size(root)
println("Tree size:")
println("  HistoryNodes = $n_h")
println("  ActionNodes  = $n_a")
println("  Root particle count = $(length(root.particles))")
println("  Root S_in = $(root.S_in)")
println()

println("T1: Done. No crashes.")
