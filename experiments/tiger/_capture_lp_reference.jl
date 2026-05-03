# experiments/tiger/_capture_lp_reference.jl
#
# Capture-once Julia reference for the Phase-2 Python LP verification.
#
# Usage (run ONCE from the repo root with the Julia project active):
#   julia --project=. experiments/tiger/_capture_lp_reference.jl
#
# Writes experiments/tiger/lp_reference.json — a list of (p_nominal, rho, c) inputs
# with Julia's HiGHS-via-JuMP optimal_value and optimal_p, plus a degeneracy flag
# obtained by re-solving with a slightly perturbed objective and checking whether
# optimal_p shifts materially despite optimal_value staying constant.
#
# Per plan §1.3 #2 and §5 Phase 2, this is a SOFT cross-check (1e-3 tolerance on
# optimal_value), so the file commit can be regenerated harmlessly.

using JSON

# Direct include rather than `using RobustOnlinePOMDP`. Reason: at top level the
# RobustOnlinePOMDP namespace surfaces both POMDPs.jl's and JuMP's `value`,
# making `value(d_plus[i])` ambiguous (Session 8 issue). Including the LP file
# alone leaves only JuMP in scope, so the binding resolves cleanly.
include(joinpath(@__DIR__, "..", "..", "src", "uncertainty", "projected_sets.jl"))

# ---------------------------------------------------------------------------
# Grid of inputs
# ---------------------------------------------------------------------------

inputs = []

# n=2 grid (Tiger-shaped)
for p_nom in ([0.5, 0.5], [0.85, 0.15], [0.3, 0.7])
    for ρ in (0.01, 0.05, 0.1, 0.3)
        for c in ([1.0, -1.0], [10.0, -10.0], [5.0, 5.0], [-1.0, 0.5])
            push!(inputs, (p_nominal=p_nom, ρ=ρ, c=c))
        end
    end
end

# n=3 grid (slightly larger problem shape, for variety)
for p_nom in ([0.4, 0.4, 0.2], [0.7, 0.2, 0.1])
    for ρ in (0.05, 0.2)
        for c in ([1.0, -1.0, 0.5], [-2.0, 1.0, 1.0])
            push!(inputs, (p_nominal=p_nom, ρ=ρ, c=c))
        end
    end
end

# ---------------------------------------------------------------------------
# Solve each input + degeneracy probe
# ---------------------------------------------------------------------------

results = []
for inp in inputs
    p_nom = collect(Float64, inp.p_nominal)
    ρ = Float64(inp.ρ)
    c = collect(Float64, inp.c)

    val, p_star = minimize_over_projected_tv_ball(p_nom, ρ, c)

    # Degeneracy probe: re-solve with a slightly perturbed objective.
    # If the optimum is unique, p_star moves by O(perturb) (~1e-7).
    # If degenerate (multiple optimal vertices), p_star can jump to a
    # different vertex — we threshold at 1e-4 to flag this.
    perturb = [1e-7 * i for i in 1:length(c)]
    _, p_star_perturbed = minimize_over_projected_tv_ball(
        p_nom, ρ, c .+ perturb
    )
    p_diff = maximum(abs.(p_star .- p_star_perturbed))
    is_degenerate = p_diff > 1e-4

    push!(results, Dict(
        "p_nominal"     => p_nom,
        "rho"           => ρ,
        "c"             => c,
        "optimal_value" => val,
        "optimal_p"     => p_star,
        "degenerate"    => is_degenerate,
    ))
end

# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

out_path = joinpath(@__DIR__, "lp_reference.json")
open(out_path, "w") do f
    JSON.print(f, results, 2)
end

n_total = length(results)
n_degen = count(r -> r["degenerate"], results)
println("Wrote $n_total reference LPs to $out_path  ($n_degen flagged degenerate)")
