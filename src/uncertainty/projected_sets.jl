"""
Projected uncertainty set optimization for TV distance.

Solves the core LP used by the robust Bellman backup:
  min  Σ p_i * c_i
  s.t. p ∈ hat{P}(p_o, ρ)   (projected TV ball)

The projected TV ball is characterized by (Piece 8 from design doc):
  p_i ≥ 0
  Σ p_i ≤ 1                           (sub-probability)
  Σ |p_i - p_o_i| + |Σ p_i - m_o| ≤ ρ  (projected TV constraint)

where m_o = Σ p_o_i (nominal in-support mass).

This function is used by both:
  - SolveObservationLP: p_o = P_Z^o(z|s') on Z_in, c = V_children
  - SolveTransitionLP:  p_o = P_T^o(s'|s,a) on S_in, c = w(s')
"""

using JuMP
using HiGHS

"""
    minimize_over_projected_tv_ball(p_nominal, ρ, objective_coeffs)

Minimize a linear objective over the projected TV ball.

Args:
    p_nominal: nominal probability vector restricted to the support (p_o_i for i ∈ support).
               Does NOT need to sum to 1 (the rest is out-of-support mass).
    ρ: TV distance radius
    objective_coeffs: coefficients c_i for the objective Σ p_i * c_i

Returns:
    (optimal_value, optimal_p): the minimum objective value and the optimal sub-probability vector
"""
function minimize_over_projected_tv_ball(p_nominal::Vector{Float64}, ρ::Float64,
                                          objective_coeffs::Vector{Float64})
    n = length(p_nominal)
    @assert length(objective_coeffs) == n "p_nominal and objective_coeffs must have same length"

    m_nominal = sum(p_nominal)  # nominal in-support mass

    # If radius is 0, the only feasible point is p = p_nominal
    if ρ ≤ 0.0
        val = sum(p_nominal[i] * objective_coeffs[i] for i in 1:n)
        return (val, copy(p_nominal))
    end

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # Variables: deviations d⁺, d⁻ for each element, and δ⁺, δ⁻ for out-of-support mass change
    @variable(model, d_plus[1:n] >= 0)  # how much EACH entry moves above nominal Pr
    @variable(model, d_minus[1:n] >= 0) # how much EACH entry moves below nominal Pr
    @variable(model, δ_plus >= 0)       # aggregate OUT-of-support mass INCREASE 
    @variable(model, δ_minus >= 0)      # aggregate OUT-of-support mass DECREASE 

    # p_i = p_nominal_i + d_plus_i - d_minus_i
    # Constraint: p_i ≥ 0
    for i in 1:n
        @constraint(model, d_plus[i] - d_minus[i] >= -p_nominal[i])
    end

    # TV budget: Σ(d⁺ + d⁻) + δ⁺ + δ⁻ ≤ ρ
    @constraint(model, sum(d_plus[i] + d_minus[i] for i in 1:n) + δ_plus + δ_minus <= ρ)

    # Mass balance: Σ(d⁺ - d⁻) = δ⁺ - δ⁻
    # (the net mass change in-support equals the net mass change out-of-support)
    @constraint(model, sum(d_plus[i] - d_minus[i] for i in 1:n) == δ_plus - δ_minus)

    # Sub-probability: Σ p_i ≤ 1, i.e., m_nominal + Σ(d⁺ - d⁻) ≤ 1
    @constraint(model, sum(d_plus[i] - d_minus[i] for i in 1:n) <= 1.0 - m_nominal)

    # Objective: min Σ (p_nominal_i + d⁺_i - d⁻_i) * c_i
    # = Σ p_nominal_i * c_i + Σ (d⁺_i - d⁻_i) * c_i
    # The first term is constant, so we minimize the second term
    @objective(model, Min,
        sum((d_plus[i] - d_minus[i]) * objective_coeffs[i] for i in 1:n))

    optimize!(model)

    # Reconstruct optimal p
    optimal_p = Vector{Float64}(undef, n)
    for i in 1:n
        optimal_p[i] = p_nominal[i] + value(d_plus[i]) - value(d_minus[i])
    end

    # Full objective value (including the constant term)
    optimal_value = sum(optimal_p[i] * objective_coeffs[i] for i in 1:n)

    return (optimal_value, optimal_p)
end
