"""
Generic episode harness for evaluating any planner on a tabular POMDP.

The harness drives the standard POMDP loop:
  belief → plan → execute (true world) → observe → Bayes update → repeat

The planner sees ONLY the nominal model it believes (passed implicitly inside
`planner_act`). The world transitions and emits observations under the *true*
model, which may differ from the nominal one in perturbed-world experiments.
This decoupling is what makes robustness experiments meaningful.

Both the BasicPOMCP planner and our `robust_pomcp_plan` are wrapped by closures
of type `(belief::Vector{Float64}) -> action::Int` so the harness can stay
agnostic to the planner identity.
"""

using Random
using Distributions: Categorical

"""
    EpisodeStep

One row of the trajectory log.
"""
struct EpisodeStep
    t::Int
    state::Int
    belief::Vector{Float64}
    action::Int
    reward::Float64
    obs::Int
    next_state::Int
end

"""
    run_episode(planner_act, belief_update, model_true, belief_init,
                horizon, rng; initial_state=nothing)
        -> (total_reward, trajectory)

Run a single Tiger-style episode for `horizon` steps.

Args:
- `planner_act(belief) -> action::Int`     — closure capturing the planner config
- `belief_update(belief, action, obs) -> belief'` — agent's belief filter
- `model_true::TabularPOMDP`               — the *real* environment dynamics
- `belief_init::Vector{Float64}`           — agent's starting belief
- `horizon::Int`                           — number of timesteps
- `rng::AbstractRNG`                       — random number generator (for reproducibility)
- `initial_state`                          — if `nothing`, sampled from `belief_init`

Returns `(total_reward::Float64, trajectory::Vector{EpisodeStep})`.
"""
function run_episode(planner_act, belief_update,
                     model_true::TabularPOMDP,
                     belief_init::Vector{Float64},
                     horizon::Int, rng::AbstractRNG;
                     initial_state::Union{Int, Nothing}=nothing)
    state = initial_state === nothing ?
        rand(rng, Categorical(belief_init)) : initial_state
    belief = copy(belief_init)
    total_reward = 0.0
    trajectory = EpisodeStep[]

    for t in 1:horizon
        action = planner_act(belief)
        r = reward(model_true, state, action)
        next_state = rand(rng, Categorical(Vector{Float64}(transition_dist(model_true, state, action))))
        obs = rand(rng, Categorical(Vector{Float64}(observation_dist(model_true, next_state))))

        push!(trajectory, EpisodeStep(t, state, copy(belief), action, r, obs, next_state))

        total_reward += r
        state = next_state
        belief = belief_update(belief, action, obs)
    end

    return (total_reward, trajectory)
end
