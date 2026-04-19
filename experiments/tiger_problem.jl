"""
Tiger POMDP problem definition.

Classic Kaelbling et al. (1998) Tiger problem:
- 2 states: tiger-left (s=1), tiger-right (s=2)
- 3 actions: open-left (a=1), open-right (a=2), listen (a=3)
- 2 observations: hear-left (z=1), hear-right (z=2)

Rewards:
- Listen: -1
- Open correct door: +10
- Open wrong door: -100

Listening accuracy: 85% correct observation.
After opening a door, the state resets uniformly (game restarts).
"""

using RobustOnlinePOMDP

"""
    make_tiger_pomdp(; listen_accuracy=0.85)

Build the Tiger POMDP as a TabularPOMDP.

Args:
    listen_accuracy: probability of hearing the tiger correctly (default 0.85)

Returns:
    TabularPOMDP with n_states=2, n_actions=3, n_obs=2
"""
function make_tiger_pomdp(; listen_accuracy::Float64=0.85)
    n_states = 2   # 1=tiger-left, 2=tiger-right
    n_actions = 3  # 1=open-left, 2=open-right, 3=listen
    n_obs = 2      # 1=hear-left, 2=hear-right

    # --- Transition matrix T[s, a, s_next] ---
    T = zeros(n_states, n_actions, n_states)

    # Open-left (a=1) or Open-right (a=2): state resets uniformly
    for s in 1:n_states
        T[s, 1, 1] = 0.5; T[s, 1, 2] = 0.5   # open-left
        T[s, 2, 1] = 0.5; T[s, 2, 2] = 0.5   # open-right
    end

    # Listen (a=3): state unchanged
    for s in 1:n_states
        T[s, 3, s] = 1.0
    end

    # --- Observation matrix O[s_next, z] ---
    # Observation is based on the state (used after any action, but only meaningful after listen)
    O = zeros(n_states, n_obs)
    O[1, 1] = listen_accuracy         # tiger-left → hear-left
    O[1, 2] = 1 - listen_accuracy     # tiger-left → hear-right
    O[2, 1] = 1 - listen_accuracy     # tiger-right → hear-left
    O[2, 2] = listen_accuracy         # tiger-right → hear-right

    # --- Reward matrix R[s, a] ---
    R = zeros(n_states, n_actions)

    # Open-left (a=1)
    R[1, 1] = -100.0   # tiger-left + open-left = bad
    R[2, 1] = 10.0     # tiger-right + open-left = good

    # Open-right (a=2)
    R[1, 2] = 10.0     # tiger-left + open-right = good
    R[2, 2] = -100.0   # tiger-right + open-right = bad

    # Listen (a=3)
    R[1, 3] = -1.0
    R[2, 3] = -1.0

    return TabularPOMDP(T, O, R)
end

# Named constants for readability
const TIGER_ACTION_OPEN_LEFT  = 1
const TIGER_ACTION_OPEN_RIGHT = 2
const TIGER_ACTION_LISTEN     = 3

const TIGER_STATE_LEFT  = 1
const TIGER_STATE_RIGHT = 2

const TIGER_OBS_HEAR_LEFT  = 1
const TIGER_OBS_HEAR_RIGHT = 2
