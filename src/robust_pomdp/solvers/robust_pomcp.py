"""
Solver A: Robust-Value POMCP.

Builds a search tree via Monte Carlo simulations (under the nominal model),
then computes robust values via worst-case Bellman backups using projected
uncertainty sets.

Tunable parameters:
    sims_per_backup: simulations between successive robust backups.
        sims_per_backup == budget is two-phase; sims_per_backup == 1 is fully
        interleaved.
    ucb_mode: "nominal" (optimistic) or "robust" (pessimistic) exploration.

Observability:
    Pass a SolverLog as `logger` and the solver emits fine-grained events
    (sim_start, sim_step, expand, rollout, backup_nominal_step, sim_end,
    robust_backup) in chronological order for later replay.

Preserved Julia invariants (Sessions 5–7):
    - V_rollout fallback for unvisited branches (HistoryNode methods).
    - V_nominal repropagation after every Q_nominal update in simulate.
    - Dirty-flag invariant: clean node => no descendant is dirty. Backup
      short-circuits at the first clean node hit.
    - HistoryNode.N bumped on every visit (in add_particle).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from robust_pomdp.core.pomdp_types import TabularPOMDP
from robust_pomdp.core.tree import ActionNode, HistoryNode, NodeIdCounter
from robust_pomdp.debug.logger import ObsLPResult, SolverLog, TransLPResult
from robust_pomdp.uncertainty.projected_sets import ProjectedTVBall
from robust_pomdp.uncertainty.uncertainty_sets import UncertaintySets


UCBMode = Literal["nominal", "robust"]


# ---------------------------------------------------------------------------
# Top-level planner
# ---------------------------------------------------------------------------

def robust_pomcp_plan(belief: np.ndarray,
                      model: TabularPOMDP,
                      uncertainty: UncertaintySets,
                      horizon: int,
                      budget: int,
                      sims_per_backup: int,
                      *,
                      ucb_mode: UCBMode = "nominal",
                      exploration_constant: float = 1.0,
                      logger: SolverLog | None = None,
                      rng: np.random.Generator | None = None
                      ) -> tuple[int, HistoryNode]:
    """Plan a single action with Robust-Value POMCP.

    Returns:
        (best_action, root) — best_action is 0-based.
    """
    if rng is None:
        rng = np.random.default_rng()
    belief = np.asarray(belief, dtype=np.float64)

    id_counter = NodeIdCounter()
    root = HistoryNode(id_counter.next_id(), depth=0)

    # LP reuse: cache ProjectedTVBall instances by problem size n. The LP
    # structure depends only on n; (p_nominal, rho, c) are passed per solve.
    lp_cache: dict[int, ProjectedTVBall] = {}

    backup_counter = 0
    n_batches = -(-budget // sims_per_backup)  # ceil(budget / sims_per_backup)

    for batch in range(n_batches):
        n_sims_this_batch = min(sims_per_backup, budget - batch * sims_per_backup)

        for _ in range(n_sims_this_batch):
            sim_index = _next_sim_index(logger)
            s = sample_from_belief(belief, rng)

            if logger is not None:
                logger.emit_sim_start(sim_index=sim_index, initial_state=s, root_node_id=root.id,)

            total_ret, _ = _simulate(s,
                                     root,
                                     horizon,
                                     model,
                                     ucb_mode,
                                     exploration_constant,
                                     id_counter,
                                     logger,
                                     sim_index,
                                     step_counter=0,
                                     rng=rng
                                     )

            if logger is not None:
                logger.emit_sim_end(sim_index=sim_index, total_return=total_ret)

        backup_counter = _robust_backup(root, model, uncertainty, logger, backup_counter, lp_cache,)

    # Pick action with highest Q_robust at the root.
    best_action = -1
    best_value = -np.inf
    for a, action_node in root.children.items():
        if action_node.Q_robust > best_value:
            best_value = action_node.Q_robust
            best_action = a

    return best_action, root


def _next_sim_index(logger: SolverLog | None) -> int:
    """Return 1 + (number of sim_start events emitted so far). 0 if no logger."""
    if logger is None:
        return 0
    count = sum(1 for ev in logger.events if ev.type == "sim_start")
    return count + 1


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _simulate(s: int,
              node: HistoryNode,
              depth: int,
              model: TabularPOMDP,
              ucb_mode: UCBMode,
              c: float,
              id_counter: NodeIdCounter,
              logger: SolverLog | None,
              sim_index: int,
              step_counter: int,
              rng: np.random.Generator
              ) -> tuple[float, int]:
    """One POMCP simulation from state s at the given history node.

    Returns (total_return_from_here, updated_step_counter).
    """
    if depth == 0:
        return 0.0, step_counter

    node.add_particle(s)

    # Leaf: expand and rollout.
    if node.is_leaf():
        created_ids, created_actions = node.expand(model.n_actions, id_counter)
        if logger is not None:
            logger.emit_expand(sim_index=sim_index,
                               node_id=node.id,
                               depth=node.depth,
                               created_action_nodes=created_ids,
                               created_action_indices=created_actions
                               )

        rollout_return, rollout_steps = _rollout_with_log(s, depth, model, rng)
        node.V_rollout = rollout_return
        node.has_rollout = True

        if logger is not None:
            logger.emit_rollout(sim_index=sim_index,
                                leaf_node_id=node.id,
                                start_state=s,
                                start_depth=node.depth,
                                rollout_steps=rollout_steps,
                                total_return=rollout_return
                                )

        return rollout_return, step_counter

    # Select action via UCB.
    a, ucb_qs, ucb_ns = _ucb_select_with_info(node, ucb_mode, c, model.n_actions)

    # Sample next state, observation, reward (under nominal model).
    s_next = model.sample_transition(s, a, rng)
    z = model.sample_observation(s_next, rng)
    r = model.reward(s, a)

    action_node = node.children[a]
    child, child_was_created = action_node.get_or_create_child(z, node.depth + 1, id_counter)

    if logger is not None:
        step_counter += 1
        logger.emit_sim_step(sim_index=sim_index,
                             step_in_sim=step_counter,
                             active_history_id=node.id,
                             depth=node.depth,
                             state=s,
                             ucb_qs=ucb_qs,
                             ucb_ns=ucb_ns,
                             action_chosen=a,
                             action_node_id=action_node.id,
                             s_next=s_next,
                             obs=z,
                             reward=r,
                             next_history_id=child.id,
                             created_action_nodes=[],
                             created_history_nodes=[child.id] if child_was_created else []
                             )

    # Recurse.
    future_return, step_counter = _simulate(s_next,
                                            child,
                                            depth - 1,
                                            model,
                                            ucb_mode,
                                            c,
                                            id_counter,
                                            logger,
                                            sim_index,
                                            step_counter,
                                            rng
                                            )
    total_return = r + future_return

    # Update Q_nominal (incremental mean) and dirty the action node.
    prev_Q = action_node.Q_nominal
    prev_N = action_node.N
    action_node.N += 1
    action_node.Q_nominal += (
        (total_return - action_node.Q_nominal) / action_node.N
    )
    action_node.dirty = True

    if logger is not None:
        logger.emit_backup_nominal_step(sim_index=sim_index,
                                        action_node_id=action_node.id,
                                        previous_Q_nominal=prev_Q,
                                        new_Q_nominal=action_node.Q_nominal,
                                        previous_N=prev_N,
                                        new_N=action_node.N,
                                        total_return_from_here=total_return
                                        )

    # V_nominal of `node` must be recomputed from its children because the
    # max-Q child may not be the one we just updated (Session 6 fix).
    node.update_nominal_value()

    return total_return, step_counter


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def _rollout_with_log(s: int,
                      depth: int,
                      model: TabularPOMDP,
                      rng: np.random.Generator
                      ) -> tuple[float, list[dict[str, Any]]]:
    """Random rollout. Returns (total_return, steps_list)."""
    total_return = 0.0
    current_s = s
    steps: list[dict[str, Any]] = []

    for _ in range(depth):
        a = int(rng.integers(model.n_actions))
        r = model.reward(current_s, a)
        s_next = model.sample_transition(current_s, a, rng)
        steps.append({
            "state":  current_s,
            "action": a,
            "reward": r,
            "s_next": s_next,
        })
        total_return += r
        current_s = s_next

    return total_return, steps


# ---------------------------------------------------------------------------
# UCB action selection
# ---------------------------------------------------------------------------

def _ucb_select_with_info(node: HistoryNode,
                          ucb_mode: UCBMode,
                          c: float,
                          n_actions: int
                          ) -> tuple[int, list[float], list[int]]:
    """UCB selection that also returns the (Q values, visit counts) seen.

    Returns (best_action, ucb_qs, ucb_ns) where ucb_qs[a] / ucb_ns[a] are
    aligned to action index a in 0..n_actions-1.
    """
    ucb_qs: list[float] = []
    ucb_ns: list[int] = []

    for a in range(n_actions):
        an = node.children[a]
        q = an.Q_robust if ucb_mode == "robust" else an.Q_nominal
        ucb_qs.append(q)
        ucb_ns.append(an.N)

    # Visit unvisited actions first.
    for a in range(n_actions):
        if node.children[a].N == 0:
            return a, ucb_qs, ucb_ns

    # Standard UCB.
    log_N = np.log(node.N)
    best_action = -1
    best_ucb = -np.inf
    for a in range(n_actions):
        an = node.children[a]
        q = an.Q_robust if ucb_mode == "robust" else an.Q_nominal
        ucb_value = q + c * np.sqrt(log_N / an.N)
        if ucb_value > best_ucb:
            best_ucb = ucb_value
            best_action = a

    return best_action, ucb_qs, ucb_ns


# ---------------------------------------------------------------------------
# Robust backup (bottom-up, dirty-aware)
# ---------------------------------------------------------------------------

def _robust_backup(node: HistoryNode,
                   model: TabularPOMDP,
                   uncertainty: UncertaintySets,
                   logger: SolverLog | None,
                   backup_counter: int,
                   lp_cache: dict[int, ProjectedTVBall]
                   ) -> int:
    """Bottom-up robust value computation. Recomputes only dirty nodes.

    Invariant: clean node => no descendant is dirty. We short-circuit the
    entire subtree at the first clean node.

    Returns updated backup_counter.
    """
    if not node.dirty:
        return backup_counter

    for a, action_node in node.children.items():
        if not action_node.dirty:
            continue

        # Recurse into observation children first (bottom-up).
        for child in action_node.children.values():
            backup_counter = _robust_backup(child,
                                            model,
                                            uncertainty,
                                            logger,
                                            backup_counter,
                                            lp_cache
                                            )

        prev_Qr = action_node.Q_robust
        if not action_node.children:
            # No observation children yet — the action was visited at a leaf
            # but never followed up. Q_robust falls back to Q_nominal.
            action_node.Q_robust = action_node.Q_nominal
        else:
            new_Qr, backup_counter = compute_robust_q(node,
                                                       a,
                                                       action_node,
                                                       model,
                                                       uncertainty,
                                                       logger,
                                                       backup_counter,
                                                       prev_Qr,
                                                       lp_cache
                                                       )
            action_node.Q_robust = new_Qr

        action_node.dirty = False

    node.update_robust_value()
    node.dirty = False

    return backup_counter


# ---------------------------------------------------------------------------
# Two-step LP decomposition for robust Q
# ---------------------------------------------------------------------------

def compute_robust_q(h: HistoryNode,
                     a: int,
                     action_node: ActionNode,
                     model: TabularPOMDP,
                     uncertainty: UncertaintySets,
                     logger: SolverLog | None,
                     backup_counter: int,
                     previous_Q_robust: float,
                     lp_cache: dict[int, ProjectedTVBall] | None = None,
                     belief_override: np.ndarray | None = None
                     ) -> tuple[float, int]:
    """Compute Q_robust(h, a) via the two-step LP decomposition.

    Step 1 (per s'): worst-case observation distribution -> w(s').
    Step 2 (per s):  worst-case transition distribution -> σ(s).
    Q_val = Σ_s belief[s] · (R(s, a) + σ(s)).

    Returns (Q_val, updated_backup_counter).

    `lp_cache` may be None (e.g. when called from a verification script that
    only does one or two backups). A fresh dict is allocated in that case.

    `belief_override`, if given, replaces the particle-empirical belief with an
    explicit vector aligned with `sorted(h.S_in)`. Used by A1's exact tree
    evaluator (no Monte-Carlo). POMCP-style callers leave it as `None`.
    """
    if lp_cache is None:
        lp_cache = {}

    S_in = sorted(h.S_in)
    Z_in = sorted(action_node.Z_in)

    if belief_override is not None:
        assert belief_override.shape == (len(S_in),), \
            f"belief_override shape {belief_override.shape} != ({len(S_in)},)"
        belief = belief_override
    else:
        belief = compute_belief(h.particles, S_in)

    V_children = [action_node.children[z].V_robust if z in action_node.children else 0.0 for z in Z_in]

    obs_lp_results: list[ObsLPResult] = []
    trans_lp_results: list[TransLPResult] = []

    n_z = len(Z_in)
    n_s = len(S_in)
    full_support_obs = (n_z == model.n_obs)
    full_support_trans = (n_s == model.n_states)
    if n_z not in lp_cache:
        lp_cache[n_z] = ProjectedTVBall(n=n_z)
    if n_s not in lp_cache:
        lp_cache[n_s] = ProjectedTVBall(n=n_s)
    lp_obs = lp_cache[n_z]
    lp_trans = lp_cache[n_s]

    # Step 1: observation LP per s'.
    w = np.empty(n_s, dtype=np.float64)
    for i, s_next in enumerate(S_in):
        p_nominal_obs = [model.observation_prob(s_next, z) for z in Z_in]
        rho_z = uncertainty.observation_radius(s_next)
        w_val, optimal_p = lp_obs.solve(p_nominal_obs, rho_z, V_children,
                                        full_support=full_support_obs)
        w[i] = w_val
        if logger is not None:
            obs_lp_results.append(ObsLPResult(s_next=s_next,
                                              nominal_p=list(p_nominal_obs),
                                              radius=float(rho_z),
                                              optimal_p=optimal_p.tolist(),
                                              objective_value=float(w_val)
                                              ))

    # Step 2: transition LP per s.
    Q_val = 0.0
    for i, s in enumerate(S_in):
        p_nominal_trans = [model.transition_prob(s, a, s_next) for s_next in S_in]
        rho_t = uncertainty.transition_radius(s, a)
        sigma, optimal_p = lp_trans.solve(p_nominal_trans, rho_t, w.tolist(),
                                          full_support=full_support_trans)
        Q_val += belief[i] * (model.reward(s, a) + sigma)
        if logger is not None:
            trans_lp_results.append(TransLPResult(state=s,
                                                  nominal_p=list(p_nominal_trans),
                                                  radius=float(rho_t),
                                                  optimal_p=optimal_p.tolist(),
                                                  objective_value=float(sigma)
                                                  ))

    if logger is not None:
        backup_counter += 1
        logger.emit_robust_backup(backup_index=backup_counter,
                                  history_node_id=h.id,
                                  action_node_id=action_node.id,
                                  node_depth=h.depth,
                                  action=a,
                                  S_in=list(S_in),
                                  Z_in=list(Z_in),
                                  belief=belief.tolist(),
                                  V_children=list(V_children),
                                  obs_lp_results=obs_lp_results,
                                  trans_lp_results=trans_lp_results,
                                  previous_Q_robust=float(previous_Q_robust),
                                  new_Q_robust=float(Q_val)
                                  )

    return float(Q_val), backup_counter


# ---------------------------------------------------------------------------
# Helpers (public)
# ---------------------------------------------------------------------------

def compute_belief(particles: list[int], S_in: list[int]) -> np.ndarray:
    """Empirical belief over S_in from particles. Returns vector aligned with S_in."""
    counts = np.zeros(len(S_in), dtype=np.float64)
    state_to_idx = {s: i for i, s in enumerate(S_in)}
    for s in particles:
        if s in state_to_idx:
            counts[state_to_idx[s]] += 1.0
    total = counts.sum()
    if total == 0.0:
        return counts
    return counts / total


def sample_from_belief(belief: np.ndarray, rng: np.random.Generator) -> int:
    """Sample a state from a belief vector."""
    return int(rng.choice(len(belief), p=belief))


def tree_size(root: HistoryNode) -> tuple[int, int]:
    """Count (HistoryNodes, ActionNodes) in the tree rooted at `root`."""
    n_history = 1
    n_action = 0
    for action_node in root.children.values():
        n_action += 1
        for child in action_node.children.values():
            nh, na = tree_size(child)
            n_history += nh
            n_action += na
    return n_history, n_action
