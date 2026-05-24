---
name: Project Progress
description: Session-by-session work log
type: project
originSessionId: 60e505b2-035f-4545-ad11-905ae1961031
---
## Session Log

### 2026-05-24 — Session 17: A1 Phases 1-4 implementation + Phase 5 design pause

- **Implementation plan written + plan mode workflow.** After Explore + Plan
  agents, wrote `C:\Users\TomerMassas\.claude-thesis\plans\lets-now-proceed-to-golden-wren.md`
  (10 phases mapping onto roadmap Phase 3 + 4). Workflow lock: 1-by-1 with
  validation, inline diff preview before every Edit, run sanity check after
  each phase, only continue when green.
- **venv set up.** Created `.venv/` at repo root with Python 3.11. Fresh
  `pip install -e ".[viz]" pytest` resolved `numpy 2.4.6 + scipy 1.17.1 +
  matplotlib 3.10.9` — these are mutually compatible (unlike the global
  Python's old-matplotlib + new-numpy combo that was crashing PyCharm's
  debugger via `sitecustomize.py`). PyCharm now points at
  `.venv\Scripts\python.exe`.
- **Phase 1 landed.** `experiments/a1/synthetic_pomdp.py` (`make_synthetic_pomdp`,
  `make_initial_belief`, action constants, `RewardSpec` dataclass) and
  `experiments/a1/fixed_policy.py` (`warning_threshold_policy`). Defaults
  match the A1 design PDF. Smoke verifies T/O row sums, boundary clipping
  at s=0 and s=N-1, per-class reward signs, belief class masses,
  observation peak alignment, and the policy rule at threshold boundaries.
- **Phase 2 landed.** `src/robust_pomdp/bounds/support_picker.py` with
  `uniform_pick` and `exponential_decay_pick`. Decay scheme: bin index
  range, per-bin allocation `∝ exp(-decay · b) · bin_size`, largest-remainder
  for fractional residuals, water-fill on saturated bins. At
  `target=0.3, N=100, decay=0.5`: bin 0 saturates at 10, decays to 0 by
  bin 8. At high target_fraction with steep decay, the front bins saturate
  and the algorithm degrades gracefully into "front filled + smooth tail."
- **Phase 3 landed.** Extended `compute_robust_q` in
  `solvers/robust_pomcp.py` with optional `belief_override: np.ndarray | None
  = None`. ~5-line change in the solver; POMCP path unchanged when None.
  Regression: `experiments/tiger/verify_robust_backup.py` still passes
  byte-identical (Q_robust = -5.12 to machine precision).
- **Phase 4 landed.** `experiments/a1/tree_evaluator.py` with `NodeData`,
  `project_belief`, `projected_bayes_update`, `build_fixed_policy_tree`,
  `evaluate_robust_value`. Tree expands one action per HistoryNode (per
  the fixed policy), |Z_in| obs children per ActionNode. A1-specific
  metadata (exact belief vector + policy action + terminal flag + depth)
  lives in a parallel `dict[node.id, NodeData]` — **zero touch on the
  existing `core/tree.py`**. Smoke verifies tree shape (10 nodes for
  N=4/M=3/H=2 full tree), beliefs sum to 1 at every node, V at ρ=0 with
  full support matches independent recursive nominal expectation
  (2.121231), projected tree is smaller (7 nodes for |S_in|=2,|Z_in|=2).
- **Smoke organization.** Per Tomer's directive, smokes live in
  `experiments/a1/smoke_test/` (kept for reuse — no longer transient).
- **Tree visualization.** Iterated through three designs:
  - ASCII tree printer (rejected — "i dont understand what is going on
    there, i want an image like that i will zoom in").
  - matplotlib-based static PNG with custom hierarchical layout — proposed,
    self-rated 6/10 (custom Reingold-Tilford-lite layout uneven for
    non-uniform branching; scales poorly past Phase 4).
  - Final: `viz/tree_viz.py` — generates standalone HTML with cytoscape.js
    from CDN. Sibling to `viz/app.py`; reuses cytoscape conventions
    (history vs action node shapes, dagre layout, label format) but skips
    the 666-line event-replay machinery (A1's tree is static).
    Color-coded action nodes (inspect=grey, proceed=green, mitigate=amber,
    abort=red), terminal HistoryNodes dashed red. Header has Save-PNG +
    Fit buttons + legend. Browser-native pan/zoom; cytoscape.js's
    `cy.png()` exports static images on demand. Repo-root `sys.path`
    insertion in the smoke makes `from viz.tree_viz import ...` resolve
    when run from anywhere.
- **A1 vs Solver A clarification.** Tomer briefly got confused why the
  full tree shows one action child per history (not multiple). Re-anchored:
  A1 is an *evaluator* of a fixed policy, not a planner. Single action per
  history is by construction (paper's "tree policy" wording, Eq. 16). The
  branching is over observations only. Distinct from Solver A which UCB-
  expands all actions.
- **Phase 5 paused — load-bearing math correction.**
  - Tomer asked which paper equations Phase 5 implements: **Eq. 71**
    (`Δ_K^rob`) via the **Eq. 47** decomposition (under `Σ_z P_Z = 1`
    simplifies to `‖P_T − P_T^in‖_1 + Σ P_T^in(s')·‖P_Z(·|s') − P_Z^in(·|s')‖_1`).
  - Discovered the plan's LP formulation for Δ_Z and Δ_K^rob is
    **mathematically unbounded**: `max Σ u_z s.t. u_z ≥ ±(P − P^in)_z` pushes
    every `u_z → +∞` since `u_z` has only lower bounds in the LP.
  - Walked through why the `u ≥ x ∧ u ≥ -x` trick only works for
    *minimizing* `|x|`. The sign-flip `max f = -min(-f)` doesn't help —
    `‖x‖_1` is convex, so `min(-‖x‖_1)` is `min of concave` which is
    equally hard.
  - Tomer pushed for academic rigor — explicitly does **not** want a loose
    closed-form bound. I corrected my earlier framing: **Eq. 47 IS exactly
    computable, just not via a single LP.** The right tool is **MILP**
    (mixed-integer linear program) with binary sign-pattern variables +
    big-M linearization, OR orthant enumeration (2^d LPs). HiGHS supports
    MILP natively — same library as the existing LP path. I owed Tomer
    that correction; earlier framing of "no clean LP → uncomputable" was
    sloppy.
  - Eq. 47 vs Eq. 48 discussed. Eq. 48 (looser) eliminates the inner-outer
    nesting but doesn't escape the max-L1 hardness per piece. Eq. 47 +
    MILP remains the academic-rigorous path.
- **Where Phase 5 stops.** Decision point conceptually agreed: MILP-based
  exact computation. But Tomer explicitly wants to walk through the math
  step-by-step *before any code lands* — these are the actual numbers that
  go into the conference paper; he needs to fully own the derivation.
- **Files this session (uncommitted at /session-end).**
  - New: `experiments/a1/{synthetic_pomdp,fixed_policy,tree_evaluator}.py`
  - New: `experiments/a1/smoke_test/{phase1,phase2,phase4}_smoke.py`
  - New: `src/robust_pomdp/bounds/{__init__,support_picker}.py`
  - New: `viz/tree_viz.py`
  - Modified: `src/robust_pomdp/solvers/robust_pomcp.py` (added
    `belief_override` kwarg, ~5 lines)
  - New (outside repo): plan file at
    `C:\Users\TomerMassas\.claude-thesis\plans\lets-now-proceed-to-golden-wren.md`
  - New (gitignored): `.venv/` at repo root
- **Next session.** Walk Phase 5 math step-by-step with Tomer until he
  fully internalizes the Eq. 47 → MILP path. Then implement
  `bounds/delta_z.py` and `bounds/delta_k.py` (MILP via highspy with
  binary sign vars), plus a Phase 5 smoke that cross-checks the MILP
  answer against brute-force orthant enumeration on a tiny config
  (|Z|=3, 8 orthants).

### 2026-05-19 — Session 16: A1 (Fixed-Policy Projected Robust Tree Evaluator) design discussion

- **Big-picture pivot.** Tomer rethought experiments + planners. Drafted A1
  design in `pdfs/planner A - design.pdf`: a *theory-validation evaluator* (not a
  planner) on a synthetic ordered-risk POMDP. Goal — empirically validate
  the projection theorems Eq. 63 (non-robust) and Eq. 74 (robust) by
  computing full robust V vs. projected robust V̂ for a *fixed* policy on a
  finite-horizon tree, sweeping projection size. Maps onto Phase 3 + Phase 4
  of the roadmap, both currently open.
- **A1 design overview.** Synthetic POMDP: S = {0..N−1} ordered by hidden
  risk (safe / risky / catastrophic partitioning by fraction). A = {inspect,
  proceed, mitigate, abort} (abort terminal). Z = {0..M−1} ordered warning
  severities. Belief is class-mass + intra-class exponential decay.
  Transitions push toward higher risk for inspect/proceed, lower for
  mitigate. Observation is exp-decay around state's target_z. Rewards
  penalize `proceed` heavily in risky/cat states. Fixed policy: depth-0 →
  mandatory `inspect`; else action keyed on last-observation warning level
  (warning < 0.4 → proceed, < 0.75 → mitigate, else → abort).
- **Read the PDF carefully + flagged 7 issues to discuss.** Confirmed
  understanding before locking design. Held positions where I had real
  conviction (don't enrich single policy — multi-policy sweep adds more
  meaning); honest about cosmetic-vs-load-bearing options.
- **Locked decisions for first A1 experiment.**
  - **Robust case only** (Eq. 74). Non-robust (Eq. 63) skipped.
  - **Eq. 47-form kernel mismatch** (tighter than Eq. 48). Nothing in
    `bounds/` exists yet — confirmed via grep — fresh code.
  - **Compute scaling target**: N ∈ [10, 200], M ∈ [4, 12], H ∈ [3, 6].
    Scale up during tests until full robust becomes intractable; use to
    compare full vs. projected runtimes.
  - **Uniform projection at every node** (same S_in[m], Z_in[m] at every
    history h). Per-node varying — closer to online sampling — deferred to
    A1 test 2.
  - **Non-uniform "spread" projection**: deterministic, exponential decay
    of inclusion percentage per bin along the risk index. Denser at safe
    end, sparser at catastrophic. Same scheme on Z_in (low-warning dense,
    high-warning sparse). NOT "top 30 % safest" — spreads picks across the
    full range. Example: if target is 30 % of states, bin 0 keeps ~95 %,
    bin 9 keeps ~5 %.
  - **Uniform ρ_T(s,a), ρ_Z(s)** across all (s,a). Localized radii deferred.
  - **Single fixed policy** for first run (PDF's warning-threshold rule).
    Multi-policy sweep deferred.
- **Two reframes worth flagging.**
  - On policy enrichment: Tomer's "does it add meaning?" filter cut
    belief-dependent root (cosmetic, skipped) and made depth-dependent
    thresholds optional (only if we want per-depth bound breakdown plots).
    The thing that actually adds demonstration robustness is running A1
    over **2-3 distinct fixed policies** (leakage-heavy / mismatch-heavy /
    balanced) — deferred to follow-up.
  - On N-scaling: explicitly clarified that under translation-invariant
    transitions, scaling N exercises belief-vector size + leakage geometry
    + LP-runtime, but **not** structural diversity of state dynamics.
    Worth noting in writeup so we don't oversell "scales to diverse state
    spaces." Heterogeneous-dynamics follow-up added to todos (safer →
    higher stay-probability, riskier → lower).
- **Codebase reuse picture for A1.** Heavy reuse: `TabularPOMDP`,
  `UncertaintySets`, `ProjectedTVBall`, the robust-backup LPs from Solver A
  all carry over (replace UCB action selection with `action = π(h)`). New
  code: synthetic domain builder, full-tree expander under fixed policy,
  restrict→projected, bound-term computation (Δ_b, Δ_T, Δ_Z, Δ_K^rob,
  leakage), sweep driver.
- **Files changed this session.** `.thesis/todos.md` — added 2026-05-19
  section with three A1 follow-ups (per-node varying projection /
  heterogeneous dynamics / policy variations). `pdfs/planner A - design.pdf` was
  staged at session start.
- **Next session.** Lock remaining implementation-level details before
  writing code: (a) exact projection-decay shape parameterization (how
  steep, target % range); (b) ρ_T, ρ_Z defaults; (c) belief-shape params
  (lambda_*, mass_*); (d) reward defaults (PDF proposes some); (e) sweep
  output layout (csv/json). Then write the implementation plan and start
  on a new A1 experiment folder.

### 2026-05-13 — Session 15: FrozenLake episode viewer (v1+v2+v3) + `/audit-plan` slash command
- **Built the FrozenLake episode viewer end-to-end through v3.** Single
  Dash app at `viz/episode_app.py`. Loads JSON produced by new
  `experiments/frozenlake/record_episode.py`. Layered, all toggleable:
  - **v1 (grid)**: 8×8 grid (holes light grey, goal gold-rimmed +
    star, start green-rimmed). Belief heatmap (red gradient, per-step
    max for legibility). Agent disc (black) + NSEW observation halo
    (sensor bits as dark/light wedges N/S/E/W around the disc).
    Chosen-action arrow (per-action color). Q_robust bars at root.
    Step readout + cumulative return + per-step true→next state.
    Slider + Prev/Next buttons. Result badge in header (REACHED GOAL
    / FELL IN HOLE / TIMED OUT). Metadata as chip pills. White cards
    on light-grey bg.
  - **v2 (flow field)**: thin grey arrow per belief-supported cell to
    its modal next-cell under the chosen root action, derived from
    `sim_step` events at `depth=0` filtered to
    `action_chosen == a_t`. Width = per-cell normalized confidence
    `count[s, s'*] / total_from_s`. Skips agent's true cell (action
    arrow already conveys it) and no-movement transitions
    (`s'* == s`).
  - **v3 (adversary chevrons)**: small colored dot at each cell
    `s' ∈ S_in` whose radius encodes
    `|Σ_{s ∈ S_in} b(s) · (optimal_p_s[s'] − nominal_p_s[s'])|`.
    Sienna (`#a0522d`) for adversary added mass, teal (`#00a0a0`) for
    removed. Bottom-right of cell, threshold |delta| > 0.01.
  - **Reward heatmap**: static green per-cell overlay (`rgba(40,180,90,α)`,
    `α ∝ r(s)/max_r`). Shows the Tamir-style `1/(d+1)^3` landscape.
- **`record_episode.py` driver** (inline episode loop, doesn't touch
  `evaluation/episode.py`). CLI: `--horizon`, `--plan-horizon`,
  `--budget`, `--sims-per-backup`, `--p`, `--epsilon`, `--seed`,
  `--robust`, `--rho-t`, `--rho-z`, `--eta-obs`, `--eta-trans`,
  `--ucb-mode`, `--strip-events`. JSON output: per-step `{state,
  belief, action, obs, reward, next_state, q_robust, planner_events}`
  + scenario metadata. Early-exits on terminal absorption.
- **`--plan-horizon` decouples episode length from per-sim depth.**
  Original `horizon=args.horizon - t` conflated them — budget gets
  diluted across deep, mostly-irrelevant futures at long episodes.
  Now `plan_h = min(args.plan_horizon, args.horizon - t)` caps the
  planner's lookahead independently. At
  `--horizon 100 --plan-horizon 40 --budget 500 --p 0.95
  --strip-events`, agent reaches goal cleanly (total_R ≈ 1.6).
- **Two bugs caught and fixed mid-session:**
  - Recorder didn't early-exit on terminal absorption — JSON had a
    useless tail of "nothing happens" steps after the agent fell in a
    hole or reached the goal. Fixed: `if state == TERMINAL_STATE:
    break` after the Bayes update.
  - Viewer halo at step `t` used `record["obs"]` which is generated
    from `next_state` (the cell the agent *will* be in), not `state`
    (the cell it *is* in). So wedges described neighbors of the wrong
    cell. Fixed to use `episode[t-1]["obs"]` (the obs that informed
    the current belief). At `t=0`, no halo (no prior obs yet).
- **Critical bug caught by `/audit-plan` on the v3 draft.** Initial
  aggregation pseudocode used `belief[s]`, but
  `last_backup["belief"]` is `|S_in|`-indexed, not `N_STATES`-indexed
  (per `compute_belief()` at `robust_pomcp.py:482-488`). Plan + impl
  corrected to build `state_to_w = {s: belief[j] for j, s in
  enumerate(S_in_list)}` first, then `w = state_to_w.get(s, 0.0)`. If
  shipped as written, would have crashed on first run.
- **`/audit-plan` slash command created** at
  `~/.claude-thesis/commands/audit-plan.md`. Mirrors Tomer's
  personal-profile `audit-plan` but adapted to the thesis context:
  added **Math correctness** lens (index conventions, distance metric,
  sub-prob semantics); added **Viz / plot correctness** lens;
  replaced "Sensitive paths" with **Numerical / degenerate cases**
  lens (`ρ=0`, singleton `S_in`, terminal absorption, LP infeasibility).
  Action list uses two separated tables (`5a` auto-fixes + `5b`
  decisions, with per-row `#` for easy reference). Used twice this
  session; caught the belief-indexing bug above + several smaller
  fixes (marker corner offset, smoke-coverage gaps, etc.).
- **Plans written**: `~/.claude-thesis/plans/v2-flow-field.md`,
  `~/.claude-thesis/plans/v3-adversary-chevrons.md`. Audited and
  revised. (The original `i-put-you-in-transient-lagoon.md` from
  earlier in the session covers v1.)
- **Random-rollout limitation surfaced as a real research issue.**
  POMCP's uniform-random rollout on 8×8 FrozenLake with 10 holes
  essentially never reaches the goal during rollouts — planner has no
  goal signal in MC returns; Q-values driven entirely by the very-
  peaked shaped reward. At `b=500/h_plan=40/p=0.95` agent reaches goal;
  at smaller budgets it doesn't. **Discussed but deferred** the
  goal-aware rollout fix (Manhattan-greedy or ε-greedy toward goal) —
  would unblock E1/E3 quality but bakes scenario knowledge into the
  solver. Tomer's not interested in fixing this right now; flagged
  for later.
- **Reward shape tuning discussed.** `1/(d+1)^3` is very peaked (start
  cell ≈ 3e-4, ratio d=1:d=14 ≈ 420×). Considered `1/(d+1)^2` (ratio
  60×) and `1/(d+1)` (ratio 7.5×). Tomer leaning toward changing it;
  the actual swap is just `**3` → `**2` or `**1` at
  `frozenlake_problem.py:300`. Not changed this session.
- **Viewer iterations driven by Tomer's eyeball feedback**: red→black
  agent disc (red was fighting the new red belief heatmap), Q-bar
  width clipping fix, `dcc.Store` removed (unused — was shipping the
  full episode JSON to the browser per request), unified flow + slider
  callback to kill double-roundtrip lag, NSEW wedge colors revisited
  twice, adversary marker corner offset + color choice (mulberry
  rejected — too close to belief red, swapped to sienna), per-toggle
  inline color swatches + explanations, multi-cause availability note
  with localized-ρ_T-aware messaging.
- **Two prompts drafted at session end** for Tomer to take to a chat
  Claude conversation: (a) full thesis context for experimental
  design, (b) algorithm-only deep dive. Both copy-paste ready.
- **Next session.** E1 baseline equivalence on FrozenLake is the
  cleanest unblocked move — vanilla `BasicPOMCP` vs `robust(ρ=0)`
  should agree within sampling noise on the nominal world (same as
  the Tiger E1 result). After that, E3 design needs reward-shape +
  rollout-policy decisions first. The episode viewer is now solid
  enough to be the eyeball-first debugging tool when sweeps surface
  weird per-config outliers.

### 2026-05-10 — Session 14: scenario refactor + seed fix + FrozenLake constructor + smoke test
- **Big-picture decision.** Loaded Tamir et al. 2025 (RSS — Robust
  Sparse Sampling). Stole their design lesson: *localize uncertainty
  to where misspecification is catastrophic*, not uniform. Tamir
  uses FrozenLake (8×8) and CartPole as MDP toys with localized
  perturbations. We adopt FrozenLake as our next POMDP toy.
  CheeseMaze and RockSample go into deferred-toys list (after
  FrozenLake yields a clean robust-vs-vanilla picture).
- **Why retire Tiger as headline.** |S|=2, |Z|=2 makes projection
  degenerate (the paper's novelty barely exercises). One policy
  structure (listen-then-open) → robustness only shifts threshold.
  Reward asymmetry confounds the story. E3 robust-vs-vanilla gap is
  ~1σ at the calibrated edge.
- **Scenario refactor (framework-wide).** Replaced Tiger-specific
  plumbing in `evaluation/sweep.py` with a generic `Scenario`
  dataclass at `evaluation/scenario.py`. Each toy provides
  `make_<name>_scenario()` returning `(name, nominal_model,
  perturb_world, metrics_fn, metric_columns, results_dir,
  uncertainty_factory)`. Migrated `build_perturbed_tiger`,
  `tiger_metrics` to `experiments/tiger/tiger_problem.py`. Deleted
  `evaluation/perturbation.py`. Updated `__init__.py`. Verified:
  Tiger E1 BasicPOMCP byte-identical to historical (modulo
  date+runtime).
- **Pre-existing seeding bug found + fixed.**
  `robust_pomcp_plan(...)` at `solvers/robust_pomcp.py:64-65`
  creates an unseeded RNG when `rng=None`; `make_robust_planner`
  in sweep.py never passed one. So every historical robust_pomcp
  result was non-deterministic. Fix: thread per-trial planner_rng,
  derive world+planner rngs from one seed via
  `np.random.default_rng(seed).spawn(2)`, build robust planner
  closure inside the trial loop. BasicPOMCP planner stays outside
  (uses Python stdlib `random` re-seeded per trial; moving its
  build inside changes pomdp_py behavior in some
  object-identity-dependent way — investigated, dropped, shift
  was within 1σ noise). Verified: two back-to-back post-fix runs
  produce byte-identical CSVs (modulo runtime).
- **Historical CSV caveat.** Pre-fix robust_pomcp results
  (2026-05-03, 2026-05-08) are valid samples but not
  point-reproducible. Worth re-running Tiger E2/E3 once for
  canonical reproducible baselines.
- **`Scenario.uncertainty_factory` added.** Supports localized
  uncertainty (Tamir-style hole-adjacent ρ_T for FrozenLake) while
  Tiger keeps uniform. The factory is `(rho_T, rho_Z) →
  UncertaintySets`; Tiger's wraps `uniform_uncertainty`,
  FrozenLake's builds zero-everywhere-except-HOLE_ADJACENT for ρ_T.
  `run_config` calls `scenario.uncertainty_factory(...)` instead of
  hardcoded `uniform_uncertainty(...)`.
- **FrozenLake design locked.**
  - 8×8 gym `FrozenLake-v1` standard layout. |S|=65 (64 grid + 1
    absorbing terminal). |A|=4. |Z|=16. 10 holes, 26 hole-adjacent
    frozen cells.
  - Initial belief concentrated at start cell (0,0).
  - Reward Tamir-shaped: `r(s) = 1/(d(s)+1)^3` for frozen, +1 at
    goal, 0 at hole and terminal.
  - Slip: p intended, (1−p)/2 each perpendicular, 0 backward.
    Off-grid stays in place. Default p° = 0.6 (less slippery than
    Tamir's 0.4).
  - Obs: 4-bit hole-adjacency (N/S/E/W: "is that direction a
    hole?"). Bits flip independently with probability ε. Off-grid
    edges read as 0. Terminal observation uniform over 16. Default
    ε° = 0.05.
  - **Z uncertainty uniform across cells** (world ε°+η_obs;
    planner ρ_Z uniform).
  - **T uncertainty localized to HOLE_ADJACENT** (Tamir-spirit).
    World uses p° outside, p°−η_trans inside (more slippery).
    Planner ρ_T(s,a) = 0 outside, ρ_T inside.
  - Calibration math for E3 (deferred to E3 design): ρ_Z ≈ 8·η_obs
    (joint TV of 4 indep Bernoullis), ρ_T = 2·η_trans (slip row L1
    TV).
- **`UncertaintySets` already had per-(s,a) ρ_T support.** No
  LP/solver-side change needed for localized uncertainty.
- **Files created.** `experiments/frozenlake/{.gitkeep,
  frozenlake_problem.py, frozenlake_test.py}`. Constructor sanity
  passed: T/O rows sum to 1; cell (2,2) at ε=0.05 gives
  P(z=4)=0.8145 (matches deep-dive math); localized ρ_T zero
  outside HOLE_ADJACENT, set inside; perturbation correctly
  uniform-Z + localized-T. Smoke test runs robust_pomcp_plan and a
  full episode (horizon=30, budget=50): trial-1 fell in a hole
  (sensible at low budget on slippery world), metrics shape
  verified.
- **Two pushbacks worth flagging.** (1) Tomer caught me
  rushing to a fix when the "BasicPOMCP behavior shifted" symptom
  appeared post-seed-fix — pushed me to diagnose first instead of
  patch first. Probes confirmed neither Python random nor numpy
  global state was consumed by `make_basic_pomcp_planner`, so it's
  some downstream object-identity thing in pomdp_py; shift was
  within sampling noise so we accepted and moved on. (2) Tomer
  flagged my model layout was unclear (5+6 conflated) — forced me
  to write out concretely "where is uncertainty, where isn't, where
  does world differ from nominal" in tabular form. The clarification
  (ε is sensor noise level; ρ is planner's hedge over what ε might
  be — independent dimensions) was load-bearing for the rest of the
  design.
- **Next session.** Step 4 of FrozenLake plan:
  `experiments/frozenlake/run_experiments.py` with E1 (baseline
  equivalence) + E3_Z (2D ρ_Z × η_obs) + E3_T (2D ρ_T × η_trans).
  Run `--only E1` first to gauge runtime; estimated ~50 min total
  at n=50/budget=100/horizon=30. Step 5 (FrozenLake-specific
  plotting) after first results land.

### 2026-05-08 — Session 13: results dir restructure + E3 redesigned as 2D rho_Z × eta sweep
- **Walked through `compute_robust_q`** (lines 428–441 — Step 1 obs LP).
  Wrote down model-structure semantics, projected TV-ball conventions
  (`‖q-p_o‖₁ + |Σq - m_o| ≤ ρ`, each unit moved costs 2 TV). Set up
  calibration math for later in the session.
- **Results directory restructured** into per-experiment folders:
  `results/<Ek>/csv/*.csv` and `results/<Ek>/figures/*.png`. Touched
  `sweep.py` (`run_sweep` writes nested; `append_to_index` arg renamed
  `filename → relative_path`) and `viz/plot_results.py` (`find_csvs`,
  every plot function, `FIGURES_DIR` removed, usage docstring rewritten).
  Migrated E1 CSVs into the new layout. Deleted 8 stale smoke/perf
  CSVs from earlier development. INDEX.md mojibake (`�` from earlier
  encoding-mismatched writes) replaced with proper em-dashes.
- **E1 at n=100 confirmed equivalence.** BasicPOMCP −40.42 vs
  robust(ρ=0) −45.92, gap 5.5 ≈ 0.5σ — sampling noise. Bimodal violin
  structure now legible (top bulge = successful listen-then-open;
  second bulge ~ −85 = single-wrong-open episodes).
- **Stale editable install** was masking `import robust_pomdp` — old
  `.pth` pointed at the deleted Julia migration worktree
  (`.claude/worktrees/quizzical-mendel-48380d`). Tomer reinstalled with
  `pip install -e .` from repo root.
- **E2 docstring added** describing the pessimism-tax framing. Result at
  n=100/budget=100 is essentially flat (Session 9's surprising upward
  slope did not replicate).
- **Calibration math written down.** Codebase TV is `‖P-Q‖₁` (no ½).
  Tiger's symmetric reduce-accuracy perturbation gives row-TV = 2·η, so
  the calibrated hedge for an η-perturbation is `ρ_Z = 2·η`. Original
  E3 grid (ρ_Z up to 0.30 at η=0.20) was systematically
  under-calibrated.
- **E3 redesigned as 2D ρ_Z × η grid.** Replaced the original
  single-η=0.20 sweep. For each ρ_Z in {0.0, 0.05, 0.1, 0.2, 0.3},
  sweep 5 evenly-spaced η in `[0, ρ_Z/2]` (over- to fully-calibrated;
  truth always inside ball). Plus BasicPOMCP at each distinct η
  (deduped — vanilla doesn't read ρ_Z): **21 robust + 11 vanilla = 32
  configs.** Tomer ran at n=100/budget=100; CSV at
  `E3/csv/E3_2026-05-08_rho_z_eta_sweep.csv`. Headline plot is
  fan-of-curves (x=η, one curve per ρ_Z, vanilla as dashed spanning
  reference). Removed `plot_e3_violins`, `plot_e3_actions`,
  `plot_e2_e3_crossover`, `plot_wrong_open_rate_e2_vs_e3` — they don't
  fit the 2D structure cleanly. Old single-η CSVs and 5 stale PNGs
  deleted.
- **Two pushbacks worth flagging.** (1) After Tomer's "csv + figures
  subdirs" amendment I jumped to parallel tool calls without restating
  the plan; he intercepted, wanted explicit plan-then-go even on small
  mods. (2) The 2× calibration factor — he initially proposed
  η ∈ [0, ρ_Z]; we untangled that this would probe up to 2× ball-radius
  in TV-distance terms, settled on η ∈ [0, ρ_Z/2].
- **Parked**: show in the future that the paper's projection novelty
  enables a tractable online solution + display runtime performance.
- **Next session:** open the new E3 fan-of-curves plot, discuss what
  the data says. If clean → document the robust-vs-vanilla headline.
  If noisy → bump budget to 500. Beyond E3: E4 currently uses
  `robust_pomcp(ρ=0)` as the implicit-vanilla baseline (Session-8
  trick); update to BasicPOMCP for consistency with E3. Flag for
  thesis writeup: the "ρ_Z in this codebase is 2× standard TV radius"
  convention is confusion-prone.

### 2026-05-08 — Session 12: `verify_robust_backup` replaced with a real projected-robust test
- Tomer flagged the existing two-layer `verify_robust_backup.py` had blind
  spots: Layer 1 used `rho=0` (uncertainty set degenerates to a singleton,
  robust math is not exercised) and Layer 2 used Tiger with `Z_in = Z` and
  `S_in = S` (projection is not exercised either). Both layers passed even
  though they only validated the nominal Bellman backup.
- Replaced both layers with a single check on a synthetic 4-state, 2-action,
  4-obs POMDP. `S_in = {0, 1}`, `Z_in = {0, 1}` (strict subsets);
  `rho_T = rho_Z = 0.2` uniform; `belief = [0.6, 0.4]`; `V_children = [+10, -20]`
  (mixed signs to force a real LP trade-off). Step 1 LP optimum is reached via
  within-`Z_in` shift; Step 2 LP optimum involves drawing mass from `S_out`
  into `S_in` — both LP modes exercised in one test.
- Hand-derived `Q_robust = -5.12` (intermediate `w = [-5, -11]`,
  `σ = [-7, -5.8]`). Full derivation lives in
  `experiments/tiger/verify_robust_backup_derivation.md` — matrices, per-LP
  "compare moves" tables (per-TV-unit gain for each adversary strategy), and
  a debug-time quick-reference. Test asserts `|Q + 5.12| ≤ 1e-6`. **Passes.**
- File no longer imports `tiger_problem`; location at `experiments/tiger/`
  is now slightly misleading. Left put — can move to a `tests/` dir later.
- **Next session:** unchanged from Session 11 — run viz plotter on E1 to
  verify it lands `violin_returns.png` + `action_breakdown.png`, then
  E2 (rho_Z sweep, nominal Tiger), then E3 (perturbed world, headline).

### 2026-05-07 — Session 11: Memory restructure + viz reimplemented + Julia legacy purged
- **Memory restructure.** Adopted dinov3-style indirection: profile-layer
  `MEMORY.md` is now a 7-line pointer; the canonical index lives at
  `.thesis/MEMORY.md` and lists every memory file to load. CLAUDE.md
  required-reading section trimmed to a pointer at that index. The
  `/session-start` command now reads the in-repo index and walks its entries
  (single source of truth — no more drift between command body and memory list).
  `todos.md` added to the index as always-load.
- **`/session-start` step 4 amended.** Now surfaces candidate next-steps from
  `project_progress.md`'s most-recent "Next session" line, `todos.md`, and
  roadmap priorities before asking focus. Today's session opened without
  surfacing them — that's the gap that triggered this change.
- **Viz plots reimplemented (todos.md #2 → done).** `viz/plot_results.py`
  rewritten from the lost commit `253b292`: `plot_e1_violin`, `plot_e1_actions`,
  `plot_e2_line`, `plot_e2_violins`, `plot_e2_actions`, wrappers, and
  `plot_e3_placeholder`. `read_latest()` now disambiguates summary vs raw via
  `raw: bool = False`. Output directories computed inside each function rather
  than at module top, so `--results-dir` overrides take effect. Default
  `RESULTS_DIR` repointed at `experiments/tiger/results/` (per-scenario layout).
- **Julia legacy + descendants purged.** Deleted: 15 `.jl` files in `src/`,
  `src/RobustOnlinePOMDP.jl`, the directory tree (`src/core/`, `src/debug/`,
  `src/evaluation/`, `src/solvers/`, `src/uncertainty/`, untracked empty
  `src/bounds/`); `Project.toml` + `Manifest.toml`; Julia entrypoints at
  `experiments/` root + `experiments/tiger_run.json`; legacy `experiments/results/`
  Julia data dir; `experiments/tiger/_capture_lp_reference.jl` +
  `lp_reference.json` + `verify_lp_correctness.py` (the Julia LP cross-check
  trio); migration smoke scripts `_smoke_phase1.py` / `_smoke_phase3.py`;
  `docs/julia_cheatsheet.md`. Layer 3 of `verify_robust_backup.py` (the
  hardcoded-Julia-Q cross-check) stripped out; layers 1 + 2 retained.
- **Memory rule saved.** New `feedback_temp_files.md`: auto-delete tmp/smoke
  scripts during cleanups, don't ask per file. Triggered by Tomer's "you
  should auto delete them if it is just a tmp file" today.
- **Branch cleanup.** Stale `claude/quizzical-mendel-48380d` migration branch
  deleted (already merged into main).
- **Cleanup committed + pushed** as `ff1aaaf` (42 files changed, +448/−192880).
  GitHub `main` now matches local: Python-only repo.
- **Paper auto-load wired.** Added `.thesis/reference_paper.md` (small wrapper
  pointing at `Robust_Online_POMDP paper.pdf`, 11 pages); registered in
  `.thesis/MEMORY.md` under a new "Theory reference" group; updated
  `/session-start` step 1 to Read the PDF after the memory files. Two repo
  changes uncommitted at session end (MEMORY.md modified + reference_paper.md
  new); session-start command lives outside the repo.
- **Next session:** start the actual research work — run viz plotter on E1
  to verify it lands `violin_returns.png` + `action_breakdown.png` in
  `experiments/tiger/results/figures/E1/`, then move into E2 (ρ_Z sweep on
  nominal Tiger). Beyond E2: E3 is the headline experiment (perturbed world).
  Tomer's framing: "actually start working on the experiments algorithms and
  all the real stuff" — infrastructure plumbing is done; this was the runway.

### 2026-04-12 — Session 1: Project Setup
- Created Claude memory directory with user profile, working guidelines, project context, roadmap, and progress log.
- Read paper draft to extract project context and key mathematical entities.
- Repo initialized with paper draft already present.

### 2026-04-15 — Session 2: Deep Planning Discussion
- Resolved: scope, distance metric, POMDPs.jl hybrid, problem sizes, solver strategy
- Created persistent discussion log in `docs/discussion/`
- See `docs/discussion/04_session_log.md` for full details

### 2026-04-16 — Session 3: Solver A Design
- Broke Solver A into 10 pieces, confirmed Option 3D (two-step LP), full rectangularity
- Both two-phase and interleaved variants, K as tunable parameter

### 2026-04-18 — Session 4: Solver A Implementation
- Wrote full pseudo-code (`docs/discussion/01b_solver_A_pseudocode.md`)
- Implemented all 5 source files for Solver A
- Set up Project.toml + module file

### 2026-04-21 — Session 5: Tiger Test + First Bug
- Tiger problem + test script + run diagnostics
- Julia cheatsheet written; walked Tomer through VSCode/REPL basics
- Fixed bug 1 (Categorical/SubArray type)
- Fixed bug 2 (unvisited action Q_robust=0 polluting V_robust) — added V_rollout field, updated max logic to exclude unvisited
- **Open:** Q_robust vs Q_nominal still looks odd. Next session: debug interactively with breakpoints

### 2026-04-24 — Session 6: Event-Replay UI + Dirty-Flag Pruning
- After trying D3Trees + Pluto (both rejected), settled on event-based logging: Julia emits chronological JSON event log → Python Dash replays it
- Added unique `id` per tree node (NodeIdCounter), `V_rollout`/`has_rollout` fields
- Event types: sim_start, sim_step, expand, rollout, backup_nominal_step, sim_end, robust_backup (with ObsLPResult/TransLPResult)
- Built `viz/app.py`: dash-cytoscape tree + step/sim/backup navigation + per-event detail panel + node detail panel
- Fixed UI replay bug: V_nominal wasn't propagating to parent history node after Q_nominal updates
- **Dirty-flag pruning in robust_backup!**: short-circuit clean subtrees using invariant (no dirty descendants under a clean node). Verified: 81 events for budget=50/K=5
- UI polish: all sidebar tables (UCB, rollout, LP obs/trans) now use shared TABLE_STYLE with per-cell borders for proper grid rendering
- **Open:** still verifying Q_robust vs Q_nominal behavior using the replay UI

### 2026-04-30 — Session 10: Decision to migrate to Python; plan written; Phase 0 done
- **Big decision:** Tomer pulled the plug on Julia/VSCode. Adopting VSCode for Julia has been a productivity drag and he wants to move everything back to Python/PyCharm. The math/algorithms aren't the issue — the tooling was.
- **Migration plan written:** `C:\Users\TomerMassas\.claude\plans\if-i-tell-you-velvety-tulip.md`. Seven sections (scope, library choices, repo layout, translation conventions, migration order/verification, opportunistic fixes, risks). Iterated section-by-section with substantive pushback from Tomer on three items: (a) parallelism/scaling concerns made me add the `ProjectedTVBall` solver-swap escape hatch + thread-pool path; (b) docs preservation including gitignored files; (c) per-scenario `experiments/<name>/` folder structure; (d) verification framing — Julia is a soft cross-check, not a bit-for-bit bar (he was right; I had over-claimed).
- **Locked decisions worth recalling next session:**
  - LP solver: `highspy` (same HiGHS as Julia → numerical parity), wrapped behind a `ProjectedTVBall` class so swapping to Gurobi/OR-Tools is a one-day change later if scale demands.
  - POMCP baseline: `pomdp_py` (replaces BasicPOMCP for E1's independent-baseline property; was originally going to be in-repo vanilla until Tomer pushed back).
  - Repo layout: src-layout (`src/robust_pomdp/`), per-scenario `experiments/tiger/` (future scenarios get sibling folders, e.g. `experiments/rocksample/`).
  - viz unchanged except one additive CLI flag each: `--json` on `viz/app.py`, `--results-dir` on `viz/plot_results.py`. Defaults preserve current behavior. Soft amendment to "viz untouched" constraint, documented in §3.5 of the plan.
  - Verification: functional correctness via hand-derived references + qualitative sanity (e.g. `Q(listen) > Q(open-*)` at uniform belief). Julia outputs are a soft cross-check (1e-3 LP, 1e-2 backup), not strict bars.
  - Indexing: 0-based throughout Python (Julia is 1-based) — flagged as the dominant translation hazard in §4.7 of the plan.
  - LP perf fix lands as part of Phase 2 (single persistent `Highs` instance + parameter updates) — addresses the Session 8/9 ~170× slowdown for ρ>0.
- **10-phase migration order** in §5 of the plan with a §1.3 verification gate per phase. **Deletion of `.jl` files is gated on Tomer's explicit sign-off** (§1.3 #8), NOT automatic after Phases 1–8 pass.
- **Phase 0 done:** `pyproject.toml`, package skeleton at `src/robust_pomdp/` + 5 subpackages with empty `__init__.py`, `experiments/tiger/.gitkeep`, `.gitignore` extended for Python (`__pycache__/`, `*.egg-info/`, `.venv/`, etc.). Tomer set up venv in PyCharm (Python 3.11 base, `pip install -e .`) and confirmed `python -c "import robust_pomdp"` exits silent.
- **Worktree note (operational fact, easy to forget):** the migration lives on branch `claude/quizzical-mendel-48380d` in worktree at `C:\Users\TomerMassas\Documents\GitHub\Robust-POMDP\.claude\worktrees\quizzical-mendel-48380d`. Tomer's main repo path is untouched — he opened the worktree as a separate PyCharm project for the migration. After Phase 9 + sign-off, the branch merges into `main` (`git merge claude/quizzical-mendel-48380d` from the main repo), and the worktree is removed.
- **No commits yet** — all Phase 0 files are uncommitted on the worktree branch. Per Tomer's working guidelines, he drives commits.
- **Next session:** Phase 1 — port `pomdp_types.jl` → `core/pomdp_types.py`, `tree.jl` → `core/tree.py`, `uncertainty_sets.jl` → `uncertainty/uncertainty_sets.py`. Wire imports into `__init__.py`. Smoke test that types instantiate (no §1.3 gate at this phase).

### 2026-04-30 — Session 9: E2 Hands-On Run + Surprising Result
- Walked Tomer through running experiments himself: shell vs REPL invocation, REPL `include("experiments/run_experiments.jl")` with `ARGS = ["--only", "E2"]`, chaining the Python plotter via `run(`...`)`, how to interrupt with Ctrl+C (and what to do when the LP solver doesn't respond → kill from outside)
- Confirmed timing: at budget=100, ρ_Z=0 ≈ 35 ms/episode (LP short-circuits); ρ_Z>0 ≈ 6 s/episode (LP runs each backup). 170× slowdown is purely the LP setup overhead in `minimize_over_projected_tv_ball`
- Tomer ran E2 successfully at n_trials=10, budget=100. CSVs + 3 plots produced in `figures/E2/`
- **Surprising result**: trend is the OPPOSITE of the pessimism-tax hypothesis. Mean return climbs from −51 (ρ=0) to ~−12 (ρ≥0.2). Mechanism: Tiger's asymmetric reward (−100 wrong vs +10 correct) makes caution a winning strategy even on the nominal world. Robust planner opens less, more confidently → fewer catastrophic wrong opens. Wrong-open rate drops from 33% to 13% as ρ_Z grows
- Caveats noted: n=10 SEM is huge (~15-28), all error bars overlap; budget=100 is small and the effect may be a low-budget artifact (vanilla planner under-funded, can't find optimal policy without pessimism's safety net). Need to rerun at higher n / higher budget to claim the trend
- Walked through how to read the violin vs action-bar plots — the violin is per-episode return distribution, the action bars are aggregate counts; a fat violin region near −100 doesn't mean "lots of listens" but "individual episodes that hit one wrong open"
- **No code changes this session** — all interpretation and runtime
- **Next session:** still need LP reuse fix OR commit to a budget=500/n=50+ scope and let it run for an hour

### 2026-04-28 — Session 8: E1 Plot Revamp + E2 Run Hits LP Bottleneck
- E1 plots replaced (per Tomer's pushback that the bar-of-means hid the actual data):
  - `plot_e1_violin` — per-episode return distribution; `plot_e1_actions` — stacked correct/wrong/listen counts
  - Output moved to `experiments/results/figures/E1/`
- Infra: `run_config` now also returns per-trial rows; `run_sweep` writes a sibling `_raw.csv` with one row per (config × trial). Needed for the violin (KDE needs samples)
- Walked Tomer through how violins encode data, what each visual element means, and how Tiger's discrete reward structure (-1 / +10 / -100) produces the multimodal shape
- Discussed E2 conceptually: pessimism-tax experiment on nominal world, ρ_Z ∈ {0, 0.05, 0.1, 0.2, 0.3}. Clarified `model_planner` vs `model_true` decoupling
- Hit a `value` name collision (POMDPs.jl + JuMP both export it). Fixed with `JuMP.value(...)` qualification at `projected_sets.jl:86`
- **E2 ran for 46+ min on config 2/5 and got KILLED.** LP setup overhead at non-zero ρ_Z is the bottleneck — every robust backup builds a fresh JuMP model + HiGHS solve. ~200K LP solves per config
- Updated working guidelines memory: cut "ready to proceed?" pushy phrasing; calibrate length to question rather than dumping or being curt
- **Next session:** address LP overhead — either rewrite `minimize_over_projected_tv_ball` to reuse a single JuMP model with parameter updates, or reduce E2's n_trials/budget for a faster sweep

### 2026-04-25 — Session 7: LP Walkthrough + Experiment Harness
- **LP walkthrough**: stepped through robust backup #1 (open-right) and #2 (listen) with Tomer using actual screenshots from the UI. Verified all numbers match by hand (e.g., factor-of-2 effect: ρ_Z=0.1 gives 0.05 deviation per entry). Key clarifications:
  - TV in this codebase is L1 (`Σ|P-Q|`, no ½ factor) — defined at `uncertainty_sets.jl:19`, enforced in LP at `projected_sets.jl:66`
  - Lines 66 (TV budget, unsigned absolute movement) vs 70 (mass balance, signed net flow) measure different things
  - Predicted Q_robust(open-right) ≈ −139.05 and Q_robust(listen) ≈ −97.4 — both verified
- Two UI bugs fixed in `viz/app.py`: history node `N` was never updated (initialized at 0, never bumped); also `dagre` layout switched from `breadthfirst` so subtrees occupy disjoint regions
- **Experiment harness built (full pipeline ready, E1 verified)**:
  - Renamed `batch_size` → `sims_per_backup` (more self-documenting)
  - Added deps: CSV, DataFrames, POMDPs, POMDPTools, BasicPOMCP, Random, Statistics
  - New `src/evaluation/` directory: belief_update, episode (plan→act→observe→Bayes→loop), perturbation (Tiger reduced-accuracy), pomdps_adapter (TabularPOMDPWrapper for BasicPOMCP), sweep (ExperimentConfig, run_config, run_sweep with CSV+INDEX.md output)
  - Vanilla POMCP baseline = BasicPOMCP from JuliaPOMDP ecosystem (NOT the implicit ρ=0 trick)
  - `experiments/run_experiments.jl` defines E1–E7 with `--only` CLI flag
  - `viz/plot_results.py` reads CSVs, produces matplotlib figures (decoupled — replot without rerunning)
  - **E1 verified**: BasicPOMCP −46.91 vs robust(ρ=0) −38.22 (diff = 8.69, SEM-of-diff ≈ 11.5 → not significant; backends agree within sampling noise)
- **Next session**: run E2–E7 (full sweep) and analyze results
