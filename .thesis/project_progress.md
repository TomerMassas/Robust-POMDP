---
name: Project Progress
description: Session-by-session work log
type: project
originSessionId: 60e505b2-035f-4545-ad11-905ae1961031
---
## Session Log

### 2026-07-19 — Session 21: New AAAI paper (R-MCTS) read + UCB-variant planner pseudo-code (design only, no code)

First session after a ~4-week gap. Entirely design/pseudo-code — **no code
changed**. Tomer set a fresh roadmap: build the paper's **two R-MCTS planner
variants (UCB, RUDB)**, pseudo-code first and finalized before any
implementation, and implement in a **new folder** kept separate from the A1
work. Focus this session: the **UCB variant** (RUDB is next).

- **New paper loaded — `R_MCTS_AAAI_2027.pdf`** (14 pp incl. supplementary;
  replaces `R_POMDP_paper.pdf`). Title: "Robust Online POMDP/MDP MCTS Planning
  with Anytime Deterministic Guarantees." Key new content:
  - **RUDB** (Robust Upper Deterministic Bound, Eq. 19-22 / Thm 2): apply the
    robust Bellman *optimality* operator to the certified upper bound ->
    deterministic upper bound on the **optimal** robust value `V*_rect`. Its
    error term `eps` (Eq. 20/75) uses **shared** minimizing models with the
    value term (Eq. 73) — one backup yields both.
  - **Two solvers**: (i) UCB-exploration, POMCP-style + robust projected
    backups, inherits the Thm-1 fixed-policy certificate, converges to the
    robust value of its tree policy; (ii) RUDB-exploration + **exact nominal
    belief filter**, additionally attains the optimal-value guarantee.
  - **The RUDB decoupling** (its crux): exact belief update with the full
    nominal transition (per-node validity, Corollary 1) + robust backups still
    over the **projected** sets (tractability). Cost paid only in the filter.
  - Thm 1 = leakage-only certificate `eps_0 = R_max * Sum_j (1 - phi_j)` —
    exactly the current `compute_certificate`. Two-sided bracket `[V_hat +/-
    eps]` -> certified action ordering, early stop, pruning.

- **Baseline recap (comprehension, no change).** Wrote faithful pseudo-code of
  the two A1 run drivers (`runs/sweep_sampler/run.py`,
  `runs/root_action_bounds/run.py`): both thin drivers over one shared pipeline
  — `build_reference` (full-support robust expectimax = ground truth
  `V_full`/`q_root`) -> `build_sampled_projection_tree` (per-node MCTS-style
  sampling) -> `evaluate_robust_value` (`V_proj`, two-step LP) ->
  `compute_certificate` (leakage `eps`).

- **UCB variant design — locked decisions** (working name **CR-UCT**, Certified
  Robust UCT; rename free):
  - **~= Solver A wrapped as an anytime online planner**: every-sim robust
    projected backup -> certificate -> early-stop check.
  - **Two exploration modes** `ucb_mode in {nominal, robust}` (existing knob):
    nominal = sample-mean `Qbar`; robust = projected robust `Q_hat^rob`.
    Backups, final root decision (`argmax_a Q_hat^rob`) and certificate are
    identical across modes — mode only drives *exploration*.
  - **Beliefs = option B only** (deterministic projected-Bayes over the sampled
    support); **particle-count beliefs dropped entirely**. Root = unnormalized
    restriction of `b0` (Eq. 10); children renormalized. No exact Bayes *filter*
    (RUDB-only); the only exact belief used is root `b0`, which is all the
    certificate reads (leakage DP uses root belief + per-node supports/actions,
    never non-root beliefs).
  - **Incremental action expansion** (`A_hat(h)` = explored actions; expand one
    unexpanded action per visit, else UCB) — no full action support.
  - **Interleaved robust backup folded into `SIMULATE`** (belief carried down,
    robust value backed up on the way up — one recursion; Solver A's K=1 mode).
    No separate belief-recompute pass.
  - **Selective descent, one node per sim, NO descend-to-H** (Tomer rejected
    descend-to-H — wastes horizon compute on uninteresting root actions). The
    **unexpanded tail is charged as leakage**: new certificate base case — a
    frontier node at depth `k<H` with no children -> `I=0` (fully leaked beyond
    `k`), distinct from a genuine terminal (`I=1`). Deep/interesting branches ->
    tight brackets, shallow ones -> loose; the certificate concentrates
    tightness where the planner spent effort.
  - **Leaf value = `r(b_hat,a) = Sum_s b_hat(s) R(s,a)`** (immediate reward,
    truncated continuation). The tail *error* is a separate object living in
    `eps`, not in the value line.
  - **Early stopping in scope** (switchable): per-root-action bracket
    `[Q_hat^rob(a) +/- eps(a)]` (greedy continuation); stop when one action's
    lower bound clears every other's upper bound. Claim = best **tree-policy
    continuation** (-> robust-greedy as support grows), *not* `V*`-optimality
    (that's RUDB's).

- **Normalization — the paper-facing explanation Tomer needed.** It's the
  standard belief-MDP Bellman backup: renormalize each child so its value is
  *conditional on the observation*, so the observation-weighted robust backup
  (`min_{P_Z} Sum_z P_Z(z) V_child(z)`) reweights well-defined per-branch values
  instead of double-counting `P(z)`. Projected value is linear in the belief, so
  the unnormalized posterior (mass = `P_hat(z|b,a)`) already carries `P(z)`;
  three pairings — (1) normalized+weighted OK, (2) unnormalized+summed OK
  (= paper Eq. 11 flat sum), (3) unnormalized+weighted WRONG (double-counts ->
  `Sum P(z)^2 V`). Code uses (1); root stays unnormalized to carry depth-0
  leakage `1-m_in`.

- **Rollouts — corrected an over-broad "no rollouts" stance.** Hard rule is
  only: the **certified value `V_hat` and bound `eps` are computed on the
  truncated tree** (rollout noise isn't covered by `eps` -> would break the
  deterministic guarantee). Rollouts are fine for **exploration** — folded in as
  optional (`ROLLOUT` helper) feeding `Q_nom` only, gated
  `track_nominal AND use_rollouts`; never touch `BACKUP`/`V_hat`/`eps`. Fixes the
  depth-limited `Q_nom` weakness in nominal mode. Robust mode uses no rollout (a
  nominal rollout gives no robust tail estimate).

- **Open / next.**
  - **Finalize the UCB (CR-UCT) pseudo-code** — current draft lives in this
    conversation only; not yet written to a file.
  - Then **RUDB variant pseudo-code**: needs the exact-nominal-belief filter +
    the shared-minimizer RUDB backup (Eq. 19/73-75) + optimality bracket
    (Thm 2). New paper has the Thm 2 + Lemma 3 proofs — internalize before
    drafting.
  - **Only after both pseudo-codes are finalized** -> implement in a **new
    folder** (separate from A1).
  - Certificate change vs A1's `compute_certificate`: the frontier base case
    (`I=0` at unexpanded depth-`k<H` nodes) — A1 trees were always full-depth so
    it never arose.
  - Housekeeping drift noted (not fixed): `CLAUDE.md` + `.thesis/reference_paper.md`
    still name the old paper (`Robust_Online_POMDP paper.pdf`, "11 pages"); actual
    is `R_MCTS_AAAI_2027.pdf` (14 pp). `docs/discussion/{04_session_log,00_resolved_decisions}.md`
    are stale (Julia-era, stop at Session 5). `project_roadmap.md` "NEXT: Run E2"
    is stale (A1/RUDB track now).

### 2026-06-21 — Session 20: Leakage-only theory change + A1 replan (frozen greedy policy, two trees) + root-action Q-certificates

Consolidates the post-Session-19 push (unlogged working sessions; commits
`a923dd1` 06-03, `55f5580` + `738c6d0` 06-21). Throughline: Tomer updated the
paper notes, the certificate became **leakage-only**, and A1 was rebuilt around
a frozen full-robust-greedy policy evaluated on two projection schemes.

- **Theory change (load-bearing) — certificate is now leakage-only.** Projected
  belief = unnormalized restriction (Eq. 20): `b_hat(s)=b(s)` on S_in, 0 else.
  Theorem 1 (Eq. 27):
  `|V^rob - V_hat^rob| <= R_max * Sum_{j=t}^{t+H}[1 - inf_theta Sum_{tau in T_in} mu_hat(tau_{t:j})]`.
  - **Delta_K removed from the run path** (kept as legacy `delta_k.py` /
    `delta_k_exact.py`, no longer imported). **Delta_b = 0** under the
    unnormalized restriction. The j=0 term is the initial out-of-support mass
    `1 - m_in`; everything else is omitted-trajectory leakage.
  - `certificate.py` slimmed to `R_max*(delta_b + (1-phi_j))` summed over depth;
    `leakage.py` reduced to two full-TV-ball min wrappers (support encoded in the
    coefficient vector); `inside_trajectory.py` is a per-node backward DP.
  - **Belief-recursion cancellation** (why per-node works): in `compute_robust_q`
    the child renormalizer is the in-support obs likelihood `P_hat(z|b,a)`, so the
    recursion reproduces Eq. 42 (V_hat) with the ROOT belief — only the root needs
    unnormalizing, not every node. (I floated unnormalizing throughout; Tomer
    pushed back, correctly — that would double-count leakage.)

- **ProjectedTVBall full-support bug fixed.** At full support the LP still let the
  adversary drop mass (5.5 vs true 5.8). Added a `full_support` flag capping the
  delta+/delta- columns to 0 so it reduces to the plain full-TV ball (Sum=1).
  Tomer ran the standalone test himself; passed.

- **A1 replan (agreed 2.1 / 2.2 / 2.3).**
  - 2.1: plots show the trivial `+/- V_max = R_max*(H+1) = 40` envelope; the
    certificate is informative only where the band detaches from it.
  - 2.2: policy is the **full-robust-greedy** `pi_full(b)=argmax_a Q^rob(b,a)`,
    solved ONCE on the full-support expectimax tree, then FROZEN and evaluated on
    each projected sub-tree (sub-tree subset of full-tree => pi_full restricts
    cleanly; V_full is constant across a support sweep, no recompute). Resolved
    "isn't pruning-by-robust-Q cheating?" — we freeze the optimal policy and
    measure how well the projection reproduces its value; we do NOT prune by the
    bound.
  - 2.3: two projection schemes on the same POMDP — **uniform masking**
    (`runs/sweep_support_size`, fan plot vs m) and **per-node MCTS-style sampling**
    (`runs/sweep_sampler`, scatter vs avg support fraction).

- **Implementation.**
  - `greedy_solver.py`: `solve_full_greedy -> (V_full, pi_full)`; refactored into
    `_build_expectimax` + `_extract_policy`, exposing
    `solve_with_root_actions -> (V_full, q_root, pi_by_root_action)` (one
    expectimax solve; per-root-action continuation policies extracted).
  - `tree_evaluator.py`: `build_policy_tree_by_path` (obs-path-keyed policy),
    `project_belief` (unnormalized restriction), `projected_bayes_update`,
    `evaluate_robust_value`.
  - `sampled_projection.py`: per-node `S_in(child) ~ posterior`,
    `Z_in(ha) ~ obs marginal`, sampled with replacement + dedup (|support| <= K);
    root belief unnormalized, children renormalized.
  - Per-node support wired through `compute_robust_q` (full_support flags) and the
    leakage DP (in-support step iff `z in Z_in(ha) AND s' in S_in(child_z)`).
    Gold test (hand-shrunk supports, backward==forward at rho=0) confirms the DP.

- **Config cleanup.** Removed all delta_k args from `A1Config` (modules kept as
  standalone legacy). Regrouped: Sizes / Synthetic-POMDP / Belief / Uncertainty
  (shared) + Projection-uniform + Projection-sampler + Output. (YAML deferred.)

- **Root-action Q-certificate experiment — `runs/root_action_bounds/`** (the
  "can we stop planning?" plot). For each root action a, freeze
  `pi_a = "take a, then optimal"`; `Q_full(h0,a)=q_root[a]`; per (K, seed) build
  the sampled tree following pi_a and read `Q_proj(a)` + `eps(a)` (Theorem 1).
  Bands `[Q_proj +/- eps]` must contain the dashed `Q_full(a)` — and do.
  Separation = best guaranteed lower bound clears every other upper bound =>
  "stop." x-axis switched from raw K to **average support fraction** (K kept on a
  secondary top axis) per Tomer — more informative.
  - **Result (rho=0.05, N=M=10, H=3):** Q_full = proceed 13.62 (=V_full), inspect
    9.53, mitigate 9.50, abort 0. Validity holds at all K/seeds. Separation is
    **marginal**, appearing only near frac≈0.68 (K~1024): the continuing-action
    **cert-floor ~2.0** sits just under the proceed-vs-inspect half-gap ~2.05.
  - **Key insight:** the stopping point is where the cert-floor (set by rho and
    model sparsity — the adversary always leaks ~rho to states outside the nominal
    posterior support) drops below the action-value half-gap. Smaller rho or a
    wider value gap => earlier / cleaner separation.
  - **Abort caveat:** pi_abort is depth-0, but the leakage cert charges the
    initial `1-m_in` at all H+1 depths => `eps(abort)=(H+1)*R_max*(1-m_in)`,
    ~(H+1)x loose. Valid, -> 0 as m_in->1, and **never the binding constraint**
    (inspect/mitigate always bind). Flagged on the plot; not fixed.

- **Open / next:**
  - Re-run root-action bands at higher `n_seeds` (V4 was n_seeds=1 for iteration
    speed) for smooth floors.
  - rho-sweep (0.05 / 0.02 / 0.01) to visualize separation moving to smaller
    support fraction as the cert-floor drops.
  - Optional: tighten the abort cert to the depth-0 term; build the literal
    POMCP tree (phase B of 2.3 — the MCTS tree is currently approximated by
    per-node sampling).
  - `phase1_smoke` has stale assertions from model changes — Tomer said ignore.

### 2026-06-01 — Session 19: A1 exploration tooling + EXACT Delta_K + unnormalized-belief redefinition

Long session. Started from the Session-18 handoff (manual walkthrough of the
certificate smoke), then built out exploration tooling, an exact Delta_K
ground-truth tool, and a load-bearing theory change to the belief definition.
All code committed mid-session in `aecf1f1`.

- **Handoff closed.** Tomer manually walked `phase6_certificate_smoke.py`,
  confirmed, handoff archived. Added tree-viz HTML output to that smoke
  (`phase6_cert_tree_{full,projected}.html`) and removed a tautological
  assertion (`V_proj_z_fullsup` was recomputing `V_full` — dropped).

- **Synthetic POMDP model revisions** (several design calls):
  - `mitigate_drift` 0.20 -> 0.75: MITIGATE is an active risk-reduction action,
    so the drift IS the wanted behavior; 0.20 was too weak to recover within H.
  - Reward rebalance via `RewardSpec`: `mitigate_risky_cost` 8 -> 2,
    `mitigate_cat_cost` 30 -> 20. Discussion concluded uniform-ish action cost
    is cleanest; PROCEED carries the state-dependent value.
  - **Observation model is now class-aware.** Old model used a linear
    state-index -> obs-index map (`round(s/(N-1)*(M-1))`) that didn't respect
    safe/risky/cat class boundaries. New scheme: warning-threshold-driven obs
    ranges (safe -> [0, low_z], risky -> [low_z+1, high_z], cat ->
    [high_z+1, M-1]) with a smooth intra-class gradient (safe anchors first
    state at z=0, cat anchors last at z=M-1). New kwargs `warning_low`,
    `warning_high` on `make_synthetic_pomdp` (kept in sync with the policy).

- **Single-run driver: `experiments/a1/config.py` + `main.py`.** `A1Config`
  dataclass exposes every knob (sizes, support via picker OR explicit S_in/Z_in,
  policy thresholds, risk_fractions [None=default], action drifts, obs_noise, H,
  rho_T/rho_Z, full RewardSpec, belief shape, delta_k_mode, output controls).
  `main.py` builds full+projected trees, computes V_full/V_proj/certificate,
  prints a component breakdown, writes tree HTMLs. Relative `out_dir` resolves
  against `experiments/a1/`.

- **`support_picker` fix**: floor `K >= 1` for any positive `target_fraction`
  in both pickers (was returning `[]` for small m, e.g. m_Z=0.1 on M=5 ->
  round(0.5)=0 -> `ProjectedTVBall(n=0)` crash).

- **Sweep tooling: `runs/sweep_support_size/run.py`.** Fan-plot sweep over joint
  support fraction (m_S=m_Z). Versioned outputs (`data_V{n}.csv`,
  `fan_plot_V{n}.png`) so runs don't overwrite. Added `plot_only(version)` to
  re-plot from a saved CSV without re-running the sweep.

- **Certificate component breakdown.** `compute_certificate` now returns
  `delta_b_contribution`, `delta_K_contribution`, `leakage_contribution`
  (sum to `certificate`). Printed in `main` (with %), and as sweep CSV columns.

- **EXACT Delta_K (paper Eq. 71) — `src/robust_pomdp/bounds/delta_k_exact.py`.**
  - Motivation: the Eq. 47 split (`DeltaKComputer`) is a loose upper bound
    (triangle-inequality decoupling + independent sups sharing no T_hat).
  - Method: `||x||_1 = max_sigma <sigma,x>`, swap the two maxima ->
    `Delta_K = max_sigma [FullSup(sigma) - ProjInf(sigma)]`. For fixed sigma,
    rectangularity makes both inner problems linear-over-a-ball (reuses the
    existing LP wrappers). Naive enumeration over `2^(|S_in|*|Z_in|)` sign
    patterns; `max_free_bits=16` guard.
  - **Verification caught a real definitional bug**: I first summed over the
    full `Z x S` (including leakage), which made exact > split at rho=0. Fixed
    to **in-support `Z_in x S_in` only** -- Delta_K feeds the inside-mismatch
    (Eq. 25/72); leakage is the SEPARATE omitted term. After the fix all 4
    checks pass (rho=0 -> 0; exact <= split; monotone; brute-force gold 2x2 +
    random 3x3).
  - **Measurement**: split overestimates Delta_K by ~30-60% (exact/split
    ~0.61-0.78). On a small config, swapping exact for split tightened the
    total certificate ~9% (delta_K is ~40-60% of it; delta_b/R_max dominate the
    rest).
  - Wired `delta_k_mode` ("split"/"exact") + `delta_k_max_free_bits` through
    config -> main -> `compute_certificate` (callable seam in
    `_max_path_sum_delta_k`).
  - Timing measured: ~2.1x per +1 free_bit; free_bits=16 ~2 min/action,
    20 ~40 min/action. Naive enum is exponential in BOTH |S_in| and |Z_in|.
  - The polynomial-in-|S_in| method (for big |S_in|, small |Z_in|) is sketched
    but NOT built -- needs an on-paper positivity-clip proof first.

- **Belief redefinition (theory change; Tomer updating the paper).**
  - Projected belief is now the **unnormalized restriction** `b_hat(s)=b(s)`
    for s in S_in, 0 else (sub-probability summing to m_in) -- parallel to the
    projected P_T^in / P_Z^in models.
  - `Delta_b` redefined as the IN-SUPPORT mismatch -> **0** under this
    definition. The dropped mass (1 - m_in) is belief leakage, carried by the
    omitted term at j=0. This removes the prior over-counting (old normalized
    `Delta_b = 2(1-m_in)` was charged at every depth AND overlapped the leakage
    term -> effectively triple-counted the initial out-of-support mass).
  - `project_belief` stops renormalizing; `projected_bayes_update` UNCHANGED.
    Verified (by reading `compute_robust_q`) that the downstream renormalization
    CANCELS against the in-support obs likelihood in the backup, so storing
    normalized child beliefs is correct -- only the ROOT scale needed changing.
    (Corrected my own earlier wrong suggestion to unnormalize throughout, which
    would double-count leakage.)
  - `compute_certificate` restructured: drop the special j=0 case, charge
    leakage at all j=0..H. `delta_b` kept in the per-depth formula (=0 now) for
    generality. `delta_b` computed EXPLICITLY (b_full vs the actual projected
    root belief `node_data[root.id].belief` over the support), not hardcoded --
    stays correct if the belief definition changes later (per Tomer's request).
  - Consequence: V_proj is now the unnormalized/leaking value (paper Eq. 17);
    numerically smaller, true_error larger, but the certificate still bounds it.
    All smokes pass; validity holds in every regime.
  - POMCP/Tiger path untouched (`robust_pomcp.py` not modified; `project_belief`
    is A1-only).

- **Key insight**: the exact-Delta_K verification suite earned its keep --
  caught the in-support-vs-full-ZxS definitional bug before we trusted the tool.

- **Open / next.**
  - Polynomial exact Delta_K (big |S_in|, small |Z_in|): work the
    positivity-clip proof on paper, then implement.
  - Paper draft: write up the unnormalized-belief definition, Delta_b=0, and
    leakage-at-all-j certificate form.
  - More exploratory sweeps/plots (rho sweep, decay sweep) as Tomer drives.

### 2026-05-28 — Session 18: Phase 6 sub-steps 2 + 3 (inside-trajectory DP, certificate assembly)

Picked up mid-Phase 6 (sub-step 1 leakage primitives already done in a prior
unlogged session). Closed out Phase 6 with two sub-steps and the integration
smoke for the Eq. 74 certificate.

- **Phase 6 sub-step 2 (inside-trajectory tree DP, paper Eq. 73).** New
  `src/robust_pomdp/bounds/inside_trajectory.py` with
  `compute_inside_traj_probs(root, node_data, leakage, b0_full, H)` — for
  each `j ∈ {0..H}`, runs a backward post-order DP over the fixed-policy
  tree computing `I_k^j(s)` = worst-case probability the trajectory stays
  inside `(S_in, Z_in)` for the remaining `j − k` steps from `h_k` with
  state `s`. Under rectangularity, each non-terminal node's DP factors into
  two nested `TVBallLinearMinLP` calls (inner per-`s'` over `P_Z`, outer
  per-`s` over `P_T`). Also added `obs_min_over_TV_ball` /
  `state_min_over_TV_ball` LP-wrapper methods on `LeakageComputer` so the
  DP reuses the persistent solvers.
- **Terminal-node convention — the load-bearing math call this session.**
  At an aborted node `h_k` with `k < j`, set `I_k^j(s) = 1` for all
  `s ∈ S_in`. Justification (after I got it wrong twice and Tomer caught
  me): aborted paths have zero V mismatch at depth `j > k` (the path ended;
  V_full and V_proj both have nothing at depth `j > k` from this path), so
  the leakage contribution should be exactly 0 — which `I = 1 → (1 − I) = 0`
  encodes. Reward-agnostic: the depth-`k` mismatch at the abort node IS
  handled separately by the inside-mismatch path (`Δ_b + Σ Δ_K^rob`), so
  the convention doesn't depend on `R[s, ABORT]`. My earlier "requires
  R = 0" framing was wrong — I conflated the depth-`k` V-mismatch (handled
  by inside-mismatch) with the depth-`j > k` leakage (which is just 0).
  Smoke verifies: backward DP exactly matches a nominal forward DP that
  freezes aborted mass; full support → `phi_j = 1`; robust ≤ nominal;
  monotonicity `phi_{j+1} ≤ phi_j`.
- **Phase 6 sub-step 3 (certificate assembly, Eq. 74).** Two new files:
  `src/robust_pomdp/bounds/belief_mismatch.py` (closed-form
  `delta_b = 2(1 − m_in)`, derived in docstring as renorm-gap + dropped-
  mass = `(1 − m_in) + (1 − m_in)`); and `certificate.py` with
  `compute_certificate(...)` returning a dict with `certificate`,
  `delta_b`, `max_path_sum_delta_K` (length H+1), `phi`,
  `leakage_per_depth`, `per_depth_bound`, `R_max`, `m_in`.
- **Three design calls in sub-step 3.**
  - **`delta_K` aggregation in a branching tree**: max-of-sum over
    root-to-depth-`j` paths (tighter than sum-of-max-per-depth, which
    stitches deltas from histories that never co-occur on a real
    trajectory — Tomer pushed back hard on me when I offered both as peer
    options). Implementation: tree-walk tracking running sum at each node.
  - **`j` range**: `j = 0..H` inclusive, BUT at `j = 0` use only
    `R_max · delta_b` (drop the `(1 − phi_0) = (1 − m_in)` leakage term,
    since it's already inside `delta_b`'s out-of-support piece). For
    `j ≥ 1`, full formula `R_max · [delta_b + max_path_sum[j] + (1 − phi_j)]`.
  - **`R_max`**: auto-derived as `max(|R.min()|, |R.max()|)` per the paper.
- **Validity smoke passes.** `experiments/a1/smoke_test/phase6_certificate_smoke.py`
  builds full + projected trees, computes `V_full` and `V_proj` via
  `evaluate_robust_value`, asserts `cert ≥ true_error` in all three
  regimes:
  - ρ=0 + full support → cert = 0, true_error = 0.
  - ρ=0 + partial support (S_in=[0,1], Z_in=[0,1]): V_full=3.70,
    V_proj=9.14, true_error=5.44, cert=213.88 (~40× loose).
  - ρ=0.1 + same partial support: V_full=0.52, V_proj=5.94,
    true_error=5.43, cert=489.69 (~90× loose).
  The looseness is expected (Eq. 47 decoupling + per-depth max aggregation
  + delta_b/leakage overlap at j ≥ 1) — Phase 7's sweeps will quantify it.
- **Two workflow corrections persisted to `feedback_working_guidelines.md`.**
  - "Default to the tightest valid form from the paper; don't soft-pitch
    looser alternatives as peer options." Triggered by my sum-of-max
    proposal. Tomer: *"you are saving code lines on me? i dont understand
    your attitude."* Math IS the contribution; code-LOC isn't a real
    trade-off at our scale.
  - "Don't bundle orthogonal design questions into a 'pick from 1/2/3'
    ask." Triggered by me lumping three independent calls (aggregation +
    j range + R_max) under one umbrella prompt. Tomer asked them
    separately, one at a time.
- **Handoff updated** at Tomer's request: next session begins with him
  walking through `phase6_certificate_smoke.py` manually to verify the
  certificate end-to-end before Phase 7 starts.
- **Files this session (uncommitted at /session-end).**
  - New: `src/robust_pomdp/bounds/{inside_trajectory,belief_mismatch,certificate}.py`
  - Modified: `src/robust_pomdp/bounds/leakage.py` (+ 2 LP-wrapper methods)
  - New: `experiments/a1/smoke_test/{phase6_inside_traj_smoke,phase6_certificate_smoke}.py`
  - Modified: `.thesis/feedback_working_guidelines.md` (2 new bullets)
  - Modified: `.thesis/handoff.md` (replaced Phase 5 mid-impl handoff with
    Phase 6 wrap-up + manual-walkthrough note; gitignored, not committed)
  - Note: prior-session files (Phase 5 implementation + Phase 6 sub-step 1)
    are also untracked and being swept into this commit.
- **Next session.** Tomer manually walks through
  `phase6_certificate_smoke.py` — the printout includes `delta_b`, `R_max`,
  `m_in`, `phi`, `max_path_sum_delta_K`, and `per_depth_bound` per test.
  After he's satisfied, Phase 7 starts: `experiments/a1/run_a1.py` CLI
  sweep driver (build full tree once, sweep projection levels via
  `support_picker`, compute certificate + V_proj per `m`, append CSV row).

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
