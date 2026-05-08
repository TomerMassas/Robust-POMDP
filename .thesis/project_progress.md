---
name: Project Progress
description: Session-by-session work log
type: project
originSessionId: 60e505b2-035f-4545-ad11-905ae1961031
---
## Session Log

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
