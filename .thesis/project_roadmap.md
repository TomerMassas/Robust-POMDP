---
name: Project Roadmap
description: Active todos, priorities, and goals
type: project
originSessionId: 60e505b2-035f-4545-ad11-905ae1961031
---
## Status: **Python migration in progress** — Migration Phase 0 done, Phase 1 (core types) is next session. All Julia-side algorithmic work paused. Plan: `C:\Users\TomerMassas\.claude\plans\if-i-tell-you-velvety-tulip.md`.

## Python Migration (active focus)
- **Plan:** `C:\Users\TomerMassas\.claude\plans\if-i-tell-you-velvety-tulip.md` — 7 sections, read at start of every migration session.
- **Worktree:** `C:\Users\TomerMassas\Documents\GitHub\Robust-POMDP\.claude\worktrees\quizzical-mendel-48380d`, branch `claude/quizzical-mendel-48380d`. Tomer's main repo path is untouched until merge.
- **10 migration phases**, gate per phase per §1.3 of the plan.
  - [x] Phase 0 — pyproject.toml, package skeleton, venv install verified (`import robust_pomdp` silent).
  - [ ] Phase 1 — port core types (pomdp_types, tree, uncertainty_sets) — **next**.
  - [ ] Phase 2 — LP layer with reuse fix + Tiger problem; gate §1.3 #2 (LP correctness).
  - [ ] Phase 3 — logger + JSON export.
  - [ ] Phase 4 — solver + tiger_test + viz `--json` flag; gates §1.3 #3, #4.
  - [ ] Phase 5 — episode + perturbation + sweep + viz `--results-dir` flag.
  - [ ] Phase 6 — pomdp_py adapter + vanilla baseline plumbing.
  - [ ] Phase 7 — experiment runner + E1; gates §1.3 #5, #6 (CSV viz contract, E1 baseline equivalence).
  - [ ] Phase 8 — LP perf gate; §1.3 #7 (≤ 1 s/episode at budget=100, ρ>0).
  - [ ] Phase 9 — cleanup (delete .jl, Project.toml, Manifest.toml; `git mv` legacy results); **gated on Tomer's explicit "shift complete" approval**, §1.3 #8.
- **Locked decisions** (see plan §2 for full rationale): LP via `highspy` (wrapped, swap-friendly); POMCP baseline via `pomdp_py`; src-layout package; per-scenario `experiments/tiger/`; viz unchanged except 2 additive CLI flags; verification = functional correctness + qualitative sanity, Julia as soft cross-check.
- **Algorithmic work (E2–E7, RockSample, bound computation) paused** until migration completes.

## Pre-migration Status (Julia side, frozen): Phase 5 complete (Solver A + experiment harness ready) → Phase 7 partial (E1+E2 prelim done, E3–E7 pending)

### Phase 1 — Foundation Types
- [x] Julia project setup (Project.toml, module structure)
- [x] Generic distance metric interface + L1/TV implementation
- [x] Core tabular POMDP types (transition/observation matrices, rewards)
- [ ] POMDPs.jl integration layer (extract matrices from POMDPs.jl models) — deferred
- [x] Uncertainty set types (parametric: nominal + radius + metric)
- [x] Belief representation (discrete distributions)

### Phase 2 — Projected Sets & Tree Structure
- [x] Projected uncertainty set computation (LP via JuMP for L1; generic interface)
- [x] History node type (stores sampled S^in, Z^in, projected belief)
- [x] Search tree structure (explicit construction + simulation-built)
- [x] V_rollout fallback for unvisited branches (bug fix from Session 5)

### Phase 3 — Value Functions & Non-Robust Bounds
- [ ] Value function computation (DP Bellman backups)
- [ ] Projected value function (over sampled support)
- [ ] Bound terms: Δ_b, Δ_T, Δ_Z, omitted trajectories
- [ ] Full non-robust bound (Eq. 63)
- [ ] Validation on tiny problems (Tiger, Baby)

### Phase 4 — Robust Bounds
- [ ] Robust kernel mismatch Δ_K^rob
- [ ] Robust omitted trajectories
- [ ] Full robust bound (Eq. 74)
- [ ] Validation: robust bounds hold on tiny problems

### Phase 5 — First Solver (Robust-Value POMCP)
- [x] POMCP tree search with robust value backups
- [x] Worst-case model computation at each node (two-step LP)
- [x] Robust UCB action selection (nominal/robust modes)
- [x] Tiger test script (T1 end-to-end passes)
- [x] Event-based logging + JSON export (src/debug/)
- [x] Python Dash replay UI (viz/app.py) — step/sim/backup navigation, LP inspection
- [x] Dirty-flag pruning in robust_backup! (short-circuit clean subtrees)
- [x] LP walkthrough verified by hand (Q_robust(open-right)=−139.05, Q_robust(listen)=−97.4 match)
- [x] UI history-node N tracking fix; dagre layout for non-overlapping subtrees
- [ ] Testing on RockSample (after Tiger experiments are done)

### Phase 6 — Additional Solvers
- [ ] Bound-Guided POMCP
- [ ] Flagship Adaptive Planner (design TBD)
- [ ] (Optional) Robust DESPOT

### Phase 7 — Benchmark & Experiments
- [x] Experiment harness: episode loop with Bayes belief update, BasicPOMCP baseline (POMDPs.jl), perturbation utility, sweep runner with CSV + INDEX.md output, Python plot_results.py decoupled from runner
- [x] E1: Baseline equivalence — BasicPOMCP −46.91 vs robust(ρ=0) −38.22, agree within sampling noise
- [x] E1 plots upgraded: violin + action breakdown, in `figures/E1/` subfolder; raw CSV plumbing added to sweep runner
- [x] E2 preliminary run (n=10, budget=100) — CSVs + 3 plots in `figures/E2/`. **Result is opposite of expected** (mean return climbs with ρ_Z); attributed to Tiger's asymmetric reward + low budget. n=10 SEM too noisy to claim a real trend
- [ ] **PRIORITY:** Optimize `minimize_over_projected_tv_ball` to reuse the JuMP model (avoid rebuilding per backup) — OR commit to a long E2 rerun at n=50+, budget=500 to validate the surprising trend
- [ ] Run E3 (ρ_Z sweep, perturbed world η=0.20) — headline experiment
- [ ] Run E4 (η sweep, vanilla vs robust ρ_Z=0.2)
- [ ] Run E5 (sims_per_backup sweep)
- [ ] Run E6 (UCB mode comparison)
- [ ] Run E7 (horizon sweep)
- [ ] Bound validation experiments
- [ ] Plots and results for thesis

See `docs/discussion/01_solver_design.md` for detailed solver analysis and `docs/discussion/` for all design discussions.
