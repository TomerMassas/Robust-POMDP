---
name: Project Roadmap
description: Active todos, priorities, and goals
type: project
originSessionId: 60e505b2-035f-4545-ad11-905ae1961031
---
## Status

**Python-only codebase.** Solver A (Robust-Value POMCP) is implemented and verified;
experiment harness E1–E7 is wired up; E1 has been run end-to-end. E2–E7 runs are
the immediate research priority. Bounds work (Phases 3–4 in the original phase
list) is the next implementation milestone.

## Phases

### Phase 1 — Foundation Types (done)
- [x] Tabular POMDP types (transition / observation matrices, rewards)
- [x] Distance metric interface + L1/TV implementation
- [x] Uncertainty set types (parametric: nominal + radius + metric)
- [x] Belief representation (discrete distributions)

### Phase 2 — Projected Sets & Tree Structure (done)
- [x] Projected uncertainty set computation (`ProjectedTVBall`, persistent Highs LP)
- [x] History node type with sampled S^in, Z^in, projected belief
- [x] Search tree structure (simulation-built)
- [x] V_rollout fallback for unvisited branches

### Phase 3 — Value Functions & Non-Robust Bounds (open)
- [ ] Value function computation (DP Bellman backups)
- [ ] Projected value function (over sampled support)
- [ ] Bound terms: Δ_b, Δ_T, Δ_Z, omitted trajectories
- [ ] Full non-robust bound (Eq. 63)
- [ ] Validation on tiny problems (Tiger, Baby)

### Phase 4 — Robust Bounds (open)
- [ ] Robust kernel mismatch Δ_K^rob
- [ ] Robust omitted trajectories
- [ ] Full robust bound (Eq. 74)
- [ ] Validation: robust bounds hold on tiny problems

### Phase 5 — Solver A: Robust-Value POMCP (done)
- [x] POMCP tree search with robust value backups
- [x] Worst-case model computation at each node (two-step LP)
- [x] Robust UCB action selection (nominal/robust modes)
- [x] Tiger smoke test (`experiments/tiger/tiger_test.py`)
- [x] Event-based logging + JSON export
- [x] Python Dash replay UI (`viz/app.py`) — step/sim/backup navigation, LP inspection
- [x] Dirty-flag pruning in robust_backup (short-circuit clean subtrees)
- [x] Robust backup correctness verification (`experiments/tiger/verify_robust_backup.py`,
      hand-trivial layer + qualitative Tiger sanity layer)
- [ ] Testing on RockSample (after Tiger experiments are done)

### Phase 6 — Additional Solvers (open)
- [ ] Bound-Guided POMCP
- [ ] Flagship Adaptive Planner (design TBD)
- [ ] (Optional) Robust DESPOT

### Phase 7 — Benchmark & Experiments (in flight)
- [x] Experiment harness: episode loop with Bayes belief update, pomdp_py POMCP
      baseline (planner_label `"BasicPOMCP"`), perturbation utility, sweep runner
      with CSV + INDEX.md output, decoupled `viz/plot_results.py`
- [x] E1: Baseline equivalence — nominal Tiger, BasicPOMCP vs robust(ρ=0). CSV +
      `_raw.csv` exist at `experiments/tiger/results/E1_2026-05-03_baseline_*`
- [x] E1 plots: violin (per-episode return distribution) + action breakdown
      (correct/wrong/listen). Output: `experiments/tiger/results/figures/E1/`
- [ ] **NEXT:** Run E2 (ρ_Z sweep on nominal world) and inspect plots
- [ ] Run E3 (ρ_Z sweep, perturbed world η=0.20) — headline experiment
- [ ] Run E4 (η sweep, vanilla vs robust ρ_Z=0.2)
- [ ] Run E5 (sims_per_backup sweep)
- [ ] Run E6 (UCB mode comparison)
- [ ] Run E7 (horizon sweep)
- [ ] Bound validation experiments
- [ ] Plots and results for thesis writeup

See `docs/discussion/01_solver_design.md` for detailed solver analysis and
`docs/discussion/` for all design discussions.
