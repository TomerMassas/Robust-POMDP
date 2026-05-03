---
name: Project Context
description: Overall description of this repo — architecture, purpose, key decisions
type: project
originSessionId: 60e505b2-035f-4545-ad11-905ae1961031
---
## Overview
MSc thesis research on **Robust Online POMDP** planning. The core problem: online POMDP solvers (like POMCP/DESPOT) sample a subset of states and observations to build a search tree, but they assume a fixed, known model. This work introduces robustness to simultaneous transition and observation model uncertainty in the online planning setting.

## Key Contributions (from paper draft)
1. Formulation of robust online POMDP planning under simultaneous transition & observation uncertainty.
2. **Projected uncertainty sets** — uncertainty sets defined over sampled supports that are consistent with the full uncertainty set. Enables robust reasoning on partial online search trees.
3. Support-based representation of history nodes for sampled state/observation spaces.
4. Explicit error decomposition: **inside-support mismatch** + **omitted trajectories contribution**.
5. Deterministic finite-horizon **non-robust error bound** between theoretical and projected value functions (accumulates through belief mismatch, model mismatch on support, and probability mass outside support).
6. Deterministic **robust error bound** — inside-support term controlled via worst-case kernel-level mismatch; omitted term via worst-case leakage outside sampled support.
7. Tractable rectangular robust formulation motivating future robust online planning algorithms.

## Core Mathematical Entities
- **Uncertainty sets**: P_T(s,a) for transitions, P_Z(s) for observations — defined via distance metric D(·||·) from nominal models P_T^o, P_Z^o with radii rho(s,a), rho(s).
- **Projected uncertainty sets**: hat{P}_T, hat{P}_Z — sub-probability vectors on sampled support, extendable to full valid distributions in the original uncertainty set.
- **History nodes**: h^k a — store sampled states {s_i^k} and observations {z_i^k}; define supp_S and supp_Z.
- **Kernels**: K(z,s'|s,pi) = P_T(s'|pi,s)·P_Z(z|s'), and projected version hat{K}.
- **Value functions**: V_t^pi (theoretical, full support) vs hat{V}_t^pi (projected, sampled support).
- **Error bound terms**: Delta_T (transition mismatch), Delta_Z (observation mismatch), Delta_b (belief mismatch), Delta_K^rob (robust kernel mismatch).
- **Non-robust bound** (Eq. 63): |V - hat{V}| <= R_max * sum over j of [inside mismatch + omitted trajectories].
- **Robust bound** (Eq. 74): replaces mismatch with sup over uncertainty set, uses Delta_K^rob.

## Architecture (in migration: Julia → Python)

**Current state (as of 2026-04-30):** Migration to Python in progress. Both languages coexist on branch `claude/quizzical-mendel-48380d` until Tomer signs off; after that, Julia is removed. Migration plan: `C:\Users\TomerMassas\.claude\plans\if-i-tell-you-velvety-tulip.md`.

**Target language: Python (≥ 3.11), src-layout package `src/robust_pomdp/`.**
- Optimization: `highspy` (HiGHS Python bindings — same solver as Julia) wrapped behind a `ProjectedTVBall` class with persistent solver instance + parameter updates. Solver-swap escape hatch (Gurobi / OR-Tools) baked into the wrapper.
- POMDP baseline interface: `pomdp_py` (replaces POMDPs.jl + BasicPOMCP). Used for the vanilla POMCP baseline in E1 to preserve the independent-baseline property.
- Numerical / data: `numpy`, `pandas`. JSON via stdlib.
- Per-scenario `experiments/<scenario>/` layout — Tiger lives at `experiments/tiger/`, future scenarios get sibling folders.

**Legacy language: Julia (until Phase 9 cleanup, gated on Tomer's sign-off).**
- POMDPs.jl + BasicPOMCP for the original baseline implementation.
- JuMP + HiGHS for LPs.
- Source under `src/*.jl` and `experiments/*.jl`. Project.toml + Manifest.toml. Removed after sign-off.

**Algorithmic decisions (uncertainty sets, projected sets, robust backups, etc.) are language-agnostic.** The Python port is functionally equivalent — same algorithms, same data structures, same JSON event log schema, same CSV column schema. Verification compares against hand-derived references and qualitative sanity, with Julia outputs as a soft cross-check (not bit-for-bit).

## Solver Strategy
Multiple solvers planned:
- **Solver A (Robust-Value POMCP):** Nominal sampling + worst-case value backups at every node
- **Solver B (Bound-Guided POMCP):** Standard POMCP + error bound for conservative action selection
- **Flagship Adaptive Planner (TBD):** Uses bound decomposition (inside mismatch vs omitted trajectories) to guide sampling
- Each may have variants. See `docs/discussion/01_solver_design.md` for full analysis.

## Infrastructure
- Git repo at GitHub (Robust-POMDP).
- Paper draft: `Robust_Online_POMDP paper.pdf` in repo root.
- **Detailed design discussions**: `docs/discussion/` (gitignored). Read `04_session_log.md` for latest status and open items.

## Debug/Visualization Stack (Session 6)
- **Event-based logging** (`src/debug/logger.jl` + `src/debug/json_export.jl`): solver emits chronological event log (sim_start, sim_step, expand, rollout, backup_nominal_step, sim_end, robust_backup). Zero overhead when `logger=nothing`.
- **Unique node ids**: every tree node gets a sequential id via NodeIdCounter (reset per planner call). Event stream references nodes by `h{id}` / `a{id}` strings.
- **Python Dash UI** (`viz/app.py`): loads JSON, replays events by index, renders tree via dash-cytoscape with dagre layout. Decision: chose event-replay over live debugging after VSCode Julia debugger crashed and Pluto notebook approach was rejected.
- **Dirty-flag pruning**: `robust_backup!` short-circuits clean subtrees. Invariant: if a node is clean, no descendant is dirty (a simulation always dirties its full root→leaf path).

## TV / L1 distance convention
- This codebase uses `D(P,Q) = ‖P-Q‖₁ = Σ|P-Q|` (NO ½ factor) — defined at `src/uncertainty/uncertainty_sets.jl:19`, enforced in LP at `src/uncertainty/projected_sets.jl:66`.
- Standard textbook TV is `½·‖P-Q‖₁`, so `ρ` here is **2 × standard TV radius**. ρ_Z = 0.1 corresponds to standard TV radius 0.05.
- Effective per-entry deviation in the projected ball is ρ/2 (factor-of-2 from either out-of-support leakage cost or in-support shuffle cost; both charged on TV budget).

## Experiment Harness (Sessions 7–8)
- **`src/evaluation/`** holds: `belief_update.jl` (discrete Bayes), `episode.jl` (run_episode plan→act→observe→update loop), `perturbation.jl` (Tiger reduced-accuracy world), `pomdps_adapter.jl` (TabularPOMDPWrapper for BasicPOMCP), `sweep.jl` (ExperimentConfig, run_config, run_sweep).
- **Vanilla baseline = BasicPOMCP from JuliaPOMDP**, NOT the implicit ρ=0 trick. Verified to agree with robust_pomcp(ρ=0) within sampling noise (E1).
- **Renamed** `batch_size` → `sims_per_backup` in `robust_pomcp_plan` (more self-documenting).
- **CSV output**: `run_sweep` writes both a summary CSV (one row per config) and a `_raw.csv` (one row per (config × trial)). The raw is needed for distributional plots (violins, histograms). INDEX.md lists each summary CSV.
- **Plot output convention**: per-experiment subfolder `experiments/results/figures/{Ek}/`. Python plotting (`viz/plot_results.py`) reads CSVs from the flat `results/` dir and routes outputs into the subfolders. Decoupled — replots don't require rerunning Julia.
- **Episode harness decoupling**: planner sees `model_planner` (nominal); world uses `model_true` (perturbed in E3+). This separation makes robustness experiments meaningful.

## Known Performance Bottleneck (being fixed during the Python migration)
- `minimize_over_projected_tv_ball` (Julia: `src/uncertainty/projected_sets.jl`; Python target: `src/robust_pomdp/uncertainty/projected_sets.py`) was rebuilding a fresh JuMP model + HiGHS solve on every call.
- **Concrete numbers (Session 9, budget=100 measurement):** ρ_Z = 0 → ~35 ms/episode (LP short-circuits), ρ_Z > 0 → ~6 s/episode. 170× slowdown was purely JuMP/HiGHS setup overhead.
- **Fix:** the Python port of this file is being written from scratch as a `ProjectedTVBall` class with a persistent `Highs` instance + parameter updates (`changeColCost`, `changeRowBounds`). Lands in migration Phase 2 of the plan. Gated by §1.3 #7 (≤ 1 s/episode at budget=100, ρ>0). The Julia code is *not* being patched — it is being replaced.
- **Architectural commitment:** the wrapper class hides the solver. If `highspy` later becomes the bottleneck at larger problem sizes (RockSample, etc.), swapping to Gurobi or OR-Tools is a one-day localized change.

## Tiger reward asymmetry — affects experiment interpretation
- Tiger rewards: listen −1, correct open +10, wrong open −100. The wrong-open penalty so dominates that **caution can pay off even on the nominal world**.
- Session 9 E2 preliminary (n=10, budget=100): mean return *increased* with ρ_Z (−51 → −12), opposite of the "pessimism tax" hypothesis. Wrong-open rate dropped from 33% (ρ=0) to 13% (ρ=0.3).
- Likely a low-budget artifact: under-funded vanilla planner can't find the optimal "listen-until-confident" policy, so pessimism acts as a useful regularizer. May disappear at budget=500. Validation pending.

## Name collisions to know about
- `value` is exported by both **POMDPs.jl** and **JuMP**. After adding POMDPs as a dep we had to qualify `JuMP.value(...)` at `src/uncertainty/projected_sets.jl:86`. Watch for similar collisions if more POMDPs/JuMP symbols come in (e.g., `actions`, `observations`, `discount`).
