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

## Architecture (Python)

**Language: Python ≥ 3.11**, src-layout package `src/robust_pomdp/`. Editable-install via `pip install -e .` (see `pyproject.toml`).

- **Optimization:** `highspy` (HiGHS Python bindings), wrapped behind a `ProjectedTVBall` class (`src/robust_pomdp/uncertainty/projected_sets.py`) with persistent solver instance + parameter updates per solve. Solver-swap escape hatch (Gurobi / OR-Tools) baked into the wrapper if scale demands.
- **POMDP baseline interface:** `pomdp_py` (used for the vanilla POMCP baseline in E1 to preserve the independent-baseline property; planner_label `"BasicPOMCP"` is kept in CSVs for schema continuity).
- **Numerical / data:** `numpy`, `pandas`. JSON via stdlib.
- **Per-scenario layout:** `experiments/<scenario>/`. Tiger lives at `experiments/tiger/` (scripts + `results/` subdir). Future scenarios (e.g. RockSample) get sibling folders.

## Source layout
- `src/robust_pomdp/core/` — `pomdp_types.py` (TabularPOMDP), `tree.py` (HistoryNode, ActionNode, NodeIdCounter)
- `src/robust_pomdp/uncertainty/` — `uncertainty_sets.py` (TVDistance, UncertaintySets), `projected_sets.py` (ProjectedTVBall)
- `src/robust_pomdp/solvers/` — `robust_pomcp.py` (robust_pomcp_plan, compute_robust_q, etc.)
- `src/robust_pomdp/debug/` — `logger.py`, `json_export.py` (event-replay logging)
- `src/robust_pomdp/evaluation/` — `belief_update.py`, `episode.py`, `perturbation.py`, `pomdp_py_adapter.py`, `sweep.py`

## Solver Strategy
Multiple solvers planned:
- **Solver A (Robust-Value POMCP):** Nominal sampling + worst-case value backups at every node. **Implemented.**
- **Solver B (Bound-Guided POMCP):** Standard POMCP + error bound for conservative action selection. Not yet implemented.
- **Flagship Adaptive Planner (TBD):** Uses bound decomposition (inside mismatch vs omitted trajectories) to guide sampling. Design pending.
- See `docs/discussion/01_solver_design.md` for full analysis.

## Infrastructure
- Git repo at GitHub (Robust-POMDP). Working branch: `main`.
- Paper draft: `Robust_Online_POMDP paper.pdf` in repo root.
- **Detailed design discussions:** `docs/discussion/` (gitignored). Read `04_session_log.md` for latest status and open items.

## Debug / Visualization Stack
- **Event-based logging** (`src/robust_pomdp/debug/logger.py` + `json_export.py`): solver emits chronological event log (sim_start, sim_step, expand, rollout, backup_nominal_step, sim_end, robust_backup). Zero overhead when `logger=None`.
- **Unique node ids:** every tree node gets a sequential id via NodeIdCounter (reset per planner call). Event stream references nodes by `h{id}` / `a{id}` strings.
- **Python Dash UI** (`viz/app.py`): loads JSON, replays events by index, renders tree via dash-cytoscape with dagre layout.
- **Dirty-flag pruning:** `robust_backup` short-circuits clean subtrees. Invariant: if a node is clean, no descendant is dirty (a simulation always dirties its full root→leaf path).

## TV / L1 distance convention
- This codebase uses `D(P,Q) = ‖P-Q‖₁ = Σ|P-Q|` (NO ½ factor) — defined in `src/robust_pomdp/uncertainty/uncertainty_sets.py`, enforced in the LP at `src/robust_pomdp/uncertainty/projected_sets.py`.
- Standard textbook TV is `½·‖P-Q‖₁`, so `ρ` here is **2 × standard TV radius**. ρ_Z = 0.1 corresponds to standard TV radius 0.05.
- Effective per-entry deviation in the projected ball is ρ/2 (factor-of-2 from either out-of-support leakage cost or in-support shuffle cost; both charged on TV budget).

## Experiment Harness
- **`src/robust_pomdp/evaluation/`** holds: `belief_update.py` (discrete Bayes), `episode.py` (run_episode plan→act→observe→update loop), `perturbation.py` (Tiger reduced-accuracy world), `pomdp_py_adapter.py` (TabularPOMDPWrapper for pomdp_py POMCP), `sweep.py` (ExperimentConfig, run_config, run_sweep).
- **Vanilla baseline = pomdp_py POMCP**, NOT the implicit ρ=0 trick. CSV planner_label is `"BasicPOMCP"` for continuity with prior outputs.
- **CSV output:** `run_sweep` writes both a summary CSV (one row per config) and a `_raw.csv` (one row per (config × trial)). The raw is needed for distributional plots (violins, histograms). INDEX.md lists each summary CSV.
- **Plot output convention:** per-experiment subfolder `experiments/tiger/results/figures/{Ek}/`. Python plotting (`viz/plot_results.py`) reads CSVs from `experiments/tiger/results/` and routes outputs into the per-Ek subfolders. `--results-dir` overrides the default for other scenarios.
- **Episode harness decoupling:** planner sees `model_planner` (nominal); world uses `model_true` (perturbed in E3+). This separation makes robustness experiments meaningful.

## LP performance
- `ProjectedTVBall.solve` keeps a persistent `Highs` instance across calls — only row bounds and objective coefficients change per solve. The LP topology (variables, sparsity pattern, mass-balance equality) is fixed at construction time.
- Designed to handle ~200K LP solves per E2 config without the per-call setup overhead that an earlier rebuild-every-call prototype incurred.
- The wrapper hides the solver. If `highspy` becomes a bottleneck at larger problem sizes (RockSample, etc.), swapping to Gurobi or OR-Tools is a localized change to this one class.

## Tiger reward asymmetry — affects experiment interpretation
- Tiger rewards: listen −1, correct open +10, wrong open −100. The wrong-open penalty so dominates that **caution can pay off even on the nominal world**.
- Earlier E2 preliminary (n=10, budget=100): mean return *increased* with ρ_Z (−51 → −12), opposite of the "pessimism tax" hypothesis. Wrong-open rate dropped from 33% (ρ=0) to 13% (ρ=0.3).
- Likely a low-budget artifact: under-funded vanilla planner can't find the optimal "listen-until-confident" policy, so pessimism acts as a useful regularizer. May disappear at budget=500. Validation pending.
