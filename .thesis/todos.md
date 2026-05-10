# Thesis Project — Deferred Todos

Captured items deferred for later. Distinct from `parked.md` (tangential
ideas during conversation) and `project_roadmap.md` (thesis research
roadmap). This file holds *Claude-system* / *workflow* todos that we
agreed to revisit later.

---

## 2026-05-10 — FrozenLake plan, remaining steps

Steps 1–3 of the FrozenLake plan landed today (Scenario refactor +
seed fix; `frozenlake_problem.py`; `frozenlake_test.py`). Open:

- **Step 4 — `experiments/frozenlake/run_experiments.py`.** E1 baseline
  equivalence (BasicPOMCP vs robust(ρ=0) on nominal FrozenLake) + E3_Z
  (2D ρ_Z × η_obs sweep on Z-perturbed world, calibrated η ∈ [0,
  ρ_Z/8]) + E3_T (2D ρ_T × η_trans sweep on T-perturbed world,
  calibrated η ∈ [0, ρ_T/2]). Defaults: n=50, budget=100, horizon=30.
  Run `--only E1` first to gauge actual runtime.
- **Step 5 — FrozenLake-specific plot functions** in
  `viz/plot_results.py`, after first results land: return distribution
  + outcome breakdown (fell / reached / timed-out); fan-of-curves for
  ρ × η on the headline plot.

## 2026-05-10 — Additional benchmark toys to add later

After FrozenLake is in and yields a clean robust-vs-vanilla picture,
layer in two more benchmarks for breadth:

- **CheeseMaze** — small T-shaped grid (~11 cells); observations are
  the wall-pattern of the current cell, causing perceptual aliasing
  across multiple cells. Wrong obs model = misidentified cell type =
  wrong direction. Small enough to compute exact V; the failure mode
  is structural rather than quantitative.
- **RockSample(n, k)** — n×n grid with k rocks of unknown good/bad
  quality; check action returns a noisy binary signal whose accuracy
  decays with distance to the rock. Sensor decay constant is the
  natural ρ_Z. Scales (RS(4,4)=256, RS(7,8)≈12k); strongest "main
  event" candidate for the headline experiment.
