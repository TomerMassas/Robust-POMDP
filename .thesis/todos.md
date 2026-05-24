# Thesis Project — Deferred Todos

Captured items deferred for later. Distinct from `parked.md` (tangential
ideas during conversation) and `project_roadmap.md` (thesis research
roadmap). This file holds *Claude-system* / *workflow* todos that we
agreed to revisit later.

---

## 2026-05-19 — A1 (Fixed-Policy Projected Robust Tree Evaluator) follow-ups

Deferred from A1 design discussion; the first A1 experiment uses the simpler versions.

- **Test 2 — per-node varying projection.** First A1 experiment uses uniform projection
  (same S_in[m], Z_in[m] at every history node). A more realistic projection — closer
  to what online sampling produces — lets S_in(h), Z_in(h) vary per history node. Add
  as a second A1 test once the uniform version is validated.
- **Heterogeneous transition dynamics.** Current A1 domain uses a translation-invariant
  transition rule (`T[a, s, s+1] = 0.1` for inspect, etc., same for every interior s).
  Initial idea: make success probability depend on risk level — safer states get higher
  stay-probability / lower drift-up; riskier states get lower stay / higher drift-up.
  This makes N-scaling exercise *dynamically different* state types, not just a longer
  uniform drift chain. Defer until the uniform-dynamics experiment yields results.
- **Policy variations.** First A1 experiment uses a single fixed policy (warning-threshold
  from the PDF design: depth 0 → `inspect`; else by last-obs warning level).
  Two follow-ups, in priority order:
  - **Multi-policy sweep.** Run A1 across 2-3 distinct fixed policies picked to span the
    bound's regimes — one leakage-heavy (aggressive, rarely aborts, trajectories drift
    into cat states), one mismatch-heavy (cautious, aborts on moderate warnings, tree
    truncates early), plus the balanced default. Stronger demonstration: the bound is
    valid + informative across policy shapes, not specific to one. One-knob extension on
    top of the single-policy infrastructure.
  - **Depth-dependent thresholds.** Mostly cosmetic for the validity claim (theorem holds
    either way), but enables per-depth bound breakdown plots showing how the policy's
    depth-shape changes which depths dominate the certificate. Add only if we want that
    plot.

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
