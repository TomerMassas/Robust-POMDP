# Robust POMCP Explorer (Python Dash UI)

Interactive browser-based visualization of a Solver A run.

## How it works

1. Julia solver writes a JSON file to `../experiments/tiger_run.json` containing the full tree, every simulation, every backup.
2. This Python app loads that JSON and gives you a browser UI with:
   - **Simulations + Tree tab**: step through each simulation, see the tree highlight the path, click any node to inspect.
   - **Backups + LPs tab**: step through each robust backup, see the two-step LP decomposition with concrete numbers.

## Install (one-time)

```bash
cd viz
pip install -r requirements.txt
```

## Run

From the repo root:

```bash
# Step 1 — generate the data (Julia side)
julia --project=. experiments/tiger_test.jl

# Step 2 — start the UI (Python side)
cd viz
python app.py
```

Open http://localhost:8050 in your browser.

## What you'll see

**Simulations tab:**
- Left: the search tree. Blue boxes = history nodes, yellow = action nodes.
- Right top: slider to pick a simulation (1..N), slider to pick a step within it.
- Right middle: details of the current step (state, action, observation, reward).
- Right bottom: info on whatever tree node you last clicked.
- Nodes on the current simulation's path are highlighted in red.

**Backups tab:**
- Left: the same tree. A backup's target node is highlighted in purple.
- Right: slider to pick a backup. Details show:
  - S_in, Z_in, belief, V_children (LP inputs)
  - For each Observation LP: nominal P_Z^o, radius, optimal (worst-case) P_Z^in, resulting w(s')
  - For each Transition LP: nominal P_T^o, radius, optimal P_T^in, resulting σ(s)
  - Final Q_robust

This lets you cross-check the LP outputs against the tree node being backed up.
