# Thesis Project — Deferred Todos

Captured items deferred for later. Distinct from `parked.md` (tangential
ideas during conversation) and `project_roadmap.md` (thesis research
roadmap). This file holds *Claude-system* / *workflow* todos that we
agreed to revisit later.

---

## 2026-05-03 — Design `/review` command

Self-review pass with two modes — context determines which:

**Plan-review mode** (after a proposal, before approval):
- Did I address what you actually asked?
- Steps in dependency order?
- Unstated assumptions baked in?
- Anything missing or over-engineered?

**Code-review mode** (after a change, before declaring done):
- Scope creep — touched only what plan said?
- Loose ends — leftover prints, TODOs, commented code?
- Tests added/updated where needed?
- Edge cases I missed?

**Output discipline:** don't fake. If nothing to flag, output is one line:
*"Self-review: nothing to flag."*

**Open design questions:**
- Name `/review` OK, or different?
- Auto-trigger after non-trivial work, manual `/review` only, or both?
- Output structure above OK?

---

## 2026-05-03 — Reimplement violin/action plot functions in viz/plot_results.py

Lost when this branch was force-merged to main, dropping commit `253b292`
from origin/main. The Julia changes in that commit will be regenerated
post-migration; the Python plot functions need to be reimplemented by hand.

**Functions to reimplement (in `viz/plot_results.py`):**

- `plot_e1_violin()` — per-episode return distribution per planner (violin),
  reads `*_raw.csv` for per-trial rows
- `plot_e1_actions()` — stacked bar (correct / wrong open / listen count)
  per planner, with count annotations on each segment
- `plot_e2_line()` — mean return vs ρ_Z (line + error bars)
- `plot_e2_violins()` — return distribution violin per ρ_Z value
- `plot_e2_actions()` — action breakdown across ρ_Z values
- Wrappers `plot_e1()` and `plot_e2()` calling the sub-plots
- `plot_e3_placeholder()` — stub for E3 (perturbed world)
- Update `read_latest()` to disambiguate summary vs raw CSVs
  (add `raw: bool = False` arg)

**Reference:** original code is at `git show 253b292:viz/plot_results.py`
while the reflog still has it (~90 days locally; the commit also exists on
GitHub's history server-side until cleaned up).
