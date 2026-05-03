# Robust Online POMDP — thesis project

This is Tomer Massas's MSc thesis on **robust online POMDP planning** —
extending POMCP/DESPOT-style online solvers to handle uncertainty in both
transition and observation models, via projected uncertainty sets defined
over sampled supports. Mathematical contributions: deterministic finite-
horizon error bounds (non-robust + robust), inside-support / omitted-
trajectory decomposition, tractable rectangular robust formulation.

## Required reading at session start

These live in `.thesis/` (project-layer memory, tracked in git):

- **`user_profile.md`** — Tomer's role, expertise, and goals as collaborator.
- **`feedback_working_guidelines.md`** — operational collaboration rules:
  length calibration, propose-before-edit, design-minimum-viable,
  *"what do you think?"* ≠ approval-to-edit, etc.
- **`personality.md`** — *read after the above two*. Defines collaboration
  disposition: exploratory, patient, idea-first. Thesis is judged by
  quality, not throughput; rushing to ship is the bad habit.
- **`project_context.md`** — full architecture, mathematical entities, key
  decisions (LP via `highspy`, POMCP baseline via `pomdp_py`, src-layout,
  per-scenario `experiments/<name>/`, TV/L1 distance convention, performance
  bottleneck and the planned fix).
- **`project_progress.md`** — session-by-session work log. Read for current
  state of work in flight.
- **`project_roadmap.md`** — phase status, locked decisions, next priorities.
- **`reference_discussion_log.md`** — pointer to `docs/discussion/`
  (gitignored design docs and ADRs).
- **`handoff.md`** *(if present)* — open thread from end of previous session.
  Read first, then archive after Tomer confirms it's been internalized.
- **`parked.md`** *(if present)* — captured tangential ideas from prior
  sessions, available to surface when relevant.

## Hard rules

- **Tomer drives commits.** Never auto-commit without explicit permission;
  never push without explicit permission; never force-push.
- **Plan, then act.** For non-trivial changes, propose first and wait for
  explicit *"go"* / *"yes"* / *"do it"*. Phrases like *"what do you think?"*
  or *"feel free to push back"* are invitations to discuss, NOT approval
  to edit.
- **Code is in service to the math.** Don't propose refactors unless asked.
  Don't add abstractions for hypothetical future requirements. Don't add
  error handling for situations that can't happen.

## Project conventions

- **Code style.** Multi-arg function calls wider than ~95 chars get stacked
  one arg per line, starting from the opening paren:
  `func(arg1,\n    arg2,\n    arg3)`. Under that width, keep on one line.
- **Detailed design discussions** live in `docs/discussion/` (gitignored).
  See `reference_discussion_log.md` for the index.

## Custom commands (project-specific slash commands)

- **`/pushback`** — switches me into critical mode for the turn: surface
  untested assumptions, ignored alternatives, weakest spots in the current
  direction. Won't fake critique — if the direction looks solid, I say so
  honestly rather than invent objections.
- **`/park <thought>`** — capture a tangential idea without derailing current
  work. Appends `[YYYY-MM-DD HH:MM] <thought>` to `.thesis/parked.md`,
  acknowledges with one word, returns to the previous topic. Reviewed at
  session-end.

## Operating context (cross-account / cross-machine)

- Tomer's profile for this thesis is `~/.claude-thesis/`
  (`CLAUDE_CONFIG_DIR` pinned by his launcher). All memory now lives in
  `.thesis/` (no two-layer split anymore); the profile-layer `MEMORY.md`
  is just an index of pointers into `.thesis/`.
- **Sibling profiles** (`~/.claude`, `~/.claude-personal`, `~/.claude-work`,
  `~/.claude-code-gui`) belong to unrelated work — **do not read or list them**.
- Tomer occasionally switches to a separate work-account session at the 5h
  rate-limit boundary. The handoff (`.thesis/handoff.md`) is the cold-start
  bridge between accounts.
