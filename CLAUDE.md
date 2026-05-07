# Robust Online POMDP — thesis project

This is Tomer Massas's MSc thesis on **robust online POMDP planning** —
extending POMCP/DESPOT-style online solvers to handle uncertainty in both
transition and observation models, via projected uncertainty sets defined
over sampled supports. Mathematical contributions: deterministic finite-
horizon error bounds (non-robust + robust), inside-support / omitted-
trajectory decomposition, tractable rectangular robust formulation.

## Required reading at session start

Project memory lives in `.thesis/` (git-tracked). The index of files is
`.thesis/MEMORY.md` — read it first, then load each file it points to.
Use `/session-start` to do this automatically.

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
  (`CLAUDE_CONFIG_DIR` pinned by his launcher). All memory lives in
  `.thesis/` inside the repo; the profile-layer `MEMORY.md` is a short
  pointer to `.thesis/MEMORY.md` (the actual index).
- **Sibling profiles** (`~/.claude`, `~/.claude-personal`, `~/.claude-work`,
  `~/.claude-code-gui`) belong to unrelated work — **do not read or list them**.
- Tomer occasionally switches to a separate work-account session at the 5h
  rate-limit boundary. The handoff (`.thesis/handoff.md`) is the cold-start
  bridge between accounts.
