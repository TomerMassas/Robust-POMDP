---
name: Paper Reference
description: Auto-load at session start — defines all the math the codebase implements
type: reference
---

The paper is at `Robust_Online_POMDP paper.pdf` in repo root (11 pages,
single Read call covers it). It contains the formal definitions, theorem
statements, and proofs that this codebase implements — the source of truth
for every mathematical entity in `project_context.md`.

`/session-start` reads it after the memory files, so every session opens
with the math in context.
