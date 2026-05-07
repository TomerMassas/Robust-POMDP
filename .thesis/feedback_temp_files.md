---
name: Auto-delete temp files during cleanups
description: Sweep up scratch/smoke/temp scripts as part of cleanup work — don't flag each one separately
type: feedback
---

When cleaning up the repo, sweep up temporary scripts (smoke runs, scratch
verifications, one-off explorations) that have served their purpose, in the
same operation. Don't gate each tmp file on a separate approval.

**Why:** During the Julia → Python cleanup, Tomer noticed `_smoke_phase1.py`
and `_smoke_phase3.py` were lingering in `experiments/tiger/`. He didn't want
to be asked one-by-one — said "you should auto delete them if it is just a
tmp file".

**How to apply:**
- Recognize: leading `_` prefix, names containing `smoke`, `tmp`, `scratch`,
  `wip`, `debug_oneoff`, `*_phase{N}.py`. Files that are clearly one-shot
  reference captures or per-phase verifications that have already passed.
- Sweep them in the same commit as the cleanup that contextually retires them.
  List them in the proposal so they're visible, but don't ask separately.
- Don't sweep files that still gate something. E.g., `verify_robust_backup.py`
  has Layers 1+2 that are language-independent Python tests — those stay even
  if Layer 3's Julia cross-check is removed.
