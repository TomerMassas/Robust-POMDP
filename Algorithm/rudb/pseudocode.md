# RUDB — Robust Upper Deterministic Bound planner (R-MCTS variant 2)

Finalized pseudo-code for the **RUDB planner**. Derived with Tomer from
`R_MCTS_AAAI_2027.pdf` (Eq. 19–23, Thm 1/2, Cor 1/3), session 2026-07-21. Spec that
`Algorithm/rudb/` builds to — edit here first, then the code.

**Note:** the paper's Thm-2 *bracketing* is not written yet; the asymmetric `[L,U]`
below is the version we derived for it. (The paper's symmetric `[V̂^π ± ε]` bracket
lives under Thm 1 — that's the fixed-policy bracket, a different object.)

**Convention.** `H` reward steps at depths `0..H-1`; terminal at depth `H-1`.
`abort` is a generic optional terminal (`None` if a domain has none), special-cased
only in BACKUP. `d = node.depth`, `mass_in = Σ_{s∈S_in} b(s)` (the **exact** belief's
in-support mass), `R_max = max(|R.min|,|R.max|)`.

**How it differs from CR-UCT** (`ucb/pseudocode.md`):
- **Belief = exact nominal Bayes over the full `S`** (Cor 1 ⇒ per-node validity),
  computed **once per node**, restricted to `S_in` at backup. Not projected-Bayes.
- **Two-sided `V*` bracket, asymmetric**, per action:
  `L(a) = V̂^π(a) − ε₁(a)` [Thm 1] and `U(a) = RUDB(a)` [Thm 2].
- **Exploration = optimism on the upper**: `π = argmax_a U(a)` (Eq. 22). No `c_ucb`,
  no `Q_nom`, no rollouts — the error is the optimism.
- **Certifies optimality** (`V*_rect`, Cor 3) and **prunes** certified-suboptimal
  subtrees at every node (config flag).

**Node attributes.**
```
History node:  .depth .b(EXACT full belief) .S_in .child[a] .pruned(set)
               backed up: .V̂ (lower value of RUDB policy) .c[·] (survival) .U (upper = max_a U(a))
Action  node:  .Z_in .S_next .child[z]
               backed up: .V̂π  .c[·]  .ε₁  .L  .U
```
Parent's **lower** reads `child.V̂`,`child.c`; parent's **upper** reads `child.U`.

**Reuse vs new.** Reuse `robust_q`, `leakage_c`, `ProjTVball`, tree structures,
trajectory sampling. New: `FILTER` (exact Bayes), the shared-minimizer upper backup,
optimistic `SELECT_ACTION`, per-node pruning, the asymmetric bracket. Cost ≈ 1.5×
CR-UCT/node (lower = 2 LP passes reused, upper = 1 shared-min pass).

---

## PLAN — one planning step

```
PLAN(b0):
    root ← HistoryNode(depth=0);  root.b ← b0                  # exact full prior
    for sim = 1..budget:
        s ~ b0;  root.S_in ← root.S_in ∪ {s}
        SIMULATE(s, root)
        if early_stop and CERTIFIED_BEST_ACTION(root) ≠ ⊥: return it
    return argmax_{a ∈ expanded(root)} root.child[a].V̂π        # budget-exhausted fallback
```

## SIMULATE — returns nothing (no Q_nom / rollout)

```
SIMULATE(s, node):
    a ← SELECT_ACTION(node);  an ← node.child[a]
    if node.depth = H-1 or a = abort:                          # terminal
        BACKUP(node); return
    s' ~ T(·|s,a);  z ~ O(·|s')                                # full nominal — drives descent + support
    an.Z_in ∪= {z};  an.S_next ∪= {s'}
    child, is_new ← an.get_or_create_child(z)
    child.S_in ∪= {s'}
    if is_new:
        child.b ← FILTER(node.b, a, z)                         # exact nominal Bayes, full S
        BACKUP(child)                                          # frontier
    else:
        SIMULATE(s', child)
    BACKUP(node)
```

## FILTER — exact nominal belief update (full state space)

```
FILTER(b, a, z) -> b':                                         # standard Bayes, FULL nominal T,O
    b'(s') ← O(z|s') · Σ_{s∈S} T(s'|s,a)·b(s)     ∀ s'∈S
    return b' / Σ_{s'} b'(s')                                  # normalize over full S
```

## SELECT_ACTION — optimism on the upper bound

```
SELECT_ACTION(node):
    if ∃ unexpanded action a (a ∉ node.pruned): node.child[a] ← ActionNode(); return a
    return argmax_{a ∈ expanded(node), a ∉ node.pruned}  node.child[a].U     # Eq. 22
```

## BACKUP — lower (Thm 1) + upper (Thm 2), fused per action

```
BACKUP(node):
    b := node.b restricted to sorted(S_in)   (sub-probability);  mass_in := Σ b;  d := node.depth

    if node has no expanded actions:                           # frontier node (just created)
        node.V̂ ← max_a Σ_{s∈S_in} b(s)·R(s,a)
        node.c ← {s: 1 ∀ s∈S_in};   node.U ← R_max·(H−d)
        return

    for each expanded action a  (an := node.child[a]):
        r_in ← Σ_{s∈S_in} b(s)·R(s,a)
        if (a = abort) ∨ (d = H-1):                            # genuine terminal
            an.V̂π ← r_in;   an.c ← {s: H−d}
            an.U  ← r_in + R_max·(1 − mass_in)                 # tight upper on V*(a)
        else if an.child = ∅:                                  # frontier action (tail unexplored)
            an.V̂π ← r_in;   an.c ← {s: 1}
            an.U  ← R_max·(H−d)                                # optimistic
        else:                                                  # interior
            # lower  [Thm 1] — reuse CR-UCT (two independent infs):
            an.V̂π ← robust_q (b, a, S_in, Z_in, S_next, V_child = {z: an.child[z].V̂})
            an.c  ← leakage_c(a, S_in, Z_in, S_next, child_c = {z: an.child[z].c})
            # upper  [Thm 2] — shared minimizer, ONE combined LP, coeff g(z)=child.U−R_max:
            for s' ∈ S_next:  wg[s'] ← min_{q ∈ ProjTVball(O(s',·)|Z_in, ρ_Z(s'))}  Σ_{z∈Z_in} q[z]·(an.child[z].U − R_max)
            for s  ∈ S_in:    σg[s]  ← min_{p ∈ ProjTVball(T(s,a,·)|S_next, ρ_T(s,a))} Σ_{s'∈S_next} p[s']·wg[s']
            an.U  ← r_in + R_max + Σ_{s∈S_in} b(s)·σg[s]
        an.ε₁ ← R_max·((H−d) − Σ_{s∈S_in} b(s)·an.c[s])        # Thm-1 leakage (per-node; Cor 1)
        an.L  ← an.V̂π − an.ε₁

    π ← argmax_{a expanded} an.U                               # RUDB policy (Eq. 22, optimism on upper)
    node.V̂ ← node.child[π].V̂π;  node.c ← node.child[π].c;  node.U ← node.child[π].U
    if prune:                                                  # per-node pruning (Cor 1) — config flag
        a_lo ← argmax_{a expanded} node.child[a].L
        node.pruned ← { a expanded : node.child[a].U < node.child[a_lo].L }
```

## CERTIFIED_BEST_ACTION — anytime optimal-action certificate

```
CERTIFIED_BEST_ACTION(root) -> action or ⊥:
    if not all actions expanded at root: return ⊥
    a* ← argmax_a root.child[a].L
    return a*  if  root.child[a*].L ≥ max_{a≠a*} root.child[a].U  else ⊥
```

---

**Config knobs:** `budget`, `early_stop`, `prune`, `seed`. (No `c_ucb` / `ucb_mode` /
`use_rollouts` — RUDB has a single exploration mode.)

**Bracket recap (asymmetric):** `L(a)=V̂^π(a)−ε₁(a)` is the Thm-1 lower (valid for `V*`
since `V̂^π−ε₁ ≤ V^π ≤ V*`); `U(a)=RUDB(a)` is the Thm-2 upper (`≥ V*(a)`). Certified
when the best lower clears every other upper.
