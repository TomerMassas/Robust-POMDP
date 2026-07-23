# CR-UCT — Certified Robust UCT (UCB variant of R-MCTS)

Finalized pseudo-code for the **UCB planner** (Milestone 4). Source: `method.docx`
(Tomer's screenshots, session 2026-07-20). This is the spec the implementation in
`Algorithm/ucb/` builds to — edit here first, then the code.

**Convention.** `H` reward steps at depths `0..H-1`; a node at depth `H-1` is
terminal (last reward step). `abort` is terminal (0 continuation). Beliefs are
option-B projected-Bayes: root = **unnormalized** restriction of `b0` (Eq. 10),
children renormalized. Next-state support `S_next = ⋃_z child_z.S_in`. The root's
immediate reward uses the **exact** `b0` (cached as `.r_exact` per root action —
the Eq. 41 cancellation), so its depth-0 error is 0; the restricted root belief
carries only the continuation.

**Node attributes.**
```
History node:  .S_in  .N  .V_rob  .delta[·] (continuation-survival δ, greedy policy)  .child[a]
Action  node:  .Z_in  .S_next (sampled next-states)  .N  .Q_rob  .Q_nom  .delta[·]  .r_exact  .child[z]
```

**Modes.** `ucb_mode ∈ {nominal, robust}` drives *exploration only*: nominal uses
the sample-mean `Q_nom` (with optional rollouts), robust uses the projected
`Q_rob`. The final root decision (`argmax_a Q_rob`) and the certificate are
identical across modes.

---

## PLAN — one planning step

```
PLAN(b0):
    root ← HistoryNode(depth = 0)
    for sim = 1 .. budget:
        s ~ b0;   root.S_in ← root.S_in ∪ {s}
        SIMULATE(s, root, restrict(b0, root.S_in))        # unnormalized restriction (Eq.10)
        a* ← CERTIFIED_BEST_ACTION(root)
        if a* ≠ ⊥: return a*
    return argmax_{a ∈ expanded(root)} root.child[a].Q_rob
```

## SIMULATE

```
SIMULATE(s, node, b) -> G:                               # G consumed only if ucb_mode = nominal
    node.N ← node.N + 1
    a ← SELECT_ACTION(node);   an ← node.child[a];   an.N ← an.N + 1
    r ← R(s, a)
    if node.depth = H-1:                                 # terminal (last reward step)
        BACKUP(node, b)
        if ucb_mode = nominal: an.Q_nom ← an.Q_nom + (r - an.Q_nom)/an.N
        return r
    s' ~ T(·|s,a);   z ~ O(·|s')
    an.Z_in ← an.Z_in ∪ {z};   an.S_next ← an.S_next ∪ {s'}
    is_new ← (z ∉ an.child)
    if is_new: an.child[z] ← HistoryNode(depth = node.depth + 1)
    child ← an.child[z];   child.S_in ← child.S_in ∪ {s'}
    b' ← normalize( Σ_{u∈node.S_in} b[u]·T(u,a,·)·O(·,z)  restricted to child.S_in )
    if is_new:
        BACKUP(child, b')                                # frontier leaf
        G' ← ROLLOUT(s', child.depth) if (ucb_mode = nominal ∧ use_rollouts) else 0
    else:
        G' ← SIMULATE(s', child, b')
    BACKUP(node, b)
    if ucb_mode = nominal: an.Q_nom ← an.Q_nom + ((r + G') - an.Q_nom)/an.N
    return r + G'
```

## SELECT_ACTION

```
SELECT_ACTION(node):
    if ∃ unexpanded action a:
        node.child[a] ← ActionNode()
        if node.depth = 0: node.child[a].r_exact ← Σ_s b0(s)·R(s,a)   # exact belief at root
        return a
    Q(a)      ← an.Q_nom if ucb_mode = nominal else an.Q_rob
    Q_norm(a) ← (Q(a) + V_max) / (2·V_max)        # V_max = R_max·H; UCB1 assumes values in [0,1]
    return argmax_{a ∈ expanded} [ Q_norm(a) + c_ucb·√(ln node.N / node.child[a].N) ]
```

## BACKUP — robust value + continuation-survival δ, fused

```
BACKUP(node, b):
    for each expanded action a  (an := node.child[a]):
        imm ← an.r_exact if an.r_exact ≠ ⊥ else Σ_{s∈node.S_in} b[s]·R(s,a)   # exact at root
        if (a is abort) ∨ (node.depth = H-1):            # genuine terminal — no leakage
            an.Q_rob ← imm
            an.delta[s] ← (H-1 - node.depth)   ∀ s∈node.S_in   # continuation survives; = 0 at depth H-1
        else if an.child = ∅:                            # frontier (not yet deepened) — tail leaks
            an.Q_rob ← imm
            an.delta[s] ← 0   ∀ s∈node.S_in
        else:                                            # interior — projected ball, shared by both objectives
            for s' ∈ an.S_next:                          # observation step
                B_Z ← ProjTVball(O(s',·)|Z_in, ρ_Z(s'))
                w[s']   ← min_{q∈B_Z} Σ_{z∈Z_in} q[z]·an.child[z].V_rob
                w_δ[s'] ← min_{q∈B_Z} Σ_{z∈Z_in} q[z]·( 1 + an.child[z].delta[s']  if s'∈S_in(an.child[z]) else 0 )
            for s ∈ node.S_in:                           # transition step
                B_T ← ProjTVball(T(s,a,·)|an.S_next, ρ_T(s,a))
                σ[s]        ← min_{p∈B_T} Σ_{s'∈an.S_next} p[s']·w[s']
                an.delta[s] ← min_{p∈B_T} Σ_{s'∈an.S_next} p[s']·w_δ[s']   # continuation only, no self +1
            an.Q_rob ← imm + Σ_{s∈node.S_in} b[s]·σ[s]
    a* ← argmax_{a∈expanded} node.child[a].Q_rob
    node.V_rob ← node.child[a*].Q_rob ;   node.delta ← node.child[a*].delta    # greedy policy
```

## ROLLOUT — nominal random sim to H-1; feeds Q_nom only

```
ROLLOUT(s, d) -> G:
    G ← 0;  depth ← d
    loop:
        a ~ rollout_policy(s);   G ← G + R(s, a)
        if depth = H-1: return G
        s ~ T(·|s,a);   depth ← depth + 1
```

## CERTIFIED_BEST_ACTION — anytime early stop

```
CERTIFIED_BEST_ACTION(root) -> action or ⊥:
    if not all actions expanded at root: return ⊥
    for each root action a  (an := root.child[a]):
        ε_a ← R_max·( (H-1) - Σ_{s∈S_in(root)} b0(s)·an.delta[s] )   # Eq. 49 (drops j=0)
        L[a] ← an.Q_rob - ε_a ;   U[a] ← an.Q_rob + ε_a
    a* ← argmax_a L[a]
    return a*  if  L[a*] ≥ max_{a≠a*} U[a]  else ⊥
```
