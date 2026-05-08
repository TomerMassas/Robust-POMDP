# Robust Backup — Hand Derivation

Companion to `verify_robust_backup.py`. Walks the worst-case backup
`compute_robust_q` performs on a tiny synthetic POMDP, all the way from
matrices to the final `Q_robust`, so you can step through the function
in the debugger and check every intermediate against this file.

The setup is deliberately chosen so that:
- both `rho_T > 0` and `rho_Z > 0` (LPs are non-trivial),
- both supports are strict subsets (`S_in ⊊ S`, `Z_in ⊊ Z`) — projection is engaged,
- `V_children` has mixed signs — the adversary has a real choice.

---

## Setup

### POMDP shape

- States: 4 — `S = {0, 1, 2, 3}`
- Actions: 2 — `A = {0, 1}`
- Observations: 4 — `Z = {0, 1, 2, 3}`

### Sampled supports

- `S_in = {0, 1}` → `S_out = {2, 3}`
- `Z_in = {0, 1}` → `Z_out = {2, 3}`

### Transition matrix `T[s, a, s']` (action `a = 0`, the action under test)

| from \ to | s'=0 | s'=1 | s'=2 | s'=3 |
|---|---|---|---|---|
| s=0 | 0.3 | 0.4 | 0.2 | 0.1 |
| s=1 | 0.5 | 0.2 | 0.1 | 0.2 |

Rows for `s ∈ S_out` and for `a = 1` are set to uniform placeholders. They
never enter the LPs because `compute_robust_q` only reads
`T[s, a, S_in]` for `s ∈ S_in` and the fixed action `a = 0`.

### Observation matrix `O[s', z]`

| s' \ z | z=0 | z=1 | z=2 | z=3 |
|---|---|---|---|---|
| s'=0 | 0.4 | 0.3 | 0.2 | 0.1 |
| s'=1 | 0.2 | 0.5 | 0.1 | 0.2 |

`O[s', :]` for `s' ∈ S_out` is uniform placeholder; doesn't enter the test.

### Reward `R[s, a=0]`

- `R(0, 0) = 1.0`
- `R(1, 0) = 2.0`

### Uncertainty radii

- `rho_T(s, a) = 0.2` for all `(s, a)`
- `rho_Z(s') = 0.2` for all `s'`

### Belief

Empirical belief over `S_in` (from particle list `[0]*6 + [1]*4`):
- `b(0) = 0.6`
- `b(1) = 0.4`

### `V_children` (V_robust on the observation children)

- `V_robust(child for z=0) = +10`
- `V_robust(child for z=1) = -20`

Set directly on the `HistoryNode` children of the action node.

---

## LP semantics (cheat sheet)

The projected TV-ball constraint used by `ProjectedTVBall.solve` is

```
sum_i |q_i - p_nominal_i|  +  |sum_i q_i - m_o|  ≤  rho
```

where:
- `q ∈ R^n_+` is the sub-probability vector inside the support
- `p_nominal` is the nominal sub-vector (projected to in-support)
- `m_o = sum p_nominal_i` is the in-support nominal mass
- the second term is the absolute change in in-support mass — i.e. the
  TV cost of the implied out-of-support adjustment

The objective is `min sum q_i · c_i`, where `c` is `V_children` (Step 1)
or `w` (Step 2).

**Per-unit cost in TV budget** for any single move:

| Move type | TV cost per unit `δ` |
|---|---|
| Shift mass `q_i → q_j` within in-support | 2 |
| Leak `q_i → out-of-support` | 2 |
| Draw `out-of-support → q_j` | 2 |

So with `rho = 0.2`, the adversary has a budget that funds exactly
**`δ = 0.1` of mass movement**.

---

## Step 1 — Observation LP per `s' ∈ S_in`

For each `s' ∈ S_in`, solve

```
w(s')  =  min_{q ∈ R^|Z_in|_+}  Σ_{z ∈ Z_in}  q_z · V_children(z)
          s.t.  q ∈ projected TV ball around O[s', Z_in], radius rho_Z(s')
```

### s' = 0

- `p_nominal_obs = O[0, Z_in] = [0.4, 0.3]`, `m_o = 0.7`
- Coefficients (from `V_children`): `[+10, -20]`
- Adversary wants `q_0` small (positive coeff) and `q_1` large (negative coeff)

**Compare moves** (each costs 2 TV per unit `δ`):

| Move | Δq_0 | Δq_1 | Δm | Drop per `δ` | Drop per TV |
|---|---|---|---|---|---|
| A. Within-shift q_0 → q_1 | −δ | +δ | 0 | (+10)δ + (+20)δ = **30δ** | **15** |
| B. Leak q_0 → out | −δ | 0 | −δ | (+10)δ | 5 |
| C. Draw out → q_1 | 0 | +δ | +δ | (+20)δ | 10 |

Best: **A (within-shift)**, 15/TV. Budget exhausted at `δ = 0.1`.

Optimal: `q* = (0.3, 0.4)`, `Δm = 0` (no leakage at the optimum).

TV check: `|0.3-0.4| + |0.4-0.3| + |0.7-0.7| = 0.1 + 0.1 + 0 = 0.2` ✓

```
w(s'=0)  =  0.3 · 10  +  0.4 · (-20)  =  3 - 8  =  -5
```

### s' = 1

- `p_nominal_obs = O[1, Z_in] = [0.2, 0.5]`, `m_o = 0.7`
- Same `V_children`, same `rho_Z`

Same trade-off; A still wins. `δ = 0.1`.

Optimal: `q* = (0.1, 0.6)`, `Δm = 0`.

TV check: `|0.1-0.2| + |0.6-0.5| + |0.7-0.7| = 0.2` ✓

```
w(s'=1)  =  0.1 · 10  +  0.6 · (-20)  =  1 - 12  =  -11
```

### Step 1 result

```
w  =  [-5, -11]            (aligned with sorted S_in = [0, 1])
```

---

## Step 2 — Transition LP per `s ∈ S_in`

For each `s ∈ S_in`, solve

```
σ(s)  =  min_{q ∈ R^|S_in|_+}  Σ_{s' ∈ S_in}  q_{s'} · w(s')
         s.t.  q ∈ projected TV ball around T[s, 0, S_in], radius rho_T(s, 0)
```

### s = 0

- `p_nominal_trans = T[0, 0, S_in] = [0.3, 0.4]`, `m_o = 0.7`
- Coefficients (from `w`): `[-5, -11]`
- Both negative → adversary wants both `q_0` and `q_1` *large*

**Compare moves**:

| Move | Δq_0 | Δq_1 | Δm | Drop per `δ` | Drop per TV |
|---|---|---|---|---|---|
| A. Within-shift q_0 → q_1 | −δ | +δ | 0 | (−5)(−δ) + (−11)(δ) interpreted as drop = (+11 − 5)δ = 6δ | 3 |
| C. Draw out → q_1 | 0 | +δ | +δ | (+11)δ | **5.5** |
| D. Draw out → q_0 | +δ | 0 | +δ | (+5)δ | 2.5 |

(For "drop" with a negative coefficient: increasing `q_j` by `δ` decreases the objective by `|c_j|·δ`.)

Best: **C (draw out → q_1)**, 5.5/TV. Budget exhausted at `δ = 0.1`.

Optimal: `q* = (0.3, 0.5)`, `Δm = +0.1` (mass drawn in from out-of-support).

TV check: `|0.3-0.3| + |0.5-0.4| + |0.8-0.7| = 0 + 0.1 + 0.1 = 0.2` ✓

```
σ(s=0)  =  0.3 · (-5)  +  0.5 · (-11)  =  -1.5 - 5.5  =  -7
```

### s = 1

- `p_nominal_trans = T[1, 0, S_in] = [0.5, 0.2]`, `m_o = 0.7`
- Same `w`, same `rho_T`

Same trade-off; C still wins. `δ = 0.1`.

Optimal: `q* = (0.5, 0.3)`, `Δm = +0.1`.

TV check: `|0.5-0.5| + |0.3-0.2| + |0.8-0.7| = 0.2` ✓

```
σ(s=1)  =  0.5 · (-5)  +  0.3 · (-11)  =  -2.5 - 3.3  =  -5.8
```

### Step 2 result

```
σ  =  [-7, -5.8]
```

---

## Aggregation

```
Q_robust(h, a=0)  =  Σ_{s ∈ S_in}  b(s) · [ R(s, 0) + σ(s) ]
                  =  0.6 · (1 + (-7))  +  0.4 · (2 + (-5.8))
                  =  0.6 · (-6)        +  0.4 · (-3.8)
                  =  -3.6              +  (-1.52)
                  =  -5.12
```

# **Expected `Q_robust = -5.12`**

---

## Debug-time quick reference

When you stop at the breakpoint inside `compute_robust_q` you should see:

| Variable | Expected value | Where in code |
|---|---|---|
| `S_in` | `[0, 1]` | line 409 |
| `Z_in` | `[0, 1]` | line 410 |
| `belief` | `array([0.6, 0.4])` | line 412 |
| `V_children` | `[10.0, -20.0]` | line 414 |
| `w` after Step 1 loop | `array([-5., -11.])` | filled lines 430–434 |
| `Q_val` after `s=0` iter | `-3.6` | accumulating at line 449 |
| `Q_val` final | `-5.12` | line 449 |
| Return value | `-5.12` | line 480 |

Inside the LP calls, the returned `optimal_p` for each (`s'`, `s`) should match:

| Call | Expected `optimal_p` |
|---|---|
| `lp_obs.solve` for `s_next=0` | `[0.3, 0.4]` |
| `lp_obs.solve` for `s_next=1` | `[0.1, 0.6]` |
| `lp_trans.solve` for `s=0` | `[0.3, 0.5]` |
| `lp_trans.solve` for `s=1` | `[0.5, 0.3]` |

Note `lp_trans` returns sub-probability vectors that sum to 0.8 (mass drawn in), while `lp_obs` returns sub-probability vectors that sum to 0.7 (no leakage at the optimum).
