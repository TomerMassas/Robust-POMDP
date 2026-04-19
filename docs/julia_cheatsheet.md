# Julia / VSCode Cheatsheet

Quick reference for working with Julia in VSCode (coming from Python/PyCharm).

## Terminal / REPL basics

### Starting the REPL
- Command line: `julia --project=.` (from repo root — `--project=.` activates the project env)
- VSCode: `Ctrl+Shift+P` → "Julia: Start REPL"

### REPL modes (press these keys at the `julia>` prompt)
| Key | Mode | Prompt |
|-----|------|--------|
| (default) | Julia code | `julia>` |
| `]` | Package manager | `(ProjectName) pkg>` |
| `;` | Shell commands | `shell>` |
| `?` | Help mode | `help?>` |
| Backspace | Return to Julia mode | `julia>` |

### Package manager (pkg mode — after pressing `]`)
| Command | What it does |
|---------|--------------|
| `activate .` | Activate the project env in current directory |
| `activate` | Activate the default env |
| `status` | Show installed packages |
| `add PkgName` | Install a package |
| `remove PkgName` | Uninstall |
| `instantiate` | Install all deps from Project.toml |
| `update` | Update packages |
| `test` | Run tests in test/runtests.jl |

**Check which env you're in:** the pkg-mode prompt shows the env name. `(@v1.12) pkg>` = default (bad), `(RobustOnlinePOMDP) pkg>` = project (good).

## Running code

### From the command line (fresh process, slow first run)
```
julia --project=. experiments/tiger_test.jl
```

### From the REPL (persistent, fast re-runs)
```julia
julia> include("experiments/tiger_test.jl")
```

### Sending code from the editor to the REPL (interactive workflow)
| Shortcut | What it does |
|----------|--------------|
| `Alt+Enter` | Run current line (or selection) in REPL |
| `Shift+Enter` | Same as Alt+Enter on some setups |
| `Ctrl+Enter` | Run current cell (cells delimited by `##`) |

After running, variables persist in the REPL — inspect them by typing the name:
```julia
julia> action
3
julia> root.children[3].Q_robust
0.935
```

## Debugging

### Setting a breakpoint
Click in the gutter (left of line numbers) next to a line → red dot appears. Or press `F9` on the line.

### Running with debugger
Two options:

**Option A (debug whole file):**
`Ctrl+Shift+P` → "Julia: Debug File in New Process"

**Option B (debug at REPL — more flexible):**
```julia
julia> using Debugger
julia> @run robust_pomcp_plan(belief, tiger, uncertainty, 5, 500, 500)
```

### Debug controls
| Key | Action |
|-----|--------|
| `F5` | Continue |
| `F10` | Step over |
| `F11` | Step into |
| `Shift+F11` | Step out |
| `Shift+F5` | Stop debugger |

When paused, the **debug pane** shows current variables, call stack, and the REPL becomes a debug prompt where you can type any expression to inspect state.

## Common Julia commands

| Need | Command |
|------|---------|
| Check current Julia version | `VERSION` or `julia --version` |
| Exit REPL | `exit()` or `Ctrl+D` at empty prompt |
| Interrupt running code | `Ctrl+C` |
| Clear screen | `Ctrl+L` |
| Access previous command | Up arrow |
| Search command history | `Ctrl+R` then type |
| Measure time | `@time expr` |
| Measure time (accurate, multiple runs) | `@btime expr` (needs BenchmarkTools) |
| Inspect variable | just type its name at `julia>` |
| Check type | `typeof(x)` |
| See what methods a type has | `methodswith(TypeName)` |
| Help on a function | `?function_name` (in help mode) |

## Common gotchas (Python → Julia)

1. **Arrays are 1-indexed.** `arr[1]` is the first, `arr[end]` is the last. NOT `arr[0]`.
2. **`=` vs `==`.** Assignment vs equality. Same as Python.
3. **`!=` → `!=` (same) or `≠`** (Unicode works).
4. **No `self`.** Methods aren't inside classes. Functions are separate from structs.
5. **`include()` is not `import`.** It pastes file content; use `using Module` for packages.
6. **REPL keeps state.** Re-`include()`-ing a file redefines everything (but can't redefine constants/structs — restart REPL if you change those).
7. **Broadcasting with `.`.** `sin.(x)` applies sin elementwise to array x (equivalent to Python's `np.sin(x)`).
8. **Tuple unpacking:** `(a, b) = (1, 2)` same as Python.
9. **String interpolation:** `"x is $x"` or `"sum is $(a + b)"`.

## Workflow tips

- **Keep the REPL alive.** Most Julia dev happens in one long-running REPL session. Re-running `include("myfile.jl")` after edits is the normal loop.
- **Restart REPL when you redefine struct fields.** Structs can't be redefined in the same session — you'll get an error. Restart REPL (exit + reopen).
- **First run is slow** (precompilation). Subsequent runs in the same REPL are fast. This is normal — Julia is JIT-compiled.
- **Check environment before running.** `]` → see prompt → `activate .` if needed.

## This project specifically

To run the Tiger test from scratch:
```
1. cd to repo root
2. julia --project=.
3. julia> include("experiments/tiger_test.jl")
```

Or fully from command line:
```
julia --project=. experiments/tiger_test.jl
```
