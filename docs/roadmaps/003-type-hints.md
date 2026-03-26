# Roadmap 003: Type Hints

| Field | Value |
|-------|-------|
| **Status** | In Progress |
| **Created** | 2026-03-16 |
| **Last updated** | 2026-03-26 |
| **Author** | Claude |
| **Parent** | [001 — Modernize Repository Standards](001-modernize-repo-standards.md), Phase 4 |

## Objective

Add type annotations to every function in the puffins codebase. Type hints
improve editor support, catch bugs early, and serve as machine-readable
documentation — especially valuable in a physics codebase where function
signatures encode units and coordinate conventions.

## Success Criteria

- Type annotations on all function parameters and return values
- Tests accompany each newly annotated module
- mypy passes (initially non-blocking, eventually required in CI)

---

## Group 0: Infrastructure

- [x] Create `puffins/_typing.py` with shared type aliases (`Scalar`, `ArrayLike`, `XarrayObj`)
- [x] Add per-module mypy strict overrides in `pyproject.toml`
- [x] Add mypy to CI (non-blocking, `continue-on-error: true`)
- [ ] Promote mypy CI check to blocking (remove `continue-on-error`)
- [ ] Promote global strict mode once all modules are annotated

## Group 1: Simple Values & Utilities

- [x] `constants.py` — module-level constants, no functions (completed 2026-03-18)
- [x] `names.py` — string constants, no functions (completed 2026-03-18)
- [x] `longitude.py` — `Longitude` class and utilities; 51 tests added (completed 2026-03-19)
- [x] `dates.py` — date utilities; 20 tests added (completed 2026-03-19)

## Group 2: Core Numerical Utilities

- [x] `calculus.py` — differentiation, integration, surface area; 57 new tests added (completed 2026-03-20)
- [x] `interp.py` — interpolation utilities; 35 new tests added (completed 2026-03-22)
- [ ] `num_solver.py` — numerical solvers

## Group 3: Physical Calculations

- [ ] `dynamics.py` — Coriolis parameter, angular momentum, vorticity, Rossby number
- [x] `thermodynamics.py` — thermodynamic calculations; 60 tests added, removed duplicate function definitions (completed 2026-03-26)
- [ ] `vert_coords.py` — vertical coordinate transformations
- [ ] `tropopause.py` — tropopause diagnostics
- [ ] `lcl.py` — lifted condensation level

## Group 4: Climate Dynamics

- [ ] `had_cell.py` — Hadley cell / meridional overturning diagnostics
- [ ] `grad_bal.py` — gradient wind balance
- [ ] `eq_area.py` — equal-area coordinate transformations
- [x] `hides.py` — Hide's theorem (completed 2026-03-16)
- [x] `radiation.py` — Planck function and Wien's law (completed 2026-03-18)

## Group 5: Theoretical Models

- [ ] `held_hou_1980.py` — Held-Hou 1980 model
- [ ] `lindzen_hou_1988.py` — Lindzen-Hou 1988 model
- [ ] `plumb_hou_1992.py` — Plumb-Hou 1992 model
- [ ] `kuo_el.py` — Kuo-Eliassen equation solver
- [ ] `fixed_temp_tropo.py` — fixed tropopause temperature model
- [ ] `polar_amp.py` — polar amplification diagnostics
- [ ] `therm_inert.py` — thermal inertia calculations

## Group 6: Statistics & Analysis

- [ ] `stats.py` — statistical analysis tools
- [ ] `bootstrap.py` — bootstrap methods
- [ ] `eofs.py` — empirical orthogonal functions
- [ ] `budget_adj.py` — column budget adjustment (**priority**: has mypy `no-any-return` error)

## Group 7: Visualization & Notebooks

- [ ] `plotting.py` — matplotlib helpers
- [ ] `nb_utils.py` — Jupyter notebook utilities

## Group 8: CI Enforcement

Moved to Group 0 (Infrastructure). See above.

---

## Notes

- Groups are ordered simplest → most complex, but can be worked in any order.
- Each module should be annotated and tested in the same PR where practical.
- Use `xr.DataArray` for xarray inputs/outputs; use `float` for physical
  constants that default to Earth values.
- For functions accepting mixed types, import from `puffins._typing`:
  `ArrayLike` (DataArray | ndarray | scalar), `Scalar`, or `XarrayObj`.
- When annotating a module, add it to the `[[tool.mypy.overrides]]` module
  list in `pyproject.toml`.
- This roadmap is a subset of Phase 4 in
  [Roadmap 001](001-modernize-repo-standards.md).
