# Roadmap 003: Type Hints

| Field | Value |
|-------|-------|
| **Status** | In Progress |
| **Created** | 2026-03-16 |
| **Last updated** | 2026-07-18 |
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

## Progress

**20 of 30 modules fully annotated** (in the `pyproject.toml` mypy strict
overrides) as of 2026-07-18. Remaining (10): `budget_adj` (priority — has a
`no-any-return` error), `eq_area`, `grad_bal`, `kuo_el`, `held_hou_1980`,
`lindzen_hou_1988`, `plumb_hou_1992`, `fixed_temp_tropo`, `plotting`,
`nb_utils`. mypy is still non-blocking in CI; current source-file errors: 4
(`budget_adj` ×1, `eq_area` ×2, `grad_bal` ×1).

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
- [x] `num_solver.py` — numerical solvers; 25 new tests added (completed 2026-03-25)

## Group 3: Physical Calculations

- [x] `dynamics.py` — Coriolis parameter, angular momentum, vorticity, Rossby number; 58 tests added (completed 2026-03-26)
- [x] `thermodynamics.py` — thermodynamic calculations; 60 tests added, removed duplicate function definitions (completed 2026-03-26)
- [x] `vert_coords.py` — vertical coordinate transformations; 62 tests (59 pass, 3 xfail for pre-existing xr.concat bug #26 in pfull_simm_burr) (completed 2026-03-29)
- [x] `tropopause.py` — tropopause diagnostics; 18 tests (8 pass for `tropo_wmo`, 10 xfail for six legacy implementations broken since ~2020 per SAH source note); fixed hardcoded `level` dim string in `tropo_wmo` to use the `p_str` argument (completed 2026-04-18)
- [x] `lcl.py` — lifted condensation level; fixed the missing-import breakage (issue #41 part 1) so the module is importable and re-exported from `__init__`; forwarded the thermodynamic constants into the internal `sat_vap_press_liq_wat` calls (they were silently ignored) and dropped its unused `r_d` parameter. Resolved issue #41 part 2 by deleting `lift_cond_temp` (a misnamed duplicate of the height) and implementing `pres_lift_cond_level` for the previously-missing LCL pressure (Romps 2017 Eq. 22b); renamed `lift_cond_level` → `height_lift_cond_level` for symmetry with the temperature/pressure functions. The module now cleanly implements Eqs. 22a (temperature), 22b (pressure), and 22c (height), and deliberately defines its own Romps-optimized physical constants (R_V, C_VL, etc.) instead of importing `puffins.constants`, whose textbook values would degrade the paper's ~5 m accuracy. 41 tests with raw-numpy known-value reconstructions over several parcels, constant-forwarding regression tests, and constant-provenance pinning (completed 2026-07-17)

## Group 4: Climate Dynamics

- [x] `had_cell.py` — Hadley cell / meridional overturning diagnostics; type hints + tests added, `cell_edges_sigma` custom-dim fix and unused `frac_thresh` drop (PR #54). Surfaced three latent bugs filed as issues #55, #56, #57 (completed 2026-07-17)
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
- [x] `polar_amp.py` — polar amplification diagnostics; type hints + tests added, `denom_bounds` given teeth in tests (completed 2026-07-16)
- [x] `therm_inert.py` — thermal inertia calculations; type hints + tests added, `temp_rad_eq_eff` annual-cycle coefficient and phase pinned (completed 2026-07-01)

## Group 6: Statistics & Analysis

- [x] `stats.py` — statistical analysis tools; 66 tests added; fixed `rmse` (`squared=` removed in sklearn ≥1.4) and `quantile_regress` (returned a length-1 `coef_` array that broke `apply_ufunc`) to work with modern sklearn/numpy (completed 2026-07-16)
- [x] `bootstrap.py` — bootstrap methods; type hints + tests added, seedable `boot_risk_ratio`, wider `rand_states`, NaN handling (completed 2026-07-15)
- [x] `eofs.py` — empirical orthogonal functions; type hints + tests added, `lat_str` coverage and unified RNG (completed 2026-07-13)
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
