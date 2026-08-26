# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`puffins` is a Python library for large-scale atmospheric and climate dynamics research. It provides functions for computing physical quantities and creating visualizations, primarily for use in Jupyter notebooks. The codebase is organized around atmospheric physics calculations, theoretical climate models, and data analysis tools.

## Build and Installation

Install the package in editable mode:
```bash
pip install -e .
```

Run tests:
```bash
pytest
```

## Module Architecture

The package is structured into several functional groups:

### Core Utilities
- `_typing.py`: Shared type aliases (`Scalar`, `ArrayLike`, `XarrayObj`, plus `SolverParamRange` and `SolverGuessRange`, which admit plain sequences and keep `num_solver` aligned with its callers)
- `constants.py`: Physical constants for Earth, Mars, Saturn, Titan, and Venus
- `names.py`: String constants for coordinate/dimension names (lat, lon, lev, time, etc.)
- `nb_utils.py`: Jupyter notebook utilities including coordinate array creation and trigonometric helpers
- `calculus.py`: Derivatives, integrals, and averages; also grid geometry (cell bounds, surface area, latitude circumference) and meridional transport diagnostics
- `interp.py`: Interpolation utilities
- `num_solver.py`: Numerical solvers
- `dates.py`: Date utilities
- `longitude.py`: Longitude utilities and `Longitude` class
- `bootstrap.py`: Bootstrap statistical methods

### Physical Calculations
- `dynamics.py`: Fundamental dynamical quantities (Coriolis parameter, absolute angular momentum, vorticity, Rossby number)
- `thermodynamics.py`: Thermodynamic calculations
- `tropopause.py`: Tropopause diagnostics
- `vert_coords.py`: Vertical coordinate transformations and pressure-level utilities (mass-weighted column integrals and averages, Simmons-Burridge full-level pressures)
- `lcl.py`: Lifted condensation level calculations
- `radiation.py`: Blackbody radiation (Planck function, Wien's displacement law)

### Climate Dynamics
- `had_cell.py`: Hadley cell and meridional overturning circulation diagnostics (streamfunction, cell strength/extent)
- `grad_bal.py`: Gradient balance and thermal wind balance, plus the angular-momentum-conserving and uniform-Rossby-number wind and potential temperature expressions
- `eq_area.py`: Equal-area model analytical solutions and numerical solvers (Held-Hou 1980, Lindzen-Hou 1988 variants); belongs conceptually with the Theoretical Models below
- `eofs.py`: Empirical Orthogonal Function analysis
- `stats.py`: Statistical analysis tools
- `budget_adj.py`: Column budget adjustment via spherical harmonic wind inversion

### Theoretical Models
- `held_hou_1980.py`: Held-Hou 1980 model implementation
- `lindzen_hou_1988.py`: Lindzen-Hou 1988 model
- `plumb_hou_1992.py`: Plumb-Hou 1992 model
- `kuo_el.py`: Kuo-Eliassen equation solver
- `fixed_temp_tropo.py`: Fixed tropopause temperature model
- `hides.py`: Hide's theorem calculations
- `polar_amp.py`: Polar amplification diagnostics
- `therm_inert.py`: Thermal inertia calculations

### Visualization
- `plotting.py`: Matplotlib helpers with custom styling, latitude axis formatting (sine-latitude and standard), and faceted plotting integration

## Key Design Patterns

### xarray Integration
Most functions operate on `xarray.DataArray` objects with standardized dimension names from `names.py` (LAT_STR, LON_STR, LEV_STR, TIME_STR, etc.).

### Physical Constants
Use constants from `constants.py` as default parameters. Functions typically accept planet-specific parameters (e.g., `grav=GRAV_EARTH`, `radius=RAD_EARTH`, `rot_rate=ROT_RATE_EARTH`) to enable calculations for other planets.

### Coordinate Conventions
- Latitude: degrees, -90 to 90
- Pressure levels: typically Pascal, but functions often have `hpa_to_pa` flags
- Streamfunctions: signed such that counter-clockwise circulation in meridional plane is positive

### Working-Tree Roles and Consumption by Other Projects
Two local copies of this repo exist, with strictly separated roles:

- `~/Dropbox/py/puffins` (this repo): the development tree. It may sit on any branch at any time; nothing else should import from it.
- `~/Dropbox/py/puffins-main`: a consumer clone permanently on `master`, updated only via `git pull`, never developed or committed on. Other projects' environments install puffins from this path (`pip install -e ~/Dropbox/py/puffins-main --no-deps`; the `--no-deps` is because project environments provide the dependencies themselves), so they always import pushed, CI-green master regardless of what branch the development tree is on.

A project that needs an unmerged branch gets its own temporary clone or worktree pinned to that branch, removed once the branch merges. To freeze an analysis (e.g., at paper submission), replace the editable install with a non-editable one pinned to a commit; the setuptools-scm version string records the SHA.

The former `set_proj_puff_branch.py` script and `nb_utils.setup_puffins()` function, which switched the single shared working tree between branches per notebook, were removed in favor of this arrangement.

## Code Standards

### Software Quality
This is research code, but it must be well-crafted software. Write clean, maintainable code with clear structure and appropriate documentation.

### Vectorization
Always use vectorized operations. Leverage array operations from numpy, xarray, and scipy. Only fall back to explicit loops over array elements when there is truly no vectorized alternative.

### Prefer Existing Implementations
Use builtin methods and functions whenever possible:
1. First choice: xarray, numpy, scipy, and existing package dependencies
2. Second choice: well-established packages that fit the need
3. Last resort: custom implementations only when no existing solution exists

### Git Practices
Always create a feature branch before starting work. Never commit directly to master. Branch naming: `<topic>` (e.g., `add-lcl-tests`, `fix-streamfunc-sign`). Clear commit messages, logical commits, and clean history.

A PR touching numerical code gets two independent reviewers with distinct lenses, one on physical correctness and one on test rigor, each walled off from the PR body and commit message. Their *disagreement* is the signal: chase it with the experiment that separates the two explanations rather than picking a side. *Case: incidents.md#reconstruction-shares-discretization*

### Type Hints
All new code must include type hints for function parameters and return values.

### Testing
All new code must have tests. Run all tests and ensure they pass before considering work complete.

For functions with a nontrivial coefficient chain or closed-form expression, include at least one known-value test that reconstructs the full expected output from raw numpy, not from the module's own helper functions. Then confirm the test has teeth by mutation: perturb one coefficient in the source, verify only that test fails, then revert. Limiting-case, symmetry, and monotonicity tests alone leave coefficient magnitudes and phases unconstrained.

**A raw-numpy rebuild is not independent when it rebuilds the same discretization (added 2026-08-26).** For a function that discretizes a continuous operator, an integral, derivative, interpolation or filter, the known-value anchor is a closed-form solution, never a rebuild: a numpy paraphrase of the same algorithm inherits its quadrature error and every test passes. The check is mechanical: if the test helper carries the same `cumsum`, `gradient` or axis-reduction structure as the source, it is a paraphrase. Where no closed form exists, say so and mark the reconstruction as coefficient-pinning only. *Case: incidents.md#reconstruction-shares-discretization*

The mutation-check applies beyond closed-form formulas. Any function with nontrivial internal logic (a partition or split, an index or slice offset, a branch, a disjointness condition) needs at least one test that fails when that logic is perturbed, confirmed by mutation. A limiting-case anchor can be blind to the very logic it appears to cover: `boot_risk_ratio`'s constant-array test passed even with its numerator/denominator split mutated to overlap, because a constant array yields risk ratio 1 for any split, and the gap survived until an independent review. Where the function seeds its RNG internally, add a `seed` parameter so a reconstruction test can reproduce the draw and pin the logic exactly.

Before considering a module's tests complete, enumerate every parameter that affects the output and confirm each is exercised with a non-default value in at least one test whose assertion would fail if that parameter's handling were broken (mutation-checked). A parameter used only at its default in every test has no teeth: `polar_amp`'s `denom_bounds` shipped that way in the first draft (every test used the global `(-90, 90)` default), and hardcoding the denominator to the global range passed all 12 tests until an independent review caught it. Applying the mutation-check to the headline formula alone is not enough; sweep it across the full parameter surface.

### Status Documents
The four roadmaps (`docs/roadmaps/001-004`) cite shared status metrics: total test count, test-file count, line-coverage percentage, module-annotation count, mypy CI-blocking status, and module count. When a change moves any such metric, update every occurrence across all of these documents in the same PR, not only the document most on-topic. Scope the doc edits by which metric moved, not by which document is nearest the change. (Promoting mypy to blocking in PR #64 moved metrics in all four roadmaps, but the first pass updated only roadmaps 001 and 003.)

## Important Notes

- This is a personal research tool with no official support
- Functions assume specific input shapes and coordinate conventions
- Many calculations assume axisymmetric (zonally-averaged) conditions
- Pressure coordinate ordering matters: functions check for monotonicity and handle both increasing/decreasing vertical coordinates
