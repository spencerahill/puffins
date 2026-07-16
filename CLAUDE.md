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
- `_typing.py`: Shared type aliases (`Scalar`, `ArrayLike`, `XarrayObj`)
- `constants.py`: Physical constants for Earth, Mars, Saturn, Titan, and Venus
- `names.py`: String constants for coordinate/dimension names (lat, lon, lev, time, etc.)
- `nb_utils.py`: Jupyter notebook utilities including coordinate array creation and trigonometric helpers
- `calculus.py`: Numerical differentiation and integration operations
- `interp.py`: Interpolation utilities
- `num_solver.py`: Numerical solvers
- `dates.py`: Date utilities
- `longitude.py`: Longitude utilities and `Longitude` class
- `bootstrap.py`: Bootstrap statistical methods

### Physical Calculations
- `dynamics.py`: Fundamental dynamical quantities (Coriolis parameter, absolute angular momentum, vorticity, Rossby number)
- `thermodynamics.py`: Thermodynamic calculations
- `tropopause.py`: Tropopause diagnostics
- `vert_coords.py`: Vertical coordinate transformations
- `lcl.py`: Lifted condensation level calculations
- `radiation.py`: Blackbody radiation (Planck function, Wien's displacement law)

### Climate Dynamics
- `had_cell.py`: Hadley cell and meridional overturning circulation diagnostics (streamfunction, cell strength/extent)
- `grad_bal.py`: Gradient wind balance calculations
- `eq_area.py`: Equal-area coordinate transformations
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

### Type Hints
All new code must include type hints for function parameters and return values.

### Testing
All new code must have tests. Run all tests and ensure they pass before considering work complete.

For functions with a nontrivial coefficient chain or closed-form expression, include at least one known-value test that reconstructs the full expected output from raw numpy, not from the module's own helper functions. Then confirm the test has teeth by mutation: perturb one coefficient in the source, verify only that test fails, then revert. Limiting-case, symmetry, and monotonicity tests alone leave coefficient magnitudes and phases unconstrained.

The mutation-check applies beyond closed-form formulas. Any function with nontrivial internal logic (a partition or split, an index or slice offset, a branch, a disjointness condition) needs at least one test that fails when that logic is perturbed, confirmed by mutation. A limiting-case anchor can be blind to the very logic it appears to cover: `boot_risk_ratio`'s constant-array test passed even with its numerator/denominator split mutated to overlap, because a constant array yields risk ratio 1 for any split, and the gap survived until an independent review. Where the function seeds its RNG internally, add a `seed` parameter so a reconstruction test can reproduce the draw and pin the logic exactly.

## Important Notes

- This is a personal research tool with no official support
- Functions assume specific input shapes and coordinate conventions
- Many calculations assume axisymmetric (zonally-averaged) conditions
- Pressure coordinate ordering matters: functions check for monotonicity and handle both increasing/decreasing vertical coordinates
