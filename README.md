# puffins

**A Python toolkit for large-scale atmospheric and climate dynamics research.**

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)
![Status: Pre-Alpha](https://img.shields.io/badge/status-pre--alpha-orange)

puffins provides functions for computing physical quantities, analyzing
climate data, and creating publication-ready visualizations. Built on
[xarray](https://xarray.dev), it is designed for use in Jupyter notebooks
for both interactive exploration and generating figures for papers and
presentations.

> **Note:** This is a personal research tool. You are welcome to use it,
> but I cannot provide support or guarantee correctness. Use at your own
> risk.

## Features

### Core Utilities
Physical constants for Earth, Mars, Saturn, Titan, and Venus.
Numerical differentiation, integration, interpolation, and solvers.
Standardized coordinate and dimension names for consistent xarray workflows.

### Physical Calculations
Coriolis parameter, absolute angular momentum, vorticity, Rossby number,
thermodynamic quantities, tropopause diagnostics, and vertical coordinate
transformations.

### Climate Dynamics
Hadley cell and meridional overturning circulation diagnostics
(streamfunction, cell strength and extent), gradient wind balance,
equal-area coordinates, Empirical Orthogonal Functions, and column budget
adjustment via spherical harmonic wind inversion.

### Theoretical Models
Implementations of classic models from the literature:
[Held & Hou (1980)](https://doi.org/10.1175/1520-0469(1980)037<0515:NASCIA>2.0.CO;2),
[Lindzen & Hou (1988)](https://doi.org/10.1175/1520-0469(1988)045<2416:HCFSSF>2.0.CO;2),
[Plumb & Hou (1992)](https://doi.org/10.1175/1520-0469(1992)049<1790:TROTAN>2.0.CO;2),
the Kuo-Eliassen equation, and Hide's theorem.

### Visualization
Matplotlib helpers with custom styling, sine-latitude and standard
latitude axis formatting, and integration with
[faceted](https://github.com/spencerahill/faceted) for multi-panel plots.

## Quick Examples

```python
import puffins

# Coriolis parameter at 30°N
f = puffins.dynamics.coriolis_param(30)

# Same calculation for Mars
f_mars = puffins.dynamics.coriolis_param(
    30, rot_rate=puffins.constants.ROT_RATE_MARS
)

# Meridional mass streamfunction from v-wind and pressure thickness
psi = puffins.had_cell.merid_streamfunc(v, dp)

# Held-Hou 1980 Hadley cell edge latitude
phi_h = puffins.held_hou_1980.hc_edge_hh80(delta_h=1/6)
```

Functions accept planet-specific parameters (gravity, radius, rotation
rate) so the same code works across planetary atmospheres.

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/spencerahill/puffins.git
cd puffins
pip install -e .
```

### Dependencies

puffins requires Python 3.10+ and the following packages (installed
automatically):

- [xarray](https://xarray.dev)
- [numpy](https://numpy.org)
- [scipy](https://scipy.org)
- [matplotlib](https://matplotlib.org)
- [eofs](https://ajdawson.github.io/eofs/)
- [faceted](https://github.com/spencerahill/faceted)
- [statsmodels](https://www.statsmodels.org)
- [scikit-learn](https://scikit-learn.org)
- [pymannkendall](https://github.com/mmhs013/pyMannKendall)

The spherical-harmonics wind inversion in `budget_adj` additionally
requires [windspharm](https://ajdawson.github.io/windspharm/), which is
optional (the `fortran` extra) and on modern setups must be installed
via conda: `conda install -c conda-forge windspharm`.

## Using puffins in individual science projects

puffins is developed continuously, so a development clone may be checked
out to any branch at any moment. Science projects should not import from
a clone that is also used for development. The intended arrangement:

- **Maintain a second, consumer-only clone** (e.g. `puffins-main`) that
  stays permanently on `master` and is only ever updated via `git pull`.
  Install it into each project's environment in editable mode
  (`pip install -e /path/to/puffins-main`, adding `--no-deps` when the
  environment already provides puffins' dependencies, e.g. from conda).
  Every project then imports
  pushed, CI-tested `master`, and a single `pull` updates all of them at
  once.
- **If a project needs an unmerged branch**, give that project its own
  temporary clone or git worktree pinned to that branch, install that
  copy into the project's environment only, and delete it once the
  branch merges.
- **To freeze an analysis** (e.g., when a paper is submitted), replace
  the editable install with one pinned to a commit or tag:
  `pip install git+https://github.com/spencerahill/puffins@<sha-or-tag>`.
  The setuptools-scm version string records the exact commit.

Earlier versions of puffins instead switched a single shared clone
between branches from the top of each notebook
(`nb_utils.setup_puffins()` and the `set-proj-puff-branch` script);
those utilities have been removed.

## License

This project is licensed under the Apache License 2.0 — see the
[LICENSE](LICENSE) file for details.

## Author

[Spencer A. Hill](https://github.com/spencerahill)
