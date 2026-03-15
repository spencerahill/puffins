# Documentation Roadmap

A phased plan to bring `puffins` documentation from its current state to comprehensive, state-of-the-art research software docs.

## Current State Assessment

- **Docstring coverage**: ~71% overall, ranging from 33% (`hides.py`) to 100% (`dates.py`, `therm_inert.py`, `budget_adj.py`)
- **Docstring style**: NumPy-style where present, but quality varies from one-liners to full Parameters/Returns sections
- **Type hints**: Sporadic; most modules lack them
- **Doc build system**: None (no Sphinx, no generated API docs)
- **Tutorials/examples**: None beyond two one-liners in `README.md`
- **No dedicated `docs/` directory**

---

## Phase 1: Foundation — Docstring Coverage and Consistency

**Goal**: Every public function has a complete, standardized NumPy-style docstring. Establish the docstring format contract.

### 1.1 Define the docstring standard

Adopt NumPy-style (already the dominant convention) with these mandatory sections for all public functions:

```python
def merid_streamfunc(v, dp, grav=GRAV_EARTH, radius=RAD_EARTH):
    """Meridional mass streamfunction.

    Longer description if the one-liner isn't sufficient. Include
    physical meaning, assumptions, and relevant equations.

    Parameters
    ----------
    v : xarray.DataArray
        Meridional wind field.
    dp : xarray.DataArray
        Pressure thickness of each gridbox, in Pascal.
    grav : float, optional
        Gravitational acceleration (m/s^2). Default: Earth.
    radius : float, optional
        Planetary radius (m). Default: Earth.

    Returns
    -------
    xarray.DataArray
        The meridional mass streamfunction, signed such that
        counter-clockwise circulation is positive.

    References
    ----------
    .. [1] Held & Hou (1980), J. Atmos. Sci., 37, 515-533.

    See Also
    --------
    had_cell_strength : Compute Hadley cell strength from streamfunction.

    Examples
    --------
    >>> psi = merid_streamfunc(v, dp)
    """
```

Sections by priority: Summary (required) → Parameters (required) → Returns (required) → References (when applicable) → See Also (when applicable) → Examples (add in Phase 3).

### 1.2 Triage modules by coverage gap

**Priority 1 — High-impact, low-coverage (<60%)**:
- `hides.py` (33%) — 4 functions need docstrings
- `polar_amp.py` (50%) — 2 functions
- `tropopause.py` (55%) — 5 functions
- `eq_area.py` (56%) — 14 functions
- `held_hou_1980.py` (56%) — 4 functions
- `vert_coords.py` (56%) — 8 functions

**Priority 2 — Core modules, moderate coverage (60–70%)**:
- `dynamics.py` (60%) — 6 functions
- `nb_utils.py` (60%) — 10 functions
- `bootstrap.py` (60%) — 2 functions
- `plotting.py` (61%) — 11 functions
- `had_cell.py` (64%) — 9 functions
- `interp.py` (67%) — 4 functions
- `lindzen_hou_1988.py` (59%) — 7 functions

**Priority 3 — Upgrade existing one-liner docstrings** to full NumPy-style across all modules.

### 1.3 Add type hints to all public functions

Add type hints alongside docstring work — one pass per module.

### 1.4 Add module-level docstrings

Each `.py` file should have a module docstring describing its purpose, key functions, and physical context.

### Deliverable

100% public function docstring coverage, consistent NumPy-style, type hints on all public APIs.

---

## Phase 2: API Reference — Automated Doc Generation

**Goal**: Sphinx-based API reference auto-generated from docstrings, hosted and versioned.

### 2.1 Set up Sphinx

```
docs/
├── conf.py
├── index.rst
├── installation.rst
├── api/
│   ├── index.rst
│   ├── dynamics.rst
│   ├── thermodynamics.rst
│   ├── had_cell.rst
│   ├── ... (one page per module)
```

Key extensions:
- `sphinx.ext.autodoc` — pull docstrings into docs
- `sphinx.ext.napoleon` — parse NumPy-style docstrings
- `sphinx.ext.intersphinx` — link to xarray, numpy, scipy docs
- `sphinx.ext.mathjax` — render equations
- `sphinx.ext.viewcode` — link to source

Theme: `pydata-sphinx-theme` (standard for scientific Python).

### 2.2 Organize API reference by functional group

Mirror the groups from `CLAUDE.md`:

1. **Core Utilities**: `constants`, `names`, `nb_utils`, `calculus`, `interp`, `num_solver`, `dates`, `longitude`, `bootstrap`
2. **Physical Calculations**: `dynamics`, `thermodynamics`, `tropopause`, `vert_coords`, `lcl`
3. **Climate Dynamics**: `had_cell`, `grad_bal`, `eq_area`, `eofs`, `stats`, `budget_adj`
4. **Theoretical Models**: `held_hou_1980`, `lindzen_hou_1988`, `plumb_hou_1992`, `kuo_el`, `fixed_temp_tropo`, `hides`, `polar_amp`, `therm_inert`
5. **Visualization**: `plotting`

### 2.3 Add mathematical notation

Use LaTeX in docstrings for key equations. Example:

```python
"""Coriolis parameter.

.. math::

    f = 2 \\Omega \\sin(\\phi)

where :math:`\\Omega` is the planetary rotation rate and
:math:`\\phi` is latitude.
"""
```

Priority: theoretical model modules (`held_hou_1980`, `lindzen_hou_1988`, `plumb_hou_1992`, `grad_bal`, `kuo_el`).

### 2.4 CI integration

- Add a `docs` tox/nox environment or Makefile target
- Build docs in CI; fail on warnings (catches broken cross-references)
- Add `docs` extras to `pyproject.toml`

### Deliverable

Browsable, searchable API reference with full cross-linking, equations, and source links. Built automatically on every push.

---

## Phase 3: Narrative Documentation — Tutorials and How-To Guides

**Goal**: Guide users from installation to productive research use.

### 3.1 Getting Started guide

- Installation (pip, editable mode, dependencies)
- Quick-start code showing a complete workflow
- Coordinate conventions and input expectations

### 3.2 Topical tutorials (Jupyter notebooks executed via `nbsphinx` or `myst-nb`)

Suggested notebooks:

| Tutorial | Modules Used |
|----------|-------------|
| Computing Hadley cell diagnostics from reanalysis data | `had_cell`, `dynamics`, `vert_coords` |
| The Held-Hou 1980 model: theory and code | `held_hou_1980`, `constants`, `plotting` |
| Gradient wind balance and thermal wind | `grad_bal`, `dynamics`, `thermodynamics` |
| Working with equal-area coordinates | `eq_area`, `calculus` |
| Plotting with `puffins` | `plotting`, `nb_utils` |
| Multi-planet calculations | `constants`, `dynamics`, `thermodynamics` |

### 3.3 How-to guides (task-oriented, concise)

- How to compute streamfunction and Hadley cell metrics
- How to interpolate to isentropic coordinates
- How to run EOF analysis
- How to switch planetary constants
- How to format latitude axes on plots

### 3.4 Inline examples in docstrings

Add `Examples` sections to the most-used functions (start with `dynamics.py`, `had_cell.py`, `thermodynamics.py`, `plotting.py`). Use `doctest`-compatible format.

### Deliverable

4–6 executed tutorial notebooks, 5+ how-to guides, docstring examples on high-traffic functions.

---

## Phase 4: Explanations and Theory — Deep Reference Material

**Goal**: Connect code to the underlying atmospheric dynamics theory for researchers new to the field or the specific implementations.

### 4.1 Concept pages

- **Coordinate conventions**: latitude, pressure, sine-latitude, equal-area — why puffins uses each and how to convert
- **Streamfunction sign convention**: detailed explanation with diagrams
- **Axisymmetric assumptions**: what breaks when inputs aren't zonally averaged
- **Vertical coordinate handling**: pressure ordering, sigma-to-pressure, isentropic coordinates

### 4.2 Theory pages for model modules

For each theoretical model module, a dedicated page covering:
- The original paper's key results and equations
- How the code implements the theory (mapping equations to functions)
- Known limitations and edge cases
- Reproduction of key figures from the original papers

Modules: `held_hou_1980`, `lindzen_hou_1988`, `plumb_hou_1992`, `kuo_el`, `fixed_temp_tropo`.

### 4.3 Glossary

Define terms used throughout the codebase: streamfunction, Hadley cell edge/strength, thermal Rossby number, Coriolis parameter, lifted condensation level, etc.

### Deliverable

Concept and theory pages bridging textbook knowledge to code, glossary, diagrams.

---

## Phase 5: Polish and Sustainability

**Goal**: Make documentation self-maintaining and contributor-friendly.

### 5.1 Hosting

Host on Read the Docs (free for open-source) with versioned docs tied to releases.

### 5.2 Contributor guide

- How to write docstrings (link to the standard from Phase 1)
- How to add a tutorial notebook
- How to build docs locally

### 5.3 Docstring enforcement

- Add `pydocstyle` or `ruff` docstring rules to CI
- Enforce NumPy-style format and minimum coverage on new code
- Consider `interrogate` for coverage reporting

### 5.4 Changelog

Maintain a `CHANGELOG.md` tied to releases and linked from docs.

### 5.5 README refresh

Update `README.md` with badges (docs build status, test coverage, PyPI version) and a link to the full documentation site.

### Deliverable

Hosted, versioned docs. CI-enforced docstring standards. Contributor documentation workflow.

---

## Summary Timeline

| Phase | Focus | Key Metric |
|-------|-------|-----------|
| **1** | Docstrings + type hints | 100% public API coverage |
| **2** | Sphinx API reference | Browsable, auto-generated docs |
| **3** | Tutorials + how-tos | 4–6 notebooks, 5+ guides |
| **4** | Theory + concepts | Concept pages for all model modules |
| **5** | Hosting + CI + sustainability | Read the Docs live, CI-enforced standards |

Each phase builds on the previous. Phase 1 is the foundation — without complete docstrings, auto-generated API docs are hollow. Phases 3–4 are where the docs become genuinely valuable to researchers beyond the author. Phase 5 keeps it all from decaying.
