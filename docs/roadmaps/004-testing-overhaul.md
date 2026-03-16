# Roadmap 004: Testing Overhaul

| Field | Value |
|-------|-------|
| **Status** | Not Started |
| **Created** | 2026-03-16 |
| **Last updated** | 2026-03-16 |
| **Author** | Spencer A. Hill |

## Objective

Take puffins from near-zero test coverage to near-comprehensive coverage
with best-practice infrastructure, targeting at least one test per public
function and enforced coverage thresholds in CI.

## Current State

- **3 test files** (`test_budget_adj.py`, `test_calculus.py`, `test_kuo_el.py`), covering 3 of ~31 modules
- **~328 public functions**, the vast majority untested
- CI workflow (`ci.yml`) and pytest configuration in `pyproject.toml` now exist (added in Roadmap 001)
- pytest and pytest-cov are project dev dependencies

The existing tests are well-structured and serve as good templates: they use xarray fixtures, descriptive test class names, and test both correctness and type contracts.

---

## Phase 1: Infrastructure

Most infrastructure was delivered by [Roadmap 001](001-modernize-repo-standards.md). Remaining items:

### 1.1 Create `conftest.py` with shared fixtures

Extract and generalize the helper functions from existing test files into `puffins/tests/conftest.py`. Provide reusable fixtures for common test data:

- 1D latitude arrays (uniform and Gaussian)
- 1D pressure/level arrays
- 2D lat-lon grids
- 3D (lat, lon, time) and 4D (lat, lon, lev, time) DataArrays
- Standard coordinate names from `names.py`

Keep fixtures minimal — just enough structure to construct inputs. Individual test files build on these as needed.

### 1.2 Add `slow` marker

Register a `slow` pytest marker in `pyproject.toml` so expensive tests can be deselected with `-m "not slow"`.

---

## Phase 2: Pure-Function and Constants Modules (Quick Wins)

These modules have simple input/output contracts, no I/O, and few dependencies. They yield high coverage per unit of effort.

### Priority order and rationale

| Module | Public funcs | Why prioritize |
|--------|-------------|----------------|
| `constants.py` | ~0 (values) | Sanity-check constants haven't drifted; trivial to write |
| `names.py` | ~0 (strings) | Same — verify string constants exist and are consistent |
| `dynamics.py` | 15 | Pure math on arrays; foundational to other modules |
| `thermodynamics.py` | 24 | Pure math; heavily used downstream |
| `calculus.py` | 14 | Differentiation/integration — easy to validate against analytical solutions |
| `num_solver.py` | 5 | Small module, testable against known roots |
| `dates.py` | 5 | Small, pure utilities |
| `longitude.py` | 3 + class | Small, self-contained |

### Testing strategy for pure functions

- **Known-answer tests**: Compare against hand-calculated or textbook values. For example, Coriolis parameter at 45°N should be `2 * ROT_RATE_EARTH * sin(π/4)`.
- **Limiting/special cases**: Zero inputs, poles, equator, tropopause boundaries.
- **Symmetry and sign tests**: Coriolis is antisymmetric about the equator; absolute angular momentum is positive in the tropics.
- **Dimension/coordinate preservation**: Output DataArray has the same dims and coords as input.
- **Multi-planet parameters**: Functions accept planet constants — test with at least Earth defaults and one non-Earth case to confirm the parameterization works.

---

## Phase 3: Coordinate and Interpolation Modules

| Module | Public funcs | Notes |
|--------|-------------|-------|
| `vert_coords.py` | 17 | Vertical coordinate transforms; test round-trip consistency |
| `interp.py` | 11 | Interpolation; test against scipy directly and known profiles |
| `eq_area.py` | 15 | Equal-area transforms; test invertibility and area conservation |
| `lcl.py` | 6 | LCL calculations; test against published tables |

### Testing strategy

- **Round-trip tests**: Transform to a new coordinate and back; result should match original within tolerance.
- **Conservation tests**: Equal-area transforms should conserve integrated quantities.
- **Monotonicity**: Interpolated profiles on pressure levels should preserve monotonicity where physically expected.
- **Edge cases**: Surface pressure at sea level, top-of-atmosphere, single-level inputs.

---

## Phase 4: Climate Diagnostics

These modules are more complex, with internal state and multi-step pipelines.

| Module | Public funcs | Notes |
|--------|-------------|-------|
| `had_cell.py` | 15 | Streamfunction, cell edge/strength detection |
| `grad_bal.py` | 21 | Gradient wind balance |
| `tropopause.py` | 6 | Tropopause detection algorithms |
| `stats.py` | 32 | Statistical tools (largest module) |
| `eofs.py` | 1 | EOF analysis |
| `bootstrap.py` | 5 | Bootstrap statistics |
| `budget_adj.py` | 2 | Already tested — expand edge cases |

### Testing strategy

- **Synthetic-data tests**: Construct idealized fields with known answers. For example, a streamfunction from a simple overturning circulation where cell edges are analytically known.
- **Regression tests**: For complex pipelines (e.g., `had_cell` edge detection), save expected outputs from a validated run and compare. Store small reference datasets in `puffins/tests/data/`.
- **Statistical tests**: For `stats.py` and `bootstrap.py`, test against `scipy.stats` and `statsmodels` on identical inputs. For bootstrap, verify that confidence intervals contain the true parameter at the expected rate over many trials (use a fixed random seed).
- **Sign conventions**: Streamfunction sign, circulation direction — these are documented in CLAUDE.md and should be enforced by tests.

---

## Phase 5: Theoretical Models

| Module | Public funcs | Notes |
|--------|-------------|-------|
| `held_hou_1980.py` | 8 | Held-Hou 1980 axisymmetric theory |
| `lindzen_hou_1988.py` | 16 | Lindzen-Hou 1988 off-equatorial forcing |
| `plumb_hou_1992.py` | 3 | Plumb-Hou 1992 |
| `fixed_temp_tropo.py` | 12 | Fixed tropopause temperature |
| `hides.py` | 3 | Hide's theorem |
| `kuo_el.py` | 5 | Kuo-Eliassen equation |
| `polar_amp.py` | 4 | Polar amplification |
| `therm_inert.py` | 6 | Thermal inertia |

### Testing strategy

- **Published-result tests**: These modules implement well-known papers. Compare against figures/tables from the original publications (with appropriate tolerances).
- **Analytical limits**: Held-Hou at zero thermal Rossby number, equinox symmetry, angular momentum conservation in the free troposphere.
- **Parameter sensitivity**: Verify that outputs change in the expected direction when parameters are perturbed (e.g., increasing rotation rate narrows Hadley cell extent).
- **Consistency between models**: Where models share common limits (e.g., Lindzen-Hou reducing to Held-Hou when forcing is on the equator), test that they agree.

---

## Phase 6: Visualization and Notebook Utilities

| Module | Public funcs | Notes |
|--------|-------------|-------|
| `plotting.py` | 25 | Matplotlib helpers |
| `nb_utils.py` | 21 | Notebook utilities, git helpers |

### Testing strategy

- **Smoke tests**: Call each plotting function with minimal valid input and verify it returns a matplotlib `Figure`/`Axes` without raising. Use `matplotlib.use("Agg")` backend.
- **Property tests**: Verify axis labels, tick formatting (e.g., sine-latitude ticks are at expected positions), colorbar presence.
- **Do not test visual appearance**: Avoid pixel-comparison or image-based tests — they are brittle and slow. Test structure, not rendering.
- **`nb_utils.py`**: Test pure-computation functions (coordinate creation, trig helpers) normally. For git-dependent functions, mock `git.Repo` or skip with `@pytest.mark.skipif`.

---

## Phase 7: Coverage Gating and Maintenance

### 7.1 Add coverage threshold

Once coverage reaches a meaningful level (~70%), add a minimum threshold to CI:

```toml
[tool.pytest.ini_options]
addopts = "--strict-markers -v --cov=puffins --cov-fail-under=70"
```

Ratchet upward over time as new tests are added.

### 7.2 Pre-commit hook (optional)

Add a lightweight pre-commit check that runs `pytest -x -q` (fail-fast, quiet) to catch regressions before push.

### 7.3 Ongoing discipline

- **All new code must have tests** (already stated in CLAUDE.md — enforce via review).
- **All bug fixes must include a regression test** that fails before the fix and passes after.
- **Mark slow tests** with `@pytest.mark.slow` so the fast suite stays under 30 seconds.

---

## Suggested Execution Order

```
Phase 1  ██████████  Infrastructure         (do first, prerequisite for all else)
Phase 2  ██████████  Pure functions          (quick wins, high coverage gain)
Phase 3  ████████    Coords & interpolation  (moderate complexity)
Phase 4  ██████████  Climate diagnostics     (highest complexity, most value)
Phase 5  ████████    Theoretical models      (domain-specific, needs paper references)
Phase 6  ████        Viz & notebook utils    (lowest priority, hardest to test well)
Phase 7  ██          Gating & maintenance    (ongoing, after ~70% coverage)
```

Phases 2 and 3 can proceed in parallel. Phase 4 is the core of the project and where test coverage matters most for correctness. Phase 6 has diminishing returns — focus on smoke tests, not exhaustive coverage.

---

## Estimated Scale

| Phase | Approx. test files | Approx. test functions |
|-------|-------------------|----------------------|
| 1 | 1 (conftest.py) | 0 (fixtures only) |
| 2 | 7 | 80–120 |
| 3 | 4 | 50–70 |
| 4 | 6 | 100–150 |
| 5 | 7 | 60–90 |
| 6 | 2 | 30–50 |
| **Total** | **~27 files** | **~320–480 tests** |

This roughly matches the ~328 public functions — the goal is at least one test per public function, with additional tests for edge cases and integration points.
