# Roadmap 001: Modernize Repository Standards

| Field | Value |
|-------|-------|
| **Status** | In Progress |
| **Created** | 2026-03-16 |
| **Last updated** | 2026-03-16 |
| **Author** | Spencer A. Hill |

## Objective

Bring puffins up to modern Python open-source project standards: reproducible
builds, automated CI/CD, enforced code quality, comprehensive tests, and full
type coverage.

## Success Criteria

- All code formatted and linted by ruff with no suppressed rules
- ≥80% test coverage for all modules (except `plotting.py` and `nb_utils.py`)
- Type annotations on every public function, enforced by mypy in CI
- Automated publish-to-PyPI on tagged releases
- Generated API documentation hosted online

---

## Phase 1: Build & Tooling Foundation — DONE

Modernize the build system and local developer tooling.

- [x] Modern `pyproject.toml` with `setuptools-scm` for git-based versioning
- [x] `uv` for dependency management with `uv.lock` for reproducible environments
- [x] Ruff for linting and formatting (replaces black, isort, flake8)
- [x] Pre-commit hooks (ruff, trailing whitespace, EOF fixer)
- [x] `.editorconfig` for consistent editor settings
- [x] Apply ruff format to entire codebase and fix safe lint errors

## Phase 2: CI/CD & Automation — DONE

Automate quality checks and releases.

- [x] GitHub Actions CI: lint + test matrix (Python 3.10–3.13)
- [x] GitHub Actions publish workflow (PyPI trusted publishing)
- [x] Dependabot for monthly dependency and Actions updates
- [x] `windspharm` moved to optional `[fortran]` extra (requires conda)

## Phase 3: README & Install Docs Refresh

Update documentation to reflect the new tooling.

- [ ] Update README badges (Python 3.10+, add CI status badge)
- [ ] Add `uv` installation instructions alongside pip
- [ ] Note windspharm's move to optional `[fortran]` extra
- [ ] Add "Contributing" section covering dev setup, pre-commit, and CI

## Phase 4: Type Hints

Add type annotations incrementally, module by module. Type hints improve
editor support, catch bugs early, and serve as machine-readable documentation
— especially valuable in a physics codebase where function signatures encode
units and coordinate conventions.

Suggested order (simplest → most complex):

- [ ] `constants.py`, `names.py`, `longitude.py` — simple values and utilities
- [ ] `calculus.py`, `interp.py`, `num_solver.py` — core numerical utilities
- [ ] `dynamics.py`, `thermodynamics.py`, `vert_coords.py` — physics modules
- [ ] `had_cell.py`, `grad_bal.py`, `eq_area.py` — climate dynamics
- [ ] `held_hou_1980.py`, `lindzen_hou_1988.py`, `plumb_hou_1992.py` — theoretical models
- [ ] Remaining modules (`stats.py`, `bootstrap.py`, `eofs.py`, `plotting.py`, etc.)
- [ ] Add mypy to CI (initially non-blocking, then required)

## Phase 5: Test Coverage

Current state: 1 test file (skipped without windspharm), effectively 0%
coverage. Tests are essential for safe refactoring and verifying that type
annotations don't change behavior.

Suggested priority order (core utilities first, then physics, then models):

- [ ] `calculus.py` — differentiation, integration, surface area
- [ ] `dynamics.py` — Coriolis, angular momentum, vorticity, Rossby number
- [ ] `constants.py` — verify constant values against references
- [ ] `names.py` — trivial but good for completeness
- [ ] `interp.py` — interpolation utilities
- [ ] `thermodynamics.py` — thermodynamic calculations
- [ ] `had_cell.py` — streamfunction, cell strength/extent
- [ ] `eq_area.py` — equal-area coordinate transforms
- [ ] `grad_bal.py` — gradient wind balance
- [ ] `held_hou_1980.py`, `lindzen_hou_1988.py`, `plumb_hou_1992.py` — theoretical models
- [ ] `longitude.py`, `dates.py`, `num_solver.py` — smaller utility modules
- [ ] `vert_coords.py`, `tropopause.py`, `lcl.py` — vertical coord / tropopause
- [ ] `stats.py`, `bootstrap.py`, `eofs.py` — statistical tools
- [ ] `plotting.py` — smoke tests for plot generation (image comparison optional)
- [ ] `nb_utils.py` — notebook utilities (where testable)
- [ ] `budget_adj.py` — already has tests; improve coverage

Target: ≥80% line coverage for all modules except `plotting.py` and `nb_utils.py`.

## Phase 6: Tighten Lint Rules

Incrementally remove items from `ruff.lint.ignore` in `pyproject.toml`.
Each rule removal should be its own PR with any necessary code fixes.

- [ ] `F841` — remove unused local variables
- [ ] `F811` — fix redefined functions (e.g. duplicate `moist_enthalpy`)
- [ ] `F821` — resolve undefined name references
- [ ] `F401` — audit unused imports vs intentional re-exports
- [ ] `B904` — add `raise ... from err` where appropriate
- [ ] `B905` — add `strict=` to `zip()` calls
- [ ] `E501` — reduce remaining long lines
- [ ] `SIM108` — adopt ternary expressions where they improve readability

## Phase 7: Documentation Site

Set up generated API documentation.

- [ ] Choose framework (Sphinx with autodoc, or MkDocs with mkdocstrings)
- [ ] Scaffold docs structure mirroring module groups
- [ ] Add docstring coverage check to CI
- [ ] Host on GitHub Pages or Read the Docs
- [ ] Add usage guides / tutorials for key workflows

## Phase 8: Packaging & Release

Prepare for proper PyPI releases.

- [ ] Create first git tag (e.g. `v0.1.0`) so setuptools-scm produces clean versions
- [ ] Configure PyPI trusted publishing environment in GitHub repo settings
- [ ] Test publish workflow with a release
- [ ] Add `CHANGELOG.md` or adopt automated changelog generation

---

## Notes

- **Phases 3–6 can be worked in parallel** — they are largely independent.
- **Phase 4 (type hints) and Phase 5 (tests) pair well** — add types as you
  write tests for each module, or type a module first to catch signature issues
  before writing tests against the annotated interface.
- Each rule removal in Phase 6 should be its own PR to keep diffs reviewable.
- This is a living document. Update it as phases are completed.
