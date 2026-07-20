# Roadmap 001: Modernize Repository Standards

| Field | Value |
|-------|-------|
| **Status** | In Progress |
| **Created** | 2026-03-16 |
| **Last updated** | 2026-07-19 |
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

**Now tracked in detail by [Roadmap 003 — Type Hints](003-type-hints.md).**
The per-module checklist lives there to avoid duplication.

Status (2026-07-19): **23 of 30 modules fully annotated** under the
`pyproject.toml` mypy strict overrides, with **0 source-file mypy errors** at
the mypy version CI pins (the remaining 62 errors are all in test files).
mypy runs in CI but is still non-blocking (`continue-on-error: true`).
Remaining: `kuo_el`, `held_hou_1980`, `lindzen_hou_1988`, `plumb_hou_1992`,
`fixed_temp_tropo`, `plotting`, `nb_utils`; then promote the mypy check to
blocking and enable global strict mode. Promoting it to blocking also requires
fixing the pre-existing `overload-cannot-match` ordering in `dates.py` and
`vert_coords.py`, which newer mypy flags; see
[Roadmap 003](003-type-hints.md).

## Phase 5: Test Coverage

**Now tracked in detail by [Roadmap 004 — Testing Overhaul](004-testing-overhaul.md).**
The per-module checklist and testing strategies live there.

Status (2026-07-19): from effectively 0% to **781 tests passing across 23 test
files, 88% total line coverage** (1 skipped, 10 xfailed). 22 modules meet the
≥80% target. The remaining gap is the theoretical-model cluster
(`kuo_el` 19%, `lindzen_hou_1988` 36%, `fixed_temp_tropo` 39%,
`held_hou_1980` 54%, `plumb_hou_1992` 78%), plus `budget_adj` at 16%.
`budget_adj` has a full test file, but its numerical tests need
`windspharm`/`pyspharm`, which will not install in CI, so they skip there and
its measured coverage understates what is actually tested. A CI coverage gate
(`--cov-fail-under`) is not yet enabled.

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

## Phase 7: Documentation Site (mostly done)

Set up generated API documentation. Delivered in PR #59; detailed planning
lives in [Roadmap 002](002-documentation.md), which supersedes this phase.

- [x] Choose framework (Sphinx with autodoc, napoleon, intersphinx, mathjax,
      viewcode; `pydata-sphinx-theme`)
- [x] Scaffold docs structure mirroring module groups (`docs/api/`, one stub
      per module, grouped in `docs/api/index.rst`)
- [x] Host on Read the Docs: live at
      [puffins.readthedocs.io](https://puffins.readthedocs.io), configured by
      `.readthedocs.yaml`. The CI `docs` job also builds with `-W`, so broken
      cross-references fail the build.
- [ ] Add docstring coverage check to CI (no `interrogate`/`pydocstyle` gate
      yet; see Roadmap 002 Phase 5.3)
- [ ] Add usage guides / tutorials for key workflows (only `installation.rst`
      exists; see Roadmap 002 Phase 3)

## Phase 8: Packaging & Release (mostly done)

Prepare for proper PyPI releases.

- [x] Create first git tag so setuptools-scm produces clean versions
- [x] Configure PyPI trusted publishing environment in GitHub repo settings
- [x] Test publish workflow with a release: **`puffins` 0.2.0 is live on
      [PyPI](https://pypi.org/project/puffins/)**, published 2026-07-18
- [ ] Add `CHANGELOG.md` or adopt automated changelog generation

### Why the first release is 0.2.0, not 0.1.0

A `puffins` project already existed on PyPI, with `0.1` and `0.1.1` published
in March 2021. PyPI refuses new file uploads to a release older than 14 days,
so tagging `v0.1.0` produced a `400 Bad Request` and the publish workflow
failed. Re-tagging as `v0.2.0` published cleanly. The practical consequence is
that the PyPI version history jumps from a 2021-era `0.1.1` straight to
`0.2.0`; there is no `0.1.0` and there never can be.

---

## Notes

- **Phases 3–6 can be worked in parallel** — they are largely independent.
- **Phase 4 (type hints) and Phase 5 (tests) pair well** — add types as you
  write tests for each module, or type a module first to catch signature issues
  before writing tests against the annotated interface.
- Each rule removal in Phase 6 should be its own PR to keep diffs reviewable.
- This is a living document. Update it as phases are completed.
