# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `random_plane` directions are now orthogonal under layer- and filter-wise normalisation.
  Previously the second direction was orthogonalised before normalisation, which rescales
  layers or filters by different factors and does not preserve angles, so the grid was
  sheared relative to the perpendicular axes it is plotted on. The second direction is now
  made orthogonal after normalisation by alternating Gram-Schmidt projection and
  renormalisation (at most 20 iterations, typically 1-3), ending on a projection so that
  orthogonality always holds. Its layer/filter norms match the start point's to within a
  small tolerance, or only approximately for models with many single-parameter filters.
  Results with `normalization="filter"` or `"layer"` will differ from 4.0.1.
- `random_plane` raises `ValueError` for a model with fewer than two parameters instead of
  failing with an unrelated error, and its docstring now describes the 2-D return value.

### Added

- `ModelParameters.orthogonalize_`, an in-place Gram-Schmidt step, now also used by
  `orthogonal_to`.

## [4.0.1] - 2026-10-31

### Fixed

- Model-wise normalisation now calculates one scale factor before modifying any parameter
  tensor, preserving direction and matching the reference norm independently of tensor order
  or partitioning. Both norms use the requested `order`.
- Model-wise normalisation no longer rotates the directions used by `random_line` and
  `random_plane`. Results with `normalization="model"` may differ from 4.0.0 because the
  previous behaviour was incorrect. Layer/filter normalisation and other known numerical
  defects are unchanged.

## [4.0.0] - 2026-10-30

Repository modernisation. **No library code changed**: the package builds from the same
twelve source modules as before. The dependency contract deliberately changes from
the published 3.0.6 release: Python 3.11–3.15 and PyTorch 2.15 are now required. Users
on older versions must upgrade their environment before installing this release.

### Added

- `pyproject.toml` with `hatchling` and `hatch-vcs`, replacing `setup.py`, `setup.cfg` and
  `MANIFEST.in`. Versions now come from git tags.
- `uv` for dependency management, with a committed `uv.lock`.
- A `pytest` suite and five isolated tox wheel-install environments for CPython 3.11–3.15
  with PyTorch 2.15, plus source-archive installation verification on Python 3.12. Linux
  x86-64 CPU CI runs all five combinations on pull requests and primary-branch pushes.
- Standard tox logs/results and pytest JUnit reports, runtime/import-origin validation,
  rejection of ordinary skips, and documented exact-version diagnostic reproduction.
- `ruff`, `mypy`, `gitleaks` and `pre-commit`; GitHub Actions for CI and for releases via
  PyPI Trusted Publishing; Dependabot for action updates.

### Changed

- Runtime metadata now requires Python `>=3.11,<3.16` and PyTorch `>=2.15,<2.16`;
  compatibility runs refresh stable patches independently of the development lock.
  GPU execution and additional platform guarantees remain follow-up work.

- Moved the package to a `src/` layout.
- Replaced `.gitignore` with GitHub's Python template. The previous file excluded `tests/`.
- Renamed `LICENCE.txt` to `LICENSE`, the name `MANIFEST.in` and `setup.cfg` had been
  pointing at all along — so no release has ever contained the licence text.

### Fixed

- PyTorch is now a declared runtime dependency, so a clean installation can import
  the library without a separate manual dependency install. The obsolete 3.0.6 metadata guard
  has been removed. Known numerical defects remain unchanged and explicitly tracked.

- The wheel no longer contains ~60 stale modules from the 1.x and 2.x refactors, which the
  3.0.6 wheel shipped because it was built over a dirty `build/` directory.
- The sdist no longer ships test files that were never committed to git.
  `find_packages(exclude='tests')` was passed a string rather than a list, so setuptools
  iterated it character by character and excluded nothing. The sdist still contains the
  committed `tests/` directory, which is conventional; the wheel contains neither.

Several long-standing library defects were found while writing the tests. They are pinned
by strict `xfail` markers rather than fixed, so that this release stays an infrastructure
change. Each marker carries the diagnosis and a `TODO:`; grep the test suite for them.

## [3.0.6] - 2019-08-30

No changelog was kept before this file was introduced; see
the git history for earlier versions.

[Unreleased]: https://github.com/marcellodebernardi/loss-landscapes/compare/v4.0.1...HEAD
[4.0.1]: https://github.com/marcellodebernardi/loss-landscapes/compare/v4.0.0...v4.0.1
[4.0.0]: https://github.com/marcellodebernardi/loss-landscapes/compare/v3.0.6...v4.0.0
[3.0.6]: https://github.com/marcellodebernardi/loss-landscapes/releases/tag/v3.0.6
