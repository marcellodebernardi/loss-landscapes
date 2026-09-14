# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Repository modernisation. **No library code changed**: the package builds from the same
twelve source modules as before, and declares the same dependency contract as the
published 3.0.6 release.

### Added

- `pyproject.toml` with `hatchling` and `hatch-vcs`, replacing `setup.py`, `setup.cfg` and
  `MANIFEST.in`. Versions now come from git tags.
- `uv` for dependency management, with a committed `uv.lock`.
- A `pytest` suite, run in CI on every pull request across Python 3.10 to 3.13.
- `ruff`, `mypy`, `gitleaks` and `pre-commit`; GitHub Actions for CI and for releases via
  PyPI Trusted Publishing; Dependabot for action updates.

### Changed

- Moved the package to a `src/` layout.
- Replaced `.gitignore` with GitHub's Python template. The previous file excluded `tests/`.
- Renamed `LICENCE.txt` to `LICENSE`, the name `MANIFEST.in` and `setup.cfg` had been
  pointing at all along — so no release has ever contained the licence text.

### Fixed

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

Last release published to PyPI. No changelog was kept before this file was introduced; see
the git history for earlier versions.

[Unreleased]: https://github.com/marcellodebernardi/loss-landscapes/compare/v3.0.6...HEAD
[3.0.6]: https://github.com/marcellodebernardi/loss-landscapes/releases/tag/v3.0.6
