# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Repository modernisation. **No library code changed in this release**: the package builds
from the same twelve source modules as before, and declares the same dependency contract
as the published 3.0.6 release, verified in CI by `scripts/compare_with_published.py`.

### Added

- `pyproject.toml` replaces `setup.py`, `setup.cfg` and `MANIFEST.in`, with `hatchling`
  as the build backend and `hatch-vcs` deriving the version from git tags.
- `uv` for dependency management, with a committed `uv.lock`.
- A `pytest` suite covering the parameter-space linear algebra, the normalization
  routines, the geometry of the sampled subspaces, the metrics, and the contents of the
  built distribution.
- GitHub Actions: `ci.yml` runs ruff, mypy and the test suite on every pull request to
  `master`; `release.yml` builds and publishes through PyPI Trusted Publishing.
- `ruff` for linting and formatting, `mypy` for type checking, and `pre-commit` hooks
  running the same checks locally.
- Dependabot for GitHub Actions updates.
- `.gitattributes` normalising line endings to LF.

### Changed

- Moved the package to a `src/` layout.
- Replaced `.gitignore` with GitHub's Python template. The previous file excluded
  `tests/`, which is why the repository had no test suite.
- Renamed `LICENCE.txt` to `LICENSE.txt`, which is the name `MANIFEST.in` and `setup.cfg`
  had been pointing at all along — meaning no published release has ever contained the
  licence text. It is now included in both the sdist and the wheel.

### Fixed

- The built wheel no longer contains roughly sixty stale modules left over from the 1.x
  and 2.x refactors. The published 3.0.6 wheel shipped them because it was built over a
  dirty `build/` directory.
- The distribution no longer ships the `tests/` package. `find_packages(exclude='tests')`
  was passed a string rather than a list, so setuptools iterated it character by character
  and excluded nothing.

### Known issues

These are pre-existing defects, recorded here and pinned by `xfail` tests rather than
fixed, so that this release stays a pure infrastructure change. Each is scheduled for the
follow-up PR.

- `torch` is a hard runtime dependency but is not declared, so a clean
  `pip install loss-landscapes` produces a package that cannot be imported.
- `requires-python` claims `>=3.5`; no Python below 3.10 can install a usable torch.
- `ModelWrapper.parameters()` and `.named_parameters()` chain a list of generators rather
  than the generators themselves, so they yield generator objects. `LossGradient` depends
  on this and cannot work.
- `model_normalize_` recomputes the vector's norm inside its loop, so on a model with more
  than one layer the result does not carry the reference point's norm.
- `random_plane` orthogonalises its two directions and only then normalises them, which
  rotates them apart again. With normalisation enabled — the default — the sampled plane
  is skewed.
- The `*_norm` methods omit the absolute value, so odd norm orders return a signed sum
  rather than a norm. The default order of 2 is unaffected.
- `loss_landscapes.contrib` has been unimportable since the 2019-07-17 refactor. It
  imports `model_interface.model_interface`, which no longer exists, and calls the removed
  `get_parameter_tensor`/`set_parameter_tensor` API.
- `LossPerturbations` accepts an `alpha` argument that it never uses.

## [3.0.6] - 2019-08-30

Last release published to PyPI. See the git history for earlier versions; no changelog was
kept before this file was introduced.

[Unreleased]: https://github.com/marcellodebernardi/loss-landscapes/compare/v3.0.6...HEAD
[3.0.6]: https://github.com/marcellodebernardi/loss-landscapes/releases/tag/v3.0.6
