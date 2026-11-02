# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [4.3.0] - 2026-11-02

### Added

- `ignore_1d` parameter for `random_line` and `random_plane`, and for
  `ModelParameters.model_normalize_`, `layer_normalize_` and `filter_normalize_`. When true,
  parameters with at most one dimension, such as biases and normalization layers' weights, are
  left unperturbed: their direction is set to zero, as by the reference implementation's
  `--xignore biasbn`. It defaults to true for `normalization="filter"`, matching the setup of
  Li et al. (2018), and to false otherwise.

### Fixed

- Filter normalization no longer normalizes each element of a 1-D tensor separately, which kept
  only the sign of the random direction and set each element's magnitude to that of the
  corresponding parameter. With `ignore_1d=False`, such tensors are now normalized as a whole, as
  layer normalization does. Filter normalization of 0-D parameters no longer raises `TypeError`.
- Layer and filter normalization leave a block that is zero in the direction at zero, instead of
  raising `ZeroDivisionError`.
- `random_plane` resamples its second direction when normalization maps it onto the first, which
  happens often when every perturbed filter is a single parameter, instead of raising
  `ZeroDivisionError`. A direction counts as collapsed only when what projection leaves is too
  small for the perturbed parameters' dtype to resolve, under 10 machine epsilons of its norm,
  so planes through models whose filter norms differ by orders of magnitude are kept where the
  dtype allows it.
- `ModelParameters.orthogonalize_`, used by `orthogonal_to` and `random_plane`, computes its
  subtraction in float64 (float32 on MPS) before rounding to each tensor's dtype, so nearly
  parallel vectors keep an accurate, orthogonal remainder in low precision.
- `random_line` and `random_plane` raise a descriptive `ValueError` when normalization leaves
  nothing to perturb, or, for `random_plane`, fewer than two dimensions, instead of
  `ZeroDivisionError`.

### Changed

- With the default `normalization="filter"`, `random_line` and `random_plane` no longer perturb
  1-D parameters, so their results differ from 4.2.0. Layer, model and no normalization are
  unchanged unless `ignore_1d=True` is passed.

## [4.2.0] - 2026-11-02

### Removed

- `loss_landscapes.metrics.ExpectedReturnMetric` and the `rl_metrics` module. It had not worked
  since the 2019 refactor: it called the model wrapper it was given, which is not callable, and
  it used the pre-0.26 Gym API, so it also failed with Gymnasium. Reinforcement learning remains
  possible with a custom `Metric` and `GeneralModelWrapper`, and built-in support is planned for
  a later release.

## [4.1.0] - 2026-11-02

### Added

- GPU support. Everything now runs on the device each model parameter is on, including CUDA and
  Apple MPS, and results are returned in host memory as before. Previously `random_line`,
  `random_plane` and `LossPerturbations` created their random directions on the CPU and failed
  with a device mismatch, and `LossGradient` failed converting device tensors to NumPy.
  Metric inputs and targets must be on the model's device.

### Fixed

- `rand_u_like` and `rand_n_like` now give each tensor its example tensor's own dtype, rather
  than the dtype of the first one.
- Parameter norms and dot products on MPS, which has no float64, are accumulated in float32.

## [4.0.8] - 2026-11-02

### Fixed

- Parameter norms and dot products are now accumulated in float64, whatever the parameters'
  dtype. Previously they were summed in the parameters' own precision, so for float16 models
  with tensors of more than about 65,000 elements the sum overflowed. `random_line` and
  `random_plane` then raised `ZeroDivisionError` or silently left the overflowing tensors
  unperturbed, and `LossPerturbations` returned only zeros. Results for float32 models may
  differ in the last few digits.

## [4.0.7] - 2026-11-02

### Fixed

- `LossPerturbations` now implements the random-perturbation method of Ahmed et al. (2019).
  For each direction, drawn uniformly from the unit sphere, it evaluates the loss at
  `theta + alpha * d` and `theta - alpha * d` and returns the pair of changes from the loss at
  `theta`, as an array of shape `(n_directions, 2)`. Their sum estimates the curvature along `d`
  and their difference the slope. Previously `alpha` was ignored, each step was an unscaled,
  all-positive uniform vector of length roughly `sqrt(n_parameters / 3)`, and only the positive
  side was evaluated, so the results could not be used to estimate curvature. The model's
  parameters are now restored even if evaluating the loss raises. The docstring now credits the
  paper's first author.

### Known issues

- Parameter norms are computed in the parameters' own precision, so normalisation, and hence
  `random_line`, `random_plane` and `LossPerturbations`, fails for float16 models with tensors of
  more than about 65,000 elements. This predates 4.0.7 and is pinned by a strict `xfail` test.

## [4.0.6] - 2026-11-02

### Fixed

- `random_line`, `random_plane` and `orthogonal_to` now sample directions from a standard
  Gaussian, as in Li et al. (2018), instead of uniformly from [0, 1). Uniform directions had
  only positive entries, so every direction moved all parameters the same way and two
  independently sampled directions were strongly correlated. Results from `random_line` and
  `random_plane` will differ from 4.0.5. `LossPerturbations` still samples uniformly.

## [4.0.5] - 2026-11-02

### Fixed

- `LossGradient` now returns the gradient of the loss with respect to the model parameters as
  a flat array in parameter order. Previously it always raised, because `wrap_model()` disables
  gradient tracking. It now enables tracking only for its own evaluation and restores the
  previous state afterwards, and gives a zero gradient for parameters the loss does not use.
  It raises `ValueError` if the loss uses none of the wrapped parameters, which indicates a
  misconfigured wrapper, and supports sparse gradients such as those of
  `torch.nn.Embedding(sparse=True)`.
- `ModelWrapper.parameters()` and `named_parameters()` now yield the parameters (or
  `(name, parameter)` pairs) of every wrapped module. Previously they yielded one generator
  object per module.

## [4.0.4] - 2026-11-02

### Fixed

- Model-, layer- and filter-wise norms now take absolute parameter values before raising
  them to the requested power. L1 and other positive odd-order norms no longer suffer
  cancellation, negative results or domain errors for negative parameters. Normalisation
  using these orders now uses the corrected magnitudes; default L2 results are unchanged.

## [4.0.3] - 2026-11-02

### Removed

- `loss_landscapes.contrib`, which could not be imported since the 2019 refactor and was
  mostly unimplemented stubs.

## [4.0.2] - 2026-10-31

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
- Layer- and filter-wise normalisation set a block whose reference norm is zero (such as a
  zero-initialised bias) to zero without dividing by its own norm. Filter normalisation
  previously raised `ZeroDivisionError` for zero biases, and repeated normalisation in
  `random_plane` did so for zero layers too.
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

[Unreleased]: https://github.com/marcellodebernardi/loss-landscapes/compare/v4.3.0...HEAD
[4.3.0]: https://github.com/marcellodebernardi/loss-landscapes/compare/v4.2.0...v4.3.0
[4.2.0]: https://github.com/marcellodebernardi/loss-landscapes/compare/v4.1.0...v4.2.0
[4.1.0]: https://github.com/marcellodebernardi/loss-landscapes/compare/v4.0.8...v4.1.0
[4.0.8]: https://github.com/marcellodebernardi/loss-landscapes/compare/v4.0.7...v4.0.8
[4.0.7]: https://github.com/marcellodebernardi/loss-landscapes/compare/v4.0.6...v4.0.7
[4.0.6]: https://github.com/marcellodebernardi/loss-landscapes/compare/v4.0.5...v4.0.6
[4.0.5]: https://github.com/marcellodebernardi/loss-landscapes/compare/v4.0.4...v4.0.5
[4.0.4]: https://github.com/marcellodebernardi/loss-landscapes/compare/v4.0.3...v4.0.4
[4.0.3]: https://github.com/marcellodebernardi/loss-landscapes/compare/v4.0.2...v4.0.3
[4.0.2]: https://github.com/marcellodebernardi/loss-landscapes/compare/v4.0.1...v4.0.2
[4.0.1]: https://github.com/marcellodebernardi/loss-landscapes/compare/v4.0.0...v4.0.1
[4.0.0]: https://github.com/marcellodebernardi/loss-landscapes/compare/v3.0.6...v4.0.0
[3.0.6]: https://github.com/marcellodebernardi/loss-landscapes/releases/tag/v3.0.6
