# Contributing

## Development setup

The project uses [uv](https://docs.astral.sh/uv/). Install it, then:

```bash
uv sync --all-groups
uv run pre-commit install
```

That creates a virtual environment in `.venv`, installs the library in editable mode along
with the test and lint tooling, and wires up the pre-commit hooks.

`uv.lock` is committed, so everyone and CI resolve to the same versions. Do not edit it by
hand — run `uv lock` and commit the result.

## Everyday commands

```bash
uv run pytest                  # run the test suite
uv run ruff check .            # lint
uv run ruff format .           # format
uv run mypy                    # type-check the package
uv run pre-commit run -a       # everything the hooks would run
```

## Tests

`pytest` picks up everything under `tests/`. Tests must run on CPU in seconds: the
fixtures use a two-layer MLP — 23 parameters across four tensors — which is enough to
exercise the layer-wise and filter-wise code paths.

`xfail_strict` is on, so an `xfail` that starts passing fails the build. This is
deliberate: the suite currently pins a number of known defects with strict `xfail`
markers, so fixing one of them makes the corresponding test fail until the marker is
removed in the same change. When you fix a bug, delete its marker.

Every `xfail` carries a `reason` explaining the defect. Keep that up: those markers are the
project's bug tracker as much as the issue list is.

## Packaging

Version numbers come from git tags via `hatch-vcs` — there is no version string to edit.
Tag a release as `vX.Y.Z` and the build picks it up; between tags the version is a `.devN`
suffix on the next patch.

The packaging rewrite was intended to leave the published dependency contract untouched.
CI enforces that:

```bash
uv run python scripts/compare_with_published.py --version 3.0.6
```

builds the current tree under the published version number and compares the resulting
`Name`, `Version`, `Requires-Python` and `Requires-Dist` against the wheel on PyPI. If you
are intentionally changing what the package depends on, update the `--version` argument in
[ci.yml](.github/workflows/ci.yml) to the release you now want to be compared against, or
drop the step.

## Releasing

Releases go out through the `Release` workflow, which uses PyPI Trusted Publishing rather
than a stored API token. It is currently `workflow_dispatch` only and defaults to TestPyPI;
see the comment at the top of [release.yml](.github/workflows/release.yml) for what needs to
change before it publishes to PyPI on a tag.

## Commit and PR conventions

Commit messages use a `type: summary` prefix (`feat`, `fix`, `build`, `ci`, `test`, `docs`,
`style`, `chore`).

Pure formatting commits are recorded in `.git-blame-ignore-revs`. If you land one, add its
hash there so `git blame` keeps pointing at the commit that last changed the code's
meaning. Configure git to use the file once, locally:

```bash
git config blame.ignoreRevsFile .git-blame-ignore-revs
```
