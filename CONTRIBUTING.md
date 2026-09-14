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

While the package still declares the dependency contract of 3.0.6, CI checks that it has
not drifted:

```bash
uv run python scripts/compare_with_published.py --version 3.0.6
```

This builds the current tree under the published version number and compares `Name`,
`Version`, `Requires-Python` and `Requires-Dist` against the wheel on PyPI. Once the
contract changes deliberately, delete the script and its step in
[ci.yml](.github/workflows/ci.yml).

## Releasing

Push a `vX.Y.Z` tag. The `Release` workflow builds it and publishes through PyPI Trusted
Publishing, gated on the `pypi` environment, so no API token is stored in the repository.
To rehearse, tag a pre-release such as `v3.1.0rc1`.

## Commit and PR conventions

Commit messages use a `type: summary` prefix (`feat`, `fix`, `build`, `ci`, `test`, `docs`,
`style`, `chore`).

Pure formatting commits are recorded in `.git-blame-ignore-revs`. If you land one, add its
hash there so `git blame` keeps pointing at the commit that last changed the code's
meaning. Configure git to use the file once, locally:

```bash
git config blame.ignoreRevsFile .git-blame-ignore-revs
```
