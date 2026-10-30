# Contributing

## Development setup

The project uses [uv](https://docs.astral.sh/uv/). Install it, then:

```bash
uv sync --all-groups
uv run pre-commit install
```

That creates a virtual environment in `.venv`, installs the library in editable mode along
with the test and lint tooling, and wires up the pre-commit hooks. Run
`uv run pre-commit install` separately in each checkout; hooks are not installed by cloning
the repository or syncing dependencies.

`uv.lock` pins the development and driver tools. Compatibility test environments resolve
their own dependencies independently. Do not edit the lockfile by hand — run `uv lock`
and commit the result.

## Everyday commands

```bash
uv run pytest                  # run the test suite
uv run ruff check .            # lint
uv run ruff format .           # format
uv run mypy                    # type-check the package
uv run pre-commit run -a       # everything the hooks would run
```

## Spec-driven development

This project uses Spec Kit for substantive features, API changes, and numerical
redesigns. Read [.specify/memory/constitution.md](.specify/memory/constitution.md) before
planning or implementing changes.

Use the installed Spec Kit commands or skills to define the specification,
clarify requirements, create a plan, generate tasks, analyse their consistency,
and implement the change. Keep the artifacts under `specs/` aligned with the
implementation. Small fixes and editorial changes may use an issue or PR
description instead.

Codex and Claude integrations are available and share the same constitution
and specification artifacts.

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

## Compatibility checks

The supported matrix is PyTorch 2.15 on standard, GIL-enabled CPython 3.11–3.15.
Linux x86-64 CPU is the verified platform; native macOS/Windows runs provide local feedback
where upstream wheels exist. Alternative interpreters, free-threaded builds and GPU execution
are outside this support claim. Existing numerical defects remain tracked by strict xfails.

After development setup, provision the latest stable interpreter patches in a test-only directory:

```bash
uvx uv@latest python install --install-dir .tox/python --upgrade --no-bin --no-registry 3.11 3.12 3.13 3.14 3.15
uv run --no-sync tox list -d --no-desc
uv run --no-sync tox run -r -e py312-torch215 --result-json .tox/result.json
uv run --no-sync tox run -r --result-json .tox/result.json
```

For one combination, provision its Python minor and 3.12 (the shared build interpreter).
`uvx uv@latest` resolves current stable uv independently of the driver lock; older
interpreter catalogs may select stale patches or prereleases. No shell
exports are required. These commands leave `.venv`, `.python-version` and `uv.lock` alone;
missing interpreters fail rather than silently reducing coverage. Downloads require network
access; allow disk space for separate PyTorch installations. macOS requires Apple Silicon
and a version supported by upstream wheels; Linux requires manylinux_2_28 compatibility.

Tox recreates environments, installs a wheel, and resolves the latest compatible stable dependencies
without using `uv.lock`. CPU backend selection is internal to test installation; library users
can install their own supported PyTorch build. Refresh interpreters with the provisioning command
before a fresh matrix run. New Python/PyTorch minor versions require an explicit support-policy change.

Pytest's header reports exact Python/PyTorch/NumPy versions, platform and import location. Tox's
`.tox/ENV/log/` contains installed dependency listings and setup/test logs; `junit.xml` contains
pytest results. `--result-json` retains standard tox orchestration results. Installation failures
may have no JUnit report. Ordinary skips, unexpected passes and failures prevent compatibility
success; known strict xfails are reported separately. Missing or cancelled checks are not passes.

To test a particular artifact, build with `uv build`, then pass the actual filename:

```text
uv run --no-sync tox run -r -e py312-torch215 --installpkg PATH_TO_WHEEL
uv run --no-sync tox run -r -e sdist --installpkg PATH_TO_SDIST
```

The non-default `sdist` environment installs the actual source archive on Python 3.12 and runs
the full suite. CI builds once and supplies the same wheel to every matrix job.

For diagnostic reproduction, retain the candidate wheel and hash, exact Python patch and installed
dependency listing from a failed run. On the same platform, provision that exact interpreter and
create a constraints file from the listing, excluding the candidate's local artifact reference.
Then use tox's fully qualified native TOML overrides (substitute real values):

```text
uv run --no-sync tox run -r -e diagnostic --installpkg PATH_TO_ORIGINAL_WHEEL -x env_list=diagnostic -x tool.tox.env.diagnostic.base_python=PATH_TO_EXACT_PYTHON -x tool.tox.env_run_base.constraints=PATH_TO_CONSTRAINTS --result-json .tox/replay-result.json
```

The temporary `diagnostic` name avoids a tox-uv 1.37.2 limitation that ignores exact
interpreter patches/paths for Python-named environments. It does not change the default
matrix. Supply an absolute interpreter path, not just a patch version.

Constraints apply to both test and runtime dependencies. Check the recorded versions match;
unavailable historical downloads must be reported rather than replaced silently. Linux CPU pins
are not portable to macOS. This explicit diagnostic command opts out of latest-patch selection.
See [the feature quickstart](specs/001-supported-versions-update/quickstart.md) for validation evidence.

### Required GitHub checks

The maintainer configures these check names in GitHub rulesets/branch protection:

- `Compatibility (py311-torch215)`
- `Compatibility (py312-torch215)`
- `Compatibility (py313-torch215)`
- `Compatibility (py314-torch215)`
- `Compatibility (py315-torch215)`
- `Build distributions`
- `Lint and type-check`

All must succeed for the candidate revision before merge/release acceptance. Repository CI publishes
these checks; GitHub settings enforce merge requirements. No repository script manages those settings.

## Packaging

Version numbers come from git tags via `hatch-vcs` — there is no version string to edit.
Tag a release as `vX.Y.Z` and the build picks it up; between tags the version is a `.devN`
suffix on the next patch.

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
