# Contract: Compatibility Workflow

This extends contributor tooling and package installation contracts. No new public library API is added.
Commands below describe the implemented contributor workflow.

## Package installation

- Python: `>=3.11,<3.16`; PyTorch: `>=2.15,<2.16`; NumPy remains declared.
- Unsupported interpreter/PyTorch combinations cannot satisfy normal dependency resolution.
- The wheel and sdist expose the existing package/import surface and implementation.
- CI installs CPU-only PyTorch because these tests do not use a GPU. The library dependency remains
  `torch>=2.15,<2.16`, allowing users to install the appropriate CPU or accelerator build; passing
  the CPU tests does not establish GPU support.

## Test selection

| Request | Meaning |
| --- | --- |
| `uv run --no-sync tox list -d --no-desc` | List exactly five default version environments |
| `uv run --no-sync tox run -r -e py311-torch215` | Fresh wheel regression run on Python 3.11/PyTorch 2.15 |
| `uv run --no-sync tox run -r` | Run all five default combinations |
| `uv run --no-sync tox run -r -e sdist --installpkg PATH` | Install and validate the supplied candidate sdist on 3.12 |
| `--installpkg PATH` with a version environment | Test that specific candidate wheel, rather than a local rebuild |

Other default names are `py312-torch215`, `py313-torch215`, `py314-torch215`, `py315-torch215`.
The `test` dependency group supplies test tools. The tox driver is separately installed from the
locked `compatibility` group. A test invocation must neither synchronise the development environment
to each requested version nor rewrite its lockfile.

## Prerequisites and environment selection

Routine runs use the newest available stable patch in each supported minor series. Provision test
interpreters into `.tox/python` using uv's `--install-dir` option. Configure the directory and download
policy inside tox/tox-uv, including interpreter discovery, so contributors need not export environment
variables before testing. CPU selection is also configured internally. Missing interpreters produce
non-zero status and incomplete coverage. Confirm these settings apply during environment creation,
not only while executing pytest.

Use CPU backend selection during test dependency installation and highest compatible stable resolution.
Recreate environments for fresh runs. Driver versions may remain locked; test dependency versions must
not be selected by the development lockfile. Reject a wrong minor, alternative/free-threaded interpreter,
wrong PyTorch series, checkout-shadowed import or non-CPU Linux build before claiming compatibility.

## Evidence and status

Use tox's ordinary console/setup logs and built-in `--result-json` output, its installed-dependency
listing, and pytest's standard `--junitxml` reports. Print interpreter, platform, package import path,
and relevant runtime versions in pytest's report header through the existing test configuration.
CI identifies the source revision and retains the original distribution artifacts. Do not invent an
`environment.json` schema, custom reporting command, or mandatory replay-mode flag.

An installation failure may have only tox/installer logs; it is still an unsuccessful result.

A successful process exit requires successful installation and regression tests, no ordinary skips,
and no unexpected passes. Existing strict xfails are reported distinctly. A missing/cancelled check
or absent report cannot be interpreted as success. Local native results state their platform; only
all five successful Linux CI checks establish official compatibility.

## Diagnostic replay

Reproduce a failure using the original wheel, matching platform, exact interpreter and dependency
constraints derived from the recorded installed-dependency listing. The exercised native
override command and tox-uv exact-interpreter limitation are documented in the quickstart. Use tox's native `base_python`
and `constraints` overrides; exclude the candidate's local artifact reference from dependency
constraints because the wheel is supplied explicitly. These overrides and logs already distinguish
an exact-version run from a routine latest-patch run. No project-specific mode flag, replay service,
report schema, or package archival system is needed.

## CI and maintainer responsibilities

CI derives its five entries from the default tox list, builds the candidate once, and runs all entries
with independent outcomes (`fail-fast: false`). Standard Actions jobs expose these stable check names:

- `Compatibility (py311-torch215)`
- `Compatibility (py312-torch215)`
- `Compatibility (py313-torch215)`
- `Compatibility (py314-torch215)`
- `Compatibility (py315-torch215)`
- `Build distributions` (includes sdist validation)
- `Lint and type-check`

The maintainer requires these checks in GitHub rulesets/branch protection. Repository changes only
publish checks and document this handoff. No custom aggregate merge-policy code or automated ruleset
updates are part of the contract. Release readiness requires checking the candidate revision's results;
publishing and release-workflow redesign remain follow-up work.
