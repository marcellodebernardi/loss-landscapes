# Quickstart: Validate Supported-Version Compatibility

These are the intended validation commands after implementation, not claims that the tooling has
already been added or the matrix has passed. Run from the repository root. See
[the workflow contract](contracts/compatibility-workflow.md) for outcome and replay rules.

## Prerequisites

- Current stable uv with Python management and PyTorch CPU backend selection.
- Network access to interpreter downloads, PyPI and the official PyTorch CPU index.
- Official verification: Linux x86-64 compatible with manylinux_2_28; no GPU required.
- Native feedback: Windows amd64 or macOS 14+ Apple Silicon where upstream wheels are available.
  An Intel Mac or another unsupported platform may fail dependency installation; report that limitation.
- Sufficient disk space for five separate PyTorch environments and interpreters. Run one combination
  at a time locally when disk space is limited; CI jobs run separately.

## Prepare the driver and isolated test interpreters

Initial development setup intentionally installs/updates the development tools:

```bash
uv sync --all-groups
uv run --no-sync tox --version
uv run --no-sync tox list -d --no-desc
```

Expect exactly the five names listed in the contract. The driver uses the committed development lock;
individual test environments do not.

Provision the latest available patches into the test-only directory:

```bash
uv python install --install-dir .tox/python --upgrade --no-bin --no-registry 3.11 3.12 3.13 3.14 3.15
```

To test only one combination, provision only its minor. The command works without shell environment
exports; tox/tox-uv configuration must select the same directory for interpreter discovery and handle
its download policy internally. These settings need validation during implementation, including on
native developer platforms. Provisioning must not replace global executables, register interpreters,
switch `.python-version`, or rewrite `uv.lock`.

## Run one combination or the full matrix

```bash
uv run --no-sync tox run -r -e py311-torch215
uv run --no-sync tox run -r
```

Expected: requested environments install the candidate wheel into independent environments, verify
actual versions and import locations, and execute the full suite. Existing documented xfails remain
visible; new failures, XPASS and ordinary skips result in failure. The full request succeeds only when
all five environments complete. A missing interpreter is a failure, not a successful skip.

Inspect the ordinary tox setup logs/result JSON, dependency listing and pytest output/JUnit report.
The pytest header identifies platform, exact runtime versions and import location. CI retains the
original candidate artifacts. Setup failure may prevent a pytest report; tox must still report failure.

## Validate built distributions

```bash
uv build
uvx twine check dist/*
```

On PowerShell, pass the actual distribution filenames explicitly if wildcard expansion is unavailable.
Supply the actual candidate wheel filename to test precisely the artifact intended for distribution:

```text
uv run --no-sync tox run -r -e py311-torch215 --installpkg PATH_TO_CANDIDATE_WHEEL
uv run --no-sync tox run -r -e sdist --installpkg PATH_TO_CANDIDATE_SDIST
```

Replace the two `PATH_TO_...` arguments with real filenames, preserving the original distribution
filenames. The sdist environment builds/installs from that archive on Python 3.12 and runs the existing
suite. Inspect metadata for Python `>=3.11,<3.16` and PyTorch `>=2.15,<2.16`, declared NumPy, and updated
classifiers. Exercise a clean install without manually supplying torch to prove runtime dependencies
are declared. An unsupported interpreter or a constraint requiring PyTorch 2.14 must fail resolution.

Verify the library implementation is unchanged:

```bash
git diff 00abb16 -- src/loss_landscapes
```

Expected: no diff. Compare `.python-version` and `uv.lock` before/after test runs too; selecting matrix
environments must not change either file. Intentional metadata/lock edits during implementation are
not test-run side effects.

## Replay a recorded environment

Use the same source revision/candidate wheel and platform as the recorded result. Retain its original
filename and verify its hash. Provision the recorded exact Python patch into the test-only directory,
and obtain its executable path. A replay does not upgrade to newer patches.

Create an ordinary dependency constraints file from the recorded installed-dependency listing,
excluding the candidate's local wheel reference. Use native tox overrides for the chosen environment's
`base_python` and shared `constraints`.
For example, with actual absolute paths substituted:

```text
uv run --no-sync tox run -r -e py311-torch215 --installpkg PATH_TO_ORIGINAL_WHEEL -x env.py311-torch215.base_python=PATH_TO_EXACT_PYTHON -x env_run_base.constraints=PATH_TO_RECORDED_CONSTRAINTS
```

No custom replay flag or report format is needed. The exact interpreter and dependency overrides
are explicit in the tox command. Validate them against the pinned tox/tox-uv versions before delivery.
Confirm the actual installed versions match the recorded run, report unavailable historical artifacts,
and do not reuse Linux CPU constraints unchanged on another operating system.

## Prove failure handling

Run the focused compatibility tooling tests added during implementation:

```bash
uv run pytest tests/test_compatibility.py
```

These exercise controlled failures, strict unexpected passes, ordinary skips, wrong environment identity
and import shadowing through isolated toy sessions/subprocesses. They must verify observable status and
evidence rather than mock the runner into success. Also request a missing interpreter with downloads
disabled and confirm a non-zero result. No permanent failing test belongs in the main regression suite.

## Validate automation and hand off GitHub settings

Open a test PR after implementation. Confirm five independently named compatibility checks, successful
sdist validation and lint/build results. Verify a controlled one-environment failure produces a failing
check while other jobs still report their results; uploaded logs should remain available on failure.
Remove the controlled failure before acceptance.

The maintainer configures the required checks listed in the contract through GitHub rulesets/branch
protection. No repository script edits these settings. Do not publish/tag a release as part of this
validation; record complete Linux evidence for the candidate and leave publishing to follow-up work.
