# Quickstart: Validate Supported-Version Compatibility

Run from the repository root. Validation evidence is recorded at the end; native results
do not establish Linux certification. See
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
uv run --no-sync uv python install --install-dir .tox/python --upgrade --no-bin --no-registry 3.11 3.12 3.13 3.14 3.15
```

To test only one combination, provision its minor and 3.12 for shared packaging. The command works without shell environment
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
uv run --no-sync tox run -r -e diagnostic --installpkg PATH_TO_ORIGINAL_WHEEL -x env_list=diagnostic -x tool.tox.env.diagnostic.base_python=PATH_TO_EXACT_PYTHON -x tool.tox.env_run_base.constraints=PATH_TO_RECORDED_CONSTRAINTS
```

The temporary `diagnostic` environment uses standard tox overrides: tox-uv 1.37.2
discards patch versions and exact paths for Python-named environments, so replay must
use a name without a Python factor. The default five-environment matrix is unchanged.
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

## Implementation validation record

- Baseline at `0fb0267`, native macOS arm64: CPython 3.12.14, PyTorch 2.14.0,
  NumPy 2.5.3, pytest 9.1.1. Existing suite: 81 passed, 9 strict xfails, no ordinary skips
  (90 collected). Library source matched `00abb16`; `.python-version` remained 3.12.

- Foundational proof: tox 4.67.4, tox-uv 1.37.2, bundled uv 0.13.6. Default list
  contains exactly five environments. Missing managed Python failed with exit 2 and downloads
  disabled. After provisioning, `py312-torch215 --notest` succeeded using managed CPython
  3.12.16, independently resolving torch 2.15.0 and NumPy 2.5.4 (development lock: NumPy 2.5.3).
  Use the project-installed uv for provisioning; the older system uv 0.12.9 selected a prerelease.

- Initial native wheel matrix: all five environments passed (83 passed, 9 xfailed each),
  using CPython 3.11.18/3.12.16/3.13.18/3.14.10/3.15.2, torch 2.15.0,
  and independently resolved NumPy (2.4.6 on 3.11; 2.5.4 on the others).
  Source archive on 3.12.16: 83 passed, 9 xfailed. No ordinary skips.
- Candidate `loss_landscapes-3.0.7.dev22+g0fb026720.d20261030`: wheel SHA256
  `95848673cf41c4d975e6101f3c4862a067819b24fc347008903c97ecb5982d52`;
  sdist SHA256 `8ab49dccb053c2fde7878b3d3d44a3ecc853cbc890f7434d08005a50802b91ca`.
  Twine accepted both; Name/Version/Requires-Python/Requires-Dist match. Normal resolver
  rejected a torch<2.15 constraint and a Python 3.16 target. Runtime dependencies were
  installed by the candidate metadata, not the test group. Source diff remained empty.
- Outcome-policy tests were red before the option/hooks existed, then passed after implementation.
  Collection-only skips preserve pytest's existing exit 5; other ordinary skips produce exit 1.
  XFAIL succeeds, strict XPASS and ordinary failures fail. Real subprocess checks reject wrong
  interpreter minors and checkout-shadowed imports. Development suite: 94 passed, 9 xfailed.

- Exact replay: the documented `-x env_list=diagnostic` and fully qualified environment
  overrides selected an absolute CPython 3.12.14 path despite 3.12.16 being installed.
  Replaying the recorded 3.12.16 wheel result with `.tox/replay-constraints.txt` then passed
  all 94 tests plus 9 xfails. Installed dependency pins matched byte-for-byte; constraints
  appeared in both test and runtime installation commands. A patch-only override on a
  Python-named environment was found ineffective and is not documented as a working replay.
- Type-checking finding: NumPy 2.5 stubs use 3.12 syntax. Mypy stays on development Python
  3.12; Ruff checks 3.11 syntax and the matrix verifies actual 3.11 execution. No new NumPy
  bound or library source change was introduced. Pre-commit also normalised the missing
  final newline in Spec Kit's workflow registry.

- Final native wheel matrix with compatibility hooks: all five environments passed, each
  with 94 passed and 9 xfailed; no ordinary skips. Development dependency listing and
  SHA256 of `uv.lock`/`.python-version` matched before/after matrix selection and replay.
  All pre-commit hooks passed, including gitleaks, Ruff, lock consistency and mypy.
