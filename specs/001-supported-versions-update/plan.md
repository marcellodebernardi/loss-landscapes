# Implementation Plan: Supported Versions and Compatibility Verification

**Branch**: `feature/supported-versions-update` | **Date**: 2026-10-30 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-supported-versions-update/spec.md`

## Summary

Publish an accurate support contract for PyTorch 2.15 with standard CPython 3.11–3.15 and
establish repeatable CPU compatibility verification without changing library source. Five tox
wheel-install environments share one definition between local runs and Linux CI. A non-lock
uv runner selects CPU PyTorch and fresh compatible dependencies; the development lock remains
separate. Built-in tox logs/results and pytest reports support diagnosis and exact-version reproduction.
The maintainer configures GitHub required checks; no custom merge-enforcement system is introduced.

## Technical Context

**Language/Version**: Existing Python library; supported interpreters are CPython 3.11–3.15,
standard GIL-enabled builds. Keep Python 3.12 as the development and build-driver default.

**Primary Dependencies**: Runtime `torch>=2.15,<2.16` and existing `numpy` requirement.
Hatchling/hatch-vcs remain the build backend. Add tox >=4.28 and tox-uv >=1 to a compatibility
tool group. Keep pytest, pytest-cov and Hypothesis in the existing test group; declare PyTorch
only as a runtime dependency. Driver versions are resolved into `uv.lock` during implementation.

**Storage**: Ephemeral `.tox/` environments/interpreters, tox logs/result JSON, dependency listings
and pytest JUnit reports; ordinary GitHub Actions distribution artifacts. No custom report schema.

**Testing**: Full existing pytest suite against installed wheels in five environments, installed
sdist verification on 3.12, and focused tests of the compatibility checks themselves. Keep strict
xfails. Compatibility-mode pytest hooks reject ordinary skips without treating xfails as skips.

**Target Platform**: Official verification on Linux x86-64 CPU, using a hosted runner compatible
with manylinux_2_28. Native macOS/Windows local feedback where upstream wheels exist; not additional
official platform certification. No GPU, free-threaded or alternative-interpreter support claim.

**Project Type**: Reusable Python library with development and CI tooling.

**Performance Goals**: Preserve the fast CPU regression suite. Reuse downloaded wheels/interpreters
through normal caches while recreating test environments. No new throughput benchmark or timing SLA.

**Constraints**: `src/loss_landscapes/` unchanged from `00abb16`; no algorithm, signature, metric,
state-handling or `contrib` repair. Do not change the release workflow or GitHub rulesets. Use isolated
test environments, exact installed-version evidence, and no development lockfile for test resolution.

**Scale/Scope**: Five default compatibility environments, one sdist environment, one CI matrix,
standard reporting and small pytest hooks; no new runtime API or reporting framework.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle or rule | Pre-research gate | Post-design gate |
| --- | --- | --- |
| I. Simplicity and earned abstractions | Pass: ordinary tox/CI machinery | Pass: no runner framework, dashboard or custom merge gate |
| II. Mathematical correctness | Pass: numerical changes deferred | Pass: valid tests retained; known defects remain explicit |
| III. Explicit public contracts | Pass: support matrix and side-effect preservation specified | Pass: bounded metadata and installed-artifact validation; no API changes |
| IV. Readable, documented code | Pass: existing conventions apply | Pass: standard tooling, small pytest hooks and contributor instructions |
| V. Behaviour-based verification | Pass: complete suite and installation checks required | Pass: installed wheel/sdist checks, negative controls and strict outcomes |
| Dependency and support declarations | Pass: intentional contract update in scope | Pass: Python/PyTorch bounds, classifiers, docs and executable matrix aligned |
| Legacy non-compliance | Pass: no unrelated rewrite required | Pass: xfails retained; source changes are blockers for this scope |

No constitutional exceptions are required. Both gates pass. Research conclusions and implementation
validation gates are recorded in [research.md](research.md); no unresolved clarifications remain.

## Project Structure

### Documentation (this feature)

```text
specs/001-supported-versions-update/
├── spec.md
├── checklists/requirements.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── contracts/compatibility-workflow.md
```

Task generation is the next phase; this command does not create `tasks.md`.

### Source Code (repository root)

```text
pyproject.toml                      # Metadata, tool groups, native tox config, version targets
uv.lock                             # Refreshed development/tool resolution
.python-version                     # Retain 3.12
.github/workflows/ci.yml             # Build artifacts, derive matrix, run five tox jobs
.github/workflows/release.yml        # Unchanged
scripts/compare_with_published.py   # Remove obsolete 3.0.6 contract guard
tests/conftest.py                    # Standard report header and opt-in skip policy
tests/test_compatibility.py          # Focused checks of the small project-specific hooks
src/loss_landscapes/                 # Entire subtree unchanged
README.md
CONTRIBUTING.md
CHANGELOG.md                        # Describe policy, validation, limitations and migration
```

**Structure Decision**: Extend existing scaffolding and tests. Place tox settings in the existing
project configuration. Keep reports under ignored `.tox/`; no runtime package additions or new
persistent application data. Only add narrowly scoped tooling needed by the agreed requirements.

## Design and Execution Boundaries

### Package and development contract

Set Python metadata to `>=3.11,<3.16`, PyTorch to `>=2.15,<2.16`, update classifiers and language
targets, and remove the legacy uv Python-range workaround. Retain NumPy's existing requirement;
record actual compatible resolutions instead of imposing an unevidenced new NumPy policy.
Remove `scripts/compare_with_published.py`, its CI step and its contributor documentation now that
the contract deliberately changes. Refresh the development lock and include compatibility tooling
in the dev group. Do not run automated source rewrites when raising Ruff's target.

### Tox and artifact isolation

Define default environments `py311-torch215` through `py315-torch215` explicitly, plus a non-default
`sdist` environment on Python 3.12. Use `uv-venv-runner`, wheel packaging, a shared pure-Python wheel
build environment, `dependency_groups = ["test"]`, fresh environment creation, and hard failure for
missing interpreters. Set CPU backend selection and interpreter-directory configuration internally
in tox/tox-uv, applying them to environment creation as well as test commands. Do not use `uv-venv-lock-runner`, editable installation, `PYTHONPATH=src`, or the
candidate project installed in the tox driver as the package under test.

Use small additions to the existing package tests to verify the interpreter minor, standard CPython
build, PyTorch series, installed package origin and Linux CPU build. Print versions/platform in a
normal pytest report header. In CI the build job supplies the identical wheel to all five matrix
jobs using tox's `--installpkg` option. The `sdist` environment receives the actual source archive,
builds and installs its wheel, then runs the existing tests. This checks that the published source
archive contains everything required to build; it is not a sixth version combination.

### Interpreter and patch policy

Routine test runs use the newest available stable patch releases within each supported minor series.
Provision test interpreters with current uv into `.tox/python` using the explicit `--install-dir`
option together with `--upgrade --no-bin --no-registry`. Configure that directory, interpreter
preference and download policy inside tox/tox-uv. Contributors should not need to set shell variables
for ordinary test runs. Validate that configuration reaches interpreter discovery/environment creation;
setting it only for pytest would be insufficient.

CI starts from a clean directory; local provisioning refreshes only the test-specific installations.
Keep downloads cached but recreate resolved test environments. Record exact versions through standard
logs and installed-dependency listings. Reproduce failures using tox's native interpreter/constraint
and `--installpkg` overrides with the original artifact. No `COMPAT_REPLAY` flag or custom report schema
is required. See [contracts/compatibility-workflow.md](contracts/compatibility-workflow.md).

### Outcome policy and evidence

Keep pytest's strict xfail behaviour. Add opt-in compatibility hooks to reject ordinary collection,
setup, call or teardown skips; do not confuse `wasxfail` reports with ordinary skips. Current ordinary
skip baseline is empty. Future intentional exemptions must be explicitly reviewed and documented;
no generic skip-allowlist framework is required now. Do not add a fixed expected-xfail count, which
would unnecessarily obstruct the later numerical fixes this setup is intended to protect.

Use pytest's standard JUnit output, tox's built-in result JSON/setup logs and installed-dependency
listing. These provide ordinary failure diagnostics; CI also records the source revision and retains
the original distributions. A small pytest report-header hook makes actual Python, platform, PyTorch,
NumPy and import-location details visible. Do not introduce a custom JSON schema, evidence helper,
mandatory dependency-pin file layout or replay-mode protocol.

Preserve the tools' original non-zero exit statuses. Always upload available standard reports/logs
after a CI failure. An installation failure can have only setup logs; absent pytest output is not
success. The only additional outcome rule is rejection of ordinary skips, as required by FR-005;
pytest does not provide that rule by default, so its small project-specific hook must be explicit.

### CI and maintainer handoff

The preparation/build job installs the locked compatibility driver group, derives matrix entries
from `tox list -d --no-desc`, builds wheel/sdist, and validates the sdist on 3.12. A small workflow-local
conversion emits each environment name and its Python minor. This is formatting the existing list,
not a second hand-maintained support matrix or custom orchestrator.

Five Linux jobs provision the requested Python, install only the driver tools, consume the candidate
wheel, and run the matching tox environment with `fail-fast: false`. Keep existing lint and distribution
validation. Use stable check names `Compatibility (py311-torch215)` through
`Compatibility (py315-torch215)`, plus `Build distributions` and `Lint and type-check`.

Document these names for the maintainer to require in GitHub settings. Do not edit settings here
or invent merge enforcement. The maintainer verifies that the candidate revision passed all checks
before release; the tag-based release workflow remains unchanged.

## Validation Strategy

1. Check package metadata, support documentation, tox environment enumeration and CI matrix agree
   on exactly five combinations. Reject incompatible Python/PyTorch environments through metadata.
2. Build distributions and inspect metadata and contents; compare the `src/loss_landscapes/` subtree
   with `00abb16` and require no source changes. Confirm declared dependencies alone install the package.
3. Run all five Linux wheel environments and the 3.12 sdist environment. Require the existing suite,
   strict xfails, no ordinary skips, correct environment identity, and installed-package import paths.
4. Use focused subprocess/pytester checks to prove ordinary failure, XPASS, skip, wrong interpreter,
   shadowed imports and missing interpreter do not produce success. Test through observable outputs
   and exit status; do not write tests that merely repeat configuration values.
5. Reproduce one recorded result with its exact interpreter, candidate artifact and dependency pins;
   verify actual versions match. Confirm `.venv`, `.python-version` and `uv.lock` are not rewritten by
   matrix selection or replay. Initial intentional lock regeneration is distinct from running tests.
6. Follow the quickstart locally on an available native platform and identify unavailable combinations
   honestly. Only Linux CI establishes the support claim. Document wheel availability constraints.
7. Verify a controlled failure in one CI environment cannot be hidden by other passing jobs, and
   cancelled/missing checks remain unsuitable for maintainer approval. Validate artifact upload on failure.
8. Run applicable lint, formatting, typing, packaging and pre-commit checks after implementation.

## Complexity Tracking

No constitutional violations or exceptions. The tox-uv plugin is justified by the existing uv workflow
and platform-aware CPU installation. Prefer standard tox/pytest features and documented configuration.
Project-specific code is limited to necessary package assertions, a readable pytest header and the
specification's skip rule; no custom report schema, replay flag, scheduler or policy engine.
