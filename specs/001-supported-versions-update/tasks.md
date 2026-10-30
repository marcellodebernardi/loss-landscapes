# Tasks: Supported Versions and Compatibility Verification

**Input**: Design documents from `specs/001-supported-versions-update/`

**Prerequisites**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md),
[data-model.md](data-model.md), [workflow contract](contracts/compatibility-workflow.md),
and [quickstart.md](quickstart.md).

**Tests**: The specification explicitly requires compatibility, distribution-installation and failure
handling verification. Include focused tests for new project-specific behaviour; do not test third-party
tool internals or duplicate configuration values without testing their effect.

**Organization**: Tasks are grouped by user story. Paths are relative to the repository root.
No task authorises library implementation changes, publishing a release, or changing GitHub settings.

## Format: `[ID] [P?] [Story] Description`

- `[P]` marks work that can proceed alongside the other named tasks once its phase prerequisites pass.
- `[US1]`, `[US2]`, and `[US3]` map to the specification's user stories.
- Follow explicit dependencies below; tasks editing the same file run sequentially.
- All checkboxes start unchecked. Record validation evidence when completing the corresponding task.

## Path Conventions

- `pyproject.toml`: runtime metadata, tool groups and the single executable tox matrix.
- `.github/workflows/ci.yml`: standard build, matrix and reporting jobs.
- `tests/`: existing regression tests plus minimal compatibility assertions/hooks.
- `specs/001-supported-versions-update/quickstart.md`: validation procedure and concise result notes.
- `.tox/`: ignored environments, test interpreters and normal tool reports; not committed evidence files.
- `src/loss_landscapes/` and `.github/workflows/release.yml`: unchanged throughout this feature.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish the existing behavioural baseline before changing the dependency contract.

- [X] T001 Verify `feature/supported-versions-update`, run the existing CPU suite with the current development environment, and record the revision, exact Python/PyTorch/NumPy versions, existing xfails and ordinary-skip baseline in `specs/001-supported-versions-update/quickstart.md`; confirm `src/loss_landscapes/` matches `00abb16` and `.python-version` remains 3.12. Do not fix failures or weaken existing assertions/markers.

**Checkpoint**: Baseline limitations and any pre-existing execution problems are distinguished from
regressions introduced by the compatibility work.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish the support contract and shared isolated test configuration.

- [X] T002 Update `pyproject.toml` to Python `>=3.11,<3.16`, runtime `torch>=2.15,<2.16` and the existing NumPy requirement; add Python 3.11–3.15 classifiers, remove the legacy uv Python-range workaround and duplicate test-group torch entry, set Ruff/mypy language baselines to 3.11, and add a development-included `compatibility` group with tox >=4.28 and tox-uv >=1. Preserve `.python-version`, the existing development CPU source, numerical tests and all library source.
- [X] T003 Regenerate `uv.lock` from the intentional `pyproject.toml` changes and install the development/tool groups; verify the tox driver and plugin load with their recorded versions. Do not hand-edit the lockfile or force a NumPy >=2.5 floor that would exclude Python 3.11.
- [X] T004 Define exactly five default wheel environments (`py311-torch215` through `py315-torch215`) in native TOML within `pyproject.toml`, using `uv-venv-runner`, highest compatible stable dependency resolution, the existing `test` dependency group, a shared pure-Python wheel build, recreation and hard failure for missing interpreters. Configure CPU dependency installation internally; do not use the lock runner, editable packages, `PYTHONPATH=src`, or a portable runtime `+cpu` pin.
- [X] T005 Configure and exercise interpreter discovery/environment creation in `pyproject.toml` using uv-managed interpreters provisioned into `.tox/python` by the documented `--install-dir --upgrade --no-bin --no-registry` command; verify directory/preference/download settings apply before pytest starts, require no shell exports, and fail explicitly for a missing interpreter. Record the working invocation in `specs/001-supported-versions-update/quickstart.md`; report a tooling limitation rather than silently using the development interpreter or inventing a custom runner.

**Checkpoint**: T002–T005 are complete before story work begins. The executable policy is
“Python 3.11–3.15 only; PyTorch 2.15 only.” Interpreter prerequisites and tool behaviour have been
exercised, not merely assumed from configuration syntax.

---

## Phase 3: User Story 1 - Install in a Supported Environment (Priority: P1) — MVP

**Goal**: Install and exercise the unchanged library from both published distribution formats with
accurate dependency metadata.

**Independent Test**: Build candidate distributions, install the wheel into clean supported version
environments, and build/install the source archive on Python 3.12. Confirm working CPU sampling,
declared dependencies, version boundaries and installed import origins.

### Tests for User Story 1

- [X] T006 [P] [US1] Extend `tests/test_packaging.py` with meaningful installed-distribution checks for declared runtime dependencies, effective Python/PyTorch support bounds and package origin inside the isolated test environment. During compatibility runs verify the requested interpreter minor, standard CPython/GIL build, PyTorch 2.15 series and Linux CPU build; enforce “candidate imports resolve inside the isolated environment, not `src/` or the development environment”. Keep ordinary development tests usable and existing known-failure tests unchanged.

### Implementation and Validation for User Story 1

- [X] T007 [P] [US1] Remove `scripts/compare_with_published.py`, its preservation step from `.github/workflows/ci.yml`, and the obsolete 3.0.6-contract section from `CONTRIBUTING.md`; retain the rest of the contributor guidance and the distribution build/metadata checks.
- [X] T008 [P] [US1] Add a non-default `sdist` environment on Python 3.12 in `pyproject.toml` that accepts the actual candidate archive via `--installpkg`, builds/installs it and runs the existing suite; ensure the default list still contains only the five wheel combinations. The source archive must be tested independently of the checkout, not replaced by a newly generated archive during validation.
- [X] T009 [US1] Follow the distribution scenarios in `specs/001-supported-versions-update/quickstart.md`: build wheel/sdist, run metadata checks, prove declared dependencies alone provide an importable installation, and demonstrate resolution rejection for the candidate with unsupported Python or PyTorch 2.14 constraints. Compare wheel/sdist metadata and verify `src/loss_landscapes/` is unchanged from `00abb16`; record filenames, hashes and results using standard tool output.
- [X] T010 [US1] Execute the five wheel environments and Python 3.12 source-archive validation configured in `pyproject.toml`, against the candidate artifacts from T009; require the existing full CPU suite with strict xfails and no new failures. Record the actual versions and platform in `specs/001-supported-versions-update/quickstart.md`; native-only results are provisional until Linux CI passes, and any source-level incompatibility is a scope blocker.

**Checkpoint**: US1 is independently demonstrable. Installation and initial compatibility evidence
exist; ongoing automated enforcement is delivered by US2, not claimed by this checkpoint.

---

## Phase 4: User Story 2 - Detect Compatibility Regressions (Priority: P1)

**Goal**: Produce ordinary, independently identifiable CI checks for all five combinations and prevent
new failures, unexpected passes or ordinary skips from looking successful.

**Independent Test**: Run the full Linux matrix and a controlled single-environment failure; verify
that other jobs still report results, the affected check fails, and standard logs/reports remain available.

### Tests for User Story 2

- [X] T011 [P] [US2] Add focused subprocess/pytester cases in `tests/test_compatibility.py` for the planned compatibility outcome rule and report header: a passing session, a documented strict xfail, a strict unexpected pass, an ordinary failure and ordinary collection/setup/call/teardown skips. Verify observable exit status/output, including “zero ordinary skips and zero unexpected passes”; make the new skip-rule tests fail before implementing that rule. Reuse existing pytest facilities without mocking tool internals.

### Implementation and Validation for User Story 2

- [X] T012 [US2] Add the minimal opt-in skip policy and normal pytest report header in `tests/conftest.py`, enabled by the tox compatibility configuration in `pyproject.toml`; distinguish `wasxfail` from ordinary skips, preserve existing error exit statuses, and report actual interpreter, platform, PyTorch/NumPy versions and package import location. Pass T011 without a custom JSON format, reporting script, replay flag, fixed xfail count or generic skip-allowlist framework.
- [X] T013 [P] [US2] Update the preparation/build portion of `.github/workflows/ci.yml` to install the locked compatibility driver, derive environment names/Python minors from `tox list -d --no-desc`, build one candidate wheel and sdist, validate that actual sdist on 3.12, and retain the original distribution files and hashes. Use a short workflow-local JSON conversion rather than a second hardcoded matrix or new orchestration script.
- [X] T014 [US2] Replace the old version jobs in `.github/workflows/ci.yml` with the five derived Linux x86-64 CPU jobs, provisioning current stable interpreter patches and passing the identical candidate wheel to each selected tox environment via `--installpkg`; run on PRs to `main` and pushes to `main`, keep `fail-fast: false`, and name checks `Compatibility (py311-torch215)` through `Compatibility (py315-torch215)`. Retain `Build distributions` and `Lint and type-check`.
- [X] T015 [US2] Configure standard tox result JSON/dependency listings and pytest JUnit output through `pyproject.toml`, and always upload available logs/reports from `.github/workflows/ci.yml`, including on failure. Keep revision and artifact identity visible, treat setup-only logs as failure evidence rather than success, and rely on native exit/status semantics instead of adding an aggregate merge gate or custom evidence-file convention.
- [ ] T016 [US2] Exercise the workflow and failure cases described in `specs/001-supported-versions-update/quickstart.md`: wrong interpreter/import origin, unavailable dependency or interpreter, strict unexpected pass, ordinary skip, and a controlled failure in only one CI environment. Confirm non-zero/failed results, other matrix results and retained diagnostics; remove temporary failure injections before acceptance and leave `.github/workflows/release.yml` and GitHub settings untouched.

**Checkpoint**: All five Linux checks succeed for the real candidate; controlled failures cannot
be mistaken for full compatibility. Missing or cancelled checks are not positive evidence.

---

## Phase 5: User Story 3 - Reproduce Checks Locally (Priority: P2)

**Goal**: Document and demonstrate selection of one/all combinations, native developer feedback and
exact-version reproduction without switching the everyday development environment.

**Independent Test**: Follow the published local instructions, inspect real versions/platform, and
reproduce one recorded run with the original wheel, exact interpreter and dependency constraints.

### Implementation and Validation for User Story 3

- [X] T017 [P] [US3] Update `CONTRIBUTING.md` with setup/provisioning prerequisites and standard commands to list/select one/all tox environments, refresh stable patches, inspect tox/pytest reports and reproduce a failure through native interpreter/constraint overrides. Explain the driver lock versus independently resolved test dependencies, explicit missing coverage and the absence of required shell exports or project-specific replay flags.
- [X] T018 [P] [US3] Update `README.md` with the five-combination Python/PyTorch support table, latest-compatible-patch policy, Linux CPU verification scope, native macOS/Windows feedback where wheels exist, and known numerical/legacy limitations. Distinguish installation compatibility from numerical correctness and defer GPU/API/algorithm improvements to follow-up work; do not promise untested accelerator support or unsupported native platforms.
- [X] T019 [P] [US3] Reproduce one recorded combination using the original candidate wheel, matching exact interpreter/platform and dependency constraints derived from tox's installed-dependency listing; verify native `base_python`/`constraints` overrides cover test and runtime dependencies, exclude the explicitly supplied candidate reference, and preserve exact versions. Record the verified command in `specs/001-supported-versions-update/quickstart.md`; respect “Linux CPU pins need not install on macOS” and report unavailable historical downloads rather than silently upgrading.
- [X] T020 [US3] Validate single/full selection on an available native developer platform using `specs/001-supported-versions-update/quickstart.md`, with missing prerequisites reported as incomplete coverage. Check before/after state of `.venv`, `.python-version` and `uv.lock` to prove matrix selection/reproduction neither switches nor rewrites the development environment; record platform limitations honestly without treating native feedback as Linux certification.
- [X] T021 [US3] Document the five exact compatibility check names plus `Build distributions` and `Lint and type-check` in `CONTRIBUTING.md`, and explain the maintainer-owned GitHub ruleset/branch-protection handoff and candidate-revision check before release. Do not implement custom merge enforcement, edit repository settings, publish a release or extend `.github/workflows/release.yml`.

**Checkpoint**: Contributor commands are validated, results can be diagnosed/reproduced with standard
tools, and the external GitHub configuration responsibility is explicit.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Finish the release-facing documentation and demonstrate the complete feature without
expanding its scope.

- [X] T022 [P] Add the support/dependency-contract change, installation fix, compatibility workflow and unchanged numerical limitations to `CHANGELOG.md`; explain migration from the obsolete Python/PyTorch declarations without selecting a final release number or publishing anything.
- [ ] T023 Re-run the end-to-end scenarios in `specs/001-supported-versions-update/quickstart.md` after final changes, confirming all five Linux wheel checks, actual sdist installation, clean automatic dependency installation, strict expected-failure behaviour and normal reports; compare `README.md`, `CONTRIBUTING.md`, `pyproject.toml` and `.github/workflows/ci.yml` for one consistent policy and record final evidence.
- [X] T024 Run applicable Ruff, formatting, mypy, build/metadata and pre-commit checks from `pyproject.toml`/`.pre-commit-config.yaml`; confirm `git diff 00abb16 -- src/loss_landscapes` and changes to `.github/workflows/release.yml` are empty, `.python-version` remains unchanged, and no `.tox/` outputs are staged. Review task completion against `specs/001-supported-versions-update/spec.md` before declaring implementation complete.

---

## Dependencies & Execution Order

### Phase Dependencies

```text
T001 baseline
  → T002 metadata/tools → T003 lock → T004 tox matrix → T005 interpreter proof
  → US1: T006 + T007 + T008 → T009 → T010
  → US2: T011 → T012
          T013 (parallel with T011/T012)
          T012 + T013 → T014 → T015 → T016
  → US3: T017 + T018 + T019 → T020; T017 → T021
  → T022 + T023 → T024
```

- Setup and foundational work block all stories.
- US1 provides the installable candidate and baseline for US2. US2 requires it to establish real CI
  outcomes rather than checks against an editable development installation.
- US3's documentation and reproduction validation use US2's working reports/commands. Documentation
  drafts may start earlier, but the story is not complete until instructions have been exercised.
- T022 may run alongside T023 after the story phases; T024 follows both.
- T013 edits only the CI workflow and may proceed while the outcome hook is implemented. T014 waits
  for that hook and the build preparation. All edits to the same file remain sequential.

### Within Each User Story

Write focused tests for newly introduced behaviour before implementing it. Existing installation and
regression suites remain the verification baseline; do not manufacture failing tests or rewrite valid
numerical expectations merely to satisfy a process. Perform the independent validation before moving
past a story checkpoint. Stop and report source-level incompatibilities outside this feature's scope.

### Parallel Opportunities

Parallel work means independent file ownership, not permission to bypass prerequisites. The `[P]`
markers identify possible work scheduling; they do not require spawning agents or running every check
simultaneously. Five CI version jobs may run in parallel after the build preparation succeeds.

## Parallel Example: User Story 1

After T005, independently author installed-package checks (T006: `tests/test_packaging.py`), remove
the legacy contract guard (T007: script/CI/contributor text), and configure source-archive validation
(T008: `pyproject.toml`). Join all three before T009's distribution validation.

## Parallel Example: User Story 2

After US1, add the outcome-policy tests (T011: `tests/test_compatibility.py`) and then its hooks
(T012: `tests/conftest.py`/`pyproject.toml`) while T013 prepares artifact/matrix setup in
`.github/workflows/ci.yml`. Join before wiring the five jobs in T014.

## Parallel Example: User Story 3

After US2, draft contributor guidance (T017: `CONTRIBUTING.md`), update the user support table
(T018: `README.md`), and validate/document exact reproduction (T019: feature `quickstart.md`) in
parallel. Sequence T020 after T019 and T021 after T017 to avoid shared-file conflicts.

## Implementation Strategy

### MVP First (User Story 1 Only)

Complete T001–T010. Deliver bounded runtime metadata and independently verified wheel/source-archive
installation with the existing implementation. Do not yet claim ongoing CI coverage or that known
numerical defects are repaired. This is the smallest useful installation milestone.

### Incremental Delivery

1. US1: accurate installable distributions and initial supported-version results.
2. US2: repeatable five-combination Linux CI and trustworthy failure reporting.
3. US3: exercised local workflow, exact-version reproduction and maintainer settings handoff.
4. Final validation: consistent docs, clean checks and unchanged library/release implementation.

### Requirement Coverage

| Requirement | Primary tasks |
| --- | --- |
| FR-001: five supported pairs | T002, T004, T010, T014 |
| FR-002: runtime metadata/obsolete contract removal | T002, T003, T006, T007, T009 |
| FR-003: clean wheel installation | T004, T006, T009, T010, T014 |
| FR-004: actual source archive | T008, T009, T013, T023 |
| FR-005: full suite and strict outcomes | T010, T011, T012, T016 |
| FR-006: ongoing CI/maintainer enforcement | T013, T014, T015, T016, T021 |
| FR-007: local selection and isolation | T005, T017, T020 |
| FR-008: actual versions and outcomes | T006, T012, T015, T016 |
| FR-009: patch refresh/exact reproduction | T004, T005, T017, T019 |
| FR-010: compatible dependency resolution | T002, T003, T004, T010 |
| FR-011: unchanged library behaviour | T001, T009, T010, T024 |
| FR-012: documentation and limitations | T017, T018, T021, T022 |
| FR-013: aligned support policy | T002, T004, T018, T023 |

SC-001 and SC-006 are demonstrated by T009–T010 and T023; SC-002 by T011–T016; SC-003 by
T012/T015/T019; SC-004 by T017/T020; SC-005 by T023.

## Notes

- Prefer standard tox/uv/pytest features; custom code is limited to necessary assertions and the
  small report-header/skip hooks. Do not revive custom evidence schemas or replay protocols.
- Keep known defects explicit. No broad xfail/skip changes or reduced matrix may hide regressions.
- Use existing reporting and ordinary GitHub artifacts. A missing report or cancelled run is not a pass.
- A missing dependency distribution or interpreter is evidence of incomplete verification, not an
  invitation to silently omit a supported combination.
- Maintainer GitHub configuration is a documented handoff, not a task for a repository policy script.


## Acceptance status

Implementation and local validation are complete. T016 and T023 remain unchecked because GitHub's
run/check API still reports queued for the probe and clean-candidate runs, even though their
uploaded tox/JUnit results are complete. Probe evidence and clean Linux results are recorded in
quickstart.md. Confirm completed failing/successful workflow statuses before marking these tasks
complete; do not treat uploaded reports alone as fulfilled required-check acceptance.
