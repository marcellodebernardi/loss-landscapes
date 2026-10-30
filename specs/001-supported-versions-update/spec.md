# Feature Specification: Supported Versions and Compatibility Verification

**Feature Branch**: `feature/supported-versions-update`

**Created**: 2026-10-30

**Status**: Draft

**Input**: User description: "Support PyTorch 2.15 and its supported Python versions, with a testing setup that maintains compatibility across future functionality changes."

## Clarifications

### Session 2026-10-30

- Q: Should contributors be able to run compatibility tests directly on macOS and Windows, while Linux remains the only officially verified platform? → A: Allow native local tests where dependencies are available; only Linux CI establishes official compatibility.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Install the library in a supported environment (Priority: P1)

As a researcher, I can install the modernised distribution with its declared dependencies
and use the existing library on a documented supported version combination, without having
to discover or install an undeclared dependency manually.

**Why this priority**: A modern re-release must first provide an accurate installation and
support contract while preserving the existing library implementation.

**Independent Test**: Install a candidate distribution into clean supported environments,
verify the installed package and dependencies, and exercise existing working sampling functionality.

**Acceptance Scenarios**:

1. **Given** any supported version combination, **When** a user installs the candidate wheel
   with its declared dependencies, **Then** the public package imports and its working CPU
   sampling functionality passes the existing regression tests.
2. **Given** a clean environment with no preinstalled runtime dependencies, **When** the user
   installs the distribution, **Then** the required runtime dependencies are installed automatically.
3. **Given** the candidate source distribution, **When** a user builds and installs it in a
   supported environment, **Then** it produces an importable package with the same declared support
   boundaries and public functionality as the candidate wheel.
4. **Given** an environment outside the declared version boundaries, **When** installation is
   attempted with normal dependency resolution, **Then** it cannot silently satisfy this release's
   declared compatibility requirements.

---

### User Story 2 - Detect compatibility regressions before accepting changes (Priority: P1)

As a maintainer, I receive a separate compatibility result for every supported combination
when a pull request or primary-branch change is checked, so a regression cannot hide behind
success on a different dependency version.

**Why this priority**: Compatibility needs continuing evidence after the initial re-release.

**Independent Test**: Run the compatibility checks, then introduce a controlled failure in
one environment and verify that the affected combination and overall result report failure.

**Acceptance Scenarios**:

1. **Given** a pull request or a push to the primary branch, **When** automated compatibility
   verification runs, **Then** all five supported combinations execute the complete regression suite
   against the installed candidate distribution and report individually identifiable results.
2. **Given** a failure in one supported combination, **When** other combinations pass,
   **Then** overall compatibility remains unsuccessful and identifies the failing environment.
3. **Given** an existing documented expected failure, **When** its test fails for the recorded
   reason, **Then** that known limitation remains visible without failing the compatibility result.
   A new ordinary failure or unexpected pass produces an unsuccessful result.
4. **Given** a missing interpreter, unavailable required dependency, installation error, or
   unexecuted matrix entry, **When** verification finishes, **Then** full compatibility is not reported
   as successful; the unavailable or incomplete combination is identified.

---

### User Story 3 - Reproduce and maintain compatibility checks locally (Priority: P2)

As a contributor, I can select one supported combination or the full matrix for local test runs
and see which versions actually ran, using the same environment definitions and acceptance rules
as automation. These isolated test runs do not replace my everyday development environment.

**Why this priority**: Contributors need to diagnose failures without repeatedly pushing changes,
and maintainers need one coherent support policy as dependencies evolve.

**Independent Test**: Follow the contributor instructions to select one combination, run it,
inspect its resolved versions, and compare its acceptance rules with the automated checks.

**Acceptance Scenarios**:

1. **Given** the prerequisites for a supported combination, **When** a contributor requests it,
   **Then** only that combination runs and reports its actual interpreter and runtime dependency versions.
2. **Given** all required local interpreters, **When** the contributor requests the full matrix,
   **Then** the same five combinations and regression checks used by automation run locally.
3. **Given** a local interpreter is missing, **When** the contributor requests the full matrix,
   **Then** the result identifies missing coverage instead of claiming the full matrix passed.
4. **Given** a later support-policy change, **When** the maintainer updates the supported combinations,
   **Then** documentation, package declarations, and automated and local checks remain aligned.
5. **Given** a contributor on macOS or Windows with the required interpreters and dependencies,
   **When** they request native local compatibility tests, **Then** the selected combinations run
   without requiring Linux, and the results identify the local platform without establishing official
   compatibility for that platform.

### Edge Cases

- Python 3.10 with PyTorch 2.15 is unsupported and must not be scheduled as a required environment.
- An existing development environment or lockfile must not override the requested test interpreter
  or dependency versions, or prevent eligible patch updates. Requested and installed versions must
  be checked for agreement; a universal lockfile may legitimately contain different resolutions
  for different interpreters or platforms.
- A newly available stable patch release must be eligible for subsequent verification within its
  supported minor series; a new minor series or prerelease must not silently expand support.
- A new failure must not be hidden by broadening an expected-failure marker, adding an unexplained
  skip, or reducing the matrix. Unexpected passes require investigation rather than automatic acceptance.
- Existing test skips must be accounted for. New or unexplained skips cannot count as complete coverage.
- A supported pair whose required dependencies cannot be installed is a failed compatibility check,
  not a reason to silently omit that pair.
- Numeric comparisons use justified tolerances; compatibility does not imply identical random samples
  or bitwise-identical floating-point results across dependency versions.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The supported matrix MUST contain exactly these five combinations of standard CPython
  and the PyTorch 2.15 stable minor series:

  | PyTorch minor series | Supported Python minor versions | Combinations |
  | --- | --- | --- |
  | 2.15 | 3.11, 3.12, 3.13, 3.14, 3.15 | 5 |

- **FR-002**: Package metadata MUST declare the runtime dependencies, including PyTorch and NumPy,
  and express the supported version boundaries consistently with FR-001. Obsolete Python classifiers
  and the requirement to preserve the published 3.0.6 dependency contract MUST be removed.
- **FR-003**: Candidate wheel installations MUST be verified in clean, isolated environments for
  every supported combination, without depending on an editable checkout or preinstalled dependencies.
- **FR-004**: The source distribution MUST be built and installed successfully in at least one
  supported combination, with import and working CPU sampling verification and metadata consistent
  with the wheel. The full regression matrix MUST run against the wheel.
- **FR-005**: Every supported combination MUST execute the existing complete regression suite on CPU.
  Existing documented defects MUST remain tracked by strict expected-failure markers. New failures,
  unexpected passes, and new or unexplained skips MUST prevent a successful compatibility result.
- **FR-006**: The full supported matrix MUST run for every pull request to the primary branch and
  every push to that branch. All combinations MUST succeed before a candidate is considered compatible
  for merge or release; incomplete or cancelled checks MUST NOT count as success. Repository changes
  MUST expose the check results and document which checks need to be required. The maintainer will
  configure GitHub rulesets or branch protection to enforce merge requirements; custom enforcement
  code and changes to the release workflow are not required by this feature.
- **FR-007**: Contributors MUST be able to select one combination or the full matrix for isolated
  local test runs, using the same environment definitions and acceptance rules as automated verification.
  Selecting a test combination MUST NOT require changing the everyday development environment or
  rewriting the committed development lockfile. Native local runs MUST be available on macOS and
  Windows where the required interpreters and dependencies are available. Local results MUST identify
  their platform; only Linux CI results establish official compatibility. Unavailable combinations
  MUST remain explicit incomplete coverage rather than being silently omitted.
- **FR-008**: Each result MUST identify the requested combination, actual Python, PyTorch and NumPy
  versions, and pass/failure, skip and expected-failure outcomes. Installation or execution failures
  MUST identify the affected combination. Resolved versions MUST match the requested combination.
- **FR-009**: Routine matrix runs MUST select the latest available compatible stable patch releases
  of each named Python and PyTorch minor series at environment resolution time. Re-running a recorded
  combination with its recorded exact dependency versions MUST be possible for diagnosis.
- **FR-010**: NumPy and other dependency constraints MUST be satisfiable in every supported combination
  and consistent with tested installations. This feature MUST NOT imply exhaustive testing of every
  possible dependency patch or NumPy version combination.
- **FR-011**: This change MUST preserve the library source implementation, public signatures,
  algorithms, output conventions and existing state side effects. Compatibility failures that require
  source changes MUST be reported as blockers for this scope rather than masked or silently fixed.
- **FR-012**: Support documentation and release notes MUST state the matrix, patch-selection policy,
  verified platform and CPU scope, commands and prerequisites for local checks, and known limitations.
  They MUST distinguish successful installation from correctness of the existing numerical algorithms.
- **FR-013**: Support-policy changes MUST update declarations, documentation and verification together.
  Adding a new Python or PyTorch minor version MUST require an explicit reviewed policy change.

### Key Entities *(include if feature involves data)*

- **Supported combination**: A named Python minor version and PyTorch minor series within the support
  policy, with the verification platform and build variant stated explicitly.
- **Compatibility result**: Evidence for a candidate revision and supported combination, including
  actual dependency versions, installation outcome, and regression-test outcomes.
- **Known-failure baseline**: Existing documented defects and their strict expected-failure tests;
  this records limitations without asserting that the defects are intended library behaviour.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A release candidate installs, imports and completes regression verification in five of five
  supported combinations, with zero new failures, unexpected passes, or new or unexplained skips.
- **SC-002**: Verification detects a controlled regression in any one selected combination and produces
  an unsuccessful overall result even if every other combination passes.
- **SC-003**: Every compatibility result identifies the candidate and actual runtime dependency versions,
  allowing a contributor to reproduce an individual failure without guessing which versions ran.
- **SC-004**: Contributors can select one combination or all supported combinations using the documented
  instructions on Linux and, where dependencies are available, natively on macOS and Windows. Missing
  prerequisites result in explicit incomplete coverage rather than a false pass; local results
  identify their platform and do not expand official platform support.
- **SC-005**: All supported combinations appear in both the published support table and automated
  verification, with no contradictory support declarations or automatic inclusion of unreviewed minors.
- **SC-006**: Both distribution formats pass their installation checks, and the library source is
  unchanged from the pre-feature revision.

## Assumptions

- Initial support targets PyTorch 2.15 only and its upstream-supported standard CPython versions
  3.11 through 3.15. PyTorch 2.14 and Python 3.10 are excluded. Future expansion requires demonstrated
  need and a reviewed support-policy change; "latest" does not automatically advance to a new minor.
- The initial automated guarantee is Linux x86-64 CPU with standard, GIL-enabled CPython. It does not
  establish macOS, Windows, alternative interpreter, or free-threaded build compatibility. Native
  macOS and Windows test runs provide contributor feedback where dependencies are available; Linux
  CI remains authoritative for the official support claim.
- GPU support, numerical bug fixes, API redesign, new metrics and repair/removal of broken `contrib`
  modules are deferred to follow-up work rather than included in this compatibility update.
  Existing working functionality and documented failures form the baseline; this feature does not
  implement or prescribe the design of those follow-ups.
- This specification defines compatibility and verification outcomes. Selection of the local test
  runner and orchestration tools belongs in the implementation plan; tox has been discussed as a candidate.
- Publishing a release, selecting its final version number, and redesigning the release process are
  deferred to follow-up release work; the deliverable is a release-ready compatibility policy and
  continuing verification. The maintainer owns GitHub merge-protection configuration separately from
  repository-content changes; documentation must identify the required configuration for handoff.
- Verification requires available upstream interpreter and dependency distributions. Availability errors
  must remain visible and must not weaken the support claim.
- Upstream support was checked on 2026-10-30 against the
  [PyTorch release compatibility matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix).
  The [2.15 release announcement](https://pytorch.org/blog/pytorch-2-15-release-blog/)
  provides additional release context. These links inform this fixed matrix; later upstream changes
  do not automatically amend it.
