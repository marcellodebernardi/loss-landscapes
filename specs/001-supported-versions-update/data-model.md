# Data Model: Compatibility Verification

These are transient configuration and evidence records, not runtime library objects or database tables.

## Supported combination

- **Identity:** tox environment name, e.g. `py311-torch215`.
- **Attributes:** Python minor, PyTorch minor series, standard CPython/GIL build, CPU execution.
- **Authority:** five explicit default environments in project tox configuration.
- **Validation:** Python 3.11–3.15 only; PyTorch 2.15 only. Linux x86-64 establishes official support;
  native macOS/Windows results identify their actual platform and remain developer feedback.
- **Lifecycle:** proposed policy change → reviewed configuration/docs update → verified matrix.
  Adding a minor is a deliberate policy change, not automatic upstream discovery.

The non-default `sdist` validation environment uses Python 3.12/PyTorch 2.15 and is not a sixth
support combination.

## Candidate distribution

- **Identity:** source revision, distribution filename, version and SHA-256.
- **Attributes:** wheel or sdist; package metadata and declared dependency boundaries.
- **Relationships:** the same CI wheel is consumed by all five compatibility results. The sdist
  shares the revision/version and is verified separately on one supported combination.
- **Validation:** candidate imports resolve inside the isolated environment, not `src/` or the
  development environment; source package files remain unchanged from the pre-feature baseline.

## Compatibility result

- **Identity:** candidate + environment + run/attempt.
- **Requested configuration:** environment name, intended minor versions and any explicit interpreter
  or dependency constraints visible in the tox invocation/configuration.
- **Resolved environment:** exact Python version/build/executable, operating system/architecture,
  exact PyTorch and NumPy versions, CPU/CUDA/ROCm build indicators, installed package location,
  driver tool versions, and installed dependency pins excluding the candidate artifact reference.
- **Evidence:** ordinary tox result JSON/setup logs, installed-dependency listing, pytest header and
  JUnit reports. CI records the revision and retains the original distributions. No custom report
  schema or mandatory dependency-pin file layout. Preserve installer logs if setup prevents testing.
- **Outcomes:** environment setup, installation, tests, ordinary skips, xfails, unexpected passes,
  final process exit status and any incomplete/cancelled execution.
- **Validation:** a complete successful result requires the requested environment, installed candidate,
  passing tests, zero ordinary skips and zero unexpected passes. A recorded known xfail is allowed.

**State transitions:** requested → provisioning → installation → environment validation → testing
→ success/failure. Any earlier failure terminates unsuccessfully with whatever evidence is available.
Cancellation or an unexecuted request is incomplete, never success. Exact-version reproduction uses
explicit tox overrides, visible in the invocation; it is not a routine latest-patch verification run.

## Known-failure baseline

Existing strict xfail markers and their diagnoses remain the record of numerical/legacy defects.
They are not duplicated into a permanent external list. Ordinary skip baseline is currently empty;
compatibility-mode hooks reject ordinary skips. Later approved functionality fixes may remove xfails
without changing the meaning of matrix success.

## Replay record

The retained artifact, interpreter patch/platform, recorded dependency listing and tool versions
provide reproduction inputs. Derive ordinary dependency constraints from that listing when needed. Constraints apply to test and runtime dependencies, while the original
candidate wheel is installed explicitly. Replay requires matching platform/build availability;
Linux CPU pins need not install on macOS. Missing historical downloads remain explicit failures.
