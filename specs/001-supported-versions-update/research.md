# Research: Supported Versions and Compatibility Verification

Research date: 2026-10-30. This document records design decisions, not completed compatibility tests.

## 1. Support boundaries and dependency resolution

**Decision:** Declare `requires-python = ">=3.11,<3.16"` and runtime dependencies
`torch>=2.15,<2.16` and `numpy`. Retain the existing NumPy requirement unless installation or
runtime evidence requires a constraint. Update classifiers to Python 3.11–3.15, Ruff to `py311`,
and mypy's language baseline to 3.11. Remove the obsolete uv Python-range workaround and move
PyTorch out of the test dependency group. Regenerate `uv.lock`; retain `.python-version` at 3.12.

**Rationale:** These closed Python/PyTorch boundaries implement the agreed support policy and
exclude unreviewed minors. No inspected NumPy API establishes a new minimum. Latest compatible
resolution matters: current NumPy 2.5.4 requires Python >=3.12, whereas 2.4.6 supports 3.11.
A global NumPy >=2.5 floor would therefore invalidate one required environment.

**Alternatives considered:** Supporting older PyTorch releases was explicitly rejected. Inventing
a NumPy floor or upper bound without evidence would expand the policy change unnecessarily.

**Evidence:** [PyTorch release matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix),
[torch 2.15.0 metadata](https://pypi.org/project/torch/2.15.0/),
[NumPy 2.5.4](https://pypi.org/project/numpy/2.5.4/),
[NumPy 2.4.6](https://pypi.org/project/numpy/2.4.6/).

## 2. One matrix shared by local runs and CI

**Decision:** Use native TOML tox configuration in `pyproject.toml`, with five explicit default
environments: `py311-torch215`, `py312-torch215`, `py313-torch215`, `py314-torch215`,
`py315-torch215`. Add a `compatibility` dependency group containing `tox>=4.28` and `tox-uv>=1`,
also included by the development group. Explicitly select `uv-venv-runner`, `package = "wheel"`,
`uv_resolution = "highest"`, environment recreation, and failure on missing interpreters.
Use the existing `test` dependency group through tox's `dependency_groups` support.

**Rationale:** tox defines versioned environments and exit status; uv supplies isolated, fast
installation and platform-aware PyTorch CPU selection. The non-lock runner resolves test dependencies
independently of `uv.lock`. The driver tools can remain locked without locking the tested libraries.
The tox minimum covers native TOML, dependency groups and constraints applied to package dependencies.

**Alternatives considered:** Plain tox/pip is viable, but needs additional index handling for CPU
PyTorch. The tox-uv lock runner would defeat fresh compatibility resolution. A bespoke matrix runner
or separate local/CI dependency definitions would introduce avoidable maintenance.

**Evidence:** [tox configuration](https://tox.wiki/en/stable/reference/config.html),
[tox CLI](https://tox.wiki/en/stable/reference/cli.html),
[tox-uv runners](https://github.com/tox-dev/tox-uv/blob/main/README.md).

## 3. CPU selection and native developer platforms

**Decision:** Set `UV_TORCH_BACKEND=cpu` for test installation; do not assume `tool.uv.sources`
affects the uv pip interface used by tox. Retain the existing explicit Linux CPU source for the
separate development environment. Verify CPU-only CUDA/ROCm build information in Linux results.
Use upstream macOS/Windows wheels for native local feedback when available, with no fallback
source build or emulation requirement.

**Rationale:** The official CPU index lists standard CPython 3.11–3.15 Linux x86-64 wheels for
2.15.0, tagged manylinux_2_28. It also has Windows amd64 and macOS 14+ Apple Silicon wheels.
Intel macOS is not covered by those wheels. Availability must be reported accurately rather than
silently reducing the requested matrix.

**Alternatives considered:** Installing the default Linux CUDA distribution wastes downloads;
manually installing PyTorch into a shared development environment compromises isolation.

**Evidence:** [Official CPU wheel index](https://download.pytorch.org/whl/cpu/torch/),
[uv PyTorch backend selection](https://docs.astral.sh/uv/guides/integration/pytorch/).

## 4. Latest patches versus exact replay

**Decision:** Provision test interpreters separately under `.tox/python`. Routine runs use current
uv and `uv python install --install-dir .tox/python --upgrade --no-bin --no-registry` for the requested
minors. Configure the directory and interpreter selection in tox/tox-uv rather than requiring manual
shell exports; verify those settings reach environment creation. This avoids replacing development executables or registering
them globally. Disable automatic interpreter downloads during tox execution so missing prerequisites
remain failures. Recreate test environments on every compatibility run; caches may reuse downloads,
but not an old installed dependency set.

Record exact interpreter/platform details, driver versions, installed dependencies and the candidate
artifact using ordinary tox/pytest logs and distribution artifacts.
Exact-version reproduction uses the retained wheel, original platform/architecture, interpreter and tox constraints
covering both test and runtime dependencies. Native tox overrides select the interpreter and constraints;
no replay service or custom package store is needed.

**Rationale:** tox chooses an interpreter but does not independently guarantee the latest patch.
A cached environment can otherwise conceal new patches. Reproducibility requires recording the actual
resolution, not asserting that every future run will resolve identically.

**Alternatives considered:** Reusing the development lockfile prevents patch discovery; globally
upgrading development interpreters violates the separation agreed in the spec. Exact-patch replay
is explicitly diagnostic and does not masquerade as a fresh latest-patch certification run.

**Evidence:** [uv Python management](https://docs.astral.sh/uv/guides/install-python/),
[tox constraints and overrides](https://tox.wiki/en/stable/reference/config.html).
Local CLI help additionally confirms `--install-dir`, `--upgrade`, `--no-bin`, and `--no-registry`.

## 5. Installed artifacts and test outcome policy

**Decision:** Build one wheel and sdist in the CI preparation/build job. Pass the same wheel to
all five tox jobs using `--installpkg`; validate the actual sdist in a separate `sdist` tox environment
on Python 3.12. Local matrix runs may build a shared wheel via tox. Validate installed import location
and package metadata before testing. Keep library source byte-for-byte unchanged.

Retain strict xfails and their diagnoses. The present suite has no ordinary skip markers, so reject
ordinary skips in compatibility mode through a small pytest hook in `tests/conftest.py`; distinguish
xfail reports by pytest's `wasxfail` flag. Future intentional skips require an explicit reviewed policy
change. Do not freeze a permanent list/count of xfails: later numerical fixes must be able to remove them.
Use standard pytest JUnit reports, tox result JSON/logs and dependency listings for results. Add a
normal pytest report header for versions and platform, not a custom reporting format or replay flag.

**Alternatives considered:** A custom `environment.json` report, reporting script and replay flag
are unnecessary conventions and were rejected during review. Editable testing hides distribution problems. Plain pytest allows skips
to pass, so it cannot alone meet the specification. A general test-results dashboard or merge gate
would duplicate existing tools.

**Evidence:** Existing `tests/`, [pytest skipping and xfail](https://docs.pytest.org/en/stable/how-to/skipping.html),
[pytest hook reference](https://docs.pytest.org/en/stable/reference/reference.html).

## 6. CI and repository configuration ownership

**Decision:** Derive the five CI matrix entries from tox's default environment list, using a short
workflow-local conversion to GitHub matrix JSON. Use `fail-fast: false`, normal job exit status, and
stable per-environment check names. Always upload available reports/logs; use normal GitHub artifact
retention. Keep lint/build checks. Document required check names for the maintainer to configure.
The release workflow is unchanged; release readiness is a maintainer check of the candidate revision.

**Rationale:** This preserves one executable matrix and provides clear failures without custom
merge enforcement. A preparation failure blocks dependent jobs; cancelled or missing jobs are not
positive compatibility evidence.

**Alternatives considered:** A hardcoded second matrix can drift. Custom aggregate enforcement,
automatically editing GitHub rules, and release redesign are outside the agreed scope.

**Evidence:** Existing `.github/workflows/ci.yml`,
[GitHub matrix jobs](https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs),
[required status checks](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches).

## Research conclusion

No unresolved design questions remain. Wheel listings and upstream metadata establish feasibility,
not runtime compatibility. Implementation must still run the matrix and treat any source-level
compatibility failure as a scope blocker. Tool option interoperability, exact replay, CPU backend
selection, and interpreter isolation must be exercised in validation before claiming completion.
