# loss-landscapes Constitution

## Core Principles

### I. Simplicity and Earned Abstractions

Every implementation MUST use the simplest design that satisfies current, validated requirements.
KISS (Keep It Simple, Stupid) and YAGNI (You Aren't Gonna Need It) are the default decision rules.
Speculative features, configuration, extension points, and framework machinery MUST NOT be added
without a concrete use case in the agreed scope. Abstractions MUST improve the clarity of present
logic or support reuse required by that scope. Existing patterns MUST be preferred when they remain
correct and clear; preserving a legacy pattern is not justification for retaining a defect.
Dependencies MUST provide a concrete benefit that justifies their installation and maintenance cost.
A small scientific library must remain understandable and easy to adopt.

### II. Mathematical Correctness

Numerical features MUST specify their mathematical definition, assumptions, and sampling conventions
before implementation. Direction generation, normalisation, distances, coordinates, and array ordering
MUST have unambiguous meanings. Implementations MUST be checked against analytic examples or
independent mathematical invariants, with tolerances appropriate to the numerical operation and dtype.
A plausible-looking plot is not evidence of correctness. Undefined or degenerate cases MUST have a
documented outcome; they MUST NOT silently produce misleading results. Correctness fixes MUST take
precedence over preserving erroneous numerical behaviour, with their effects disclosed to users.

### III. Explicit Public Contracts

Public APIs MUST remain small, intentional, and usable from ordinary Python research code. Contracts
MUST describe relevant inputs, outputs, shapes, coordinate conventions, and failure modes. They MUST
state whether evaluation changes caller-owned parameters, buffers, training mode, gradient settings,
or random state. Mutation MUST be deliberate and documented; preservation guarantees MUST cover
failure paths as well as successful calls. Stochastic operations MUST expose a documented way to
reproduce results under stated conditions, without promising identical results across unsupported
platform or dependency combinations. Loss and other metrics MUST have an explicit evaluation contract
so users can extend the library without depending on private implementation details.

### IV. Readable, Documented Code

Code MUST follow Python conventions and the repository's established formatting, linting, and naming
rules. Modules MUST present their public purpose clearly, with high-level functionality before
implementation helpers where practical. Public modules, classes, functions, and constructors MUST
have docstrings describing their purpose and applicable contract. Functions MUST remain focused;
complexity MUST NOT be hidden by arbitrary splitting or layers of indirection. Non-trivial numerical
logic MUST include concise explanations of its intent, assumptions, or derivation. Comments that
summarise the purpose or algorithmic step of a block are encouraged: they let readers understand the
implementation at a glance. Comments MUST NOT merely paraphrase an individually obvious statement
without adding context. Documentation MUST distinguish implemented capabilities from proposed or
unfinished work.

### V. Behaviour-Based Verification

Changes to library behaviour MUST include tests of observable contracts, using small models and data
where possible. Numerical tests MUST include independently derived expected results or invariants;
shape checks and snapshots alone are insufficient. Tests MUST minimise mocks and dependence on private
helpers. Tests deeply coupled to implementation details do not provide useful evidence of contract
correctness: they can fail during valid refactoring while missing incorrect public behaviour. Tests
MUST remain valid when the implementation changes without changing its contract. A valid failing test
MUST drive investigation of the implementation; expectations MUST NOT
be weakened simply to make a change pass. Tests MAY change when requirements change, an expectation
is wrong, or a test depends on an implementation detail, with the reason recorded in the change.
Bug fixes MUST include regression coverage. Integration tests MUST exercise real interactions when a
change affects sampling, model handling, and metrics together, or package installation and imports.

## Engineering Standards

The project's core scope is a reusable Python library for evaluating metrics in low-dimensional
subspaces of neural-network parameter space. PyTorch is the current model framework. Plotting systems,
additional frameworks, and application infrastructure MUST require an explicit feature justification;
they MUST NOT enter the core merely to support a demonstration.

Computation MUST avoid unnecessary model copies, parameter transfers, and retained autograd graphs.
Performance changes MUST preserve numerical and state contracts, and optimisation complexity MUST be
supported by measurements on a relevant workload. Features whose cost grows with model size or sample
count MUST document material time or memory constraints.

Supported Python and dependency versions MUST be explicit and consistent with package metadata and
CI. Runtime imports MUST be backed by declared dependencies or documented, deliberately optional
features. Changes to public signatures, numerical semantics, output structure, or support policy MUST
be treated as compatibility decisions and documented in release notes. Breaking releases MUST include
migration guidance and use appropriate semantic versioning. Known incorrect results MUST NOT become
compatibility requirements.

## Delivery Workflow

Substantive features, API changes, and numerical redesigns MUST begin with a focused specification of
intended behaviour and acceptance criteria. Plans MUST identify mathematical assumptions, state
handling, compatibility effects, and verification relevant to the change. Small, scoped fixes and
editorial changes MAY use a concise issue or PR description rather than a full feature workflow.
Specifications, plans, tasks, examples, and implementation MUST remain consistent as decisions evolve.

Before merge, each change MUST be reviewed against this constitution and pass applicable repository
checks. Behaviour changes require relevant unit and integration evidence; packaging changes require
build and installation verification. The default test suite MUST remain fast and CPU-capable;
expensive or hardware-specific checks MUST be separate and identified where needed. Documentation-only
changes do not require unrelated numerical tests.

Existing defects MUST remain explicitly recorded until repaired or removed. Strict expected-failure
markers MUST identify the defect and MUST be removed when the corresponding behaviour is fixed.
Infrastructure-only changes MUST preserve the library behaviour and dependency contract specified by
their scope; deliberate changes to those contracts belong in explicitly scoped follow-up work.
Adopting this constitution does not assert that the legacy implementation already complies or require
an unrelated rewrite in every change. New and modified behaviour MUST comply, and relevant outstanding
non-compliance MUST be identified during review.

## Governance

This constitution defines the project's engineering policy. Specifications, plans, tasks, reviews,
and automation MUST follow it. Conflicts MUST be resolved explicitly rather than silently bypassing a
principle. The maintainer approves amendments and exceptions. Exceptions MUST record their scope,
rationale, and remediation or review condition; they MUST NOT silently become permanent policy.

Amendments MUST state the affected rules, the reason, and the impact on existing work. Governance
versioning uses MAJOR for incompatible principle removals or redefinitions, MINOR for new principles
or materially expanded guidance, and PATCH for clarifications without a change in meaning. Compliance
MUST be checked during planning and pre-merge review. Operational commands and tool versions belong in
contributor documentation and configuration rather than being duplicated here.

**Version**: 1.0.1 | **Ratified**: 2026-10-16 | **Last Amended**: 2026-10-16
