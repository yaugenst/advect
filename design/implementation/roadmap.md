# Implementation Status

Advect is the numerical-transformation package described by the
[core requirements](../requirements.md). Its implemented product
boundary is local array-program differentiation and explicit durable staging.
Workflow orchestration, object storage, remote execution, and the former
graph-first application surface are independent layers.

The concrete NumPy ownership refactor is complete. NumPy owns its foreign
protocol contract, the provider-neutral Array API frontend remains portable,
and both meet at canonical operation emission. The completed implementation
preserves the qualified numerical surface while removing adapter state,
identity repair, duplicated support classification, and unowned benchmark
products.

## Runtime shape

One native SSA substrate has two owners:

- `DynamicTape` owns a concrete, invocation-local trace for ordinary
  differentiation;
- `GraphBuilder` and immutable `GraphStore` own an abstract, durable staged
  program.

Python owns tracers, pytrees, provider callbacks, derivative rules, and API
orchestration. Rust owns compact topology, traversal, staged validation,
optimization schedules, execution planning, and lifetime-aware release.
Dynamic differentiation never constructs or optimizes a durable graph.

## Dynamic differentiation

The implemented dynamic surface includes:

- `grad`, `value_and_grad`, `linearize`, `jvp`, and one-shot `vjp`;
- structured Jacobians, Hessians, Hessian-vector products, and Hessian
  diagonals;
- positional and named multi-argument selection;
- built-in, registered, and inherited-protocol custom pytrees;
- real-linear complex differentiation;
- bounded multi-seed traversal for Jacobian and Hessian assembly;
- reverse-mode selection for transpose-only Jacobian primitives;
- manual dynamic checkpointing;
- matrix-free implicit roots.

All differentiable leaves use the array tracer. Selected real Python scalar
primals are boundary-lifted to zero-dimensional `float64` arrays, and their
rank-zero derivative results are returned as Python scalars. The former
`TracedValue`, scalar-operation facade, and parallel scalar derivative rules
are absent.

Ordinary NumPy remains the default frontend. Its standard `like=` parameter
preserves dependencies when traced code assembles direct or nested arrays:
`np.array(values, like=x)`. Explicit `advect.array` and `advect.asarray` remain
provider-neutral alternatives, while `advect.numpy` is a thin secondary
namespace that delegates ordinary attributes directly to NumPy. Traced
`.item()` preserves scalar extraction as a rank-zero array. `advect.is_traced`
and the dynamic-only `advect.stop_gradient` are the bounded
application-integration escape hatches. The sole process-visible NumPy patch is
the frontend's scoped ambient-RNG tripwire during abstract staging. A lock
shares it across overlapping staging scopes, calls fail only in active staging
contexts, and the last scope restores the original NumPy attributes.

## Mutation

The NumPy frontend functionalizes supported mutation into pure SSA values:

- owned local in-place operators;
- basic indexed set and additive updates;
- supported `out=` calls;
- direct named basic-slice mutation;
- view epochs and stale-view diagnostics.

Input mutation, advanced-index augmented assignment, arbitrary transformed
views, and overlap inference remain explicit errors. Mutation is never an IR
effect. Staged buffer donation is an internal physical optimization and cannot
change these semantics.

## Staging

`stage(f, *examples, kw_specs=...)` infers, or
`stage(f, specs=..., kw_specs=...)` declares, exactly one explicit signature.
`StagedProgram` owns:

- one immutable `GraphStore`;
- one prebound execution plan;
- one input and output signature;
- one constant manifest;
- one fixed `DCE -> simplify -> CSE` report;
- one compilation duration.

There is no lazy tracing or multi-signature cache inside `StagedProgram`.
Different signatures are different program objects. `grad(program)`,
`value_and_grad(program)`, and `vjp_program(program)` compile singular staged
derivative programs that execute without a dynamic tape.

A zero-leaf output pytree is a valid zero-output program. This admits standard
operations such as `unstack` on an empty axis without adding sentinel nodes or
special provenance metadata. The graph retains its input signature, and the
empty output structure survives serialization and restoration.

The detached envelope is `advect.ssa-program` version 2 and contains one program.
Its nested Rust graph remains the versioned portable graph artifact.

## Primitive system and abstract evaluation

One `@primitive` definition owns its implementation, abstract evaluation,
static and nondifferentiable arguments, JVP, and any justified explicit
transpose or residual lifecycle. Advect owns graph schema revisions; authors do
not maintain a provider dispatch table or schema number.

Abstract evaluation is organized as plain domain tables for elementwise,
creation, reductions, shape, indexing, contractions, linear algebra, FFT, and
signal operations. Shared dtype, axis, and shape logic has one focused helper
layer. There is no generated dispatcher or second schema language.

The primitive conformance inventory is the exhaustiveness control. It
mechanically joins registered operations to semantic classification, abstract
or dynamic-only staging status, derivative rules, and executable invocation
cases. Default and thorough Hypothesis profiles exercise the same complete
inventory at different depths.

Derivative registries emit canonical operation IDs without NumPy-name
intermediates or bootstrap callbacks. Abstract semantics define staged
operations; JVP and non-differentiability records contribute the `OpDef`
fragment for dynamic-only operations. Independent semantic, frontend,
evaluator, and executable-case evidence rejects unknown derivative IDs in CI,
avoiding a second maintained inventory of dynamic-only names.

## Providers

NumPy is the required, deterministically registered frontend. Portable Array
API 2022.12, 2023.12, and 2024.12 execution is exercised by the automated CPU
evidence lane. Single-device CuPy through the built-in compatibility fallback
uses a reproducible manual GPU gate for each profile; the repository does not
pretend that GitHub's CPU runners exercise CUDA.

Advect does not scan package entry points, and an unrelated installed
distribution cannot mutate input dispatch. Frameworks with their own autodiff
system are not Advect array providers; their concrete host-composition path is
the explicit first-order `advect.interop` VJP bridge.

## Scientific integrations

The single Python distribution includes bounded optional modules:

- `advect.scipy` supplies 15 SciPy 1.18 special-function call contracts, 27
  image-filter and morphology contracts, and callback factories for implicit
  root and GMRES solves;
- `advect.xarray` registers floating- and complex-valued labeled containers as
  dynamic pytrees.

The base distribution separately configures its private `array-api-compat`
fallback when `advect` is imported.

These modules reuse the same primitive, pytree, staging, and linear-map
contracts. They do not add derivative engines, solver IR, provider-specific
graph formats, or separate distributions.

## Repository evidence

Correctness is established by the ordinary suite, the registry-wide
property-based conformance suite, staged serialization round trips, pure-Rust
graph fixtures, provider qualification, and wheel smoke tests.

Reproducible benchmark and qualification programs are versioned. Generated
JSON reports are ignored by Git and uploaded by the manual
`Numerical Evidence` workflow or release process. Runtime registries and tests,
not generated reports, remain authoritative.

The compact installed NumPy support declaration owns public catalog policy.
Independent test-only invocation cases prove every declared lifetime and
derivative role. Exact bidirectional checks reject drift in either direction;
removing a case fails CI until a maintainer explicitly changes the declaration.
Cases do not generate an installed manifest or a second support schema.

The Array API has 154 direct generated `_FUNCTION_SPECS` bindings. Its public
catalog has 169 rows after adding 10 composite and 5 compile-time metadata
functions. Preservation audits report both inventories separately.

## Simplification outcome

The implementation meets both blocking code-reduction gates against baseline
`b6e05e6c`: installed runtime Python falls from 58,425 to 57,421 physical lines,
a net reduction of 1,004. The frontend-boundary, canonical-identity, and
normalization cohort falls from 57,552 lines at accepted Work package 1
(`d65f9da`) to 57,531 at the Work package 4 endpoint (`ce482f0`), a net
reduction of 21. These are endpoint counts; Git's rename similarity does not
change either result.

The normalization review found that required positional binding is already
shared through `advect.numpy._signature.normalize_required_positionals`, while
concrete and abstract constructor dispatch already share the implementations in
`advect.numpy._constructors`. No additional layer is needed for either family.
Controlled reductions and `out=` remain with their lifetime owners because the
dynamic paths consume concrete payloads, destinations, and recorder state while
the abstract paths derive graph shape, dtype, layout, and replacement semantics
without payloads. Sharing those paths would require callbacks or metadata that
represent the lifetime split instead of deleting a decision.

Linear-algebra handling remains separate for the same reason: mode-dependent
operation identity, output arity, and result restoration would require a
descriptor plus dynamic-versus-abstract branching. The normalization boundary
therefore stays explicit unless a future call family has identical inputs,
normalized outputs, errors, and semantics in both lifetimes and extraction
removes more production decision code than it adds.

The public preservation audit compares exact baseline and candidate wheels
through NumPy 2.4 and records the candidate inventory on NumPy 2.5. It finds no
removed callable, array-function handler, or ufunc call in a same-minor
comparison. Candidate NumPy row totals are 390, 393, 395, 395, 393, and 392 on
NumPy 2.0.2 through 2.5.2; the changing count follows NumPy's installed surface.
Root exports (44), direct generated Array API bindings (154), public Array API
rows (169), and built-in registry IDs (224) are exact. The nine
`numpy.lib.scimath` rows remain dynamic-only, and `register_backend` is the sole
intentional public-looking symbol removal because it was a mutable test reset,
not a backend extension contract.

Four catalog lifetime changes are explicit support corrections rather than
runtime removals. `numpy.round`, `numpy.linalg.eig`, and
`numpy.linalg.eigvals` retain documented restricted staged forms while their
whole-callable rows are conservative. `numpy.linspace` with traced endpoints
was dynamic-only in the baseline implementation as well; its old staged flags
and a duplicate private classifier were false claims and are gone.

The whole repository remains net-positive because preservation and independent
evidence are retained. Final attribution is recorded by area rather than used
as a deletion target:

| Area | Net lines versus baseline |
|---|---:|
| Runtime Python | -1,004 |
| Production Rust | -5 |
| Tests | +1,829 |
| Scripts | +700 |
| Documentation and guidance | +719 |
| CI | +54 |
| Dependencies | -5 |
| Whole repository | **+2,288** |

Qualification uses exact release wheels and retained reports. Automated lanes
cover default and thorough conformance, NumPy 2.0-2.5, all three Array API
revisions through dynamic, staged, and restored execution, native and browser
wheels, and documentation. CPU regression and memory acceptance remain manual
pre-tag evidence until the release workflow has a stable non-self reference.
CuPy is not a release claim; making it one requires retained evidence for the
exact wheel, revision, device, driver, CUDA runtime, and CuPy version.

## Intentional exclusions

Advect does not currently own:

- workflow orchestration, restart, or object storage;
- a general compiler, kernel generator, fusion system, or backend lowering;
- a public `vmap` or public graph-rewrite API;
- automatic checkpoint placement;
- arbitrary mutation through aliased views;
- dense complex Hessians;
- a PyTorch provider lane;
- automatic provider discovery;
- compatibility shims for the former product or artifact shapes.

Future additions require a concrete numerical user journey and must reuse the
singular tracer, primitive, SSA, and staged-program boundaries.
