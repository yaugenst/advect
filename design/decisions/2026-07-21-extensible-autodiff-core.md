# ADR: Extensible Autodiff Core Reset

**Date:** 2026-07-21
**Status:** Accepted
**Native storage amendment:** 2026-07-24

## Context

The original Axon implementation proved that a reusable native graph can execute differentiated programs
quickly, but its default dynamic path became a durable-graph construction path.
The implementation also accumulated several overlapping concepts:

- storage mutation and version validation in the compute IR;
- dynamic and static behavior selected through `cache=` arguments;
- JVP rules, VJP rules, VJP synthesis, and special higher-order orchestration;
- overlapping primitive, custom-JVP, and custom-VJP authoring paths;
- NumPy protocol tracing beside partial Array API provider plumbing.

Advect has no external users. Compatibility is therefore less valuable than a
small, coherent model. This decision resets the product around an extensible
autodiff core and removes features that do not earn their complexity.

## Decision

### 1. Five core concepts

The runtime is organized around only five semantic concepts:

1. Primitive definition: one operation and its implementation, abstract
   evaluation, and derivative rules.
2. `Tracer`: a value participating in one concrete or abstract trace.
3. `DynamicTape`: the native owner of one append-only concrete SSA
   linearization.
4. `LinearMap`: a real-linear derivative program with application and adjoint
   application.
5. `Pytree`: structure carried around numerical and static leaves.

The required Rust extension supplies one internal structural substrate,
`RawArena`. `DynamicTape` owns it for one concrete invocation.
`GraphBuilder` owns it during abstract construction, and `GraphStore` owns the
immutable, serializable staged result. Python owns mutable tracer wrappers,
callbacks, pytrees, and API orchestration, but no graph or tape data structure.

`RawArena` is storage shared by two lifetimes, not another user-facing semantic
concept. `GraphStore` is never constructed and canonicalized on a dynamic
derivative call.

### 2. Concrete dynamic tracing and abstract staging

`grad(f)`, `vjp(f)`, and `jvp(f)` use concrete primals by default. Python
control flow and data-dependent shapes resolve from the values observed during
that call. The resulting native `DynamicTape` uses `RawArena` positions as node
identities. Its values, literals, residuals, operand layouts, and derivative
metadata are invocation-local side tables. Rust owns arena traversal and
derivative slots; Python-authored derivative rules remain traceable and
reentrant.

A dynamic tape does not compute durable IDs, structural fingerprints, hashes,
serialization metadata, or optimization plans. It never enters the staged
optimizer.

`stage(f, specs=..., kw_specs=...)` traces one explicit signature with abstract
values. Value-dependent truth, iteration, shapes, and ambient state fail with
an actionable error. The trace records directly into native `GraphBuilder`.
Finishing construction validates and canonicalizes it, then always applies the
fixed `DCE -> simplify -> CSE` pipeline before producing `GraphStore`. Custom
and remote primitives are optimization barriers.

The staged signature owns one immutable native
`GraphExecutionPlan`, whose structure and evaluator bindings are resolved once.
Calls reuse that plan and allocate a fresh dense value table plus temporary
operand tuples. Native bulk traversal drives execution without rebuilding
Python graph nodes. A `StagedProgram` owns that one signature and one graph;
its constants, compile time, graph, and optimization report are directly
inspectable.

Dynamic tapes and staged programs share `RawArena`, stable primitive
name/schema semantics, and compact node semantics, not lifetime-specific
metadata or construction cost. Each arena assigns dense local operation IDs;
durable stores carry the table that maps them back to stable schemas.

`cache="none"` and `cache="static"` are removed. Dynamic transforms are the
default; staging is an explicit operation. There is no public `optimize=`
choice, and loaded artifacts are not silently reoptimized.

### 3. Source mutation is tracer functionalization

Mutation is never an IR effect. Frontends may opt into this capability; the
initial complete implementation is the NumPy frontend, while a generic Array
API provider may reject source mutation. A participating concrete wrapper
directly holds the current immutable SSA node ID, concrete value, ownership
bit, and epoch. Source operations such as `x += y`, `x[i] = y`, and supported
`out=` calls emit pure value-producing nodes and replace the current ID and
value. Python aliases to the same wrapper observe the replacement. Abstract
staging implements the same behavior with private wrapper-owned state; there
is no IR cell or shared cross-runtime `TracerCell` type.

Input wrappers are not writable. Input mutation fails at the operation with a
rewrite such as `x = x + y`. `copy()` creates an owned wrapper.

An aliasing view stores its root wrapper and creation epoch; an indexed view
also retains the structural index needed for augmented assignment matching.
Using a view after its root wrapper has advanced raises `StaleViewError`. This
permits normal stencil code:

```python
out = u.copy()
lap = out[2:] - 2 * out[1:-1] + out[:-2]
out[1:-1] += dt * lap
```

while rejecting behavior whose result would diverge from NumPy:

```python
out = u.copy()
v = out[2:]
out += 1
use(v)  # stale view
```

Aliasing is classified conservatively by the semantic frontend, not by the
provider's observed memory layout. Basic slicing, reshape, transpose, and
ravel are aliases in the NumPy profile. Whole-wrapper epochs deliberately
reject some disjoint-region programs; overlap analysis is not part of the core.

#### Augmented assignment to a subscript

Python evaluates `x[key] += value` as getitem, in-place operation on the
temporary result, then setitem. A view cannot distinguish that syntax from a
named `view += value` inside `__iadd__`.

For a direct basic view of an owned root, `__iadd__` emits the functional root
`index_update` immediately, advances the root epoch, refreshes that view, and
returns it. The trace context retains one completed acknowledgement so the
setitem generated for `x[key] += value` can be consumed as a no-op. With a
named `view += value`, the next operation or trace finalization discards the
acknowledgement. Dynamic basic-slice `+=` still lowers directly to
`index_update(mode="add")`; its destination getitem remains lazy.

Matching uses returned-view identity, destination-wrapper identity and epoch,
and a structural normalized index tuple. Python index-object identity is never
assumed. Sibling views remain stale after the root update.
Setitem on a view, including `x[i][j] += value`, errors and suggests a single
index such as `x[i, j] += value`.

Supported `out=` calls update an owned traced destination wrapper. Masked calls
lower masked-off positions from the old destination value. Raw destinations,
input destinations, view destinations, and unsupported alias patterns error.

### 4. Physical reuse is an executor decision

The semantic core has no ownership ledger and makes no donation promise. The
staged execution plan may reuse an internally owned concrete buffer only after
proving last use, writable storage, compatible metadata, and the absence of a
live alias. Inputs, constants, outputs, and dynamic tape values are not
donated. The logical program remains immutable regardless of physical reuse.

The concrete payload is private. Array coercion, array-interface export, and
DLPack export from a tracer raise instead of detaching computation. Any
explicit debugging escape marks the value permanently borrowed and therefore
not donatable.

### 5. Derivatives are real-linear maps

Linearization is the derivative primitive. A primitive supplies a traceable
JVP rule and, when automatic transposition is unavailable, an explicit
transpose rule. Tangent provenance is represented in the derivative program;
linearity is validated structurally rather than trusted from a user flag.

`vjp` applies the real adjoint of the linearized program. `grad`, nested
derivatives, and Hessian-vector products are compositions of `linearize`, JVP,
and adjoint application. Higher-order APIs do not select modes by catching
runtime errors.

Multi-seed application is a property of `LinearMap`; it is not a general
`vmap`. Exact Hessian diagonals retain their linear-in-input-dimension seed
cost. Stochastic diagonal estimation is a separate, explicitly approximate
API if added.

Explicit transpose rules support higher-order differentiation only when their
rule bodies trace. Capability errors name the primitive and missing rule.

### 6. Complex convention

All derivative maps are real-linear. For a real scalar loss
`L: complex[n] -> real`, Advect returns

```text
g = dL/dx + i dL/dy = 2 dL/d(conj(z))
dL = Re(vdot(g, dz))
```

Thus `grad(lambda z: abs(z)**2)(z) == 2*z`, and `z -= rate * g` is a descent
step. Adjoint transposition uses `Re(vdot(a, b))`. Rules for `real`, `imag`,
`conj`, `abs`, complex products, contractions, and FFTs follow that convention.

`grad` requires a real scalar output. There is no `holomorphic=True` escape in
the initial API; complex-output differentiation uses `linearize`, `jvp`, or
`vjp`. Dense Hessians for complex inputs are rejected until an explicit real
block or block-Wirtinger return type exists. Complex HVPs remain supported.

### 7. One primitive authoring model

An `@primitive` definition owns:

- one concrete implementation and a default name inferred from it;
- abstract output evaluation;
- explicit static/nondifferentiable arguments;
- a JVP rule;
- an optional transpose rule;
- an optional invocation-local residual contract.

The implementation signature is the one call contract, including ordinary
Python defaults. Concrete execution does not prevalidate those defaults.
Tracing validates and snapshots actual static values through the closed
artifact codec. Advect owns graph schema revisions; authors neither declare
versions nor register string-keyed providers.

Rule decorators are methods on the callable handle returned by `@primitive`.
Ordinary users do not declare capability dataclasses. Transforms use installed
rules directly, and author checks report failures without changing runtime
state.

Residual-capable implementations return
`PrimitiveResult(output, residual, release=None)`. The residual is a
process-local sidecar owned by `DynamicTape` during differentiation, never part
of `RawArena`, `GraphStore`, node attributes, output pytrees, or serialization.
It is passed only to the exact matching transpose and released
deterministically. Plain calls, including plain staged replay, release an
unused residual before returning. Staged replay under dynamic autodiff records
one atomic custom node and transfers the residual to the enclosing tape. Such
primitives are first-order-only until residual dependence has an explicit
derivative representation.

The optional `advect.testing` author kit includes concrete/abstract parity,
finite-difference JVP/VJP checks, real-adjoint dot-product checks, complex
checks, higher-order checks for traceable rules, and stage/serialization round
trips.

### 8. Array API core and NumPy frontend

The backend-neutral traced array implements `__array_namespace__` and returns
a tracing namespace that emits canonical `array.*` primitives. Concrete
execution delegates to the selected provider.

Canonical array-family semantics live in focused abstract and autodiff domain
modules. Built-in registry bootstrap joins their schemas, evaluators, JVPs,
VJPs, and output arities into complete `OpDef` records once. Frontends lower to
those records; they do not introduce placeholder operations, and transforms do
not rescan the registry after tracing. NumPy and conforming Array API namespaces
are resolved directly from runtime values.

Advect pins an Array API version and a deliberately finite operation surface
generated from its pinned signatures and primitive schemas.
Concrete promotion, weak Python scalar handling, dtype, device, and result
shape are delegated to a provider implementing that version, then recorded and
validated by the trace. Advect does not duplicate the provider's array type
system. Important weak-scalar cases are differential tested. Operations absent
from the binder and abstract-rule table fail immediately. Data-dependent-shape
operations need an explicit dynamic rule and are rejected from abstract staging
unless they expose static output semantics.

The durable `advect-array-1` semantic profile admits exactly two provider
contracts: Array API 2024.12, or NumPy 2.3 with NEP-50 promotion semantics.
Unversioned generic namespaces and other NumPy major versions reject before
execution. Advect requests `api_version="2024.12"` explicitly from every runtime
array, validates every input rather than selecting the first namespace, and
rejects mixed-provider calls. Per-node result specifications remain an
additional runtime check; call specifications also enforce weak-scalar status
and any declared device constraint.

NumPy remains a first-class frontend through `__array_ufunc__` and
`__array_function__`. It has explicit alias/mutation rules and a differential
conformance suite. The NumPy adapter is not a deprecated route to the Array API
frontend, but its derivatives use the same canonical rules and runtime namespace
resolution as every other array frontend.

Unknown operations fail rather than coercing a tracer. SciPy and other
libraries integrate through primitives, not implicit `np.asarray` conversion.

### 9. Constants and effects during staging

Concrete operands encountered during abstract tracing become constant nodes.
They do not require a special constant wrapper. Every staged artifact reports
each constant's origin category, source location, shape, dtype, byte size, and
digest, plus aggregate count and bytes. There is no separate constant-marking
API in the core.

Known ambient RNG and stateful provider entry points fail during staging.
Staged randomness uses explicit state or keys. Since arbitrary Python effects
cannot all be detected, the enforceable invariant is that concrete capture is
never unattributed or uninspectable.

Static argument and pytree metadata are copied through the closed artifact
codec before tracing. Constant identity deduplication retains each source object
for the complete compile, so Python object-ID reuse cannot merge distinct
temporaries.

### 10. Diagnostics and trace lifetime

Shipping-mode diagnostics are included in latency benchmarks. Lightweight
locations are always captured for views, pending updates, input mutation,
constant capture, stage effects, and trace boundaries. Per-node source maps are
debug-only.

Every tracer carries a trace generation. A closed or foreign-generation tracer
raises `EscapedTracerError` on read, conversion, or mutation. Trace contexts are
thread-affine; independent threads may own independent traces.

Errors name the user operation, relevant locations, and a concrete rewrite.
Silent detach is forbidden.

### 11. Durable program versions

A staged artifact records independent versions for:

- the graph container format;
- the core operation set;
- the semantic profile;
- the compiler;
- the fixed optimizer;
- Advect-owned operation schema revisions on graph nodes.

Compiler and optimizer versions participate in artifact provenance and staged
cache identity. Unknown versions fail before execution. Semantic migrations
are explicit and offline; loaders do not reinterpret old programs through
current provider behavior or rerun current optimization over a loaded artifact.
`GraphStore` is the sole durable topology, canonical-metadata, and
serialization authority.

## Removed commitments

The reset removes these concepts from the core contract:

- mutation/version nodes and backward-time mutation validation;
- `cache=` modes and implicit prepared caches on every transform;
- exception-driven higher-order fallback orchestration;
- overlapping ordinary/custom/prepared derivative APIs;
- an Advect graph built as the mandatory artifact for every dynamic call;
- workflow orchestration, object stores, user-configurable or dynamic graph
  optimization, and broad scientific wrappers as reasons to complicate the
  autodiff core.

The fixed staged optimization sequence is part of staging itself, not a public
transformation framework.

Optional integrations may be rebuilt after the core proves its value. They do
not retain compatibility authority over this reset.

## Acceptance gates

The replacement core is accepted only when it demonstrates:

1. Dynamic trace/gradient latency remains within the reference-versus-candidate
   regression gate on the agreed small-op and scientific kernels.
2. Peak live payload memory and post-backward release competitive with the
   reference, including a field/stencil workload.
3. The mutation matrix: aliases, copies, input rejection, basic indexed
   augmented assignment, stale views, nested-view errors, `out=`, and pending
   finalization.
4. Nested differentiation through traceable primitive rules without
   exception-driven mode selection.
5. Complex gradients and real-adjoint identities for the core primitive set.
6. Array API conformance through trace-and-execute round trips and a separate
   NumPy differential suite.
7. Staged constant/effect manifests, retrace visibility, and strict versioned
   round trips.
8. Deterministic staged optimization with visible pre/post node counts and
   per-pass rewrite reports.

## Consequences

Advect becomes smaller and more opinionated. Some previously implemented
features disappear. Dynamic autodiff no longer pays the cost of a durable graph,
while direct native staging still produces an optimized, inspectable artifact.
The dynamic and staged paths share one compact SSA substrate without sharing
their lifetime costs. Mutation remains useful source syntax without
contaminating derivative semantics. Complex scientific workloads receive an
explicit convention. New backends and primitives attach to one small contract
rather than several historical layers.
