# ADR: Runtime Extension Boundaries After the Core Reset

**Date:** 2026-07-24
**Status:** Accepted

## Context

The core reset deliberately removed workflow, storage, checkpoint, graph
transformation, and compatibility subsystems. That deletion established the
right dependency direction: dynamic autodiff no longer builds a durable graph,
and product infrastructure no longer shapes the tracing model.

Those removals do not all mean the same thing. Workflow persistence is a
separate product layer. Gradient checkpointing is an autodiff transform.
Buffer donation is a physical execution optimization. CuPy support is a
frontend integration. Mutation through every possible NumPy view is a large
semantic commitment, while mutation through a direct basic slice is a bounded
extension of the existing functionalization model.

We therefore distinguish four outcomes:

- **outside the core**: useful capabilities that belong in another package or
  service;
- **implemented extension**: a bounded addition with a current, tested
  contract;
- **deferred**: a plausible addition whose need or safe design has not yet been
  demonstrated;
- **excluded**: a commitment whose complexity is contrary to the current
  architecture.

This decision defines those boundaries and records the initial implementations.

## Decision

| Capability | Direction | Boundary |
| --- | --- | --- |
| Workflow orchestration and restart | Outside the core | A layer may consume staged programs and application artifacts |
| Generic key-to-bytes object storage | Outside the core | No storage protocol in tracing or autodiff |
| Large staged-constant externalization | Deferred integration | A narrow artifact serializer may store immutable blobs |
| Gradient checkpointing/rematerialization | Implemented extension | Manual, dynamic callable-level transform |
| Tape-to-graph optimization before backward | Excluded from the dynamic path | Dynamic reverse consumes the invocation-local tape directly |
| Raw user-selected graph passes | Excluded | The staged compiler owns one fixed, versioned pipeline |
| Semantic callable transforms | Allowed when earned | `stage`, `checkpoint`, and similarly bounded transforms |
| Named basic-slice mutation | Implemented extension | Replay onto an owned root SSA value |
| General NumPy view mutation and overlap semantics | Excluded | No general view-replay subsystem |
| Internal temporary-buffer donation | Implemented executor optimization | Conservative staged indexed updates only |
| Explicit input donation | Deferred | Requires an intentionally destructive public contract |
| Array API compatibility bridge | Implemented built-in frontend with a base dependency | Provider resolution remains outside the stdlib-only core module |
| CuPy GPU qualification | Designated bounded provider | One CUDA device; historical source-only scientific and donation results remain unverified without retained reports and digests |
| Other compatibility providers | Unqualified | Must justify and pass the same provider matrix before becoming a support promise |

## Storage, workflows, and residuals

`GraphStore` is not a generic object store. It is the native, immutable owner
of one staged program's topology and closed metadata. Its purpose is
canonicalization, inspection, execution planning, and serialization of an Advect
program.

A generic object store answers a different question: where arbitrary bytes
live and how a process retrieves them by key. The former Axon object-store
layer combined that concern with workflow outputs, checkpoints, and prepared
resources. Reintroducing it in the core would make local differentiation depend
on deployment policy without improving primitive, tracing, or derivative
semantics.

Workflow orchestration may be built later as an independent consumer of Advect.
It can schedule staged programs, persist application state, retry work, and
choose S3, a filesystem, a database, or another service. Advect does not define
that lifecycle.

Large immutable constants may eventually justify a narrow artifact-storage
adapter. Such an adapter would externalize serialized array blobs while the
staged artifact retains their digest, specification, and reference. It would
not become a general-purpose object-store abstraction, and it would not change
the constant-capture semantics.

Primitive residuals stay on the other side of this boundary. A residual is
live, invocation-local state needed by the adjoint of one exact primitive call.
It may be a local factorization, a device allocation, or a server-side handle.
It is retained by the enclosing `DynamicTape`, released deterministically, and
never serialized into a `GraphStore` or persisted through an object store.

This residual contract already supports split differentiation. A client-side
tape may retain an opaque handle returned by a server-side forward primitive
and present it to that primitive's server-side transpose. The durable artifact
contains only the primitive name and schema version. Reconnecting, expiring,
or recovering remote handles belongs to the primitive implementation and its
service, not to the graph format.

## Gradient checkpointing means rematerialization

We use *checkpointing* for two distinct capabilities:

1. workflow checkpointing persists job state so a process can resume later;
2. gradient checkpointing discards selected forward intermediates and
   recomputes them during the reverse pass.

Only the second belongs in Advect's autodiff scope.

The public transform is manual and callable-level:

```python
def loss(x):
    y = ad.checkpoint(expensive_block)(x)
    return objective(y)
```

The checkpoint wrapper installs one invocation-local custom primitive per input
pytree structure. The outer tape records that atomic call and its explicit
inputs, but not the region's interior operations. Its traceable JVP and
transpose rules rerun the Python callable under a nested dynamic transform.
The callable and input structure remain Python-side wrapper state; they are not
`RawArena` attributes and cannot enter durable serialization because abstract
staging rejects the boundary.

The contract is deliberately narrow:

- the region is pure and deterministic for the same explicit inputs;
- randomness is represented by explicit state or keys;
- functionalized mutation of values owned inside the region is allowed;
- mutation of external state, I/O, and other effects remains outside the
  supported contract;
- every residual-bearing primitive is a rematerialization barrier in the
  initial implementation;
- nested differentiation is supported only when the replayed primitive rules
  are themselves traceable.

The first implementation applies to concrete dynamic tracing. Abstract staging
rejects a checkpoint boundary rather than silently inlining it and losing its
memory contract. A staged region representation and memory-budget-aware planner
may be considered later, once real workloads provide memory profiles and
recomputation costs. They must lower to the same checkpoint-region semantics
rather than create a second derivative engine.

Rematerialization and buffer donation are complementary. Rematerialization can
make a forward value dead earlier; donation can then reuse its physical buffer.
Neither changes the logical SSA program.

## The dynamic tape is not an optimization artifact

A `DynamicTape` and `GraphStore` share the native `RawArena` substrate, so a
debugging or inspection tool can copy tape structure into a durable graph-like
form. That mechanical possibility does not make conversion part of reverse
mode.

The dynamic tape describes one concrete execution path and owns concrete
values, residuals, and derivative retention for one invocation. Turning it into
a `GraphStore` would add canonicalization, closed metadata, versioning, and
artifact construction that the dynamic path intentionally avoids. Running the
staged `DCE -> simplify -> CSE` pipeline before one backward sweep
would also pay optimization cost on every call, commonly to optimize work that
will execute only once.

Dynamic reverse therefore traverses the tape directly. Cheap construction-time
elision, liveness-based release, and specialized reverse traversal remain valid
tape or executor implementation techniques. They must not quietly grow into a
second graph compiler.

Programs that benefit from repeated optimization use `stage`, which traces
abstractly, optimizes once per signature, and reuses the resulting execution
plan. Advect does not infer a reusable staged program from a concrete derivative
call.

## Public transformations are semantic, not compiler-pass controls

The staged optimizer owns a fixed, deterministic, versioned pass sequence.
Users can inspect pre/post node counts and per-pass reports, but cannot choose
pass order or apply raw `Graph -> Graph` rewrites through the public API.

This keeps artifact identity reproducible and lets primitives define effect,
residual, and optimization barriers in one place. It also avoids making
internal node schemas and pass-order accidents part of Advect's compatibility
surface.

Public transformations may still be valuable when they express program
semantics rather than compiler policy. `stage` changes lifetime and execution
mode. `checkpoint` changes the memory/recomputation tradeoff. A future batching
or partition transform would likewise need its own explicit contract. Domain
operations that need special optimization belong in a custom primitive;
general third-party compiler passes would require a separately versioned plugin
boundary and are deferred until a concrete use case earns it.

## Mutation through views has a deliberately narrow tier

The current source-mutation model is tracer-level functionalization. A wrapper
points to its current immutable SSA value, and an update replaces that pointer
with a new value. Views record their root and creation epoch so stale use after
a root update fails rather than silently diverging from NumPy.

Direct mutation through a named basic-slice view rooted in an owned array is
supported:

```python
u = x.copy()
interior = u[1:-1]
interior += update
```

Advect replays this operation as an `index_update` on `u`'s current SSA value,
advance the root epoch, and refresh `interior` to the corresponding view of the
new root. Other views created from the previous root epoch become stale. Input
roots remain immutable.

This tier is feasible because the frontend already has a normalized basic
`IndexSpec`, root identity, and whole-root epochs. It does not require mutation
nodes in the IR. After direct slices, we may admit only transformations with an
unambiguous provider-independent inverse, such as a simple axis permutation,
when tests demonstrate a useful program path.

We continue to reject:

- advanced-index mutation and duplicate-index write semantics;
- mutation through broadcast or otherwise overlapping views;
- mutation through `reshape` or `ravel`, whose view-or-copy behavior depends on
  concrete layout and provider;
- arbitrary chains of slices, reshapes, transposes, and negative strides;
- attempts to infer disjointness between a view and a later root update.

Whole-root epochs intentionally produce false positives. For example, using
`u[:2]` after mutating `u[5:]` still raises. Exact overlap analysis and complete
NumPy view replay would require a substantial subsystem for layout-dependent
aliasing, chained inverse transforms, zero and negative strides, sibling
views, overlapping writes, and provider-specific write order. That subsystem
is technically possible but is not justified by the intended scientific
workloads. Unsupported cases fail with a rewrite to an explicit root update or
copy.

## Buffer donation is physical reuse

Logical SSA values are immutable. Buffer donation allows an executor to place a
new SSA value into storage previously occupied by an old, dead value. For
example, an indexed accumulation can update an owned temporary in place once
the executor knows that no later operation or derivative rule needs its
previous contents.

This can reduce peak device memory, allocation traffic, and memory bandwidth in
elementwise chains and iterative field updates. It is especially relevant on a
GPU, where allocation and saved-forward storage may dominate small amounts of
arithmetic.

The initial implementation is internal staged-executor donation. The native
execution plan computes use counts and conservative alias groups once, then
releases dead invocation values as it executes. An evaluator may declare a
donatable operand and owned output. The current Python runtime makes that
declaration only for the destination of `advect.index_update`; `advect.copy` and
prior `index_update` results are the initial owned donors.

Reuse is legal only when the executor proves that the buffer is:

- owned and writable;
- dead after the consuming operation;
- produced internally rather than supplied as an input or constant;
- not aliased by a live view;
- not a graph output or otherwise still live;
- compatible in dtype, device, shape, and storage requirements.

The bound evaluator also checks that the concrete buffer owns writable storage
and is not a tracer or provider view. The optimization cannot affect output
values, supported aliasing, or the serialized graph. If either proof fails,
`index_update` copies normally.

Dynamic donation is deferred until ownership across tracer aliases, views,
`LinearMap` retention, and residuals can be established cheaply enough to
preserve the dynamic latency target. Explicit donation of user inputs is
separately deferred because it invalidates caller-owned data and therefore
requires an intentionally destructive public API.

## Array API providers

The generic frontend accepts arrays that directly provide one of Advect's
supported `__array_namespace__` revisions. `array_api_strict` exercises that
path. Ordinary CuPy arrays do not expose that object protocol, so they use the
built-in compatibility fallback.

Advect's private bridge recognizes supported array types through
[`array-api-compat`](https://data-apis.org/array-api-compat/), a base dependency
since 2026-08-01. Importing `advect` configures one fixed fallback after direct
object-protocol discovery. The bridge requests the selected revision, preserves
the upstream namespace and its provider metadata, and hands it to the existing
generic tracer. It remains outside the internal `advect.core` module, whose
Python layer stays stdlib-only. NumPy is intentionally left to Advect's richer
first-class frontend.

This bridge is intentionally narrower than a CuPy clone of the NumPy frontend.
Code written against `x.__array_namespace__()` can remain backend-neutral once
Advect has wrapped the input. Existing functions written directly against
`cupy.*`, however, would need a dedicated protocol frontend analogous to the
NumPy adapter. We will not build that larger surface until backend-neutral code
proves insufficient.

CuPy is the first real GPU acceptance provider for the compatibility bridge.
Its single-device qualification covers:

- weak Python scalars and complex64 preservation;
- device-preserving constant capture and materialization;
- representative FFT and linear-algebra gradients;
- copied-field functionalized mutation;
- dynamic JVP/VJP and staged derivative serialization;
- peak device memory and internal donation;
- asynchronous stream synchronization in performance measurements.

Checkpoint replay and residual-bearing custom primitives are separate transform
contracts and are not part of the provider qualification claim. Unsupported
checkpoint/staging boundaries still fail before provider execution.

The fallback remains provider-neutral, but support does not. Providers with
their own autodiff systems add a second semantics and testing matrix without a
clear Advect user journey, so they remain unqualified until a concrete
interoperability requirement justifies that cost. Incidental compatibility
through `array-api-compat` is not a contract.

The CuPy contract is one process and one device. Multi-device placement and
distributed execution remain provider or orchestration concerns until Advect has
a concrete cross-device transform.

## Implementation status

The four bounded slices were implemented in this order:

1. Importing `advect` configures the private `array-api-compat` fallback while
   leaving NumPy on its first-class frontend. The bridge returns upstream
   namespaces directly; CuPy donation is a separate provider capability.
2. `ad.checkpoint` provides dynamic replay with nesting, pytree inputs,
   staging rejection, and residual barriers.
3. Direct named basic-slice mutation updates owned roots and refreshes only the
   mutated view.
4. Native staged execution releases dead values and conservatively donates
   owned temporaries to indexed updates.

Historical source notes report that the CuPy staged-donation memory path passed
its reserved-pool and runtime gate and that the scientific transform matrix
covered dynamic and staged round trips, promotion, complex values, FFT/linalg,
constants, mutation, and serialization. No retained report and digest currently
verify that pass. Provider-specific checkpoint and custom residual workloads
remain outside these bounded qualification claims.

Large-constant externalization, automatic checkpoint placement, explicit input
donation, composed-view replay, a dedicated `cupy.*` frontend, and compiler
plugins remain deferred. Workflow and object-store packages may evolve
independently without changing this sequence.

## Consequences

The reset remains a small autodiff architecture rather than a partial workflow
platform. Useful memory and backend capabilities can return at the layer where
their invariants are knowable: recomputation in autodiff, donation in the
executor, provider discovery in an optional frontend, and persistence in
orchestration.

The design accepts some explicit errors and conservative alias failures in
exchange for deterministic semantics across providers. It also keeps one clear
performance choice: concrete dynamic differentiation traverses its tape
directly, while reusable graph optimization belongs to abstract staging.
