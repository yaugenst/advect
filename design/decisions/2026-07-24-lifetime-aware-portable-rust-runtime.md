# ADR: Lifetime-Aware Portable Rust Runtime

**Date:** 2026-07-24
**Status:** Accepted
**Implementation status:** Implemented

## Context

The native SSA cutover established one compact Rust arena for concrete tapes
and staged graphs and brought dynamic differentiation to HIPS Autograd latency
parity. At the time of this decision, its measurements covered only
Python-visible allocation through `tracemalloc`; they did not establish whether
Rust reduced total process or device memory.

The pre-decision Rust package also combined two different responsibilities:

- Python integration through PyO3, including provider values, callbacks,
  residuals, and conversions;
- Python-independent graph structure, validation, canonical metadata,
  optimization, serialization, topology, and liveness planning.

Consequently, `GraphStore` was serializable from Python but the Rust
implementation was not reusable outside Python. Constants and evaluator
bindings contained `PyObject` values, and deserialization depended on Python
callbacks.

Rust is justified only if it owns a capability that is visible beyond moving
Python allocations into another allocator. Two capabilities can earn that
cost:

1. precise, low-overhead memory lifetime management for dynamic reverse and
   staged execution;
2. a portable staged SSA runtime that can validate, plan, and execute a
   `GraphStore` through a non-Python host.

Kernel compilation, code generation, backend lowering, and automatic fusion
would be a different product. They are not required to deliver either
capability and remain out of scope.

## Decision

### Rust has one narrow role

Rust owns Advect's compact, lifetime-aware SSA runtime. Python continues to own
the language and autodiff semantics.

```text
Python API, tracing, pytrees, derivative rules
                         |
                         v
                 advect-native (PyO3)
                 /                 \
                v                   v
      advect-runtime (pure Rust)   Python providers
      graph + plans + lifetime   NumPy/CuPy/custom
                ^
                |
        another Rust host
```

The implemented ownership split is:

| Layer | Owns |
| --- | --- |
| Python modules | Tracer behavior, pytrees, mutation functionalization, primitive authoring, JVP/transpose formulas, diagnostics, provider discovery, and checkpoint replay recipes |
| `advect-runtime` | `RawArena`, closed graph metadata, portable constants, `GraphStore`, validation, serialization, fixed graph cleanup, topology, execution structure, use counts, and host-independent lifetime decisions |
| `advect-native` | PyO3 conversion, Python-owned dynamic payload side tables, reentrant callbacks, residual handles, and the Python implementation of the runtime host contract |
| Array/custom providers | Concrete kernels and buffers, concrete alias/writability checks, accepted donation, and the meaning and release operation of residual payloads |

`advect-runtime` has no PyO3 or Python dependency. `advect-native` remains a
required Python extension while Python is Advect's primary frontend, but it is
an adapter rather than the authority for durable graph semantics.

### Dynamic memory management stays direct

Dynamic differentiation continues to traverse its invocation-local tape. It
does not construct a `GraphStore`, run staged graph passes, or build a reusable
compiler artifact before backward.

Each frozen reverse binding carries an explicit conservative `ReverseNeeds`
record:

- whether it needs the operation output;
- whether it needs no primal operands or all primal operands;
- whether it requires an invocation-local residual.

The implementation reuses the coarse `vjp_needs_inputs` distinction and adds
`vjp_needs_output`. Custom primitives and structurally transposed JVPs default
to retaining everything needed for correctness. Per-operand dependency masks
were not needed to pass the memory gate and are not part of the runtime.

Freezing builds reverse-use counts. A consuming one-shot reverse first prunes
payloads with zero reverse uses, then releases saved outputs, literals, primals,
and residuals at their last reverse use. A reusable `LinearMap` is
intentionally non-consuming and retains its required linearization until
`close()`.

Manual `checkpoint` remains the mechanism for reducing saved-forward memory.
Precise reverse release reduces the values carried into and through backward.
Staged liveness and donation reduce call-local execution storage. These are
separate mechanisms and are measured separately.

Dynamic buffer donation and destructive donation of caller inputs remain out
of scope.

### The staged artifact is Python-independent

The pure Rust `GraphStore` owns only closed, portable data:

- append-topological nodes and stable operation schemas;
- closed attributes and dtype descriptors;
- input and output topology;
- versioned numeric constant payloads;
- graph-format, opset, semantic-profile, producer, and optimizer provenance.

Provider objects, Python callbacks, runtime values, residuals, devices, and
resources never enter the store.

Numeric constants use a versioned canonical byte representation with explicit
dtype, shape, byte order, layout, and digest. The byte count is derived from
and validated against dtype, shape, and payload length rather than serialized
as redundant metadata. Live construction, storage, and Python/native transfer
use raw bytes. Canonical graph artifact version 2 encodes those bytes as
lowercase hexadecimal only at the JSON boundary. Large-constant
externalization remains a separate future storage concern.

The Python `StagedProgram` envelope may continue to own Python-specific call
structure such as pytrees and `Static` metadata. A non-Python host consumes
flat graph inputs and returns flat graph outputs. Advect does not make Python
pytree codecs part of the Rust runtime merely to claim portability.

### Portable execution is host-driven

`advect-runtime` defines a small host contract for:

- linking a stable operation name and schema version;
- materializing a portable constant;
- retaining another host handle when one SSA value appears in multiple flat
  output positions;
- evaluating an operation over opaque host values;
- validating every input, materialized constant, and evaluated result against
  all declared output leaves;
- declaring output ownership, aliases, and possible donation operands;
- accepting or declining a donation selected by the runtime.

Each execution plan retains an `Arc<GraphStore>`. The runtime owns dense value
slots, last-use release, conservative alias-root-set accounting, and donation
eligibility. The host owns numerical behavior and concrete buffer safety.
Declining donation always falls back to a fresh result with identical
semantics.

The independent proof is a pure Rust vector host that deserializes, validates,
executes, and reserializes the canonical fixture also emitted by Python. It is
not a second numerical library. Full NumPy, CuPy, FFT, or linear-algebra
implementations in Rust are not part of this decision. A server may implement
a host operation locally or by RPC, but scheduling, retry, persistence, and
object storage remain outside Advect.

### Residuals remain runtime sidecars

Residuals are compatible with the lifetime runtime precisely because they are
not IR:

- one concrete primitive invocation creates one residual;
- the enclosing dynamic tape owns it;
- the matching transpose borrows or consumes it according to tape lifetime;
- Rust arranges deterministic release on success, failure, and explicit close;
- it is never serialized, hashed, checkpointed, or placed in `GraphStore`.

A residual may be a remote handle. Reconnection, expiry, and recovery belong
to the primitive implementation and service. A portable staged derivative can
cross a client/server boundary only when all derivative operations are
traceable into the graph. Opaque residual primitives remain staged-derivative
barriers.

### This is not a compiler project

Abstract staging retains the fixed `DCE -> simplify -> CSE` cleanup sequence
because those passes have demonstrated useful reusable-program results. The
former no-op `fuse` stage was deleted during the portable-runtime cutover.

The following remain outside the plan:

- kernel fusion and generated kernels;
- machine-code, GPU-code, or bytecode generation;
- backend-specific lowering or scheduling;
- automatic checkpoint placement;
- autosharding or distributed execution;
- a public graph-pass or compiler-plugin API;
- a Rust clone of NumPy or the Array API.

### Implementation evidence

The implementation passed the decision's retention, latency, portability, and
source-retirement gates:

- reverse-entry provider bytes fell from 68,506,704 to 34,952,400
  (`0.510x`) for the elementwise workload and from 69,904,680 to 36,350,472
  (`0.520x`) for the stencil workload;
- manual checkpointing reduced peak RSS from 69.30 to 29.19 MiB (`0.421x`)
  and reverse-entry provider storage from 65.33 to 6.67 MiB, with four
  recomputations and `0.747x` runtime in the measured case;
- CuPy staged donation reduced reserved-pool high water from 67,109,376 to
  44,739,584 bytes (`0.667x`), avoided 22,369,792 reserved bytes—enough for
  one 22,369,616-byte destination—and ran at `0.431x` the forced-fresh control;
- taking each workload's median across three checked runs and then their
  geometric mean gives 50.58165 microseconds for current Advect versus
  48.30620 microseconds for the equivalently aggregated pre-change baseline
  (`1.0471x`); the median current-to-HIPS core ratio was `1.02159x`;
- the five-run captured-constant case for a 64 MiB byte budget stored a
  22,369,616-byte constant, reached 94,408,704 bytes peak RSS, compiled in
  66.2097 ms, and ran warm in 3.50635 ms. Provider-cache bytes equaled the
  constant size, post-close provider-owned bytes were zero, and peak-RSS
  variation was 0.01735%;
- nine-run staged functional-update medians remained within the 5% extraction
  gate:
  donation compile and warm ratios were `1.0198909x` and `0.9950918x`;
  forced-fresh ratios were `0.9921773x` and `1.0108587x`.
- the PyO3 adapter shrank from the 6,772-line, 26-file native baseline to
  3,819 lines in 13 files. The new pure runtime is 4,912 lines in 12 files;
  together they are 8,731 lines in 25 files, 28.93% more source than the old
  single crate. The split deletes 43.61% of the adapter while making the
  additional portable runtime cost explicit.

The measurements ran on Linux with Python 3.12.13, NumPy 2.3.5, CuPy 14.1.1,
an NVIDIA RTX 4080 SUPER and driver 610.43.03, Rust 1.94.0, and uv 0.11.28.
The subprocess benchmark reports RSS, provider-live bytes, provider pool
reservation, native structural bytes, and runtime separately.

The reproducible benchmark programs are checked in. Generated calibration,
retention, checkpoint, residual, donation, staged-regression, CPU-corpus, and
dynamic-latency JSON is uploaded by the evidence workflow or release process
alongside its environment and acceptance metadata.

`advect-runtime` builds without PyO3, owns the only durable graph model,
validator, canonical serializer, optimizer, and execution schedule, and
executes the cross-language fixture through a non-Python host. The replaced
durable authorities were removed from `advect-native`; its remaining graph and
execution types are thin PyO3 adapters.

## Consequences

The Python user API does not grow. `grad`, `linearize`, `checkpoint`, `stage`,
custom primitives, and staged artifacts retain their semantic roles.

The implementation gains two independently testable claims: Rust manages
provider-value lifetimes, and a staged graph is genuinely usable without
Python. Neither claim depends on calling graph interpretation a JIT.

The work requires a clean crate split and a staged-artifact version bump.
Because Advect has no users, the cutover uses one new format and removes the old
internal path rather than carrying migrations or compatibility shims.

Portability does not imply universal execution. A host must implement every
operation named by a graph. Likewise, precise lifetime release cannot remove
values that conservative custom rules declare as required.
