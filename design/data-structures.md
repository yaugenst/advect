# Core Data Structures

This document defines the numerical runtime's structures. It is
normative together with [Core Requirements](requirements.md).

## Primitive

```python
@primitive(
    static_argnames=("config",),
    nondiff_argnames=("tag",),
)
def solve(a, b, tag, *, config): ...

@solve.def_abstract
def solve_abstract(a, b, tag, *, config): ...

@solve.def_jvp
def solve_jvp(output, primals, tangents, *, config): ...

@dataclass(frozen=True, slots=True)
class PrimitiveResult:
    output: object
    residual: object
    release: Callable[[object], None] | None = None
```

`@primitive` replaces the implementation function with a callable handle to one
canonical registry record; there is no empty object to instantiate and no
second primitive table. The function signature defines the call contract and
its module plus qualified name defines the default operation identity. An
explicit `name=` is available for durable library identities. The registry
record owns operation metadata, Advect's graph schema revision, and process-local
rules. Built-in records additionally own their fixed output arity and abstract
operand/attribute schema and evaluator. Serializable programs store the
Advect-owned operation name and schema revision, inputs, result specifications,
and closed attributes. Python callables are linked at runtime. The closed
attributes include the expected output pytree, so result leaf order cannot
silently reinterpret a staged multi-output primitive.

Rules are ordinary traceable functions. Runtime transforms inspect the rules
that are actually installed. Structural transposition validates a JVP's
real-linearity when it is used; higher-order differentiation succeeds when the
rule bodies themselves trace.

Authored derivative rules receive the exact output of the matching
implementation call:

```python
@primitive.def_jvp
def jvp(output, primals, tangents, **static_attrs): ...

@primitive.def_transpose
def transpose(cotangent, primals, output, **static_attrs): ...
```

A residual primitive's transpose receives one additional `residual` argument
after `output`. Providing the primal output avoids recomputing expensive or
numerically stabilized primitives. An explicit transpose conservatively
retains that output until its reverse use.

A primitive declared with `residual=True` has one implementation return contract:
`PrimitiveResult(output, residual, release=None)`. Advect returns only `output`
and, when the call is being differentiated, retains the opaque residual for
the exact matching transpose invocation. Ownership transfers to Advect; the
optional synchronous release callback runs exactly once. A plain call,
including plain staged replay, has no transpose consumer and releases the
residual before returning. Residual primitives are first-order-only until a
future rule contract can represent the residual's derivative dependence.

## AbstractValue and ArraySpec

```python
@dataclass(frozen=True, slots=True)
class ArraySpec:
    shape: tuple[int, ...]
    dtype: DType
    device: str | None = None
    weak: bool = False

@dataclass(frozen=True, slots=True)
class AbstractValue:
    spec: ArraySpec
```

`weak=True` represents a Python scalar whose promotion follows the pinned
provider-facing Array API rules. Weakness is part of staged call identity and
is validated exactly. `device=None` leaves device unconstrained for an explicit
specification; an inferred or non-`None` device is recorded and validated.
Abstract values contain no concrete payload and cannot answer truth, iteration,
item extraction, or data-dependent shape questions.

## Native SSA substrate

The Python-independent `advect-runtime` crate owns one compact structural
representation:

```rust
type NodeId = u32;
type OpId = u16;

struct RawArena {
    nodes: Vec<NodeCore>,
    edges: Vec<NodeId>,
    ops: OpTable,
}

struct NodeCore {
    op: OpId,
    inputs: InputRef,
    flags: NodeFlags,
}

enum InputRef {
    None,
    Unary(NodeId),
    Binary(NodeId, NodeId),
    Nary { start: u32, len: u32 },
}

struct OpSchema {
    name: String,
    schema_version: u32,
}
```

`NodeId` is the arena position. Unary and binary inputs stay inline; uncommon
n-ary inputs occupy a range in the edge arena. Parent IDs are always smaller,
so forward and reverse order need no sort.

`OpId` is dense and local to one arena. `OpTable` maps it to a stable primitive
name and schema version. Serialized programs therefore never depend on a
process-global registry integer.

`RawArena` contains only SSA structure. It has no concrete values, abstract
specifications, attributes, residuals, source locations, serialization
payloads, or optimization state. Those belong to the owner whose lifetime
requires them. Python never mirrors `RawArena` with node objects.

### DynamicTape

The native `DynamicTape` owns one concrete define-by-run invocation:

```rust
struct DynamicTape {
    arena: RawArena,
    operand_layouts: Vec<OperandLayout>,
    operand_positions: Vec<u32>,
    metadata: Vec<DynamicNodeMetadata>,
    values: Vec<Option<PyObject>>,
    attrs: Vec<Option<PyObject>>,
    literals: Vec<Option<PyObject>>,
    residuals: Vec<Option<PyObject>>,
    jvp_bindings: Vec<Option<PyObject>>,
    vjp_bindings: Vec<Option<PyObject>>,
    reverse_needs: Vec<Option<ReverseNeeds>>,
    reverse_value_uses: Vec<u32>,
    inputs: Vec<NodeId>,
    outputs: Vec<NodeId>,
}
```

The side tables retain transient attributes, active operand positions, and the
compact shape/dtype information required to normalize derivatives. Inactive
concrete operands remain direct literals; they do not receive durable constant
nodes merely to make dynamic storage resemble staging. A tracer from an
enclosing trace is retained intact as an opaque literal in the inner tape.
Evaluating the inner operation records the enclosing dependence in the outer
tracer's recorder, while the inner result is recorded in the inner recorder.
Node IDs remain strictly arena-local.

After the concrete forward trace, the tape freezes its arena-local operation
table and resolves dense derivative bindings. Each reverse binding has a
conservative `ReverseNeeds { output, primals, residual }` record. Freezing
builds reverse-use counts. A consuming reverse prunes payloads with no reverse
consumer before its first callback, decrements counts after each callback, and
drops provider values and residuals at their last use. Reusable `LinearMap`
application remains non-consuming. Rust owns traversal, derivative slots,
retention, and release. Python-authored derivative callbacks receive positional
values and may reenter an enclosing trace; native borrows do not span those
callbacks.

There is no dynamic structural hash, canonicalization, serialization metadata,
topology plan, or optimization pass. Consuming a one-shot tape releases its
value, literal, and residual side tables after derivative application or
failure. A reusable `LinearMap` retains the required invocation state until
`close()`.

`checkpoint(f)` installs an atomic dynamic primitive for each observed input
pytree structure. The outer tape retains its explicit operand leaves and
output, while the wrapper retains the Python callable and pytree recipe. The
region's interior operations are omitted from the outer tape. JVP and
transpose rules replay the callable under a nested dynamic transform. Abstract
staging rejects the boundary, so the recipe never becomes a `RawArena`
attribute or crosses durable serialization.

### GraphBuilder, GraphStore, and GraphExecutionPlan

An abstract trace records directly into native `GraphBuilder`. Its side tables
own abstract value specifications, canonical attributes, constants, output
metadata, and source attribution while construction remains mutable.

Finishing construction validates and canonicalizes the program, then runs the
fixed staged pipeline:

```text
GraphBuilder -> DCE -> simplify -> CSE -> GraphStore
```

There is no placeholder fusion pass. Backend extension FMA/FMS operations are
not a provider-independent rewrite contract.

The resulting `advect-runtime::GraphStore` owns the immutable `RawArena`,
operation schemas, canonical metadata and constants, input/output topology,
the complete version header, and serialization state. `StagedProgram` does not
duplicate those fields. A dynamic tape is never promoted into a store.
Custom or remote primitives are pass barriers in the current optimizer.

Each compiled signature owns one immutable native `GraphExecutionPlan`, which
retains an `Arc<GraphStore>`, derives its dense execution structure from that
store, and resolves each evaluator binding once. Every call reuses that plan
and allocates only a dense value table plus temporary operand tuples. Native
bulk traversal therefore executes without materializing Python graph nodes or
rebinding evaluators.

Captured arrays are detached during compilation into a closed, immutable
numeric payload owned by the store. The live representation is canonical raw
little-endian bytes; Python/native transfer also uses bytes. Canonical artifact
version 2 uses lowercase hexadecimal only when serializing those bytes to JSON.
A compiled signature materializes the payload at most once for each runtime
namespace and device, then reuses the provider-local constants on warm calls.
Fresh and deserialized programs use the same path.

The staged executor has no residual table. Plain custom-primitive replay
releases an unused residual before returning. When a staged program runs under
an enclosing dynamic transform, custom replay calls the linked primitive handle;
that records one atomic custom node and transfers its residual to the enclosing
`DynamicTape`. Residuals never enter `RawArena`, `GraphStore`, constants,
attributes, output leaves, the execution plan, or serialized state.

Staged reverse compilation deliberately uses that same nesting mechanism once:
an outer `GraphBuilder` supplies abstract values, an inner `DynamicTape`
linearizes replay of the already-optimized primal program, and the ordinary
transpose rules emit their results back into the outer builder. The inner tape
is released when compilation ends. The resulting derivative is just another
optimized `GraphStore`; warm execution has no tape or reverse traversal.
Traceable custom primitives remain atomic nodes. An opaque residual primitive
cannot cross this boundary because the resulting store has nowhere to retain
its invocation-local residual.

The host-independent execution plan links stable operation schemas once and
owns dense value slots, use counts, conservative alias-root sets, last-use
release, and donation eligibility. A host materializes constants, evaluates
operations over opaque values, validates inputs, constants, and evaluated
results against every declared output leaf, retains another handle when an SSA
value occupies repeated flat output positions, declares output ownership or
aliases, and accepts or declines an offered donor.
`advect-native::PythonHost` implements retention with a Python reference clone;
the pure Rust fixture host proves that the same graph executes without Python.

Python may hold PyO3 `DynamicTape`, `GraphBuilder`, and `GraphStore` handles for
recording, append-only construction, and immutable inspection. The builder and
store handles are thin adapters over `advect-runtime`; Python owns no graph or
tape data structure. The remaining
`_graph_attrs.py` module is a stateless, backend-neutral codec for
the closed attribute algebra; it owns no nodes or topology.

## Trace frame and abstract trace

```python
@dataclass(slots=True)
class TraceFrame:
    recorder: DynamicTape | GraphBuilder
    trace_level: int
    trace_kind: str
    frame_id: int
    pending_update: object | None

@dataclass(slots=True)
class AbstractTrace:
    builder: GraphBuilder
    profile: str
    open: bool
```

Frames form a thread-local stack for nested transforms. A tracer operation is
valid only when its frame occurs on the active stack. A trace is never continued
on another thread. Constant capture belongs to the staging builder rather than
the dynamic frame.

Before every external tracer operation, the context verifies that no pending
indexed update exists. Matching setitem is the only exception.

## Mutable tracer wrappers and views

There is no IR mutation cell and no shared Python `TracerCell` abstraction.
The concrete `TracedArray` wrapper itself is the source-level cell: it directly
holds its current native node ID, concrete value, ownership bit, epoch, trace
identity, and optional `ViewState`. Functionalized mutation replaces the node
ID and concrete value on that wrapper and increments its epoch. Ordinary
Python aliases see the update because they reference the same wrapper object.
Inputs have `owned=False`; `copy()` returns a new owned wrapper.

An aliasing result is a distinct wrapper whose `ViewState` points to the root
`TracedArray` and stores the root epoch, normalized index when applicable, and
creation location. Every use first verifies that the root still has that
epoch. Getitem views defer their SSA node until the view is actually consumed;
this lets a matching basic-slice `+=` lower directly to one additive
`index_update`.

Abstract staging uses the same semantics with a private `_AbstractCell` owned
by each `AbstractArray`; an abstract view points to the root cell. This private
staging detail does not define a cross-runtime cell type.

Concrete payload fields are internal. Public coercion and buffer-export
protocols fail.

Frontend profiles classify alias-producing operations. The initial NumPy
profile treats basic slicing, reshape, transpose, and ravel as aliases even if
a concrete provider happens to allocate a copy. Advanced indexing is a copy.

Named-view mutation is narrower than this conservative alias classification.
It replays only a direct basic-slice `ViewState` onto an owned
root, advances the root epoch, and refreshes the mutated view. Reshape/ravel,
advanced, overlapping, and arbitrary composed views remain non-replayable.
This keeps view mutation in tracer metadata rather than adding alias or
mutation state to `RawArena`.

## Normalized index specification

Pending basic-index updates use a closed, hashable tuple representation:

```python
("int", value)
("slice", start, stop, step)
("newaxis",)
("ellipsis",)
```

Matching never relies on Python slice-object equality or identity. Advanced
array indices have a closed serialized attribute representation for reads but
are rejected by indexed assignment.

## PendingIndexUpdate

```python
@dataclass(frozen=True, slots=True)
class PendingIndexUpdate:
    root: TracedArray
    root_epoch: int
    index_spec: object
    replacement: TracedArray
    location: SourceLocation | None
    view_location: SourceLocation | None
```

The in-place method emits the root `index_update` immediately, refreshes the
mutated view, and returns that view. The pending object acknowledges the
optional setitem that CPython emits for indexed augmented assignment. Matching
requires replacement-view identity, root-wrapper identity, post-update epoch
equality, and structural index agreement. A matching setitem is a no-op. If no
setitem follows because the view was named, the next traced operation or trace
finalization discards the completed acknowledgement. It is never managed
through `__del__`.

## Physical storage

There is no ownership or donation structure in the semantic core. The
`advect-runtime` execution plan computes use counts and conservative alias-root
sets.
Linked hosts declare owned outputs, alias sources, and candidate donation
operands. The Python provider path may reuse an internally produced, writable
`advect.copy` or `advect.index_update` buffer only for a last-use
`advect.index_update` with no live alias and compatible shape/dtype. It releases
other dead values without reusing them. The host may decline a structurally
eligible donor; failed proof means fresh allocation. Dynamic reuse and
destructive donation of user inputs remain deferred; neither can change the
SSA model.

The exact boundaries for checkpoint replay, named views, and donation are recorded in
[Runtime Extension Boundaries](decisions/2026-07-24-runtime-extension-boundaries.md).

## Pullback

```python
class Pullback:
    def __call__(self, cotangent: object) -> object: ...
    def close(self) -> None: ...
```

`vjp` returns a `Pullback` that owns one concrete reverse trace. It is
deliberately one-shot: applying it consumes and releases the tape even when
the reverse sweep raises, while `close()` releases it without applying it.
Callers that need repeated adjoint applications use `linearize` and the
reusable `LinearMap` below.

## LinearMap

```python
class LinearMap:
    def __call__(self, tangents: object) -> object: ...
    def transpose(self) -> Callable[[object], object]: ...
    def apply_many(self, tangents: object) -> object: ...
    def transpose_many(self, cotangents: object) -> object: ...
    def close(self) -> None: ...
```

The map owns the concrete `TraceResult` and therefore its `DynamicTape`.
Operations must be real-linear in tangent inputs. Complex-linear operations
are a proven subset; general complex rules may use both a tangent and its
conjugate. Transposition uses the real inner product. `LinearMap` is a context
manager; closing it releases the tape payload lifecycle, is idempotent, and
makes later application an error.

`apply_many` and `transpose_many` are private optimization mechanisms exposed
on the derivative object, not array-axis transforms. They divide seeds into
groups of at most 16. Rust walks the tape once per group and snapshots each
active node once; Python-authored primitive rules are still called separately
for each active seed with their ordinary scalar contract. This shares graph
bookkeeping without introducing batched primitive semantics or a public
`vmap`. Dense reverse Jacobian assembly uses the same path below the pytree
boundary: provider-native basis rows become output-node cotangent tables, and
input pytrees are rebuilt only after the final blocks are assembled.
For JVP-only nodes, the native multi-seed reverse traversal invokes one
internal batched callback per node and group. Python traces and validates that
node's JVP once, then applies the resulting tangent map to every cotangent in
the group. Explicit VJP rules retain their ordinary scalar callback contract.

## StagedProgram

```python
class StagedProgram:
    @property
    def graph(self) -> GraphStore: ...

    @property
    def signature(self) -> tuple[tuple[object, ...], dict[str, object]]: ...

    @property
    def compile_seconds(self) -> float: ...

    @property
    def constants(self) -> tuple[ConstantRecord, ...]: ...

    @property
    def optimization(self) -> OptimizationReport: ...

    @property
    def array_api_version(self) -> str: ...

    def to_dict(self) -> dict[str, object]: ...

    @classmethod
    def from_dict(cls, payload: object) -> StagedProgram: ...
```

`StagedProgram` owns exactly one call signature and one immutable
`GraphStore`. Finishing its abstract `GraphBuilder` validates, canonicalizes,
and optimizes that graph once. The Python-independent `GraphStore` remains the
sole durable topology authority, while `advect-runtime` owns canonical graph
artifact version 2 validation and serialization. Compiler and optimizer
versions are artifact provenance; loading an artifact does not rerun the
optimizer.
The `advect-array-1` semantic profile records one required Array API revision
from 2022.12, 2023.12, or 2024.12 while retaining the separate NumPy 2.0-2.5
frontend contract. `array_api_version` exposes the graph-owned execution
requirement; derived staged programs preserve it.

`graph` is therefore unconditional. It exposes immutable structural
inspection—node IDs, topology, attributes, and constant IDs—but never mutable
constant payloads or raw native serialization. `StagedProgram.to_dict()` is
the public detached serialization boundary. Its envelope is:

```python
{
    "format": "advect.ssa-program",
    "version": 2,
    "program": {...},
}
```

The envelope version describes this singular-program shape. Its nested graph
header is the sole source for graph format, core opset, semantic profile,
required Array API revision, compiler, and optimizer versions. Operation schema
revisions continue to live on each graph node and in Advect's registry. Loading
uses exact current revisions; no compatibility range or migration table is part
of primitive authoring.

`grad(StagedProgram)`, `value_and_grad(StagedProgram)`, and
`vjp_program(StagedProgram)` compile its one signature and produce a new
`StagedProgram`. The envelope records exact output specifications as well as
output structure, so a VJP program can turn those outputs into one typed
`cotangent` input without executing the primal. The derived program owns its
own graph, execution plan, optimizer report, and serialization. It does not
retain a dynamic tape.

## ConstantRecord

```python
@dataclass(frozen=True, slots=True)
class ConstantRecord:
    value_id: int
    origin: Literal["closure", "global", "created"]
    location: SourceLocation | None
    shape: tuple[int, ...]
    dtype: str
    bytes: int
    digest: str
    name: str | None = None
```

The tuple in `program.constants` is staged inspection and serialization
metadata. Typed constant payloads remain private in the validated staged
artifact constant table.

## OptimizationReport

```python
@dataclass(frozen=True, slots=True)
class OptimizationPass:
    name: str
    nodes_before: int
    nodes_after: int
    removed_nodes: int
    rewritten_nodes: int

@dataclass(frozen=True, slots=True)
class OptimizationReport:
    nodes_before: int
    nodes_after: int
    rewritten_nodes: int
    passes: tuple[OptimizationPass, ...]

```

`program.optimization` contains this report for the singular graph and always
records the fixed `DCE -> simplify -> CSE` pass sequence.
`program.compile_seconds` and `program.constants` expose the other compilation
diagnostics directly. Version compatibility belongs to the serialized native
graph header rather than a second Python object. There are no cache hit, miss,
retrace, or signature-collection states on `StagedProgram`.

## Pytree structures

`TreeDef` describes container type and static metadata. Flattening produces a
stable leaf order. Dict insertion order defines dynamic leaf order; static
mapping identity is canonicalized by encoded key. Static leaves are explicit
wrappers or primitive-declared arguments rather than accidental untraceable
objects.

Built-in containers and exact registered node types use the process-local
pytree registry. Application model families may instead implement two
inherited hooks:

```python
def __advect_tree_flatten__(self):
    return (dynamic_child_1, dynamic_child_2), equality_safe_metadata

@classmethod
def __advect_tree_unflatten__(cls, metadata, children):
    return cls.from_children(metadata, children)
```

The children must be a tuple. Both hooks are required, and reconstruction is
bound to the concrete subclass recorded in `TreeDef`; one base-class protocol
therefore covers a model hierarchy without registering every leaf class.
Exact registration takes precedence over the protocol. This is a dynamic
integration contract only until a custom node also has a stable durable codec.
