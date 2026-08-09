# Architecture

Advect is an autodiff core, not an array library. NumPy — or another Array API
provider — keeps doing the numerics; Advect's job is to record what a function
does to its arrays, differentiate that record, and, when asked, turn it into a
durable program. Everything in the design follows from keeping that job small.

## Three frontend paths, one semantic core

NumPy and the Array API are separate calling contracts. The concrete NumPy
frontend owns `numpy.*` signatures, ufunc methods, aliases, `out=`, and
functionalized mutation. The provider-neutral Array API frontend owns the
versioned namespace returned by `x.__array_namespace__()`. Their array-valued
calls share code only after each frontend has bound its foreign call to a
canonical operation.

Compile-time metadata takes a third path. NumPy structural queries and the
Array API metadata functions resolve directly from the active concrete or
abstract namespace. They return the dtype, shape, or scalar metadata needed by
the trace instead of emitting a numerical graph node. A traced composition can
use that result for static attributes or trace-time control flow, but the query
does not become a primitive.

```text
numpy.* ─────────────> NumPy frontend ────┐
                                         ├─ array call ─> canonical operation
__array_namespace__ → Array API frontend ─┤                  ├─ dynamic tape
        ↑                                │                  ├─ staged graph
        └─ NumPy or CuPy provider        │                  └─ derivative rules
                                         └─ metadata ─> trace-time metadata
                                                        (no graph node)
```

CuPy reaches Advect through the designated Array API provider path; Advect does
not maintain a speculative direct `cupy.*` frontend. Its historical GPU record
is source-only until a retained report and digest verify the current gate. A
future direct frontend would bind its own calling convention and join at the
same canonical-operation boundary.

## One semantic core, two lifetimes

A supported frontend call takes one of three paths: resolve compile-time
metadata without a node, emit a canonical operation, or expand into a traceable
composition of existing operations. Each canonical operation records stable
identity and arity plus the semantic capabilities it actually supports:
abstract shape/dtype semantics for staging and a JVP when it supports forward
mode. Built-in concrete execution remains with the frontend and array provider;
a custom primitive calls its authored implementation. Reverse mode structurally
transposes the recorded JVP where possible.
The smaller set that needs a direct real adjoint carries an explicit transpose;
that rule can also stand alone to supply reverse mode without forward mode or
structural transposition. A traceable non-residual transpose can compose under
reverse mode, while a residual-bearing transpose is inherently first-order.
Canonical records mark mathematically non-differentiable operations with a
reason rather than placeholder rules. Frontend invocation evidence and public
support declarations separately say whether each call form is dynamic-only or
also stageable.

The transforms you call — `grad`, `jvp`, `vjp`, `jacobian`, and `stage` —
consume those declared capabilities. A traceable frontend composition reuses
them; it does not become another primitive merely because it has a public name.

What differs between the two execution modes is not the math but the
*lifetime* of the record:

```text
              ┌─────────────────────────────────────┐
              │ User code                           │
              │   z = np.sin(x) + y                 │
              └──────────────────┬──────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │ Canonical operation semantics       │
              │   identity/arity · abstract ·       │
              │   derivative                        │
              │   capabilities as declared          │
              └────────┬───────────────────┬────────┘
                       │                   │
    ┌──────────────────▼────────┐ ┌────────▼──────────────────┐
    │ Dynamic lifetime          │ │ Staged lifetime           │
    │   concrete SSA tape       │ │   abstract trace →        │
    │   → pullback              │ │   durable optimized graph │
    └───────────────────────────┘ └───────────────────────────┘
```

## The dynamic path

`grad(f)(x)` runs `f` with the real values, wrapped in tracers that record
each emitted canonical operation onto an invocation-local SSA tape. Because
the trace happens inside an ordinary call, Python control flow needs no special
forms: loops unroll to the iterations that actually ran, branches take the
branch the data took. The reverse sweep walks that tape backwards through the
transpose rules and then releases it. Consuming transforms such as `grad`
release before returning; `vjp` and `linearize` transfer that one invocation's
tape to the returned `Pullback` or `LinearMap` until it is consumed or closed.
None of these handles is a cross-call trace cache.

That per-call freshness is the point of the dynamic mode: a new trace can have
new shapes, new branches, even a different number of loop iterations. The cost
is that tracing work is repeated on every call.

## The staged path

`stage(f, example)` runs the same trace once with *abstract* values — shape
and dtype, no data — and keeps the result. The recorded graph is validated,
then run through a fixed optimization pipeline (dead-code elimination,
simplification, common-subexpression elimination) into an immutable program:

```{.python .run}
import numpy as np

import advect as ad


def loss(x):
    return np.sum(np.sin(x) ** 2)


gradient_program = ad.grad(ad.stage(loss, np.linspace(0.0, 1.0, 4)))
for node_id in gradient_program.graph.node_ids():
    node = gradient_program.graph.get_node(node_id)
    print(f"%{node_id} = {node.op} {node.inputs}  # {node.shape} {node.dtype}")
```

A `StagedProgram` is compiled for exactly one shape/dtype signature — it is
never a hidden multi-signature cache. The staged-producing transforms
`grad(program)`, `value_and_grad(program)`, and `vjp_program(program)` compile
the derivative graph once and return programs whose warm calls execute with no
tape and no reverse sweep. Dense higher-order transforms remain concrete
invocation-time callables even when their input function is a staged program.

Its serialized form has two nested contracts. Python core owns the versioned
outer envelope: call and output pytrees, their leaf specifications, the
constant manifest, the optimization report, and consistency among those
records and the graph. `advect-runtime` owns the enclosed canonical graph
artifact and validates the `GraphStore` before accepting it. The combined JSON
is a Python `StagedProgram` artifact because its outer envelope uses Python
pytree and codec identities. Only the enclosed flat runtime graph is portable
to a non-Python host; that host consumes the runtime contract, links the named
operations, and supplies flat inputs and outputs.

Because the staged trace has no data, it is also stricter: Python truth tests
on traced values, ambient randomness, and data-dependent shapes are rejected
at trace time rather than silently frozen. The [debugging
tutorial](tutorials/debugging.md) maps those errors to their rewrites.

## Derivatives are JVP-first

The preferred derivative rule is a JVP written as ordinary traceable code.
Advect obtains reverse mode by transposing it and validates the required
real-linearity instead of trusting it. A primitive may instead provide an
explicit transpose without a JVP: it has reverse mode but no forward mode or
structural transposition. That rule can support reverse-over-reverse when its
body is traceable and non-residual. Residual-bearing primitives use an
inherently first-order boundary when their exact adjoint needs
invocation-local state. Derivatives are real-linear throughout, which gives
complex and non-holomorphic functions one consistent convention, and a missing
rule is a named error — Advect never substitutes a numeric approximation
silently at runtime.

## Mutation is functionalized

Traced arrays are not writable in place. Supported mutation syntax — indexed
updates, augmented assignment through basic slices — is rewritten at the
tracer boundary into immutable SSA updates, and everything else raises with a
suggested rewrite. Inputs are never implicitly writable: copy first, then
update the owned copy. This keeps the recorded graph a pure dataflow program,
which is what makes transposition, optimization, and serialization tractable.

## The native layer

Advect ships as one Python distribution with a required native extension, in
three pieces:

- **`advect`** (Python) owns Python-language integration and authored
  numerical/autodiff semantics: tracer frontends, primitive bindings and
  derivative rules, pytrees, the user API, and the outer `StagedProgram`
  envelope with its cross-record validation.
- **`advect-runtime`** (pure Rust, no Python bindings) owns structural graph
  semantics and policy: the canonical artifact, closed graph data,
  `GraphStore` validation, fixed cleanup with conservative op-aware rewrites,
  execution planning, and host-independent value lifetimes.
- **`advect-native`** (PyO3) translates between Python and that runtime and
  owns the invocation-local dynamic tape.

The split is deliberate: Python never mirrors the graph with per-node objects,
and the pure runtime interprets only the portable graph structure and closed
metadata needed to validate, optimize, and execute a plan. The host supplies
the linked operation behavior and array kernels. Advect contains no kernel
compiler, code generation, or fusion; a staged program's speed comes from
tracing once, optimizing once, and executing with native traversal over
provider operations.

## Deliberate boundaries

- **No dispatch patching.** NumPy is intercepted through its own protocols
  (`__array_ufunc__`, `__array_function__`, `like=`); integrations register
  explicitly on import, never through entry-point discovery. The sole
  process-visible patch is a lock-protected, depth-counted guard around NumPy's
  ambient RNG entry points during abstract staging. Non-staging callers reach
  the originals, and the last staging scope restores the original attributes.
- **Deterministic frontend hooks.** Core's private dependency-inversion seam is
  populated at import time. A hook name is single-assignment: registering the
  same callable again is harmless, while a different callable is rejected;
  there is no unregister or reset path.
- **No compiler ambitions.** Kernel compilation, backend lowering, and
  automatic checkpoint placement are out of scope; the staged optimizer's
  pass list is fixed and raw graph rewriting is not a user API.
- **No hidden fallbacks.** Unsupported behavior fails with a rewrite
  suggestion rather than detaching gradients or degrading precision.
- **No orchestration.** Workflow scheduling, artifact storage, and remote
  execution are independent layers above the serialized program.

The contributor-level specifications behind this page — requirements, data
structures, and the architecture decision records — live in the repository's
[`design/` directory](https://github.com/yaugenst/advect/tree/main/design).
