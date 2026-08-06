# Architecture

Advect is an autodiff core, not an array library. NumPy — or another Array API provider — keeps doing the numerics; Advect's job is to record what a function does to its arrays, differentiate that record, and, when asked, turn it into a durable program. Everything in the design follows from keeping that job small.

## Two frontends, one semantic core

NumPy and the Array API are separate calling contracts. The concrete NumPy frontend owns `numpy.*` signatures, ufunc methods, aliases, `out=`, and functionalized mutation. The provider-neutral Array API frontend owns the versioned namespace returned by `x.__array_namespace__()`. They share code only after each frontend has bound its foreign call to a canonical operation.

```text
flowchart LR
    NP["numpy.*"] --> NF["Concrete NumPy frontend"]
    XP["x.__array_namespace__()"] --> AF["Array API frontend"]
    CU["CuPy provider"] --> AF
    NU["NumPy provider"] --> AF
    NF --> OP["Canonical operation and complete OpDef"]
    AF --> OP
    OP --> DY["Dynamic tape"]
    OP --> ST["Staged graph"]
    OP --> AD["JVP and transpose rules"]
```

CuPy reaches Advect through the designated Array API provider path; Advect does not maintain a speculative direct `cupy.*` frontend. Its historical GPU record is source-only until a retained report and digest verify the current gate. A future direct frontend would bind its own calling convention and join at the same canonical-operation boundary.

## One core, two lifetimes

Every operation Advect understands is a **primitive** with four faces: a concrete evaluation, an abstract evaluation over shapes and dtypes, a JVP (directional-derivative) rule, and a transpose derived from the JVP. The transforms you call — `grad`, `jvp`, `vjp`, `jacobian`, `stage` — are all built from those four faces. Adding an operation means supplying the faces, never touching the transforms.

What differs between the two execution modes is not the math but the *lifetime* of the record:

```text
              ┌─────────────────────────────────────┐
              │ User code                           │
              │   z = np.sin(x) + y                 │
              └──────────────────┬──────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │ Primitive semantics                 │
              │   concrete · abstract · JVP ·       │
              │   transpose                         │
              └────────┬───────────────────┬────────┘
                       │                   │
    ┌──────────────────▼────────┐ ┌────────▼──────────────────┐
    │ Dynamic lifetime          │ │ Staged lifetime           │
    │   concrete SSA tape       │ │   abstract trace →        │
    │   → pullback              │ │   durable optimized graph │
    └───────────────────────────┘ └───────────────────────────┘
```

## The dynamic path

`grad(f)(x)` runs `f` with the real values, wrapped in tracers that record each primitive onto an invocation-local SSA tape. Because the trace happens inside an ordinary call, Python control flow needs no special forms: loops unroll to the iterations that actually ran, branches take the branch the data took. The reverse sweep walks that tape backwards through the transpose rules and then releases it — nothing is cached between calls, and no graph outlives the invocation.

That per-call freshness is the point of the dynamic mode: a new trace can have new shapes, new branches, even a different number of loop iterations. The cost is that tracing work is repeated on every call.

## The staged path

`stage(f, example)` runs the same trace once with *abstract* values — shape and dtype, no data — and keeps the result. The recorded graph is validated, then run through a fixed optimization pipeline (dead-code elimination, simplification, common-subexpression elimination) into an immutable program:

```python
import numpy as np

import advect as ad


def loss(x):
    return np.sum(np.sin(x) ** 2)


gradient_program = ad.grad(ad.stage(loss, np.linspace(0.0, 1.0, 4)))
for node_id in gradient_program.graph.node_ids():
    node = gradient_program.graph.get_node(node_id)
    print(f"%{node_id} = {node.op} {node.inputs}  # {node.shape} {node.dtype}")
```

A `StagedProgram` is compiled for exactly one shape/dtype signature — it is never a hidden multi-signature cache. Derivatives of programs are programs: `grad(program)` and `vjp_program(program)` compile the derivative graph once, and warm calls execute it with no tape and no reverse sweep. Programs serialize to a versioned JSON artifact and load anywhere the referenced primitives are importable.

Because the staged trace has no data, it is also stricter: Python truth tests on traced values, ambient randomness, and data-dependent shapes are rejected at trace time rather than silently frozen. The [debugging tutorial](https://yaugenst.github.io/advect/dev/tutorials/debugging/index.md) maps those errors to their rewrites.

## Derivatives are JVP-first

A primitive author writes one derivative rule — the JVP — as ordinary traceable code. Reverse mode is obtained by transposing that rule, and Advect validates the required real-linearity instead of trusting it. Derivatives are real-linear throughout, which gives complex and non-holomorphic functions one consistent convention, and a missing rule is a named error — Advect never substitutes a numeric approximation silently.

## Mutation is functionalized

Traced arrays are not writable in place. Supported mutation syntax — indexed updates, augmented assignment through basic slices — is rewritten at the tracer boundary into immutable SSA updates, and everything else raises with a suggested rewrite. Inputs are never implicitly writable: copy first, then update the owned copy. This keeps the recorded graph a pure dataflow program, which is what makes transposition, optimization, and serialization tractable.

## The native layer

Advect ships as one Python distribution with a required native extension, in three pieces:

- **`advect`** (Python) owns all semantics: the tracer frontends, the primitive registry, derivative rules, pytrees, and the user API.
- **`advect-runtime`** (pure Rust, no Python bindings) owns structure and lifetime: the SSA arena, the dynamic tape, the durable graph store, the optimization passes, and the execution planner.
- **`advect-native`** (PyO3) is the thin adapter between the two.

The split is deliberate: Python never mirrors the graph with per-node objects, and Rust never interprets what an operation means. Array computation itself stays with the provider — Advect contains no kernel compiler, no code generation, and no fusion; a staged program's speed comes from tracing once, optimizing once, and executing with native traversal over provider operations.

## Deliberate boundaries

- **No dispatch patching.** NumPy is intercepted through its own protocols (`__array_ufunc__`, `__array_function__`, `like=`); integrations register explicitly on import, never through entry-point discovery. The sole process-visible patch is a lock-protected, depth-counted guard around NumPy's ambient RNG entry points during abstract staging. Non-staging callers reach the originals, and the last staging scope restores the original attributes.
- **Deterministic frontend hooks.** Core's private dependency-inversion seam is populated at import time. A hook name is single-assignment: registering the same callable again is harmless, while a different callable is rejected; there is no unregister or reset path.
- **No compiler ambitions.** Kernel compilation, backend lowering, and automatic checkpoint placement are out of scope; the staged optimizer's pass list is fixed and raw graph rewriting is not a user API.
- **No hidden fallbacks.** Unsupported behavior fails with a rewrite suggestion rather than detaching gradients or degrading precision.
- **No orchestration.** Workflow scheduling, artifact storage, and remote execution are independent layers above the serialized program.

The contributor-level specifications behind this page — requirements, data structures, and the architecture decision records — live in the repository's [`design/` directory](https://github.com/yaugenst/advect/tree/main/design).
