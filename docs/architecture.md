# Architecture

Advect records array computation while NumPy or another Array API provider
performs the arithmetic. From that record it can apply derivatives immediately
or build one reusable, serializable program. Keeping those responsibilities
separate is the central design choice.

## From array calls to one operation model

NumPy and the Array API are different calling contracts. The NumPy frontend
understands ufuncs, array functions, methods, `out=`, constructors, and local
mutation. The provider-neutral frontend understands the versioned namespace
returned by `x.__array_namespace__()`. Each frontend first binds its own public
call, then both emit the same canonical operations.

Shape, dtype, and other trace-time queries take a shorter path. They return
metadata needed by Python or by an operation attribute without adding a
numerical graph node.

```text
numpy.* ─────────────> NumPy frontend ────┐
                                          ├─ array call ─> canonical operation
__array_namespace__ → Array API frontend ─┤                  ├─ dynamic tape
        ↑                                 │                  ├─ staged graph
        └─ NumPy or CuPy provider         │                  └─ derivative rules
                                          └─ metadata ─> trace-time metadata
                                                         (no graph node)
```

This convergence matters: a derivative rule describes the operation, not every
spelling that can reach it. Frontend support remains explicit because sharing
an operation does not make two public signatures or execution modes identical.

## One semantic core, two execution modes

Dynamic transforms and staged programs use the same operation identities,
abstract shape and dtype rules, and derivative rules. They differ in how long
the recorded computation lives.

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
    │   → pullback              │ │   saved optimized graph   │
    └───────────────────────────┘ └───────────────────────────┘
```

### Dynamic execution follows Python

`grad(f)(x)` runs `f` with concrete values wrapped in tracers. Each supported
array operation appends to an invocation-local tape, and reverse mode walks
that tape backward before releasing it. Because tracing happens inside an
ordinary call, branches take the branch selected by the data and loops run for
the current number of iterations.

The fresh trace is the feature: a later call can have a different path or
shape. `vjp` retains one call's tape in a single-use `Pullback`; applying or
closing it releases the tape. `linearize` returns a reusable `LinearMap` that
retains its tape until closed. Neither handle is a cross-call cache.

### Staging fixes one signature

`stage(f, example)` traces with abstract values—shape and dtype without data—so
it can build a graph before execution. Advect validates that graph and runs a
fixed cleanup pipeline before returning an immutable `StagedProgram`.

```{.python .run}
import numpy as np

import advect as ad


def loss(x):
    return np.sum(np.sin(x) ** 2)


example = np.linspace(0.0, 1.0, 4)
gradient_program = ad.grad(ad.stage(loss, example))
print("staged gradient:", np.round(gradient_program(example), 6))
print(
    "optimized nodes:",
    f"{gradient_program.optimization.nodes_before} -> "
    f"{gradient_program.optimization.nodes_after}",
)
```

The program accepts exactly its compiled pytree, shapes, dtypes, devices,
Python scalar categories, and static values. Warm calls execute the graph
without retracing Python. `grad(program)`, `value_and_grad(program)`, and
`vjp_program(program)` compile reusable derivative programs in the same way.

A saved `StagedProgram` combines a Python call envelope with an enclosed flat
runtime graph. The envelope owns Python pytrees, input and output
specifications, the constant manifest, and cross-record consistency. The
runtime graph owns provider-neutral operations, portable constant bytes,
optimization, scheduling, and value lifetimes. Another host can consume the
flat graph contract, but the complete `StagedProgram` remains a Python artifact.

Abstract tracing is stricter by design. Data-dependent Python branches,
ambient randomness, and data-dependent shapes cannot define one reusable graph
and therefore fail during staging rather than being frozen accidentally.

## Derivatives and mutation stay composable

Advect prefers JVP rules written as ordinary traceable code. Forward mode uses
them directly; reverse mode structurally transposes them when possible. The
smaller set of operations that needs a direct real adjoint provides an explicit
transpose. Missing rules raise a named error instead of silently substituting a
numerical approximation.

All derivatives are real-linear. This gives complex and non-holomorphic
functions one consistent convention: a real loss can use `grad`, while a
complex output uses `jvp`, `vjp`, or `linearize`.

Supported mutation syntax is functionalized into immutable updates. Inputs are
never implicitly writable: copy first, then update the owned local array. This
keeps both the dynamic tape and staged graph as pure dataflow, which is what
makes transposition, optimization, and serialization fit together.

## Where Advect stops

The provider still owns numerical kernels and device execution. Advect does not
compile kernels, generate code, fuse operations, or choose devices. Staging
saves tracing and derivative construction; it does not turn provider calls into
a new numerical backend.

Integrations are equally explicit. NumPy uses its protocols, Array API
providers use their namespaces, xarray registers pytree structure, and host
autodiff bridges wrap one Advect function. Unsupported behavior fails with a
rewrite rather than detaching gradients or changing precision.

Workflow scheduling, artifact storage, and remote execution belong above the
serialized program. Contributor-level ownership and dependency rules live in
the [developer guide](development/index.md).
