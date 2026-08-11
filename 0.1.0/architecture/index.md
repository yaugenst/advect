# Architecture

NumPy or another Array API provider still performs the numerical work. Advect records how the array values were produced, then uses that record to calculate derivatives immediately or build a reusable program that can be saved and loaded. This separation keeps Advect focused on differentiation while the array library remains responsible for execution.

## Different array APIs, one set of operations

NumPy and the Array API expose similar numerical work through different Python interfaces. Advect understands NumPy ufuncs, array functions, methods, `out=`, constructors, and local mutation. It also understands the versioned namespace returned by `x.__array_namespace__()`. Each path interprets its own public call, then records the same internal operation.

Shape, dtype, and other trace-time queries take a shorter path. They return metadata needed by Python or by an operation attribute without adding a numerical graph node.

```text
numpy.* ─────────────> NumPy frontend ────┐
                                          ├─ array call ─> canonical operation
__array_namespace__ → Array API frontend ─┤                  ├─ dynamic tape
        ↑                                 │                  ├─ staged graph
        └─ NumPy or CuPy provider         │                  └─ derivative rules
                                          └─ metadata ─> trace-time metadata
                                                         (no graph node)
```

This shared model lets one derivative rule describe the numerical operation rather than every Python spelling that can reach it. The public interfaces remain separate because they can still differ in signatures and supported execution modes.

## One set of rules, two ways to run

Dynamic transforms and staged programs use the same operation identities, abstract shape and dtype rules, and derivative rules. They differ in how long the recorded computation lives.

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

`grad(f)(x)` runs `f` with concrete values wrapped in tracers. Each supported array operation appends to an invocation-local tape, and reverse mode walks that tape backward before releasing it. Because tracing happens inside an ordinary call, branches take the branch selected by the data and loops run for the current number of iterations.

Tracing each call afresh lets a later call take a different path or use a different shape. `vjp` keeps one call's tape in a single-use `Pullback`; applying or closing it releases the tape. `linearize` returns a reusable `LinearMap` that keeps its tape until closed. Both objects belong to the call that created them.

### Staging fixes one signature

`stage(f, example)` traces with abstract values—shape and dtype without data—so it can build a graph before execution. Advect validates that graph and runs a fixed cleanup pipeline before returning an immutable `StagedProgram`.

```python
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

The program accepts the same input structure, shapes, dtypes, devices, Python scalar categories, and static values it was compiled for. Later calls execute the graph without retracing Python. `grad(program)`, `value_and_grad(program)`, and `vjp_program(program)` build reusable derivative programs in the same way.

A saved `StagedProgram` combines a Python call envelope with an enclosed flat runtime graph. The envelope owns Python pytrees, input and output specifications, the constant manifest, and cross-record consistency. The runtime graph owns provider-neutral operations, portable constant bytes, optimization, scheduling, and value lifetimes. Another host can consume the flat graph contract, but the complete `StagedProgram` remains a Python artifact.

Abstract tracing is stricter by design. Data-dependent Python branches, ambient randomness, and data-dependent shapes cannot define one reusable graph and therefore fail during staging rather than being frozen accidentally.

## Derivatives and mutation stay composable

Advect prefers JVP rules written as ordinary traceable code. Forward mode uses them directly; reverse mode structurally transposes them when possible. The smaller set of operations that needs a direct real adjoint provides an explicit transpose. Missing rules raise a named error instead of silently substituting a numerical approximation.

Advect treats every derivative as a real-linear map. This gives complex and non-holomorphic functions one consistent convention: a real loss can use `grad`, while a complex output uses `jvp`, `vjp`, or `linearize`.

Supported mutation syntax is functionalized into immutable updates. Inputs are never implicitly writable: copy first, then update the owned local array. This keeps both the dynamic tape and staged graph as pure dataflow, which is what makes transposition, optimization, and serialization fit together.

## The array provider remains the numerical backend

The provider continues to own numerical kernels and device execution. Advect records, differentiates, and schedules provider calls; it does not replace them with a new numerical backend. Staging saves the work of tracing and constructing derivatives, while each operation still runs through the provider.

Each integration keeps that ownership clear. NumPy uses its protocols, Array API providers use their namespaces, xarray registers container structure, and host-autodiff bridges wrap one Advect function. When an operation cannot cross one of these boundaries, Advect raises an error instead of silently detaching gradients or changing precision.

Workflow scheduling, artifact storage, and remote execution belong above the serialized program. Contributor-level ownership and dependency rules live in the [developer guide](https://yaugenst.github.io/advect/0.1.0/development/index.md).
