# Architecture

System design and internals for Advect.

## Vision

Advect is an extensible autodiff core. Concrete dynamic tracing and abstract
staging share primitive semantics but not lifetime machinery. NumPy and Array
API code remain ordinary user code; the core owns differentiation,
functionalization, and the staged artifact boundary.

## System Overview

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
    │   → LinearMap             │ │   portable Rust GraphStore│
    └───────────────────────────┘ └───────────────────────────┘
```

## Sections

- [Requirements](requirements.md) — Current requirements (R1-R17) and bounded
  extension contracts (E1-E9)
- [Public architecture](../docs/architecture.md) — Execution and lifetime model
- [API reference](../docs/api/index.md) — Current public transforms, staging,
  primitives, arrays, and optional integrations
- [Codebase map](../docs/development/codebase.md) — Internal ownership and
  dependency boundaries
- [Decisions](decisions/README.md) — Design decisions and rationale

## Boundaries

Advect keeps deployment policy and compiler policy outside its semantic core:

1. Workflow orchestration, restart, and generic object storage are independent
   layers.
2. Dynamic tapes are differentiated directly; they are not converted into
   durable graphs and optimized before each backward pass.
3. The staged compiler owns its fixed graph-pass sequence; raw graph rewrites
   are not a user API.
4. Mutation remains tracer-level functionalization, with a bounded tier for
   direct basic-slice views rather than complete NumPy view replay.
5. Buffer donation may reuse proven-dead storage but can never change logical
   SSA semantics.

Manual dynamic rematerialization, the built-in Array API compatibility bridge,
direct named basic-slice mutation, and internal staged buffer donation are
current bounded extensions. The
[runtime extension decision](decisions/2026-07-24-runtime-extension-boundaries.md)
defines their contracts, the CuPy qualification gate, and the capabilities
that remain deferred or excluded.

Matrix-free implicit roots and the bounded built-in SciPy and xarray modules,
whose third-party dependencies are installed through extras, are the second
extension set. The
[scientific extension decision](decisions/2026-07-26-scientific-extension-contracts.md)
defines their lifetime, dependency, and metadata boundaries.

The optional `advect.interop` modules expose NumPy-backed Advect callables as
first-order PyTorch, JAX, or HIPS Autograd VJP operations without registering
those frameworks as providers. The
[host interop decision](decisions/2026-08-01-host-autodiff-vjp-bridges.md)
defines their host-transfer, replay, and complex-cotangent boundaries.

The implemented
[lifetime-aware portable runtime decision](decisions/2026-07-24-lifetime-aware-portable-rust-runtime.md)
defines Rust's current boundary: measured payload lifetime and a PyO3-free
staged host runtime, without kernel compilation. Reproducible measurements are
described by the [performance contract](implementation/performance.md) and
emitted as CI or release evidence.

The [Python distribution decision](decisions/2026-07-26-single-python-distribution.md)
defines publication separately from those semantic boundaries: built-in Python
modules share one `advect` distribution, while the Rust crates retain their own
internal ownership boundaries.
