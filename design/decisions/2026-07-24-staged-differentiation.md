# ADR: Staged Differentiation

**Date:** 2026-07-24
**Status:** Accepted

## Context

The reset gives Advect two explicit lifetimes on one SSA substrate. Ordinary
autodiff records an invocation-local `DynamicTape`; `stage` builds an optimized,
serializable `GraphStore`. Before this decision, differentiating a
`StagedProgram` reused its optimized primal execution but still recorded a new
tape and ran a reverse sweep on every call. That left the main benefit of
staging unavailable to repeated gradient workloads.

We want staged gradients without restoring a second derivative engine, a
public graph-transformation system, or graph construction on ordinary dynamic
calls.

## Decision

`grad` and `value_and_grad` preserve an explicit staged lifetime:

```python
primal = ad.stage(loss, specs=(spec,))
gradient = ad.grad(primal)

assert isinstance(gradient, ad.StagedProgram)
```

Compilation nests the existing lifetimes once. An outer abstract trace owns a
`GraphBuilder`. Inside it, Advect replays the source program's already-optimized
primal through the ordinary dynamic linearizer and transpose rules. Operations
performed by those rules are abstract values in the outer trace, so the
complete derivative is emitted directly into the outer builder. The temporary
tape is released after compilation, and the fixed
`DCE -> simplify -> CSE` pipeline produces the derivative
`GraphStore`.

The inner tape represents every outer tracer explicitly. Selected primal
leaves are active inputs; unselected positional and keyword leaves are passive
inputs. A passive leaf remains a real operand—never a captured constant—while
its activity bit prevents the inner reverse sweep from producing a gradient
for it. This makes derivative compilation independent of operand order, such
as `weight * value * value` versus `value * value * weight`.

For `grad`, the compiler seeds the rank-zero pullback with a typed graph
constant rather than an operation depending on the primal result. DCE can
therefore remove primal nodes used only to produce the discarded loss.
`value_and_grad` retains that path because the primal result is an output.

The resulting program has the same runtime contract as any other staged
program:

- warm calls execute a prebound `GraphExecutionPlan` without tracing or a
  reverse sweep;
- the graph, optimizer report, constants, schemas, and compile time are
  inspectable directly;
- `to_dict()` and `from_dict()` detach and restore the derivative artifact;
- the source program's one signature is compiled into the derivative
  immediately;
- the derivative is another singular program with an unconditional graph.

This is not a third autodiff path. It reuses the dynamic derivative rules as
the compiler frontend and the staged builder, optimizer, executor, and
serializer as the durable backend.

The durable transform surface is `grad`, `value_and_grad`, and `vjp_program`,
including multi-argument selection, pytrees, keyword `argnames`, auxiliary
outputs, complex real gradients, functionalized mutation, conforming Array API
providers, and traceable custom primitives. `vjp_program` adds one explicit
keyword-only `cotangent` input whose pytree and array specifications are copied
from the primal outputs. Ordinary `vjp` and `linearize` still return
invocation-local callable state and remain dynamic.

Opaque residual primitives are barriers. Their state belongs to one concrete
provider invocation and never enters `GraphStore`; serializing it or silently
recomputing it would violate the residual contract. A future extension would
need an explicit staged residual representation rather than an implicit
fallback.

## Consequences

Repeated gradient workloads can pay tracing, transposition, and optimization
once while ordinary `grad(f)` keeps near-Autograd define-by-run behavior.
Derivative artifacts receive the same deterministic optimization and
durability guarantees as primal artifacts, with no user-selectable pass API.

Compilation can be more expensive than one dynamic gradient call because it
performs both abstract replay and one temporary reverse construction. That cost
is explicit in `program.compile_seconds` and is amortized only when the program
is reused.
