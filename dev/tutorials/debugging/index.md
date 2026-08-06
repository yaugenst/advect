# Debugging

Start with the exception. Advect errors name the rejected behavior and usually give the safe rewrite. If that is not enough, rerun the same call in one scoped debug mode:

```python
import advect as ad
import numpy as np

with ad.debug():
    gradient = ad.grad(loss)(x)
```

The scope adds user source locations and bounded live-tracer values. Keep it around the transform call; for staging, keep it around `ad.stage(...)` so the program retains those locations.

## I got an exception

Common rewrites are:

- **Input mutation:** copy first with `owned = x.copy()`.
- **Stale view:** consume or copy the view before updating its base.
- **Unsupported conversion:** use a traced `like=` anchor, `ad.asarray`, or `ad.array`; move unsupported library calls outside the transform or wrap the numerical behavior with `@ad.primitive`.
- **Staged Python branch:** use array operations, make the decision static, or leave the function dynamic.
- **Ambient staged randomness:** pass explicit state or data as an input.
- **Complex output from `grad`:** use `linearize`, `jvp`, or `vjp`.

Missing derivative errors suggest the exact `with ad.debug():` retry. Staged provider errors retain their original type and traceback and append the failing operation, up to three inputs, and a source location when available.

## I need to see a live value

Use ordinary `print(x)` or debugger inspection. Normal mode shows trace identity, shape, and dtype. Debug mode adds a bounded preview and finite count:

```text
TracedArray(node=%2, shape=(4,), dtype=float64,
            finite=4/4, values=array([...]))
```

This is display-only; tracer payloads remain private. `ad.stop_gradient` is the explicit concrete dynamic boundary, while staging has no payload to detach.

## I got a NaN or infinity

Ask Advect to locate the first non-finite dynamic value:

```python
with ad.debug(numerics=True):
    gradient = ad.grad(loss)(x)
```

`NumericsError` reports the primal, JVP, or VJP phase, operation, source, shape/dtype, finite count, and bounded preview. This mode adds provider work and synchronization, so it is intentionally scoped.

## The gradient is finite but looks wrong

Check the composed function against numerical behavior:

```python
from advect.testing import check_gradient

check_gradient(loss, x, tangent=np.ones_like(x))
```

The unary argument may itself be a pytree. `check_gradient` compares the whole-function JVP with central differences over an epsilon sweep and checks the reverse gradient against that direction. It reevaluates the function, so use deterministic inputs and keep effects outside it.

This does not duplicate `check_primitive`: the function check covers the actual composition a user ran; the primitive check validates one extension's concrete/abstract contract, JVP, transpose, complex behavior, nesting, and staging. On a mismatch, `check_gradient` names custom primitives on that path and directs their authors to `check_primitive`.

Passing only establishes consistency with the function that ran. It cannot prove the intended mathematics, and finite differences remain unreliable near discontinuities or badly scaled regions.

## Dynamic execution works but staging fails

Dynamic transforms follow concrete Python control flow. `stage` must instead produce one reusable graph for one shape/dtype signature, so data-dependent Python branches, output shapes, ambient state, or unsupported staged operations can work dynamically and correctly fail during staging.

Inspect the compiled result directly:

```python
with ad.debug():
    program = ad.stage(function, example)

print(program)        # bounded optimized operation sequence
print(program.graph)  # immutable graph summary
print(program.optimization)
```

`program.trace` contains the in-process pre-optimization trace and ID mapping; it is absent from loaded artifacts. There is no separate `explain()` step.

## Tracers and views

A tracer retained after its transform raises `EscapedTracerError`; return an ordinary result instead. Views use conservative whole-cell epochs, so a view is stale after any update to its root. Rewrite `x[i][j] += value` as one update such as `x[i, j] += value`, or copy the view first.
