# Troubleshooting

[`debug()`](../api/errors.md#advect.debug) keeps more context during one trace.
Live tracers show bounded value previews, and recorded operations remember the
source location that produced them. The extra context helps locate the first
value or operation that differs from what you expected.

```{.python .run}
import numpy as np

import advect as ad


def show_trace(x):
    print(x)
    return np.sin(x) ** 2


sample = np.array(0.5)
print("ordinary trace")
ad.grad(show_trace)(sample)

print("debug trace")
with ad.debug():
    ad.grad(show_trace)(sample)
```

The ordinary tracer shows shape and dtype. The debug tracer also shows how many
values are finite and a bounded preview of those values. In a real failure,
the recorded location is attached to the error. Keep the context around the
transform call—or around [`ad.stage(...)`](../api/staging.md#advect.stage) when
diagnosing staging. Debug mode does not make an unsupported operation work. See
the full [error hierarchy](../api/errors.md) when handling a specific failure.

## Common rewrites

- Copy an input before mutation: `owned = x.copy()`.
- Consume or copy a view before updating its base.
- Give traced NumPy constructors a `like=` anchor, or use
  [`ad.array`](../api/arrays.md#advect.array) and
  [`ad.asarray`](../api/arrays.md#advect.asarray).
- Move unsupported library work outside the transform, or define its numerical
  contract with [`@ad.primitive`](../api/primitives.md#advect.primitive).
- Replace a data-dependent staged branch with array operations, make the choice
  static, or keep the function dynamic.
- Pass random state or sampled data explicitly when staging.
- Use [`linearize`](linear-maps.md#reuse-one-local-linear-model),
  [`jvp`](../api/transforms.md#advect.jvp), or
  [`vjp`](../api/transforms.md#advect.vjp) instead of
  [`grad`](../api/transforms.md#advect.grad) for a complex output.

## Find the first non-finite value

[`debug(numerics=True)`](../api/errors.md#advect.debug) checks dynamic primal
and derivative values and raises
[`NumericsError`](../api/errors.md#advect.NumericsError) at the first NaN or
infinity:

```{.python .run}
with np.errstate(invalid="ignore"):
    try:
        with ad.debug(numerics=True):
            ad.grad(lambda value: np.sum(np.sqrt(value)))(
                np.array([1.0, -1.0])
            )
    except ad.NumericsError as error:
        print(f"{type(error).__name__}: {error}")
```

The check can add provider work and synchronization, which is why it is scoped
instead of always enabled.

## Check a suspicious gradient

```{.python .run}
from advect.testing import check_gradient


def loss(x):
    centered = x - np.mean(x)
    return np.sum(np.sin(centered) ** 2)


x = np.linspace(-0.5, 0.5, 6)
direction = np.linspace(0.5, 1.5, x.size)
check_gradient(loss, x, tangent=direction)
print("directional gradient check passed")
```

[`check_gradient`](../api/testing.md#advect.testing.check_gradient) compares the
composed function's JVP with central differences over several step sizes and
checks reverse mode against the same direction. It reevaluates the function,
so keep effects outside it. Passing establishes consistency with the function
that ran; it cannot prove that the function is the mathematics you intended,
and finite differences remain unreliable near discontinuities.

Use [`check_primitive`](../api/testing.md#advect.testing.check_primitive) instead
when authoring one custom operation. That check owns its abstract, JVP,
transpose, nesting, complex, and staging contracts.

## When dynamic code cannot be staged

[Dynamic tracing](control-flow.md) knows concrete values.
[Staging](staging.md) knows only the declared structure, shapes, and dtypes, so
a Python truth test on an array value cannot choose one reusable graph:

```{.python .run}
def branch(value):
    if np.sum(value) > 0:
        return 2 * value
    return -value


example = np.array([0.2, 0.4])
try:
    ad.stage(branch, example)
except ad.TracingError as error:
    print(f"{type(error).__name__}: {error}")


def stageable_branch(value):
    return np.where(np.sum(value) > 0, 2 * value, -value)


program = ad.stage(stageable_branch, example)
print("stageable result:", program(example))
```

Data-dependent output shapes and ambient state have the same problem. Inspect a
compiled program with `print(program)` and
[`program.optimization`](../api/staging.md#advect.StagedProgram.optimization);
there is no separate explain step.

## Tracers and views

A tracer retained after its transform raises
[`EscapedTracerError`](../api/errors.md#advect.EscapedTracerError); return an
ordinary result instead. Views use conservative whole-array epochs, so a view
becomes [stale](../api/errors.md#advect.StaleViewError) after its root is
updated. Rewrite `x[i][j] += value` as one update such as
`x[i, j] += value`, or copy the view first.
