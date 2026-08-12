# Dynamic Control Flow and Mutation

[Dynamic transforms](../api/transforms.md) execute the Python function for
every call and differentiate the path its inputs take. Conditions, loops, and
helper functions remain ordinary Python.

## Follow data-dependent branches

```{.python .run}
import numpy as np

import advect as ad


def piecewise_loss(x):
    if np.sum(x) > 0:
        return np.sum(np.sin(x))
    return np.sum(x * x)


gradient = ad.grad(piecewise_loss)
positive = np.array([0.2, 0.4])
negative = np.array([-0.2, -0.4])

print("positive branch:", gradient(positive))
print("negative branch:", gradient(negative))
```

The first call differentiates `sin`; the second differentiates the square.
These are pathwise derivatives: Advect does not differentiate the discrete
decision itself, so the derivative may jump where the branch changes.

## Let loops run for the current input

Iteration counts can also depend on traced values.
[`has_aux=True`](../api/transforms.md#advect.grad) is handy when the function
should report what happened without differentiating the report:

```{.python .run}
def settle_loss(x):
    state = x
    steps = 0
    while np.max(np.abs(state)) > 0.25:
        state = 0.5 * state
        steps += 1
    return np.sum(state * state), steps


initial_state = np.array([2.0, -1.0])
settle_gradient = ad.grad(settle_loss, has_aux=True)
gradient, steps = settle_gradient(initial_state)
print(f"{steps} iterations; gradient:", gradient)
```

The loop is unrolled into this invocation's trace. A later call may run a
different number of iterations and gets a fresh trace. Helper functions behave
the same way: Advect records their supported numerical operations, not the
Python call boundary.

## Update an owned local array

Supported mutation syntax becomes immutable updates on the trace. Inputs are
not writable, so copy first and update the owned local value:

```{.python .run}
def smooth(field):
    updated = field.copy()
    laplacian = field[2:] - 2 * field[1:-1] + field[:-2]
    updated[1:-1] += 0.1 * laplacian
    return updated


def stencil_loss(field):
    updated = smooth(field)
    return np.sum(updated * updated)


field = np.sin(np.linspace(0.0, 2 * np.pi, 128))
updated = smooth(field)
gradient = ad.grad(stencil_loss)(field)
print("largest local update:", np.max(np.abs(updated - field)))
print("edge and center gradients:", gradient[[0, len(gradient) // 2, -1]])
```

Basic indexed updates and direct named basic-slice views are supported.
Mutating an input, updating through advanced indexing, or mutating through an
ambiguous transformed view raises at the offending operation with a suggested
rewrite. [Troubleshooting](debugging.md#common-rewrites) collects the common
fixes.

Dynamic tracing is the right model when the executed path is part of the
computation. If iterations only search for a converged state, use
[implicit differentiation](implicit-differentiation.md) instead of recording
the solver's steps.
