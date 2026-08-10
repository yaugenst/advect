# Advanced Differentiation

Gradients answer one scalar question. Advect also exposes complete local
linear models, higher-order derivatives, and an explicit computation-for-memory
tradeoff for large reverse-mode traces.

## Assemble a Jacobian

`jacobian` differentiates every output coordinate with respect to every input
coordinate. Its result shape is `output_shape + input_shape`:

```{.python .run}
import numpy as np

import advect as ad


def response(parameters):
    x, y = parameters
    return np.array(
        [x * y, np.sin(x) + y**2],
        like=parameters,
    )


parameters = np.array([0.5, 2.0])
jacobian = ad.jacobian(response)(parameters)
print("Jacobian:\n", jacobian)
```

For pytree inputs or outputs, the result is a matching tree of Jacobian blocks
rather than one flattened matrix. Advect chooses forward- or reverse-mode
assembly from the traced input and output sizes.

## Compose transforms for higher derivatives

Derivative functions remain differentiable. Nesting `grad` therefore gives
successive scalar derivatives without a separate higher-order tracing mode:

```{.python .run}
first = ad.grad(lambda value: value**4)
second = ad.grad(first)
third = ad.grad(second)

print(
    "derivatives at 2:",
    first(2.0),
    second(2.0),
    third(2.0),
)
```

Nested differentiation requires every operation on the path to have traceable
derivative rules. A deliberately first-order primitive raises instead of
silently substituting a numerical derivative.

## Use curvature without always building a matrix

`hvp` applies the Hessian to a direction without materializing the dense
matrix. This is usually the useful operation in an optimizer. `hessian` and
`hessian_diag` assemble exact dense curvature when the input is small enough:

```{.python .run}
def rosenbrock(value):
    x, y = value
    return (1.0 - x) ** 2 + 100.0 * (y - x**2) ** 2


point = np.array([-1.2, 1.0])
direction = np.array([1.0, 0.5])

loss, product = ad.hvp(rosenbrock)(point, vectors=direction)
hessian = ad.hessian(rosenbrock)(point)
diagonal = ad.hessian_diag(rosenbrock)(point)

np.testing.assert_allclose(product, hessian @ direction)
np.testing.assert_allclose(diagonal, np.diag(hessian))
print("loss:", loss)
print("H @ v:", product)
print("Hessian:\n", hessian)
```

Dense Hessians scale quadratically with the number of input coordinates. Use
an HVP when an algorithm only needs curvature along one or a few directions.

## Trade computation for memory with checkpointing

Reverse mode normally retains forward intermediates until the backward pass
uses them. `checkpoint` makes a pure function atomic on the outer dynamic tape:
Advect keeps the boundary values and recomputes the function body while taking
derivatives.

```{.python .run}
@ad.checkpoint
def nonlinear_block(value):
    return np.sin(value) ** 2 + 0.1 * value


def deep_loss(value):
    for _ in range(4):
        value = nonlinear_block(value)
    return np.sum(value)


sample = np.linspace(-0.5, 0.5, 5)
gradient = ad.grad(deep_loss)(sample)
_, curvature = ad.hvp(deep_loss)(
    sample,
    vectors=np.ones_like(sample),
)
print("checkpointed gradient:", gradient)
print("checkpointed H @ 1:", curvature)
```

Checkpointed functions must be deterministic from their explicit inputs;
mutable state and side effects are outside the contract. Checkpointing is a
dynamic rematerialization boundary, so `stage` rejects it. It is most useful
around large pure blocks whose saved intermediates cost more than replaying
the block.
