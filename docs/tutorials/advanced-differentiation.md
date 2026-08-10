# Higher-Order Differentiation

Derivative functions remain differentiable. This lets the same transforms
produce higher-order derivatives, Hessian products, and exact curvature. For
large reverse traces, checkpointing trades extra computation for lower memory.

## Compose transforms

Nesting `grad` gives successive scalar derivatives without a separate
higher-order tracing mode:

```{.python .run}
import numpy as np

import advect as ad


first = ad.grad(lambda x: x**4)
second = ad.grad(first)
third = ad.grad(second)

print(
    "derivatives at 2:",
    first(2.0),
    second(2.0),
    third(2.0),
)
```

Every operation on the nested path needs traceable derivative rules. A
first-order primitive raises at that boundary instead of silently falling back
to a numerical derivative.

## Use curvature without always building a matrix

`hvp` applies the Hessian to a direction without materializing the dense
matrix. This is usually what a second-order optimizer needs. `hessian` builds
the full matrix for small inputs, while `hessian_diag` computes its exact
diagonal without building the full matrix first.

```{.python .run}
def rosenbrock(x):
    a, b = x
    return (1.0 - a) ** 2 + 100.0 * (b - a**2) ** 2


point = np.array([-1.2, 1.0])
direction = np.array([1.0, 0.5])

loss, product = ad.hvp(rosenbrock)(point, vectors=direction)
hessian = ad.hessian(rosenbrock)(point)
diagonal = ad.hessian_diag(rosenbrock)(point)

np.testing.assert_allclose(product, hessian @ direction)
np.testing.assert_allclose(diagonal, np.diag(hessian))
print("loss:", loss)
print("H @ direction:", product)
print("Hessian diagonal:", diagonal)
```

Dense Hessian storage grows quadratically with the number of input
coordinates. Prefer HVPs when an algorithm only needs curvature along one or a
few directions.

## Checkpoint large pure blocks

Reverse mode normally retains forward intermediates until the backward pass
uses them. `checkpoint` makes a pure function atomic on the outer dynamic tape:
Advect keeps the boundary values and recomputes the body while taking
derivatives.

```{.python .run}
@ad.checkpoint
def nonlinear_block(x):
    return np.sin(x) ** 2 + 0.1 * x


def deep_loss(x):
    for _ in range(4):
        x = nonlinear_block(x)
    return np.sum(x)


sample = np.linspace(-0.5, 0.5, 5)
gradient = ad.grad(deep_loss)(sample)
_, curvature = ad.hvp(deep_loss)(
    sample,
    vectors=np.ones_like(sample),
)
print("checkpointed gradient:", gradient)
print("checkpointed H @ 1:", curvature)
```

Checkpoint placement is manual. Put it around large deterministic blocks whose
saved intermediates cost more than replaying the block. Mutable state and side
effects are outside the contract, and `stage` rejects checkpointed calls.

For a different kind of advanced derivative—one defined by a converged
equation rather than an executed algorithm—continue with [Implicit
Differentiation](implicit-differentiation.md).
