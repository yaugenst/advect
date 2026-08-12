# JVPs, VJPs, and Linear Maps

A gradient is one special view of a derivative. For a general function, the
derivative at a point is a linear map: a
[JVP](../api/transforms.md#advect.jvp) applies it to an input direction, while a
[VJP](../api/transforms.md#advect.vjp) applies its transpose to an output
cotangent. Advect exposes both without first building a dense Jacobian.

## Push a direction forward

`jvp` returns the function value and `J @ direction`:

```{.python .run}
import numpy as np

import advect as ad


def response(parameters):
    x, y = parameters
    return np.array(
        [x * y, np.sin(x) + y**2],
        like=parameters,
    )


point = np.array([0.5, 2.0])
direction = np.array([1.0, -0.25])
response_jvp = ad.jvp(response)
value, tangent = response_jvp(point, tangents=direction)

print("response:", value)
print("J @ direction:", tangent)
```

This is forward mode. Its cost follows the number of directions, which makes
it natural when the input is small or only a few directional derivatives are
needed.

## Pull a cotangent backward

[`vjp`](../api/transforms.md#advect.vjp) evaluates the function and returns a
one-shot [`Pullback`](../api/transforms.md#advect.Pullback). Applying the
pullback computes `J.T @ cotangent`:

```{.python .run}
cotangent = np.array([2.0, -1.0])
response_vjp = ad.vjp(response)
value, pullback = response_vjp(point)
input_cotangent = pullback(cotangent)

print("J.T @ cotangent:", input_cotangent)
```

The pullback owns this call's dynamic trace and releases it when consumed. Use
it as a context manager, or call `close()`, if a path may decide not to apply
it.

## Reuse one local linear model

[`linearize`](../api/transforms.md#advect.linearize) captures the derivative
once and returns a reusable [`LinearMap`](../api/transforms.md#advect.LinearMap).
It can apply the map or its transpose until it is closed:

```{.python .run}
value, linear = ad.linearize(response, point)
with linear:
    along_x = linear(np.array([1.0, 0.0]))
    along_y = linear(np.array([0.0, 1.0]))
    pulled_back = linear.pullback(np.ones(2))

print("basis directions:", along_x, along_y)
print("pullback of ones:", pulled_back)
```

The map belongs to one concrete call; it is reusable, but it is not a
[`StagedProgram`](../api/staging.md#advect.StagedProgram) or a cache across input
points.

## Materialize the Jacobian only when useful

[`jacobian`](../api/transforms.md#advect.jacobian) assembles the dense matrix and
preserves array or pytree block shapes. The products above and the matrix are
three views of the same local derivative:

```{.python .run}
jacobian = ad.jacobian(response)(point)
np.testing.assert_allclose(tangent, jacobian @ direction)
np.testing.assert_allclose(input_cotangent, jacobian.T @ cotangent)
print("Jacobian:\n", jacobian)
```

Dense assembly is convenient for small problems. For larger ones, JVPs and
VJPs avoid storing every matrix entry.

## Complex derivatives are real-linear

Advect treats complex arrays as pairs of real coordinates. A real scalar loss
therefore has the expected gradient, including for non-holomorphic functions:

```{.python .run}
z = np.array([1.0 + 2.0j, -0.5 + 0.25j])
gradient = ad.grad(lambda value: np.sum(np.abs(value) ** 2))(z)
np.testing.assert_allclose(gradient, 2 * z)

_, conjugate_tangent = ad.jvp(lambda value: np.conj(value))(
    z,
    tangents=1j * np.ones_like(z),
)
print("gradient of |z|²:", gradient)
print("D conj(z)[i]:", conjugate_tangent)
```

Complex-output [`grad`](../api/transforms.md#advect.grad) and a single dense
complex Jacobian would be ambiguous; use the product transforms above for those
real-linear maps. Dense Jacobians and
[Hessians](advanced-differentiation.md#use-curvature-without-always-building-a-matrix)
are limited to real inputs and outputs.
