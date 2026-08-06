# Gradients and Pytrees

## Differentiate a NumPy function

`grad` expects a real scalar output and differentiates the first argument by default:

```python
import numpy as np

import advect as ad


def loss(x):
    return np.sum(np.sin(x) ** 2)


x = np.linspace(-0.5, 0.5, 8)
gradient = ad.grad(loss)(x)
np.testing.assert_allclose(gradient, 2 * np.sin(x) * np.cos(x))
```

The function is traced with the concrete value on every call. Python branches, loops, and data-dependent shapes therefore behave as they do in an ordinary call, provided every numerical operation is supported by the frontend.

Real Python scalar inputs use the same array tracer. Advect lifts them to zero-dimensional `float64` arrays at the transform boundary and returns their derivatives as Python scalars:

```python
gradient = ad.grad(lambda value: value * value)(3.0)
assert gradient == 6.0
```

This is boundary convenience rather than a parallel scalar-operation frontend. Use provider operations inside functions that need transcendental functions or mixed array/scalar behavior.

## Construct arrays from traced values

Keep ordinary NumPy and give its constructor a live dispatch anchor through the standard `like=` parameter:

```python
def constructor_loss(x):
    coefficients = np.array(
        [[x[0], x[1]], [x[1], 2 * x[0]]],
        like=x,
    )
    return coefficients.sum().item()


gradient = ad.grad(constructor_loss)(np.array([1.0, 2.0]))
np.testing.assert_allclose(gradient, np.array([3.0, 2.0]))
```

`like=x` selects Advect's constructor handling and x's array provider; it does not itself create a mathematical dependence on `x`. `np.array` creates an owned value by default, while `np.asarray` preserves a direct tracer when no conversion is required. Both accept rectangular nested lists or tuples.

`ad.array` and `ad.asarray` remain explicit provider-neutral alternatives. For constructor-heavy migration code, `import advect.numpy as np` is a secondary convenience: it overrides the traced constructors and delegates every other attribute directly to the installed NumPy module.

## Multiple arguments and pytrees

Use `argnums` to select positional arguments. Dictionaries, lists, tuples, and custom pytree nodes preserve their structure:

```python
parameters = {
    "weight": np.array([1.0, 2.0, 3.0]),
    "bias": np.array([0.25, -0.5, 0.75]),
}
features = np.array([2.0, -1.0, 0.5])


def model_loss(params, inputs):
    prediction = params["weight"] * inputs + params["bias"]
    return np.sum(prediction**2)


dparameters, dfeatures = ad.grad(
    model_loss,
    argnums=(0, 1),
)(parameters, features)
```

The first result has the same dictionary structure as `parameters`. Keyword arguments can be selected with `argnames`:

```python
def scaled_loss(value, *, scale):
    return np.sum(scale * value**2)


positional, named = ad.grad(
    scaled_loss,
    argnums=(0,),
    argnames=("scale",),
)(features, scale=0.5)
```

`jacobian` uses the same selection model and preserves both sides of the derivative. For an output leaf with shape `(m,)` and an input leaf with shape `(n,)`, its block has shape `(m, n)`. Output and input pytrees remain nested rather than being flattened into one package-specific matrix:

```python
def model(params, inputs):
    return {
        "prediction": params["weight"] * inputs + params["bias"],
        "energy": np.sum(inputs**2),
    }


blocks = ad.jacobian(model, argnums=(0, 1))(parameters, features)
assert blocks["prediction"][0]["weight"].shape == (3, 3)
assert blocks["energy"][1].shape == (3,)
```
