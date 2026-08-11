# Gradients and Pytrees

Start with the NumPy function you want to differentiate. If it returns one real scalar, [`grad`](https://yaugenst.github.io/advect/0.1.0/api/transforms/#advect.grad) produces a new function that accepts the same inputs and returns the gradient.

## Differentiate a NumPy function

```python
import numpy as np

import advect as ad


def loss(x):
    return np.sum(np.sin(x) ** 2)


x = np.linspace(-0.5, 0.5, 8)
gradient = ad.grad(loss)(x)

np.testing.assert_allclose(gradient, 2 * np.sin(x) * np.cos(x))
print("gradient:", np.round(gradient, 6))
```

The transform traces the concrete call, runs reverse mode, and releases the trace before returning. A later call can take different branches, shapes, or loop counts; [Dynamic control flow](https://yaugenst.github.io/advect/0.1.0/tutorials/control-flow/index.md) develops that model.

Use [`value_and_grad`](https://yaugenst.github.io/advect/0.1.0/api/transforms/#advect.value_and_grad) when an optimizer needs the objective and gradient from the same evaluation:

```python
value, gradient = ad.value_and_grad(loss)(x)
updated = x - 0.1 * gradient

print(f"loss: {value:.6f}")
print(f"loss after one step: {loss(updated):.6f}")
```

If the function also returns diagnostics, set [`has_aux=True`](https://yaugenst.github.io/advect/0.1.0/api/transforms/#advect.value_and_grad). The auxiliary value follows the call but is not differentiated:

```python
def loss_with_metrics(x):
    value = loss(x)
    return value, {"maximum": np.max(np.abs(x)), "size": x.size}


value, gradient, metrics = ad.value_and_grad(
    loss_with_metrics,
    has_aux=True,
)(x)
print(f"loss: {value:.6f}; metrics: {metrics}")
```

## Select arguments and preserve structure

[Pytrees](https://yaugenst.github.io/advect/0.1.0/api/pytree/index.md) keep the structure of lists, tuples, dictionaries, and registered application nodes. The [`argnums` and `argnames`](https://yaugenst.github.io/advect/0.1.0/api/transforms/#advect.grad) parameters select positional and named arguments.

```python
parameters = {
    "weight": np.array([1.0, 2.0, 3.0]),
    "bias": np.array([0.25, -0.5, 0.75]),
}
features = np.array([2.0, -1.0, 0.5])


def model_loss(params, inputs, *, scale):
    prediction = params["weight"] * inputs + params["bias"]
    return scale * np.sum(prediction**2)


(dparameters, dfeatures), named = ad.grad(
    model_loss,
    argnums=(0, 1),
    argnames=("scale",),
)(parameters, features, scale=0.5)

print("weight gradient:", dparameters["weight"])
print("feature gradient:", dfeatures)
print("scale gradient:", named["scale"])
```

The gradient tree mirrors the selected input tree. Real Python scalars are accepted at the boundary and return Python-scalar derivatives; numerical work inside the function still follows the [active array provider](https://yaugenst.github.io/advect/0.1.0/tutorials/scientific-python/#write-provider-neutral-array-code).

## Stop one dependency explicitly

[`stop_gradient`](https://yaugenst.github.io/advect/0.1.0/api/arrays/#advect.stop_gradient) keeps a concrete dynamic value in the computation while removing its derivative contribution. Here the normalization scale is measured from the input but treated as fixed during differentiation:

```python
def normalized_loss(x):
    scale = ad.stop_gradient(np.max(np.abs(x)))
    return np.sum((x / scale) ** 2)


sample = np.array([-2.0, 1.0])
gradient = ad.grad(normalized_loss)(sample)
print("gradient with fixed scale:", gradient)
```

Stopping a gradient is an explicit modeling choice, not a way to hide an unsupported operation. It is available only for dynamic calls because a [staged trace](https://yaugenst.github.io/advect/0.1.0/tutorials/staging/index.md) has no concrete value to detach.

Next, see how the same transform handles [branches, loops, and local mutation](https://yaugenst.github.io/advect/0.1.0/tutorials/control-flow/index.md).
