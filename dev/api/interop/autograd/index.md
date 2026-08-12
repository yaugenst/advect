# HIPS Autograd

This adapter carries an Advect pullback into [HIPS Autograd](https://github.com/HIPS/autograd) through `autograd.grad`. See the [shared bridge contract](https://yaugenst.github.io/advect/dev/api/interop/#shared-contract) and [host-framework tutorial](https://yaugenst.github.io/advect/dev/tutorials/host-frameworks/index.md) for the boundary around the wrapped call.

## wrap

```python
wrap(
    function: Callable[..., object],
) -> Callable[..., object]
```

Wrap a NumPy-backed callable as a first-order HIPS Autograd primitive.

Every NumPy floating or complex leaf in positional or keyword arguments is selected. The bridge translates between Autograd's complex-bilinear cotangents and Advect's real-adjoint convention. The exact forward linearization remains reusable for first-order host transforms; higher-order differentiation is rejected.

```python
import autograd
import autograd.numpy as anp
import numpy as np

from advect.interop.autograd import wrap


energy = wrap(lambda value: np.sum(np.sin(value) ** 2))
gradient = autograd.grad(energy)(anp.linspace(0, 1, 8))
print(gradient)
```

The custom primitive retains the exact reusable Advect [`LinearMap`](https://yaugenst.github.io/advect/dev/api/transforms/#advect.LinearMap) from its forward call. First-order transforms such as `autograd.jacobian` may apply its pullback more than once; releasing the host VJP releases the retained linearization. A nested Autograd trace raises an explicit higher-order error.
