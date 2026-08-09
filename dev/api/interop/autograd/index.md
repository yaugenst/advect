# HIPS Autograd

## wrap

```python
wrap(function: Callable[..., Any]) -> Callable[..., Any]
```

Wrap a NumPy-backed callable as a first-order HIPS Autograd primitive.

Every NumPy floating or complex leaf in the positional arguments is selected. The bridge translates between Autograd's complex-bilinear cotangents and Advect's real-adjoint convention. Higher-order differentiation is rejected.

```python
import autograd
import autograd.numpy as anp
import numpy as np

from advect.interop.autograd import wrap


energy = wrap(lambda value: np.sum(np.sin(value) ** 2))
gradient = autograd.grad(energy)(anp.linspace(0, 1, 8))
```

The custom primitive retains the exact Advect pullback from its forward call. A nested Autograd trace raises an explicit higher-order error.
