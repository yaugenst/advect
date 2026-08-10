# HIPS Autograd

::: advect.interop.autograd.wrap

```python
import autograd
import autograd.numpy as anp
import numpy as np

from advect.interop.autograd import wrap


energy = wrap(lambda value: np.sum(np.sin(value) ** 2))
gradient = autograd.grad(energy)(anp.linspace(0, 1, 8))
print(gradient)
```

The custom primitive retains the exact Advect pullback from its forward call.
A nested Autograd trace raises an explicit higher-order error.
