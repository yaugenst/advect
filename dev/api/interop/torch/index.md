# PyTorch

## wrap

```python
wrap(function: Callable[..., Any]) -> Callable[..., Any]
```

Wrap a NumPy-backed callable as a first-order PyTorch operation.

Every tensor leaf with a NumPy floating or complex representation is an Advect input. Static configuration should be closed over by `function`. Values execute through NumPy on the host and outputs return to the inputs' common device. One PyTorch backward consumes the retained Advect pullback.

```python
import numpy as np
import torch

from advect.interop.torch import wrap


def numpy_energy(value):
    return np.sum(np.sin(value) ** 2)


energy = wrap(numpy_energy)
x = torch.linspace(0, 1, 8, requires_grad=True)
energy(x).backward()
```

Inputs are copied through host NumPy. Outputs return to the common input device, and input gradients restore each input tensor's device and dtype. Eager `torch.autograd` is the supported boundary. One backward consumes the retained Advect pullback; repeated backward over the same retained PyTorch graph is not supported. PyTorch dtypes without a NumPy representation, including `bfloat16`, Float8, and `complex32`, reject at the boundary.
