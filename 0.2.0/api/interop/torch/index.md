# PyTorch

The [shared bridge contract](https://yaugenst.github.io/advect/0.2.0/api/interop/#shared-contract) and [host-framework tutorial](https://yaugenst.github.io/advect/0.2.0/tutorials/host-frameworks/index.md) explain what crosses this boundary.

## wrap

```python
wrap(
    function: Callable[..., object],
) -> Callable[..., object]
```

Wrap a NumPy-backed callable as a first-order PyTorch operation.

Every tensor leaf in positional or keyword arguments with a NumPy floating or complex representation is an Advect input. Static configuration should be closed over by `function`. Values execute through NumPy on the host and outputs return to the inputs' common device. One PyTorch backward consumes the retained Advect pullback.

```python
import numpy as np
import torch

from advect.interop.torch import wrap


def numpy_energy(value):
    return np.sum(np.sin(value) ** 2)


energy = wrap(numpy_energy)
x = torch.linspace(0, 1, 8, requires_grad=True)
energy(x).backward()
print(x.grad)
```

Inputs are copied through host NumPy. Outputs return to the common input device, and input gradients restore each input tensor's device and dtype. Eager [`torch.autograd`](https://docs.pytorch.org/docs/stable/autograd.html) is the supported boundary. One backward consumes the retained Advect [`Pullback`](https://yaugenst.github.io/advect/0.2.0/api/transforms/#advect.Pullback); repeated backward over the same retained PyTorch graph is not supported. PyTorch dtypes without a NumPy representation, including `bfloat16`, Float8, and `complex32`, reject at the boundary.
