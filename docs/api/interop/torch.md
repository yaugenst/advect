# PyTorch

The [shared bridge contract](index.md#shared-contract) and
[host-framework tutorial](../../tutorials/host-frameworks.md) explain what
crosses this boundary.

::: advect.interop.torch.wrap

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

Inputs are copied through host NumPy. Outputs return to the common input device,
and input gradients restore each input tensor's device and dtype. Eager
[`torch.autograd`](https://docs.pytorch.org/docs/stable/autograd.html) is the
supported boundary. One backward consumes the retained Advect
[`Pullback`](../transforms.md#advect.Pullback); repeated backward over the same
retained PyTorch graph is not supported. PyTorch dtypes without a NumPy
representation, including `bfloat16`, Float8, and `complex32`, reject at the
boundary.
