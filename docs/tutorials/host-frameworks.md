# Host Autodiff Frameworks

An Advect function can appear as one differentiable operation inside
[PyTorch](../api/interop/torch.md), [JAX](../api/interop/jax.md), or
[HIPS Autograd](../api/interop/autograd.md). The host framework keeps its own
arrays and outer graph; the bridge runs the NumPy-backed Advect function and
carries its [VJP](linear-maps.md#pull-a-cotangent-backward) back into the host.

This is different from an
[Array API provider](scientific-python.md#write-provider-neutral-array-code). A
PyTorch or JAX array does not become an Advect array, and Advect does not trace
the host program around the wrapped call.

## Wrap one NumPy function

The PyTorch bridge works with ordinary eager
[`torch.autograd`](https://docs.pytorch.org/docs/stable/autograd.html):

```python
import numpy as np
import torch

from advect.interop.torch import wrap


def numpy_energy(x):
    return np.sum(np.sin(x) ** 2)


energy = wrap(numpy_energy)
x = torch.linspace(0, 1, 5, requires_grad=True)
value = energy(x)
value.backward()

print(f"loss: {value.detach():.6f}")
print("PyTorch gradient:", x.grad)
# loss: 1.463763
# PyTorch gradient: tensor([0.0000, 0.4794, 0.8415, 0.9975, 0.9093])
```

Install and import only the bridge you use:

| Framework | Extra | Entry point |
| --- | --- | --- |
| [PyTorch](../api/interop/torch.md) | `advect[torch]` | [`advect.interop.torch.wrap`](../api/interop/torch.md#advect.interop.torch.wrap) |
| [JAX](../api/interop/jax.md) | `advect[jax]` | [`advect.interop.jax.wrap`](../api/interop/jax.md#advect.interop.jax.wrap) |
| [HIPS Autograd](../api/interop/autograd.md) | `advect[autograd]` | [`advect.interop.autograd.wrap`](../api/interop/autograd.md#advect.interop.autograd.wrap) |

JAX can run eagerly without an output specification.
[`jax.jit`](../api/interop/jax.md) and abstract shape evaluation need an exact
`result_shape_dtypes` contract because JAX must know the callback result before
running it.

## Keep the boundary explicit

The [shared bridge contract](../api/interop/index.md#shared-contract) is a
first-order reverse-mode boundary. The bridges support positional and keyword
built-in pytrees whose leaves the host can differentiate, but they do not add
host forward mode, higher derivatives, Advect staging, or static argument
annotations. Close static configuration over the wrapped function.

The bridge copies values through host NumPy, so it is best for bounded
operations whose numerical work is large enough to justify the boundary. It is
not a way to make an entire host model execute inside Advect.
