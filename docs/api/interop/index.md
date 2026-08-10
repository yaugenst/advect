# Host Autodiff Interop

Advect can expose a NumPy-backed callable as one differentiable operation
inside [PyTorch](https://pytorch.org/), [JAX](https://docs.jax.dev/), or
[HIPS Autograd](https://github.com/HIPS/autograd). These adapters carry
Advect's [VJP](../transforms.md#advect.vjp) into the host framework; they do not
make host arrays into Advect providers. The
[host-framework tutorial](../../tutorials/host-frameworks.md) shows the boundary
in one complete example.

## Install and import

The base `advect` import loads no host framework. Install and import only the
bridge you use:

| Framework | Extra | Entry point | Reverse-mode execution |
| --- | --- | --- | --- |
| [PyTorch](torch.md) | `advect[torch]` | [`advect.interop.torch.wrap(function)`](torch.md#advect.interop.torch.wrap) | Retains and consumes the forward Advect pullback |
| [JAX](jax.md) | `advect[jax]` | [`advect.interop.jax.wrap(function, ...)`](jax.md#advect.interop.jax.wrap) | Executes eagerly or uses callbacks with a JIT/shape contract |
| [HIPS Autograd](autograd.md) | `advect[autograd]` | [`advect.interop.autograd.wrap(function)`](autograd.md#advect.interop.autograd.wrap) | Retains and consumes the forward Advect pullback |

## Shared contract

The callable accepts one or more positional tuple, list, or dictionary pytrees,
and every leaf is differentiated. PyTorch leaves are tensors, JAX leaves are
arrays, and HIPS Autograd also accepts NumPy or Python numeric scalars. Custom
containers are supported only when both Advect and the host framework recognize
the same structure. Close over static configuration instead of passing static
leaves or keyword arguments. Inputs and differentiable outputs use standard
NumPy floating or complex dtypes and outputs are nonempty pytrees. JAX may
additionally return a nondifferentiable auxiliary pytree with `has_aux=True`.

All three bridges are first-order reverse-mode boundaries. They do not support
Advect [staging](../staging.md), host forward mode, or higher derivatives. The
adapters handle the frameworks' different complex cotangent conventions, so
native host losses over complex outputs receive the gradient convention
expected by that host.
