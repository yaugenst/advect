# Host Autodiff Interop

Advect can wrap a NumPy-backed function as one differentiable operation inside
[PyTorch](https://pytorch.org/), [JAX](https://docs.jax.dev/), or
[HIPS Autograd](https://github.com/HIPS/autograd). The outer framework keeps
its arrays and computation graph, while Advect supplies the wrapped function's
[VJP](../transforms.md#advect.vjp). The
[host-framework tutorial](../../tutorials/host-frameworks.md) shows the pattern
in a complete example.

## Install and import

The base `advect` import loads no host framework. Install and import only the
bridge you use:

| Framework | Extra | Entry point | Reverse-mode execution |
| --- | --- | --- | --- |
| [PyTorch](torch.md) | `advect[torch]` | [`advect.interop.torch.wrap(function)`](torch.md#advect.interop.torch.wrap) | Retains and consumes the forward Advect pullback |
| [JAX](jax.md) | `advect[jax]` | [`advect.interop.jax.wrap(function, ...)`](jax.md#advect.interop.jax.wrap) | Executes eagerly or uses callbacks with a JIT/shape contract |
| [HIPS Autograd](autograd.md) | `advect[autograd]` | [`advect.interop.autograd.wrap(function)`](autograd.md#advect.interop.autograd.wrap) | Retains the reusable forward Advect linearization |

## Shared contract

The callable accepts one or more positional or keyword tuple, list, or
dictionary pytrees, and every supplied leaf is differentiated. PyTorch leaves
are tensors, JAX leaves are arrays, and HIPS Autograd also accepts NumPy or
Python numeric scalars. Custom containers are supported only when both Advect
and the host framework recognize the same structure. Close over static
configuration rather than passing static leaves. Inputs and differentiable
outputs use standard NumPy floating or complex dtypes and outputs are nonempty
pytrees. JAX may additionally return a nondifferentiable auxiliary pytree with
`has_aux=True`.

All three bridges are first-order reverse-mode boundaries. They do not support
Advect [staging](../staging.md), host forward mode, or higher derivatives. The
adapters handle the frameworks' different complex cotangent conventions, so
native host losses over complex outputs receive the gradient convention
expected by that host.
