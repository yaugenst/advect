# Host Autodiff Interop

Advect can expose a NumPy-backed callable as one differentiable operation
inside PyTorch, JAX, or HIPS Autograd. These adapters carry Advect's VJP into
the host framework; they do not make host arrays into Advect providers.

## Install and import

Install only the framework you use:

```bash
python -m pip install "advect[torch]"
python -m pip install "advect[jax]"
python -m pip install "advect[autograd]"
```

The base `advect` import loads none of these dependencies. Import the adapter
you intend to use:

| Framework | Entry point | Reverse-mode execution |
| --- | --- | --- |
| PyTorch | `advect.interop.torch.wrap(function)` | Retains and consumes the forward Advect pullback |
| JAX | `advect.interop.jax.wrap(function, has_aux=False, result_shape_dtypes=None)` | Executes eagerly or uses callbacks with a JIT/shape contract |
| HIPS Autograd | `advect.interop.autograd.wrap(function)` | Retains and consumes the forward Advect pullback |

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
Advect staging, host forward mode, or higher derivatives. The adapters handle
the frameworks' different complex cotangent conventions, so native host losses
over complex outputs receive the gradient convention expected by that host.

## PyTorch

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

Inputs are copied through host NumPy. Outputs return to the common input device,
and input gradients restore each input tensor's device and dtype. Eager
`torch.autograd` is the supported boundary. One backward consumes the retained
Advect pullback; repeated backward over the same retained PyTorch graph is not
supported. PyTorch dtypes without a NumPy representation, including `bfloat16`,
Float8, and `complex32`, reject at the boundary.

## JAX

Without a result specification, the adapter executes eagerly and observes the
actual output structure, shapes, and dtypes:

```python
import jax
import jax.numpy as jnp
import numpy as np

from advect.interop.jax import wrap


def numpy_energy(value):
    return np.sum(np.sin(value) ** 2)


energy = wrap(numpy_energy)
x = jnp.linspace(0, 1, 8)
value, gradient = jax.value_and_grad(energy)(x)
```

For auxiliary outputs, pass `has_aux=True` to both the bridge and the JAX
transformation. Only the first result participates in the Advect VJP:

```python
def energy_and_metrics(value):
    return numpy_energy(value), {"iterations": np.asarray(3, dtype=np.int32)}


energy_with_metrics = wrap(energy_and_metrics, has_aux=True)
(value, metrics), gradient = jax.value_and_grad(
    energy_with_metrics,
    has_aux=True,
)(x)
```

Auxiliary leaves may be JAX-compatible floating, complex, integer, or boolean
arrays and scalars. They remain nondifferentiable even when floating. Opaque
Python objects such as strings cannot cross this JAX operation boundary.

JAX needs output shapes and dtypes during JIT compilation and abstract shape
evaluation. Supply that contract to use `jax.jit` or `jax.eval_shape`:

```python
compiled_energy = wrap(
    energy_and_metrics,
    has_aux=True,
    result_shape_dtypes=(
        jax.ShapeDtypeStruct((), np.float32),
        {"iterations": jax.ShapeDtypeStruct((), np.int32)},
    ),
)
(value, metrics), gradient = jax.jit(
    jax.value_and_grad(compiled_energy, has_aux=True)
)(x)
```

If staging starts without `result_shape_dtypes`, the bridge raises a `TypeError`
that explains how to enable the callback path. It does not cache a contract
from an earlier eager call. A supplied specification must match the callable's
output structure, shapes, and dtypes exactly; the bridge does not cast callback
results. The callback path uses a pure host callback, so the callable must be
deterministic and free of externally visible effects. Reverse mode replays it
once at the saved primal values to construct and immediately consume an Advect
pullback. `jax.vmap` remains unsupported; an output specification does not add
batching semantics. An effectful or remote operation needs an
application-specific token/residual adapter instead.

## HIPS Autograd

```python
import autograd
import autograd.numpy as anp
import numpy as np

from advect.interop.autograd import wrap


energy = wrap(lambda value: np.sum(np.sin(value) ** 2))
gradient = autograd.grad(energy)(anp.linspace(0, 1, 8))
```

The custom primitive retains the exact Advect pullback from its forward call.
A nested Autograd trace raises an explicit higher-order error.
