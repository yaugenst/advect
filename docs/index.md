# Advect

Differentiate the NumPy you already write. Advect follows ordinary Python
control flow during dynamic transforms, and it can compile one fixed input
signature into a reusable, serializable array program.

!!! warning "Pre-release"

    Advect is under active development. Public APIs and serialized programs
    may change before 1.0.

## Install

Advect supports Python 3.12 through 3.14:

```bash
python -m pip install advect
```

[SciPy](api/scipy/index.md) and [xarray](api/xarray.md) integrations are
available with `.[scientific]`. [JAX](api/interop/jax.md),
[PyTorch](api/interop/torch.md), and
[HIPS Autograd](api/interop/autograd.md) bridges have separate extras.

## Your first gradient

[`value_and_grad`](api/transforms.md#advect.value_and_grad) evaluates a scalar
function and differentiates it in one call. Press `[ run ]` to execute the
example in your browser:

```{.python .run}
import numpy as np

import advect as ad


def energy(x):
    return np.sum(np.sin(x) ** 2)


x = np.array([0.0, 0.5, 1.0])
value, gradient = ad.value_and_grad(energy)(x)
print(f"loss: {value:.6f}")
print("gradient:", np.round(gradient, 6))
```

Each call traces the path its concrete inputs take, so
[Python branches, loops, helper functions, and supported local mutation](tutorials/control-flow.md)
keep their normal meaning.

## Keep going

- Follow the [tutorials](tutorials/index.md) from gradients through staged
  programs and custom primitives.
- Open the [playground](playground.md) to trace an expression and inspect its
  derivative graph in the browser.
- Check exact callable coverage in [Compatibility](compatibility/index.md).
- Use the [API reference](api/index.md) for signatures and contracts.
- Read [Architecture](architecture.md) for the execution model and its
  boundaries.
