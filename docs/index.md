# Advect

Advect is a focused automatic differentiation library for scientific Python,
with broad NumPy API coverage and first-class support for the Python Array API
standard. For repeated workloads, it can stage a function into a reusable,
optimized program that can itself be differentiated, saved, and loaded.

!!! tip "Try Advect in your browser"

    Open the [playground](playground.md) to trace a NumPy expression, inspect
    its derivative graph, and run it without installing anything.

## Install

```bash
python -m pip install advect
# or
conda install -c conda-forge advect
```

Add optional integrations with extras such as `advect[scipy]`,
`advect[xarray]`, `advect[jax]`, `advect[torch]`, or `advect[autograd]`.

## Your first gradient

[`value_and_grad`](api/transforms.md#advect.value_and_grad) evaluates a scalar
function and returns its gradient in the same call. Press `[ run ]` to execute
the example in your browser:

```{.python .run}
import numpy as np

import advect as ad


def energy(x):
    return np.sum(np.sin(x) ** 2)


x = np.array([0.0, 0.5, 1.0])
energy_and_gradient = ad.value_and_grad(energy)
value, gradient = energy_and_gradient(x)
print(f"loss: {value:.6f}")
print("gradient:", np.round(gradient, 6))
```

Advect differentiates the path its inputs take, so
[Python branches, loops, helper functions, and supported local mutation](tutorials/control-flow.md)
keep their normal meaning.

## Keep going

- Follow the [tutorials](tutorials/index.md) from gradients through staged
  programs and custom primitives.
- Check exact callable coverage in [Compatibility](compatibility/index.md).
- Use the [API reference](api/index.md) for signatures and contracts.
- Read [Architecture](architecture.md) for the execution model and its
  boundaries.
