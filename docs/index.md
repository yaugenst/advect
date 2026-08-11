# Advect

Advect brings automatic differentiation to NumPy code. Write a numerical
function with familiar array operations and Python control flow, then ask for
its value, gradient, or another derivative without maintaining the derivative
by hand.

Dynamic transforms follow the branches and loops taken by each call. When the
same kinds of inputs will run repeatedly, `stage` turns the function into a
reusable program that can be saved and loaded.

## Install

```bash
python -m pip install advect
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
value, gradient = ad.value_and_grad(energy)(x)
print(f"loss: {value:.6f}")
print("gradient:", np.round(gradient, 6))
```

Advect differentiates the path its inputs take, so
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
