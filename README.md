# Advect

Advect differentiates ordinary NumPy and Array API programs. Dynamic transforms
follow the Python code that runs for each call; `stage` turns one fixed input
signature into an immutable program that can be reused, saved, and loaded.

Advect is pre-1.0 software. Public APIs and serialized programs may change
before the first stable release.

## Try it

Advect requires Python 3.12 or newer. The first package release is still being
prepared, so install the current source from a checkout:

```bash
python -m pip install .
```

```python
import numpy as np

import advect as ad


def energy(x):
    return np.sum(np.sin(x) ** 2)


x = np.array([0.0, 0.5, 1.0])
value, gradient = ad.value_and_grad(energy)(x)

print(f"loss: {value:.6f}")
print("gradient:", np.round(gradient, 6))
# loss: 0.937922
# gradient: [0.       0.841471 0.909297]
```

Each dynamic call traces its own branches, loops, and supported local array
updates. When the same shape and dtype will run repeatedly, stage the same
function:

```python
program = ad.stage(energy, x)
restored = ad.StagedProgram.from_dict(program.to_dict())
print(f"restored loss: {restored(x):.6f}")
# restored loss: 0.937922
```

The core API also covers JVPs, VJPs, Jacobians, higher-order derivatives,
checkpointing, implicit differentiation, and custom primitives. Optional
integrations add SciPy functions, xarray-aware gradients, and first-order
bridges into JAX, PyTorch, and HIPS Autograd.

## Learn and contribute

Start with the [runnable tutorials](https://yaugenst.github.io/advect/latest/tutorials/),
then use the [API reference](https://yaugenst.github.io/advect/latest/api/) and
[compatibility tables](https://yaugenst.github.io/advect/latest/compatibility/)
for exact contracts. Contributors should begin with
[CONTRIBUTING.md](https://github.com/yaugenst/advect/blob/main/CONTRIBUTING.md)
and the [developer guide](https://yaugenst.github.io/advect/latest/development/).

Report security issues through GitHub's
[private advisory form](https://github.com/yaugenst/advect/security/advisories/new).
Advect is distributed under the
[MIT License](https://github.com/yaugenst/advect/blob/main/LICENSE).
