# ![Advect](docs-theme/img/logo.svg)

Advect brings automatic differentiation to NumPy code. Write a numerical
function as usual, then calculate its gradients and other derivatives without
deriving or maintaining them by hand.

You keep familiar NumPy operations, Python branches, loops, and helper
functions. Advect follows the path each call takes. For repeated work with the
same input structure, `stage` turns the function into a reusable program that
can be saved and loaded.

## Try it

```bash
python -m pip install advect
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

`value_and_grad` evaluates `energy` and differentiates it with respect to `x`.
When the same kinds of inputs will run repeatedly, stage the function once:

```python
program = ad.stage(energy, x)
restored = ad.StagedProgram.from_dict(program.to_dict())
print(f"restored loss: {restored(x):.6f}")
# restored loss: 0.937922
```

From the same code, Advect can compute JVPs, VJPs, Jacobians, Hessians, and
higher-order or implicit derivatives. The [core API](https://yaugenst.github.io/advect/latest/api/)
also covers checkpointing and custom primitives. Optional integrations add
[SciPy and xarray workflows](https://yaugenst.github.io/advect/latest/tutorials/scientific-python/)
and [bridges to other autodiff frameworks](https://yaugenst.github.io/advect/latest/tutorials/host-frameworks/).

## Related projects

Other NumPy-focused autodiff projects include
[HIPS Autograd](https://github.com/HIPS/autograd) and
[MyGrad](https://mygrad.readthedocs.io/en/latest/). Advect also offers optional
[bridges](https://yaugenst.github.io/advect/latest/api/interop/) that let a
NumPy-backed function participate in [JAX](https://docs.jax.dev/en/latest/quickstart.html)
or [PyTorch](https://docs.pytorch.org/docs/stable/).

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
