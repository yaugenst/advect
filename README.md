<!-- markdownlint-disable-next-line MD033 -->
<h1 align="center"><img src="https://raw.githubusercontent.com/yaugenst/advect/main/docs-theme/img/logo.svg" alt="Advect"></h1>

<!-- markdownlint-disable MD033 -->
<p align="center">
  <a href="https://github.com/yaugenst/advect/actions/workflows/ci.yml"><img src="https://github.com/yaugenst/advect/actions/workflows/ci.yml/badge.svg?branch=main&amp;event=push" alt="CI"></a>
  <a href="https://app.codecov.io/gh/yaugenst/advect"><img src="https://codecov.io/gh/yaugenst/advect/branch/main/graph/badge.svg" alt="Coverage"></a>
  <a href="https://pypi.org/project/advect/"><img src="https://img.shields.io/pypi/v/advect.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/advect/"><img src="https://img.shields.io/pypi/pyversions/advect.svg" alt="Python versions"></a>
</p>
<!-- markdownlint-enable MD033 -->

Advect is a focused automatic differentiation library for scientific Python,
with broad NumPy API coverage and first-class support for the Python Array API
standard. For repeated workloads, it can stage a function into a reusable,
optimized program that can itself be differentiated, saved, and loaded.

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
