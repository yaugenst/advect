# Arrays and Scientific Python

Advect works with the scientific Python stack in three complementary ways. NumPy and other Array API providers execute the numerical operations, [`advect.scipy`](https://yaugenst.github.io/advect/dev/api/scipy/index.md) supplies differentiable versions of selected SciPy functions, and [`advect.xarray`](https://yaugenst.github.io/advect/dev/api/xarray/index.md) preserves labels and coordinates around derivative results.

## Write provider-neutral array code

Ask an input for its [Array API](https://yaugenst.github.io/advect/dev/compatibility/array-api/index.md) namespace when the same function should run on more than one provider:

```python
import numpy as np

import advect as ad


def centered_energy(x):
    xp = x.__array_namespace__()
    centered = x - xp.mean(x)
    return xp.sum(centered * centered)


x = np.asarray([1.0, 2.0, 4.0], dtype=np.float32)
gradient = ad.grad(centered_energy)(x)
print("NumPy gradient:", gradient)
```

The same function accepts another qualified provider without changing its body. For example, with [`array-api-strict`](https://data-apis.org/array-api-strict/) installed:

```python
import array_api_strict as xp

import advect as ad


def centered_energy(x):
    namespace = x.__array_namespace__()
    centered = x - namespace.mean(x)
    return namespace.sum(centered * centered)


x = xp.asarray([1.0, 2.0, 4.0], dtype=xp.float32)
print(ad.grad(centered_energy)(x))
# Array([-2.6666667, -0.6666667, 3.3333333], dtype=array_api_strict.float32)
```

[With NumPy, write ordinary `numpy.*` calls](https://yaugenst.github.io/advect/dev/api/numpy/index.md); its own protocols route them through Advect. When a constructor contains traced values, give it a live dispatch anchor with `like=` or use [`ad.array`](https://yaugenst.github.io/advect/dev/api/arrays/#advect.array) or [`ad.asarray`](https://yaugenst.github.io/advect/dev/api/arrays/#advect.asarray). CuPy follows the Array API path; its [compatibility page](https://yaugenst.github.io/advect/dev/compatibility/cupy/index.md) describes the supported boundary.

## Use the differentiable SciPy namespace

Install `advect[scipy]` and import functions from `advect.scipy`. Direct `scipy.*` calls are not intercepted. The [special-function reference](https://yaugenst.github.io/advect/dev/api/scipy/special/index.md) and [SciPy compatibility table](https://yaugenst.github.io/advect/dev/compatibility/scipy/index.md) give the exact callable and lifetime contracts:

```python
import numpy as np

import advect as ad
from advect.scipy.special import logsumexp


scores = np.array([1.0, 2.0, 4.0])
value, gradient = ad.value_and_grad(logsumexp)(scores)
print(f"logsumexp: {value:.6f}")
print("softmax gradient:", gradient)
# logsumexp: 4.169846
# softmax gradient: [0.04201007 0.1141952  0.84379473]
```

The optional namespace also includes differentiable [image filters](https://yaugenst.github.io/advect/dev/api/scipy/ndimage/index.md) and [solver callbacks](https://yaugenst.github.io/advect/dev/api/scipy/solvers/index.md) used by [implicit differentiation](https://yaugenst.github.io/advect/dev/tutorials/implicit-differentiation/index.md).

## Preserve xarray labels

[`advect.xarray`](https://yaugenst.github.io/advect/dev/api/xarray/index.md) registers [`DataArray`](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html) and [`Dataset`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) as dynamic pytrees. Data buffers are differentiable; dimensions, coordinates, names, and attributes are static metadata restored on derivative results.

```python
import numpy as np
import xarray as xr

import advect as ad
import advect.xarray  # register the container types


field = xr.DataArray(
    np.arange(6.0).reshape(2, 3),
    dims=("y", "x"),
    coords={"y": [10, 20], "x": [1, 2, 3]},
    name="field",
    attrs={"units": "V"},
)


def labeled_energy(x):
    centered = x - x.mean("x")
    return (centered * centered).sum(("x", "y"))


gradient = ad.grad(labeled_energy)(field)
print(gradient.values)
print("dims:", gradient.dims, "units:", gradient.attrs["units"])
# [[-2.  0.  2.]
#  [-2.  0.  2.]]
# dims: ('y', 'x') units: V
```

Floating and complex data variables are differentiable. Integer, boolean, string, and object buffers are rejected rather than assigned meaningless zero gradients. xarray owns alignment and named operations; Advect owns the array operations they lower to.

The [xarray boundary](https://yaugenst.github.io/advect/dev/compatibility/xarray/index.md) is dynamic-only. Stage the raw array kernel, call it with `field.data`, and restore labels outside the program when reusable staging is needed.
