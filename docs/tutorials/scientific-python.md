# Arrays and Scientific Python

Advect meets the scientific Python stack at three different boundaries. Array
providers execute numerical operations, `advect.scipy` supplies explicit
differentiable SciPy functions, and `advect.xarray` teaches pytrees how to
preserve labeled containers.

## Write provider-neutral array code

Ask an input for its Array API namespace when the same function should run on
more than one provider:

```{.python .run}
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

The same function accepts another qualified provider without changing its
body. For example, with `array-api-strict` installed:

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

NumPy remains the first-class frontend, so ordinary `numpy.*` calls use NumPy's
protocols directly. When a constructor contains traced values, give it a live
dispatch anchor with `like=` or use `ad.array`/`ad.asarray`. CuPy follows the
provider-neutral path; consult its [current compatibility status](../compatibility/cupy.md)
before treating GPU support as a release claim.

## Use the differentiable SciPy namespace

Install `advect[scipy]` and import functions from `advect.scipy`. Direct
`scipy.*` calls are not intercepted:

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

The optional namespace also includes differentiable image filters and the
solver callbacks used by [implicit differentiation](implicit-differentiation.md).
The [SciPy compatibility table](../compatibility/scipy.md) is the exact list.

## Preserve xarray labels

`advect.xarray` registers `DataArray` and `Dataset` as dynamic pytrees. Data
buffers are differentiable; dimensions, coordinates, names, and attributes are
static metadata restored on derivative results.

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

Floating and complex data variables are differentiable. Integer, boolean,
string, and object buffers are rejected rather than assigned meaningless zero
gradients. xarray owns alignment and named operations; Advect owns the array
operations they lower to.

The xarray boundary is dynamic-only. Stage the raw array kernel, call it with
`field.data`, and restore labels outside the program when reusable staging is
needed.
