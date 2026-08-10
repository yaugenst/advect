# Providers and Interop

## Write provider-neutral functions

Obtain the namespace from the traced input rather than importing provider
operations inside the function:

```{.python .run}
import numpy as np

import advect as ad


def centered_energy(value):
    xp = value.__array_namespace__()
    centered = value - xp.mean(value)
    return xp.sum(centered * centered)


print(centered_energy(np.asarray([1.0, 2.0, 4.0], dtype=np.float32)))
```

The same Advect transform accepts qualified NumPy and Array API Strict inputs;
the built-in compatibility fallback also enables a CuPy path with separate GPU
qualification:

```{.python .run}
numpy_gradient = ad.grad(centered_energy)(
    np.asarray([1.0, 2.0, 4.0], dtype=np.float32)
)
print(numpy_gradient)
```

## Array providers

NumPy is the first-class frontend and needs nothing beyond `advect` itself.
Other Array API providers use the same user function through the built-in
compatibility fallback, which is configured when `advect` is imported:

| Provider | Additional installation | Qualified boundary |
| --- | --- | --- |
| NumPy 2.0-2.4 | None | First-class NumPy frontend |
| `array-api-strict` | `array-api-strict` | Array API 2022.12-2024.12 |
| CuPy | The appropriate CuPy build | [Manual qualification](../compatibility/cupy.md) required |

CuPy installation depends on the local CUDA version; a CUDA 12 environment may
use `cupy-cuda12x`, and Advect does not select the CUDA package.

The compatibility fallback is provider-neutral. CuPy has recorded GPU evidence
for all three supported profiles. The fallback handles raw provider arrays at
Advect's input boundary; it does not make plain calls to
`centered_energy(raw_provider_array)` work when that array does not itself
implement `__array_namespace__`.

The [compatibility catalog](../compatibility/index.md) documents the qualified
surface in detail — per-provider tables generated from the live registry,
including every supported function and its mode claims.

## Preserve xarray labels

With `advect[xarray]` installed, importing the built-in integration registers
`DataArray` and `Dataset` as pytrees:

```python
import xarray as xr

import advect.xarray  # explicit registration

field = xr.DataArray(
    np.arange(6.0).reshape(2, 3),
    dims=("y", "x"),
    coords={"y": [10, 20], "x": [1, 2, 3]},
    name="field",
    attrs={"units": "V"},
)


def labeled_energy(value):
    centered = value - value.mean("x")
    return (centered * centered).sum(("x", "y"))


gradient = ad.grad(labeled_energy)(field)
assert gradient.dims == field.dims
xr.testing.assert_identical(gradient.coords, field.coords)
```

Floating- and complex-valued data buffers are differentiable. Integer,
boolean, string, and object data variables are rejected instead of receiving
meaningless zero gradients. Dimensions, coordinates, names, and attributes are
copied static metadata and are restored around the gradient. Datasets expose
one leaf per data variable. xarray continues to own alignment, named indexing,
and named reductions when those operations lower to supported array
primitives.

The initial xarray contract is dynamic. To reuse a staged kernel, stage the raw
array function, call it with `field.data`, and restore labels outside the
program. Data-dependent coordinates, MultiIndex coordinates, Dask execution,
and broad groupby/rolling/interpolation coverage are not part of this first
slice.
