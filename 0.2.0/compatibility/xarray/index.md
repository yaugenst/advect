# xarray

The xarray integration preserves labeled containers; it is not another array provider. With `advect[xarray]` installed, importing `advect.xarray` registers `DataArray` and `Dataset` as pytrees. Dynamic transforms then operate on their data buffers and rebuild the labels around derivative results.

## Contract

- Floating and complex data buffers are differentiable. Integer, boolean, string, and object variables are rejected rather than assigned zero gradients.
- Dimensions, coordinates, names, attributes, and dataset variable order are static metadata.
- xarray continues to own alignment, named indexing, and named reductions when those operations lower to supported array primitives.
- `DataArray` subclasses inherit the same pytree behavior and preserve their concrete container type.

## Boundaries

This integration is dynamic-only. For a reusable staged kernel, stage the raw array function, call it with `field.data`, and restore labels outside the program. `DataArray.values` and `.to_numpy()` materialize NumPy arrays and remain outside tracing. Data-dependent coordinates, MultiIndex coordinates, Dask execution, and broad groupby, rolling, or interpolation coverage are outside the current contract.

See [Arrays and Scientific Python](https://yaugenst.github.io/advect/0.2.0/tutorials/scientific-python/#preserve-xarray-labels) for a complete example.
