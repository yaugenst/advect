# xarray

The xarray integration is a pytree contract, not a function catalog: with
`advect[xarray]` installed, `import advect.xarray` registers `DataArray` and
`Dataset` as pytree nodes, and every transform then works leaf-wise on their
data buffers. Registration is explicit — installing the package changes
nothing until the module is imported.

## The contract

- Floating- and complex-valued data buffers are differentiable. Integer,
  boolean, string, and object data variables are rejected instead of
  receiving meaningless zero gradients.
- Dimensions, coordinates, names, and attributes are copied static metadata
  and are restored around the gradient. Datasets expose one leaf per data
  variable.
- xarray continues to own alignment, named indexing, and named reductions
  when those operations lower to supported array primitives.

## Boundaries

The contract is dynamic-only. To reuse a staged kernel, stage the raw array
function, call it with `field.data`, and restore labels outside the program.
Data-dependent coordinates, MultiIndex coordinates, Dask execution, and broad
groupby/rolling/interpolation coverage are not part of this slice.

See the [interop tutorial](../tutorials/interop.md#preserve-xarray-labels)
for a worked example.
