# xarray

Importing `advect.xarray` explicitly registers floating- and complex-valued
`DataArray` and `Dataset` objects as pytrees. Data buffers are differentiable
leaves; dimensions, coordinates, names, attributes, and alignment metadata are
static and are restored on derivative results.

This is a structure integration, not an array provider. It supports dynamic
transforms and deliberately rejects staging across the labeled-container
boundary. See the [xarray compatibility contract](../compatibility/xarray.md)
for the exact data and metadata rules.

::: advect.xarray
