# xarray

Importing `advect.xarray` explicitly registers floating- and complex-valued
[`DataArray`](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html)
and [`Dataset`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html)
objects as pytrees. Data buffers are differentiable leaves; dimensions,
coordinates, names, attributes, and alignment metadata are static and are
restored on derivative results. The
[Scientific Python tutorial](../tutorials/scientific-python.md#preserve-xarray-labels)
shows a labeled gradient.

This is a structure integration, not an array provider. It supports dynamic
transforms and deliberately rejects staging across the labeled-container
boundary. See the [xarray compatibility contract](../compatibility/xarray.md)
for the supported boundary.
