"""Register xarray containers for Advect's dynamic transforms.

Importing this module makes floating and complex ``DataArray`` and ``Dataset``
buffers differentiable while dimensions, coordinates, names, and attributes
remain static metadata. This integration is dynamic-only: stage a raw array
function, call the program with ``field.data``, and restore labels outside it.
"""

from __future__ import annotations

try:
    from advect.xarray._pytree import register_xarray_pytrees as _register_xarray_pytrees
except ModuleNotFoundError as error:
    if error.name != "xarray":
        raise
    msg = (
        "advect.xarray requires the optional xarray dependency; "
        "install it with `pip install 'advect[xarray]'`."
    )
    raise ModuleNotFoundError(msg) from None

__all__: list[str]
__all__ = []

_register_xarray_pytrees()
