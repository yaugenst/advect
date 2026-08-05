"""Labeled xarray pytrees for Advect's dynamic autodiff transforms."""

from __future__ import annotations

try:
    from advect.xarray._pytree import register_xarray_pytrees
except ModuleNotFoundError as error:
    if error.name != "xarray":
        raise
    msg = (
        "advect.xarray requires the optional xarray dependency; "
        "install it with `pip install 'advect[xarray]'`."
    )
    raise ModuleNotFoundError(msg) from None

__all__ = ["register"]


def register() -> None:
    """Register xarray's two labeled containers with Advect's pytree core."""
    register_xarray_pytrees()


register()
