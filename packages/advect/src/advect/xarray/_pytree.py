# ruff: noqa: ANN401
"""Single pytree registration path for xarray labeled containers."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from advect.core import ArraySpec
from advect.pytree import register_pytree_node

_DATASET_ORDER_SIZE = 2


@dataclass(frozen=True, eq=False, slots=True)
class _Metadata:  # noqa: PLW1641
    template: xr.DataArray | xr.Dataset
    order: tuple[Any, ...]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Metadata) or self.order != other.order:
            return False
        if isinstance(self.template, xr.DataArray):
            return isinstance(other.template, xr.DataArray) and self.template.identical(
                other.template
            )
        return isinstance(other.template, xr.Dataset) and self.template.identical(other.template)


def _dummy(shape: tuple[int, ...]) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """Return a shape-only placeholder backed by one byte."""
    return np.broadcast_to(np.array(0, dtype=np.uint8), shape)


def _contains_tracer(value: Any) -> bool:
    if callable(getattr(value, "_advect_snapshot", None)):
        return True
    if isinstance(value, np.ndarray):
        return value.dtype.hasobject and any(_contains_tracer(item) for item in value.flat)
    if isinstance(value, dict):
        return any(_contains_tracer(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tracer(item) for item in value)
    return False


def _validate_static(value: Any, *, path: str) -> None:
    if callable(getattr(value, "_advect_snapshot", None)):
        msg = (
            "xarray coordinates, dimensions, names, and attributes are static; "
            f"found a traced value at {path}. Pass differentiable values as data "
            "or as a separate argument."
        )
        raise TypeError(msg)

    if value is None or isinstance(
        value,
        (bool, int, float, complex, str, bytes, dt.date, dt.datetime, dt.timedelta),
    ):
        return
    if isinstance(value, np.generic):
        if value.dtype.hasobject:
            _validate_static(value.item(), path=f"{path}.item()")
        return
    if isinstance(value, slice):
        _validate_static(value.start, path=f"{path}.start")
        _validate_static(value.stop, path=f"{path}.stop")
        _validate_static(value.step, path=f"{path}.step")
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _validate_static(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                msg = f"xarray attribute keys must be strings; got {type(key).__name__} at {path}"
                raise TypeError(msg)
            _validate_static(item, path=f"{path}[{key!r}]")
        return
    if isinstance(value, np.ndarray):
        if value.dtype.fields is not None:
            msg = f"xarray structured metadata arrays are not supported at {path}"
            raise TypeError(msg)
        if value.dtype.hasobject:
            for index, item in enumerate(value.flat):
                _validate_static(item, path=f"{path}.flat[{index}]")
        return

    msg = f"xarray static metadata at {path} has unsupported type {type(value).__name__}"
    raise TypeError(msg)


def _require_differentiable_data(data: Any, *, path: str) -> None:
    dtype = getattr(data, "dtype", None)
    kind = getattr(dtype, "kind", None)
    if kind == "O":
        items = tuple(getattr(data, "flat", ()))
        if items and all(isinstance(item, ArraySpec) for item in items):
            # Lazy stage reconstructs a custom pytree with ArraySpec children
            # before rejecting that pytree at the durable-codec boundary.
            return
    if kind not in {"f", "c"}:
        msg = (
            "advect.xarray differentiable data must have a floating or complex "
            f"dtype; {path} has dtype {dtype!s}. Move labels and other static "
            "values to coordinates or cast the data before differentiation."
        )
        raise TypeError(msg)


def _validate_coordinate(name: Any, coordinate: xr.DataArray) -> None:
    data = coordinate.data
    if _contains_tracer(data):
        msg = (
            "xarray coordinates, dimensions, names, and attributes are static; "
            f"found traced coordinate {name!r}. Pass differentiable values as data "
            "or as a separate argument."
        )
        raise TypeError(msg)

    try:
        index = coordinate.to_index()
    except ValueError:
        index = None
    if index is not None and int(getattr(index, "nlevels", 1)) > 1:
        msg = (
            f"xarray MultiIndex coordinate {name!r} is not supported by advect.xarray. "
            "Reset the index before differentiation."
        )
        raise TypeError(msg)

    _validate_static(name, path=f"coords[{name!r}].name")
    _validate_static(tuple(coordinate.dims), path=f"coords[{name!r}].dims")
    _validate_static(coordinate.to_numpy(), path=f"coords[{name!r}].values")
    _validate_static(dict(coordinate.attrs), path=f"coords[{name!r}].attrs")


def _validate_coordinates(container: xr.DataArray | xr.Dataset) -> None:
    for name, coordinate in container.coords.items():
        _validate_coordinate(name, coordinate)


def _flatten_dataarray(tree: xr.DataArray) -> tuple[tuple[Any, ...], Any]:
    _require_differentiable_data(tree.data, path="DataArray.data")
    _validate_static(tree.name, path="name")
    _validate_static(tuple(tree.dims), path="dims")
    _validate_static(dict(tree.attrs), path="attrs")
    _validate_coordinates(tree)
    metadata = _Metadata(
        template=tree.copy(deep=True, data=_dummy(tree.shape)),
        order=tuple(tree.coords),
    )
    return (tree.data,), metadata


def _unflatten_dataarray(aux_data: Any, children: tuple[Any, ...]) -> xr.DataArray:
    if not isinstance(aux_data, _Metadata) or not isinstance(aux_data.template, xr.DataArray):
        msg = "Invalid xarray.DataArray pytree metadata"
        raise TypeError(msg)
    if len(children) != 1:
        msg = "xarray.DataArray pytree requires exactly one data leaf"
        raise ValueError(msg)

    return aux_data.template.copy(deep=True, data=children[0])


def _flatten_dataset(tree: xr.Dataset) -> tuple[tuple[Any, ...], Any]:
    names = tuple(tree.data_vars)
    for name in names:
        variable = tree[name]
        _require_differentiable_data(
            variable.data,
            path=f"Dataset data variable {name!r}",
        )
        _validate_static(name, path=f"data_vars[{name!r}].name")
        _validate_static(tuple(variable.dims), path=f"data_vars[{name!r}].dims")
        _validate_static(dict(variable.attrs), path=f"data_vars[{name!r}].attrs")

    _validate_coordinates(tree)
    _validate_static(dict(tree.attrs), path="attrs")
    metadata = _Metadata(
        template=tree.copy(
            deep=True,
            data={name: _dummy(tree[name].shape) for name in names},
        ),
        order=(names, tuple(tree.coords)),
    )
    return tuple(tree[name].data for name in names), metadata


def _unflatten_dataset(aux_data: Any, children: tuple[Any, ...]) -> xr.Dataset:
    if (
        not isinstance(aux_data, _Metadata)
        or not isinstance(aux_data.template, xr.Dataset)
        or len(aux_data.order) != _DATASET_ORDER_SIZE
        or not isinstance(aux_data.order[0], tuple)
        or not isinstance(aux_data.order[1], tuple)
    ):
        msg = "Invalid xarray.Dataset pytree metadata"
        raise TypeError(msg)
    names = aux_data.order[0]
    if len(names) != len(children):
        msg = "xarray.Dataset pytree data-variable count does not match its leaves"
        raise ValueError(msg)

    return aux_data.template.copy(
        deep=True,
        data=dict(zip(names, children, strict=True)),
    )


def register_xarray_pytrees() -> None:
    """Register DataArray and Dataset as dynamic Advect pytree nodes."""
    register_pytree_node(
        xr.DataArray,
        flatten_fn=_flatten_dataarray,
        unflatten_fn=_unflatten_dataarray,
    )
    register_pytree_node(
        xr.Dataset,
        flatten_fn=_flatten_dataset,
        unflatten_fn=_unflatten_dataset,
    )
