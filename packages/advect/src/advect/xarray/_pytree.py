# ruff: noqa: ANN401
"""Single pytree registration path for xarray labeled containers."""

from __future__ import annotations

from typing import Any

import xarray as xr

from advect.core import ArraySpec
from advect.pytree import register_pytree_node
from advect.xarray._metadata import contains_tracer, freeze, thaw

_DATAARRAY_AUX_SIZE = 4
_DATASET_AUX_SIZE = 3


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


def _coordinate_spec(name: Any, coordinate: xr.DataArray) -> tuple[Any, ...]:
    data = coordinate.data
    if contains_tracer(data):
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

    return (
        freeze(name, path=f"coords[{name!r}].name"),
        freeze(tuple(coordinate.dims), path=f"coords[{name!r}].dims"),
        freeze(coordinate.to_numpy(), path=f"coords[{name!r}].values"),
        freeze(dict(coordinate.attrs), path=f"coords[{name!r}].attrs"),
    )


def _coordinate_specs(container: xr.DataArray | xr.Dataset) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        _coordinate_spec(name, coordinate) for name, coordinate in container.coords.items()
    )


def _coordinates_from_specs(specs: tuple[tuple[Any, ...], ...]) -> dict[Any, xr.Variable]:
    coordinates: dict[Any, xr.Variable] = {}
    for frozen_name, frozen_dims, frozen_values, frozen_attrs in specs:
        name = thaw(frozen_name)
        dims = thaw(frozen_dims)
        values = thaw(frozen_values)
        attrs = thaw(frozen_attrs)
        coordinates[name] = xr.Variable(dims, values, attrs=attrs)
    return coordinates


def _flatten_dataarray(tree: xr.DataArray) -> tuple[tuple[Any, ...], Any]:
    _require_differentiable_data(tree.data, path="DataArray.data")
    metadata = (
        freeze(tree.name, path="name"),
        freeze(tuple(tree.dims), path="dims"),
        freeze(dict(tree.attrs), path="attrs"),
        _coordinate_specs(tree),
    )
    return (tree.data,), metadata


def _unflatten_dataarray(aux_data: Any, children: tuple[Any, ...]) -> xr.DataArray:
    if not isinstance(aux_data, tuple) or len(aux_data) != _DATAARRAY_AUX_SIZE:
        msg = "Invalid xarray.DataArray pytree metadata"
        raise TypeError(msg)
    if len(children) != 1:
        msg = "xarray.DataArray pytree requires exactly one data leaf"
        raise ValueError(msg)

    frozen_name, frozen_dims, frozen_attrs, coordinate_specs = aux_data
    return xr.DataArray(
        children[0],
        dims=thaw(frozen_dims),
        coords=_coordinates_from_specs(coordinate_specs),
        name=thaw(frozen_name),
        attrs=thaw(frozen_attrs),
    )


def _flatten_dataset(tree: xr.Dataset) -> tuple[tuple[Any, ...], Any]:
    leaves: list[Any] = []
    variable_specs: list[tuple[Any, ...]] = []
    for name, variable in tree.data_vars.items():
        _require_differentiable_data(
            variable.data,
            path=f"Dataset data variable {name!r}",
        )
        leaves.append(variable.data)
        variable_specs.append(
            (
                freeze(name, path=f"data_vars[{name!r}].name"),
                freeze(tuple(variable.dims), path=f"data_vars[{name!r}].dims"),
                freeze(dict(variable.attrs), path=f"data_vars[{name!r}].attrs"),
            )
        )

    metadata = (
        tuple(variable_specs),
        _coordinate_specs(tree),
        freeze(dict(tree.attrs), path="attrs"),
    )
    return tuple(leaves), metadata


def _unflatten_dataset(aux_data: Any, children: tuple[Any, ...]) -> xr.Dataset:
    if not isinstance(aux_data, tuple) or len(aux_data) != _DATASET_AUX_SIZE:
        msg = "Invalid xarray.Dataset pytree metadata"
        raise TypeError(msg)
    variable_specs, coordinate_specs, frozen_attrs = aux_data
    if len(variable_specs) != len(children):
        msg = "xarray.Dataset pytree data-variable count does not match its leaves"
        raise ValueError(msg)

    data_vars: dict[Any, xr.Variable] = {}
    for spec, data in zip(variable_specs, children, strict=True):
        frozen_name, frozen_dims, frozen_variable_attrs = spec
        data_vars[thaw(frozen_name)] = xr.Variable(
            thaw(frozen_dims),
            data,
            attrs=thaw(frozen_variable_attrs),
        )

    return xr.Dataset(
        data_vars=data_vars,
        coords=_coordinates_from_specs(coordinate_specs),
        attrs=thaw(frozen_attrs),
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
