# ruff: noqa: ANN401
"""Single pytree registration path for xarray labeled containers."""

from __future__ import annotations

import copy
import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from advect.core import ArraySpec
from advect.pytree import register_pytree_node

if TYPE_CHECKING:
    from collections.abc import Mapping

_DATASET_ORDER_SIZE = 2


@dataclass(frozen=True, eq=False, slots=True)
class _Metadata:  # noqa: PLW1641
    template: xr.DataArray | xr.Dataset
    order: tuple[Any, ...]
    # Paths of the metadata arrays that xarray compares with ``==`` and that
    # do not equal their own copies that way.
    eq_arrays: tuple[str, ...]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Metadata) or self.order != other.order:
            return False
        try:
            if _identical(self.template, other.template):
                return True
        except ValueError as error:
            paths = self.eq_arrays + other.eq_arrays
            if not paths:
                raise
            raise _incomparable(paths) from error
        # Metadata that differs from its own copy can never compare equal, so
        # name the arrays instead of letting callers report a structure mismatch.
        for side in (self, other):
            if side.eq_arrays and not _equals_its_copy(side.template):
                raise _incomparable(side.eq_arrays)
        return False


def _identical(first: xr.DataArray | xr.Dataset, second: xr.DataArray | xr.Dataset) -> bool:
    if isinstance(first, xr.DataArray):
        return isinstance(second, xr.DataArray) and first.identical(second)
    return isinstance(second, xr.Dataset) and first.identical(second)


def _equals_its_copy(template: xr.DataArray | xr.Dataset) -> bool:
    try:
        return _identical(template, template.copy(deep=True))
    except ValueError:
        return False


def _incomparable(paths: tuple[str, ...]) -> TypeError:
    # Name every such array: which one a comparison reached depends on
    # attribute order and on whether xarray shared it between copies.
    named = tuple(dict.fromkeys(paths))
    arrays = "array" if len(named) == 1 else "arrays"
    msg = (
        f"xarray cannot compare the metadata {arrays} at {', '.join(named)}: it "
        "compares arrays with == unless they are direct attribute or coordinate "
        "values or list items. Store each array directly as an attribute value "
        "or inside a list."
    )
    return TypeError(msg)


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


def _validate_static(value: Any, *, path: str, eq_arrays: list[str], arrays: bool = True) -> None:
    """Validate static metadata and collect the paths of arrays ``==`` cannot compare.

    xarray compares arrays with NaN-aware ``array_equiv`` only as direct
    metadata values or inside lists (``arrays``). Names, dicts, tuples, slices,
    and object-array elements compare with ``==``, which is ambiguous for most
    arrays and false for NaN or NaT; ``_Metadata`` names such arrays if a
    comparison fails. Nested paths come first, so the innermost cause leads.
    """
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
            _validate_static(value.item(), path=f"{path}.item()", eq_arrays=eq_arrays, arrays=False)
        return
    if isinstance(value, slice):
        for part in ("start", "stop", "step"):
            _validate_static(
                getattr(value, part), path=f"{path}.{part}", eq_arrays=eq_arrays, arrays=False
            )
        return
    if isinstance(value, (tuple, list)):
        items_compare_arrays = arrays and isinstance(value, list)
        for index, item in enumerate(value):
            _validate_static(
                item, path=f"{path}[{index}]", eq_arrays=eq_arrays, arrays=items_compare_arrays
            )
        return
    if isinstance(value, dict):
        _validate_attrs(value, path=path, eq_arrays=eq_arrays, arrays=False)
        return
    if isinstance(value, np.ndarray):
        if value.dtype.fields is not None:
            msg = f"xarray structured metadata arrays are not supported at {path}"
            raise TypeError(msg)
        if value.dtype.hasobject:
            for index, item in enumerate(value.flat):
                _validate_static(
                    item, path=f"{path}.flat[{index}]", eq_arrays=eq_arrays, arrays=False
                )
        if not arrays and not _equals_its_copy_with_eq(value):
            eq_arrays.append(path)
        return

    msg = f"xarray static metadata at {path} has unsupported type {type(value).__name__}"
    raise TypeError(msg)


def _equals_its_copy_with_eq(value: np.ndarray[Any, Any]) -> bool:
    """Return whether ``==`` gives one true value against a deep copy of ``value``."""
    if value.size != 1:
        return False
    try:
        return bool(value == copy.deepcopy(value))
    except (TypeError, ValueError):
        # Object elements such as dicts of arrays compare ambiguously.
        return False


def _validate_attrs(
    attrs: Mapping[Any, Any], *, path: str, eq_arrays: list[str], arrays: bool = True
) -> None:
    """Validate an attribute mapping, whose values xarray compares one by one."""
    for key, item in attrs.items():
        if not isinstance(key, str):
            msg = f"xarray attribute keys must be strings; got {type(key).__name__} at {path}"
            raise TypeError(msg)
        _validate_static(item, path=f"{path}[{key!r}]", eq_arrays=eq_arrays, arrays=arrays)


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


def _is_multiindex(variable: xr.Variable) -> bool:
    # A MultiIndex coordinate holds tuples, so only object coordinates need a
    # pandas.Index; building one for other dtypes can warn (float16).
    if variable.dtype.kind != "O":
        return False
    try:
        index = variable.to_index()
    except ValueError:
        return False
    return int(getattr(index, "nlevels", 1)) > 1


def _validate_coordinate(name: Any, variable: xr.Variable, eq_arrays: list[str]) -> None:
    data = variable.data
    if _contains_tracer(data):
        msg = (
            "xarray coordinates, dimensions, names, and attributes are static; "
            f"found traced coordinate {name!r}. Pass differentiable values as data "
            "or as a separate argument."
        )
        raise TypeError(msg)

    if _is_multiindex(variable):
        msg = (
            f"xarray MultiIndex coordinate {name!r} is not supported by advect.xarray. "
            "Reset the index before differentiation."
        )
        raise TypeError(msg)

    prefix = f"coords[{name!r}]"
    _validate_static(name, path=f"{prefix}.name", eq_arrays=eq_arrays)
    _validate_static(variable.dims, path=f"{prefix}.dims", eq_arrays=eq_arrays)
    values = data if isinstance(data, np.ndarray) else variable.to_numpy()
    _validate_static(values, path=f"{prefix}.values", eq_arrays=eq_arrays)
    _validate_attrs(variable.attrs, path=f"{prefix}.attrs", eq_arrays=eq_arrays)


def _validate_coordinates(container: xr.DataArray | xr.Dataset, eq_arrays: list[str]) -> None:
    # Coordinate variables avoid constructing one DataArray per coordinate.
    for name, variable in container.coords.variables.items():
        _validate_coordinate(name, variable, eq_arrays)


def _flatten_dataarray(tree: xr.DataArray) -> tuple[tuple[Any, ...], Any]:
    _require_differentiable_data(tree.data, path="DataArray.data")
    eq_arrays: list[str] = []
    _validate_static(tree.name, path="name", eq_arrays=eq_arrays, arrays=False)
    _validate_static(tuple(tree.dims), path="dims", eq_arrays=eq_arrays)
    _validate_attrs(tree.attrs, path="attrs", eq_arrays=eq_arrays)
    _validate_coordinates(tree, eq_arrays)
    metadata = _Metadata(
        template=tree.copy(deep=True, data=_dummy(tree.shape)),
        order=tuple(tree.coords),
        eq_arrays=tuple(eq_arrays),
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
    eq_arrays: list[str] = []
    for name in names:
        variable = tree.variables[name]
        _require_differentiable_data(variable.data, path=f"Dataset data variable {name!r}")
        prefix = f"data_vars[{name!r}]"
        _validate_static(name, path=f"{prefix}.name", eq_arrays=eq_arrays)
        _validate_static(variable.dims, path=f"{prefix}.dims", eq_arrays=eq_arrays)
        _validate_attrs(variable.attrs, path=f"{prefix}.attrs", eq_arrays=eq_arrays)

    _validate_coordinates(tree, eq_arrays)
    _validate_attrs(tree.attrs, path="attrs", eq_arrays=eq_arrays)
    metadata = _Metadata(
        template=tree.copy(
            deep=True,
            data={name: _dummy(tree[name].shape) for name in names},
        ),
        order=(names, tuple(tree.coords)),
        eq_arrays=tuple(eq_arrays),
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
        include_subclasses=True,
    )
    register_pytree_node(
        xr.Dataset,
        flatten_fn=_flatten_dataset,
        unflatten_fn=_unflatten_dataset,
    )
