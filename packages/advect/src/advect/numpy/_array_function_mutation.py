"""Functionalize NumPy mutating functions through pure whole-array updates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as _numpy  # noqa: ICN001 - typed module and dynamic lowering namespace

from advect.core._errors import TracingError
from advect.core._protocols import _snapshot_traced

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.numpy._traced_array import TracedArray


np: Any = _numpy
NOT_FUNCTIONALIZED = object()


def _bind(
    *,
    name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    parameters: tuple[str, ...],
    required: int,
    defaults: dict[str, object] | None = None,
) -> dict[str, object]:
    if len(args) > len(parameters):
        msg = f"numpy.{name} received too many positional arguments during tracing"
        raise TracingError(msg)
    unsupported = set(kwargs) - set(parameters)
    if unsupported:
        msg = f"numpy.{name} kwargs not supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)
    values = {} if defaults is None else dict(defaults)
    for parameter, value in zip(parameters, args, strict=False):
        if parameter in kwargs:
            msg = f"numpy.{name} received {parameter} twice"
            raise TracingError(msg)
        values[parameter] = value
    values.update(kwargs)
    if any(parameter not in values for parameter in parameters[:required]):
        msg = f"numpy.{name} is missing a required argument during tracing"
        raise TracingError(msg)
    return values


def _concrete(value: object) -> object:
    if callable(getattr(value, "_advect_snapshot", None)):
        return _snapshot_traced(value)[1]
    return value


def _destination(
    value: object,
    *,
    traced_type: type[TracedArray],
    name: str,
) -> TracedArray:
    if not isinstance(value, traced_type):
        msg = f"numpy.{name} destination must be a TracedArray during tracing"
        raise TracingError(msg)
    return value


def _copyto(
    traced_type: type[TracedArray],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> None:
    values = _bind(
        name="copyto",
        args=args,
        kwargs=kwargs,
        parameters=("dst", "src", "casting", "where"),
        required=2,
        defaults={"casting": "same_kind", "where": True},
    )
    dst = _destination(values["dst"], traced_type=traced_type, name="copyto")
    src = values["src"]
    where = values["where"]
    casting = str(values["casting"])
    if not np.can_cast(np.asarray(_concrete(src)).dtype, dst.dtype, casting=casting):
        msg = (
            f"Cannot cast array data from {np.asarray(_concrete(src)).dtype!r} "
            f"to {np.dtype(dst.dtype)!r} according to the rule {casting!r}"
        )
        raise TypeError(msg)
    np.broadcast_shapes(np.shape(_concrete(src)), dst.shape)
    np.broadcast_shapes(np.shape(_concrete(where)), dst.shape)
    source = np.astype(np.broadcast_to(src, dst.shape), dst.dtype)
    dst[...] = np.where(where, source, dst)


def _putmask(
    traced_type: type[TracedArray],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> None:
    values = _bind(
        name="putmask",
        args=args,
        kwargs=kwargs,
        parameters=("a", "mask", "values"),
        required=3,
    )
    dst = _destination(values["a"], traced_type=traced_type, name="putmask")
    mask = values["mask"]
    replacements = values["values"]
    concrete_mask = np.asarray(_concrete(mask), dtype=bool)
    if concrete_mask.size != dst.size:
        msg = "putmask: mask and data must be the same size"
        raise ValueError(msg)
    concrete_replacements = np.ravel(np.asarray(_concrete(replacements)))
    if concrete_replacements.size == 0:
        return
    tiled = np.reshape(np.resize(replacements, dst.size), dst.shape)
    dst[...] = np.where(np.reshape(mask, dst.shape), tiled, dst)


def _place(
    traced_type: type[TracedArray],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> None:
    values = _bind(
        name="place",
        args=args,
        kwargs=kwargs,
        parameters=("arr", "mask", "vals"),
        required=3,
    )
    dst = _destination(values["arr"], traced_type=traced_type, name="place")
    mask = values["mask"]
    replacements = values["vals"]
    concrete_mask = np.asarray(_concrete(mask), dtype=bool)
    if concrete_mask.size != dst.size:
        msg = "place: mask and data must be the same size"
        raise ValueError(msg)
    concrete_replacements = np.ravel(np.asarray(_concrete(replacements)))
    if not np.any(concrete_mask):
        return
    if concrete_replacements.size == 0:
        msg = "numpy.place cannot use empty replacements when mask is true"
        raise TracingError(msg)
    flat_mask = np.reshape(concrete_mask, (-1,))
    selection = np.maximum(np.cumsum(flat_mask) - 1, 0) % concrete_replacements.size
    selected = np.take(np.ravel(replacements), selection)
    dst[...] = np.reshape(
        np.where(flat_mask, selected, np.ravel(dst)),
        dst.shape,
    )


def _fill_diagonal(
    traced_type: type[TracedArray],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> None:
    values = _bind(
        name="fill_diagonal",
        args=args,
        kwargs=kwargs,
        parameters=("a", "val", "wrap"),
        required=2,
        defaults={"wrap": False},
    )
    dst = _destination(values["a"], traced_type=traced_type, name="fill_diagonal")
    replacements = values["val"]
    wrap = bool(values["wrap"])
    concrete_replacements = np.ravel(np.asarray(_concrete(replacements)))
    marker = np.zeros(dst.shape, dtype=bool)
    np.fill_diagonal(marker, True, wrap=wrap)  # noqa: FBT003 - NumPy value argument
    if not np.any(marker):
        return
    if concrete_replacements.size == 0:
        msg = "numpy.fill_diagonal cannot use an empty replacement array"
        raise TracingError(msg)
    flat_marker = np.ravel(marker)
    selection = np.maximum(np.cumsum(flat_marker) - 1, 0) % concrete_replacements.size
    selected = np.take(np.ravel(replacements), selection)
    dst[...] = np.reshape(
        np.where(flat_marker, selected, np.ravel(dst)),
        dst.shape,
    )


def _normalized_put_indices(indices: object, *, size: int, mode: str) -> np.ndarray[Any, Any]:
    normalized = np.asarray(indices, dtype=np.intp)
    if mode == "wrap":
        return np.remainder(normalized, size)
    if mode == "clip":
        return np.clip(normalized, 0, size - 1)
    if mode != "raise":
        msg = f"clipmode must be one of 'clip', 'raise', or 'wrap' (got {mode!r})"
        raise TracingError(msg)
    normalized = np.where(normalized < 0, normalized + size, normalized)
    if np.any((normalized < 0) | (normalized >= size)):
        msg = "numpy.put index is out of bounds"
        raise TracingError(msg)
    return normalized


def _put(
    traced_type: type[TracedArray],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> None:
    values = _bind(
        name="put",
        args=args,
        kwargs=kwargs,
        parameters=("a", "ind", "v", "mode"),
        required=3,
        defaults={"mode": "raise"},
    )
    dst = _destination(values["a"], traced_type=traced_type, name="put")
    indices = np.ravel(
        _normalized_put_indices(
            _concrete(values["ind"]),
            size=dst.size,
            mode=str(values["mode"]),
        )
    )
    replacements = values["v"]
    concrete_replacements = np.ravel(np.asarray(_concrete(replacements)))
    if indices.size == 0:
        return
    if concrete_replacements.size == 0:
        msg = "numpy.put cannot use empty replacements when indices are present"
        raise TracingError(msg)
    last_source = np.full(dst.size, -1, dtype=np.intp)
    for source_position, destination_position in enumerate(indices):
        last_source[int(destination_position)] = source_position % concrete_replacements.size
    mask = last_source >= 0
    selected = np.take(np.ravel(replacements), np.maximum(last_source, 0))
    dst[...] = np.reshape(
        np.where(mask, selected, np.ravel(dst)),
        dst.shape,
    )


def _put_along_axis(
    traced_type: type[TracedArray],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> None:
    values = _bind(
        name="put_along_axis",
        args=args,
        kwargs=kwargs,
        parameters=("arr", "indices", "values", "axis"),
        required=4,
    )
    dst = _destination(values["arr"], traced_type=traced_type, name="put_along_axis")
    indices = np.asarray(_concrete(values["indices"]), dtype=np.intp)
    replacements = values["values"]
    axis_raw = values["axis"]
    axis = None if axis_raw is None else int(cast("Any", axis_raw))
    if axis is None:
        if indices.ndim != 1:
            msg = "`indices` and `arr` must have the same number of dimensions"
            raise ValueError(msg)
        _put(traced_type, (dst, indices, replacements), {"mode": "raise"})
        return
    if axis < 0:
        axis += dst.ndim
    if axis < 0 or axis >= dst.ndim:
        raise np.exceptions.AxisError(axis_raw, dst.ndim)
    if indices.ndim != dst.ndim:
        msg = "`indices` and `arr` must have the same number of dimensions"
        raise ValueError(msg)
    for dimension, (index_size, destination_size) in enumerate(
        zip(indices.shape, dst.shape, strict=True)
    ):
        if dimension != axis and index_size not in {1, destination_size}:
            msg = "`indices` can only broadcast against `arr` outside the indexed axis"
            raise ValueError(msg)
    normalized = np.where(indices < 0, indices + dst.shape[axis], indices)
    if np.any((normalized < 0) | (normalized >= dst.shape[axis])):
        msg = "numpy.put_along_axis index is out of bounds"
        raise TracingError(msg)
    update_shape = list(indices.shape)
    for dimension, destination_size in enumerate(dst.shape):
        if dimension != axis:
            update_shape[dimension] = destination_size
    update_shape_tuple = tuple(update_shape)
    try:
        broadcast_indices = np.broadcast_to(normalized, update_shape_tuple)
        broadcast_replacements = np.broadcast_to(replacements, update_shape_tuple)
    except ValueError as exc:
        msg = "`values` array and `indices` array shape mismatch"
        raise ValueError(msg) from exc
    coordinates = list(np.indices(update_shape_tuple))
    coordinates[axis] = broadcast_indices
    flat_destinations = np.ravel_multi_index(tuple(coordinates), dst.shape)
    _put(
        traced_type,
        (dst, flat_destinations, np.ravel(broadcast_replacements)),
        {"mode": "raise"},
    )


_FUNCTIONALIZERS: dict[object, Callable[..., None]] = {
    np.copyto: _copyto,
    np.fill_diagonal: _fill_diagonal,
    np.place: _place,
    np.put: _put,
    np.put_along_axis: _put_along_axis,
    np.putmask: _putmask,
}


def functionalize_array_function_mutation(
    self: TracedArray,
    func: object,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    """Apply a supported mutation and return ``None``, or a private sentinel."""
    functionalizer = _FUNCTIONALIZERS.get(func)
    if functionalizer is None:
        return NOT_FUNCTIONALIZED
    functionalizer(type(self), args, kwargs)
    return None
