"""Normalize NumPy call forms before emitting canonical operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as _numpy  # noqa: ICN001 - concrete namespace with dynamic protocol operands

from advect.core._errors import TracingError
from advect.numpy._array_function.emission import (
    _add_backend_node,
    _get_node,
    _get_value,
    _result_shape_and_dtype,
)
from advect.numpy._op_bindings import canonicalize_numpy_op

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike

np: Any = _numpy

_PAIR_WIDTH = 2
_BINARY_ARG_COUNT = 2


def _bind_optional_positionals(
    *,
    name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    required: int,
    optional: tuple[str, ...],
    keyword_only: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Bind the repeated required-plus-optional NumPy call shape."""
    if len(args) < required or len(args) > required + len(optional):
        msg = f"numpy.{name} received an invalid positional signature during tracing"
        raise TracingError(msg)
    unsupported = set(kwargs) - (set(optional) | set(keyword_only))
    if unsupported:
        msg = f"numpy.{name} kwargs not supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)
    values = dict(kwargs)
    for parameter, value in zip(optional, args[required:], strict=False):
        if parameter in values:
            msg = f"numpy.{name} received {parameter} twice"
            raise TracingError(msg)
        values[parameter] = value
    return values


def _normalize_axis(axis: int, ndim: int) -> int:
    axis_int = axis
    if axis_int < 0:
        axis_int += ndim
    if axis_int < 0 or axis_int >= ndim:
        msg = f"Axis {axis} is out of bounds for ndim={ndim}"
        raise TracingError(msg)
    return axis_int


def _normalize_axis_value(axis: object) -> int | tuple[int, ...]:
    if isinstance(axis, np.integer):
        return int(axis)
    if isinstance(axis, int):
        return axis
    if isinstance(axis, (tuple, list, np.ndarray)):
        return tuple(int(item) for item in np.asarray(axis).tolist())
    msg = f"Unsupported axis value {axis!r}"
    raise TracingError(msg)


def _normalize_gradient_axes(*, axis: object, ndim: int) -> tuple[int, ...]:
    """Normalize gradient axis argument to a tuple of unique in-bounds axes."""
    if axis is None:
        return tuple(range(ndim))

    axis_value = _normalize_axis_value(axis)
    raw_axes = (axis_value,) if isinstance(axis_value, int) else axis_value
    normalized = tuple(_normalize_axis(axis, ndim) for axis in raw_axes)
    if len(set(normalized)) != len(normalized):
        msg = f"numpy.gradient axis contains duplicates: {axis_value!r}"
        raise TracingError(msg)
    return normalized


def _normalize_shape(shape: object) -> tuple[int, ...]:
    arr = np.asarray(shape)
    if arr.ndim == 0:
        return (int(arr.item()),)
    return tuple(int(item) for item in arr.tolist())


def _normalize_kth(kth: object) -> int | tuple[int, ...]:
    arr = np.asarray(kth)
    if arr.ndim == 0:
        return int(arr.item())
    return tuple(int(item) for item in arr.tolist())


def _normalize_pad_width(pad_width: object) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(pad_width)
    if arr.ndim == 0:
        width = int(arr.item())
        return ((width, width),)
    if arr.ndim == 1:
        values = tuple(int(item) for item in arr.tolist())
        if len(values) != _PAIR_WIDTH:
            msg = f"Unsupported pad_width shape: {arr.shape}"
            raise TracingError(msg)
        return ((values[0], values[1]),)
    if arr.ndim == _PAIR_WIDTH and arr.shape[1] == _PAIR_WIDTH:
        return tuple((int(a), int(b)) for a, b in arr.tolist())
    msg = f"Unsupported pad_width shape: {arr.shape}"
    raise TracingError(msg)


def _normalize_constant_values(
    value: object,
) -> int | float | tuple[tuple[int | float, int | float], ...]:
    arr = np.asarray(value)
    if arr.ndim == 0:
        scalar = arr.item()
        if isinstance(scalar, (np.floating, float)):
            return float(scalar)
        if isinstance(scalar, (np.integer, int)):
            return int(scalar)
        msg = f"Unsupported constant_values scalar: {type(scalar).__name__}"
        raise TracingError(msg)
    if arr.ndim == 1:
        vals = tuple(float(item) for item in arr.tolist())
        if len(vals) != _PAIR_WIDTH:
            msg = f"Unsupported constant_values shape: {arr.shape}"
            raise TracingError(msg)
        return ((vals[0], vals[1]),)
    if arr.ndim == _PAIR_WIDTH and arr.shape[1] == _PAIR_WIDTH:
        return tuple((float(a), float(b)) for a, b in arr.tolist())
    msg = f"Unsupported constant_values shape: {arr.shape}"
    raise TracingError(msg)


def _binary_handler(
    *,
    op_name: str,
    np_func: Callable[..., Any],
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    if len(args) != _BINARY_ARG_COUNT:
        msg = f"{op_name} expects two positional arguments during tracing"
        raise TracingError(msg)
    if kwargs:
        msg = f"{op_name} kwargs not supported during tracing: {sorted(kwargs)}"
        raise TracingError(msg)

    a, b = args
    a_value = _get_value(a, traced_type)
    b_value = _get_value(b, traced_type)
    result = np_func(a_value, b_value)
    result_shape, result_dtype = _result_shape_and_dtype(result)
    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op(op_name),
        inputs=(
            _get_node(a, graph, traced_type),
            _get_node(b, graph, traced_type),
        ),
        value=result,
        attrs={},
        shape=result_shape,
        dtype=result_dtype,
    )
    return result, node_id
