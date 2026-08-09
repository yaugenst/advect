"""Trace NumPy split-family functions while preserving their list results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as _numpy  # noqa: ICN001 - concrete namespace with dynamic protocol operands

from advect.core._errors import TracingError
from advect.core._protocols import _snapshot_traced
from advect.numpy._array_function.emission import _get_value
from advect.numpy._array_function.normalization import _normalize_axis

if TYPE_CHECKING:
    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike
    from advect.numpy._array_function.emission import ArrayFunctionResult

np: Any = _numpy

_REQUIRED_SPLIT_ARGS = 2


def _split_to_getitems(
    *,
    traced_array: TracedArrayLike,
    outputs: tuple[Any, ...],
    axis: int,
) -> ArrayFunctionResult:
    _node_id, traced_value = _snapshot_traced(traced_array)
    axis_norm = _normalize_axis(axis, traced_value.ndim)
    start = 0
    traced_parts: list[TracedArrayLike] = []
    for output in outputs:
        width = output.shape[axis_norm]
        index: list[object] = [slice(None)] * traced_value.ndim
        index[axis_norm] = slice(start, start + width)
        traced_parts.append(cast("Any", traced_array)[tuple(index)])
        start += width

    # NumPy's split family returns lists.  Keep that public container even
    # though the parallel node-ID tree may use an immutable tuple internally.
    return (
        [_snapshot_traced(part)[1] for part in traced_parts],
        tuple(_snapshot_traced(part)[0] for part in traced_parts),
    )


def _split_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ArrayFunctionResult:
    if len(args) < _REQUIRED_SPLIT_ARGS:
        msg = "numpy.split expects (ary, indices_or_sections) during tracing"
        raise TracingError(msg)
    unsupported = set(kwargs) - {"axis"}
    if unsupported:
        msg = f"numpy.split kwargs not supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)

    ary = args[0]
    if not isinstance(ary, traced_type):
        msg = "numpy.split tracing requires a traced first argument"
        raise TracingError(msg)
    indices_or_sections = args[1]
    axis = int(kwargs.get("axis", 0))
    outputs = tuple(np.split(_get_value(ary, traced_type), indices_or_sections, axis=axis))
    return _split_to_getitems(traced_array=ary, outputs=outputs, axis=axis)


def _array_split_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ArrayFunctionResult:
    if len(args) < _REQUIRED_SPLIT_ARGS:
        msg = "numpy.array_split expects (ary, indices_or_sections) during tracing"
        raise TracingError(msg)
    unsupported = set(kwargs) - {"axis"}
    if unsupported:
        msg = f"numpy.array_split kwargs not supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)

    ary = args[0]
    if not isinstance(ary, traced_type):
        msg = "numpy.array_split tracing requires a traced first argument"
        raise TracingError(msg)
    indices_or_sections = args[1]
    axis = int(kwargs.get("axis", 0))
    outputs = tuple(np.array_split(_get_value(ary, traced_type), indices_or_sections, axis=axis))
    return _split_to_getitems(traced_array=ary, outputs=outputs, axis=axis)


def _hsplit_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ArrayFunctionResult:
    if len(args) != _REQUIRED_SPLIT_ARGS:
        msg = "numpy.hsplit expects (ary, indices_or_sections) during tracing"
        raise TracingError(msg)
    if kwargs:
        msg = f"numpy.hsplit kwargs not supported during tracing: {sorted(kwargs)}"
        raise TracingError(msg)

    ary = args[0]
    if not isinstance(ary, traced_type):
        msg = "numpy.hsplit tracing requires a traced first argument"
        raise TracingError(msg)

    outputs = tuple(np.hsplit(_get_value(ary, traced_type), args[1]))
    axis = 1 if _snapshot_traced(ary)[1].ndim > 1 else 0
    return _split_to_getitems(traced_array=ary, outputs=outputs, axis=axis)


def _vsplit_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ArrayFunctionResult:
    if len(args) != _REQUIRED_SPLIT_ARGS:
        msg = "numpy.vsplit expects (ary, indices_or_sections) during tracing"
        raise TracingError(msg)
    if kwargs:
        msg = f"numpy.vsplit kwargs not supported during tracing: {sorted(kwargs)}"
        raise TracingError(msg)

    ary = args[0]
    if not isinstance(ary, traced_type):
        msg = "numpy.vsplit tracing requires a traced first argument"
        raise TracingError(msg)

    outputs = tuple(np.vsplit(_get_value(ary, traced_type), args[1]))
    return _split_to_getitems(traced_array=ary, outputs=outputs, axis=0)


def _dsplit_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ArrayFunctionResult:
    if len(args) != _REQUIRED_SPLIT_ARGS:
        msg = "numpy.dsplit expects (ary, indices_or_sections) during tracing"
        raise TracingError(msg)
    if kwargs:
        msg = f"numpy.dsplit kwargs not supported during tracing: {sorted(kwargs)}"
        raise TracingError(msg)

    ary = args[0]
    if not isinstance(ary, traced_type):
        msg = "numpy.dsplit tracing requires a traced first argument"
        raise TracingError(msg)

    outputs = tuple(np.dsplit(_get_value(ary, traced_type), args[1]))
    return _split_to_getitems(traced_array=ary, outputs=outputs, axis=2)
