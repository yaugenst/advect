"""Trace NumPy split-family functions while preserving their list results."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as _numpy  # noqa: ICN001 - concrete namespace with dynamic protocol operands

from advect.core._errors import TracingError
from advect.numpy._array_function.composite import _finish
from advect.numpy._array_function.emission import _get_value
from advect.numpy._array_function.normalization import _normalize_axis

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike
    from advect.numpy._array_function.composite import CompositeResult
    from advect.numpy._array_function.emission import ArrayFunctionHandler

np: Any = _numpy


_AXIS_POSITION = 2
# hsplit, vsplit and dsplit fix their axis; split and array_split receive it.
_FIXED_AXES: dict[object, Callable[[int], int]] = {
    np.hsplit: lambda ndim: 1 if ndim > 1 else 0,
    np.vsplit: lambda _ndim: 0,
    np.dsplit: lambda _ndim: 2,
}


def _split_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    function: Callable[..., list[Any]],
) -> CompositeResult:
    """Slice the traced array into the parts NumPy's own split returns."""
    array = args[0]
    if not isinstance(array, traced_type):
        msg = f"numpy.{function.__name__} tracing requires a traced first argument"
        raise TracingError(msg)
    value: Any = _get_value(array, traced_type)
    pieces = function(value, *args[1:], **kwargs)
    fixed_axis = _FIXED_AXES.get(function)
    axis = _normalize_axis(
        fixed_axis(value.ndim)
        if fixed_axis
        else int(args[_AXIS_POSITION] if len(args) > _AXIS_POSITION else kwargs.get("axis", 0)),
        value.ndim,
    )
    parts: list[Any] = []
    start = 0
    for piece in pieces:
        width = piece.shape[axis]
        index: list[object] = [slice(None)] * value.ndim
        index[axis] = slice(start, start + width)
        parts.append(array[tuple(index)])
        start += width
    return _finish(parts, traced_type=traced_type)


def register_split_handlers(
    handlers: dict[Callable[..., Any], ArrayFunctionHandler],
) -> None:
    """Register the split family."""
    for function in (np.split, np.array_split, np.hsplit, np.vsplit, np.dsplit):
        handlers[function] = partial(_split_handler, function=function)
