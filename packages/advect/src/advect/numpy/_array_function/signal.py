"""One-dimensional signal operations with differentiable array operands."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as _numpy  # noqa: ICN001 - concrete namespace with dynamic protocol operands

from advect.core._errors import TracingError
from advect.numpy._array_function.emission import _emit, _get_value

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike
    from advect.numpy._array_function.emission import ArrayFunctionHandler

np: Any = _numpy


_BINARY_ARITY = 2


def _signal_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    function: Callable[..., object],
    default_mode: str,
) -> tuple[object, int]:
    left, right = args[:2]
    mode = str(args[2] if len(args) > _BINARY_ARITY else kwargs.get("mode", default_mode))
    if mode not in {"full", "same", "valid"}:
        msg = f"{function.__module__}.{function.__name__} mode must be full, same, or valid"
        raise TracingError(msg)
    result = function(
        _get_value(left, traced_type),
        _get_value(right, traced_type),
        mode=mode,
    )
    op = f"numpy.{function.__name__}"
    return _emit(graph, traced_type, op, (left, right), result, {"mode": mode})


def register_signal_handlers(
    handlers: dict[Callable[..., Any], ArrayFunctionHandler],
) -> None:
    """Register NumPy signal operations."""
    handlers[np.convolve] = partial(_signal_handler, function=np.convolve, default_mode="full")
    handlers[np.correlate] = partial(_signal_handler, function=np.correlate, default_mode="valid")
