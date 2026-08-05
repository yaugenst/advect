"""One-dimensional signal operations with differentiable array operands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as _numpy  # noqa: ICN001 - concrete namespace with dynamic protocol operands

from advect.core._errors import TracingError
from advect.numpy._op_bindings import canonicalize_numpy_op
from advect.numpy._protocol_array_function_common import (
    _add_backend_node,
    _get_node,
    _get_value,
    _result_shape_and_dtype,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike

np: Any = _numpy


_BINARY_ARITY = 2


def _signal_handler(
    *,
    function: Callable[..., object],
    op_name: str,
    default_mode: str,
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    if len(args) not in {_BINARY_ARITY, _BINARY_ARITY + 1}:
        msg = f"{function.__module__}.{function.__name__} expects (a, v, mode='full')"
        raise TracingError(msg)
    unsupported = set(kwargs) - {"mode"}
    if unsupported:
        msg = (
            f"{function.__module__}.{function.__name__} kwargs not supported during "
            f"tracing: {sorted(unsupported)}"
        )
        raise TracingError(msg)
    if len(args) == _BINARY_ARITY + 1 and "mode" in kwargs:
        msg = f"{function.__module__}.{function.__name__} received mode twice"
        raise TracingError(msg)

    left, right = args[:2]
    mode = str(args[2] if len(args) == _BINARY_ARITY + 1 else kwargs.get("mode", default_mode))
    if mode not in {"full", "same", "valid"}:
        msg = f"{function.__module__}.{function.__name__} mode must be full, same, or valid"
        raise TracingError(msg)
    result = function(
        _get_value(left, traced_type),
        _get_value(right, traced_type),
        mode=mode,
    )
    shape, dtype = _result_shape_and_dtype(result)
    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op(op_name),
        inputs=(
            _get_node(left, graph, traced_type),
            _get_node(right, graph, traced_type),
        ),
        value=result,
        attrs={"mode": mode},
        shape=shape,
        dtype=dtype,
    )
    return result, node_id


def _convolve_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    return _signal_handler(
        function=np.convolve,
        op_name="numpy.convolve",
        default_mode="full",
        graph=graph,
        traced_type=traced_type,
        args=args,
        kwargs=kwargs,
    )


def _correlate_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    return _signal_handler(
        function=np.correlate,
        op_name="numpy.correlate",
        default_mode="valid",
        graph=graph,
        traced_type=traced_type,
        args=args,
        kwargs=kwargs,
    )


def register_signal_handlers(
    handlers: dict[Callable[..., Any], Callable[..., Any]],
) -> None:
    """Register NumPy signal operations."""
    handlers[np.convolve] = _convolve_handler
    handlers[np.correlate] = _correlate_handler
