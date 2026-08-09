"""Trace NumPy unique-family functions and their structured results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as _numpy  # noqa: ICN001 - typed module and dynamic lowering namespace

from advect.core._errors import TracingError
from advect.core._protocols import _snapshot_traced
from advect.numpy._array_function.composite import _finish
from advect.numpy._op_bindings import frontend_lowering

np: Any = _numpy

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike
    from advect.numpy._array_function.composite import CompositeResult

_UNIQUE_ALLOWED_KWARGS = frozenset(
    {"return_index", "return_inverse", "return_counts", "axis", "equal_nan", "sorted"}
)


def _normalize_unique_axis(axis: object) -> int | None:
    if axis is None:
        return None
    if isinstance(axis, np.integer):
        return int(axis)
    if isinstance(axis, int):
        return axis
    msg = f"numpy.unique axis must be an integer or None during tracing (got {type(axis).__name__})"
    raise TracingError(msg)


def _call_numpy_unique(
    value: np.ndarray[Any, Any],
    *,
    return_index: bool,
    return_inverse: bool,
    return_counts: bool,
    axis: int | None,
    equal_nan: bool,
    sorted_values: bool,
) -> object:
    kwargs: dict[str, object] = {
        "return_index": return_index,
        "return_inverse": return_inverse,
        "return_counts": return_counts,
        "axis": axis,
        "equal_nan": equal_nan,
    }
    # ``sorted`` was added after NumPy 2.0. Its historical behavior already
    # matches the default, so only forward the keyword when a caller requests
    # the newer non-default behavior.
    if not sorted_values:
        kwargs["sorted"] = False
    return np.unique(value, **kwargs)


def _unique_result(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> object:
    positional_names = ("return_index", "return_inverse", "return_counts", "axis")
    if not args or len(args) > len(positional_names) + 1:
        msg = "numpy.unique received an invalid positional signature during tracing"
        raise TracingError(msg)

    unsupported = set(kwargs) - _UNIQUE_ALLOWED_KWARGS
    if unsupported:
        msg = f"numpy.unique kwargs not supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)

    values = dict(kwargs)
    for name, value in zip(positional_names, args[1:], strict=False):
        if name in values:
            msg = f"numpy.unique received {name} twice"
            raise TracingError(msg)
        values[name] = value

    array = args[0]
    return_index = bool(values.get("return_index", False))
    return_inverse = bool(values.get("return_inverse", False))
    return_counts = bool(values.get("return_counts", False))
    axis = _normalize_unique_axis(values.get("axis"))
    equal_nan = bool(values.get("equal_nan", True))
    sorted_values = bool(values.get("sorted", True))
    _node_id, concrete_array = _snapshot_traced(array)
    concrete_result = _call_numpy_unique(
        np.asarray(concrete_array),
        return_index=True,
        return_inverse=return_inverse,
        return_counts=return_counts,
        axis=axis,
        equal_nan=equal_nan,
        sorted_values=sorted_values,
    )
    concrete_outputs = tuple(cast("tuple[object, ...]", concrete_result))
    indices = np.asarray(concrete_outputs[1])
    source = np.ravel(array) if axis is None else array
    unique_values = np.take(source, indices, axis=axis)

    def discrete(value: object) -> object:
        concrete = np.asarray(value)
        zero = np.astype(np.sum(np.zeros_like(array)), concrete.dtype)
        return zero + concrete

    outputs: list[object] = [unique_values]
    if return_index:
        outputs.append(discrete(indices))
    cursor = 2
    if return_inverse:
        outputs.append(discrete(concrete_outputs[cursor]))
        cursor += 1
    if return_counts:
        outputs.append(discrete(concrete_outputs[cursor]))
    return outputs[0] if len(outputs) == 1 else tuple(outputs)


def _unique_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    return _finish(
        _unique_result(args, kwargs),
        traced_type=traced_type,
    )


def _unique_values_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    if kwargs:
        msg = f"numpy.unique_values kwargs not supported during tracing: {sorted(kwargs)}"
        raise TracingError(msg)
    _ = graph
    return _finish(
        _unique_result(args, {"equal_nan": False}),
        traced_type=traced_type,
    )


def _named_unique_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    function: Callable[[object], object],
    unique_kwargs: dict[str, bool],
) -> CompositeResult:
    if len(args) != 1 or kwargs:
        msg = f"numpy.{function.__name__} expects one array during tracing"
        raise TracingError(msg)
    result = _unique_result(
        args,
        {"equal_nan": False, **unique_kwargs},
    )
    outputs = cast("tuple[object, ...]", result)
    result_type = type(function(np.array([0])))
    return _finish(result_type(*outputs), traced_type=traced_type)


def register_unique_handlers(
    handlers: dict[Any, Any],
) -> None:
    """Register NumPy's classic and Array-API-style unique functions."""
    handlers[np.unique] = _unique_handler
    handlers[np.unique_values] = _unique_values_handler
    handlers[np.unique_all] = lambda graph, traced_type, args, kwargs: _named_unique_handler(
        graph,
        traced_type,
        args,
        kwargs,
        function=np.unique_all,
        unique_kwargs={
            "return_index": True,
            "return_inverse": True,
            "return_counts": True,
        },
    )
    handlers[np.unique_counts] = lambda graph, traced_type, args, kwargs: _named_unique_handler(
        graph,
        traced_type,
        args,
        kwargs,
        function=np.unique_counts,
        unique_kwargs={"return_counts": True},
    )
    frontend_lowering("array.unique_counts")(handlers[np.unique_counts])
    handlers[np.unique_inverse] = lambda graph, traced_type, args, kwargs: _named_unique_handler(
        graph,
        traced_type,
        args,
        kwargs,
        function=np.unique_inverse,
        unique_kwargs={"return_inverse": True},
    )
    frontend_lowering("array.unique_inverse")(handlers[np.unique_inverse])
