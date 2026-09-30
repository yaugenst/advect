"""NumPy 2.x aliases for canonical array-family operations."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, cast

import numpy as _numpy  # noqa: ICN001 - concrete namespace with dynamic protocol operands

from advect.core._errors import TracingError
from advect.core._protocols import _snapshot_traced
from advect.numpy._array_function.emission import _emit, _get_array_value, _get_value
from advect.numpy._composite_lowering import lower_cumulative_initial, operand_dtype
from advect.numpy._op_bindings import canonicalize_numpy_op, frontend_lowering

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike
    from advect.numpy._array_function.emission import ArrayFunctionHandler

np: Any = _numpy

# NumPy's dispatchers bind these fixed signatures before any handler runs.


def _astype_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    device = kwargs.get("device")
    if device not in {None, "cpu"}:
        msg = "numpy.astype device= must be None or 'cpu'"
        raise TracingError(msg)

    x, dtype = args
    copy = bool(kwargs.get("copy", True))
    target_dtype = np.dtype(dtype)
    value = _get_array_value(x, traced_type)
    if not copy and operand_dtype(value) == target_dtype:
        msg = (
            "numpy.astype(copy=False) would create a runtime-dependent alias when the "
            "dtype is unchanged; use x.astype(..., copy=False) to preserve wrapper identity"
        )
        raise TracingError(msg)
    # A nested trace's value is itself traced and must dispatch, not convert.
    operand = value if callable(getattr(value, "_advect_snapshot", None)) else np.asarray(value)
    result = np.astype(operand, target_dtype, copy=copy)
    if isinstance(value, np.generic):
        # From NumPy 2.1 np.astype returns a NumPy scalar for one; 2.0 rejects it.
        result = result[()]
    attrs: dict[str, Any] = {"dtype": str(target_dtype), "copy": copy}
    return _emit(graph, traced_type, "numpy.astype", (x,), result, attrs)


@frontend_lowering("array.transpose")
def _matrix_transpose_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    _kwargs: dict[str, Any],
) -> tuple[object, int]:
    x = args[0]
    value = _get_value(x, traced_type)
    result = np.matrix_transpose(value)
    axes = list(range(cast("Any", value).ndim))
    axes[-2], axes[-1] = axes[-1], axes[-2]
    return _emit(graph, traced_type, "numpy.transpose", (x,), result, {"axes": tuple(axes)})


@frontend_lowering("array.transpose")
def _permute_dims_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    x = args[0]
    axes_raw = args[1] if len(args) > 1 else kwargs.get("axes")
    axes = None if axes_raw is None else tuple(int(axis) for axis in axes_raw)
    result = np.permute_dims(_get_value(x, traced_type), axes=axes)
    return _emit(graph, traced_type, "numpy.transpose", (x,), result, {"axes": axes})


def _cumulative_alias_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    function: Callable[..., object],
    op_name: str,
) -> tuple[object, int]:
    x = args[0]
    axis = kwargs.get("axis")
    dtype = kwargs.get("dtype")
    if kwargs.get("include_initial", False):
        result = lower_cumulative_initial(function.__name__, x, axis=axis, dtype=dtype)
        node_id, concrete = _snapshot_traced(result)
        return concrete, node_id

    call_kwargs: dict[str, Any] = {"axis": axis, "include_initial": False}
    if dtype is not None:
        call_kwargs["dtype"] = dtype
    result = function(_get_value(x, traced_type), **call_kwargs)
    attrs: dict[str, Any] = {}
    if axis is not None:
        attrs["axis"] = int(axis)
    if dtype is not None:
        attrs["dtype"] = str(np.dtype(dtype))
    return _emit(graph, traced_type, op_name, (x,), result, attrs)


@frontend_lowering("array.cross")
def _linalg_cross_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    left, right = args
    axis = int(kwargs.get("axis", -1))
    result = np.linalg.cross(
        _get_value(left, traced_type),
        _get_value(right, traced_type),
        axis=axis,
    )
    return _emit(graph, traced_type, "numpy.cross", (left, right), result, {"axis": axis})


def _linalg_binary_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    function: Callable[..., object],
    op_name: str,
) -> tuple[object, int]:
    left, right = args
    attrs = dict(kwargs)
    if "axis" in attrs:
        attrs["axis"] = int(attrs["axis"])
    if "axes" in attrs:
        axes = attrs["axes"]
        attrs["axes"] = (
            int(axes)
            if isinstance(axes, (int, np.integer))
            else tuple(tuple(int(axis) for axis in group) for group in axes)
        )
    result = function(
        _get_value(left, traced_type),
        _get_value(right, traced_type),
        **kwargs,
    )
    return _emit(graph, traced_type, op_name, (left, right), result, attrs)


@frontend_lowering("array.diagonal")
def _linalg_diagonal_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    x = args[0]
    offset = int(kwargs.get("offset", 0))
    result = np.linalg.diagonal(_get_value(x, traced_type), offset=offset)
    attrs = {"offset": offset, "axis1": -2, "axis2": -1}
    return _emit(graph, traced_type, "numpy.diagonal", (x,), result, attrs)


def _linalg_norm_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    function: Callable[..., object],
) -> tuple[object, int]:
    x = args[0]
    attrs = dict(kwargs)
    axis = attrs.get("axis")
    if axis is not None:
        attrs["axis"] = (
            int(axis) if isinstance(axis, (int, np.integer)) else tuple(int(item) for item in axis)
        )
    result = function(_get_value(x, traced_type), **kwargs)
    return _emit(graph, traced_type, f"numpy.linalg.{function.__name__}", (x,), result, attrs)


@frontend_lowering("array.trace")
def _linalg_trace_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    x = args[0]
    offset = int(kwargs.get("offset", 0))
    dtype = kwargs.get("dtype")
    result = np.linalg.trace(_get_value(x, traced_type), offset=offset, dtype=dtype)
    attrs: dict[str, Any] = {"offset": offset, "axis1": -2, "axis2": -1}
    if dtype is not None:
        attrs["dtype"] = str(np.dtype(dtype))
    return _emit(graph, traced_type, "numpy.trace", (x,), result, attrs)


def register_alias_handlers(
    handlers: dict[Callable[..., Any], ArrayFunctionHandler],
) -> None:
    """Register NumPy names that lower to existing canonical operations."""
    handlers[np.astype] = _astype_handler
    handlers[np.matrix_transpose] = _matrix_transpose_handler
    handlers[np.linalg.matrix_transpose] = _matrix_transpose_handler
    handlers[np.permute_dims] = _permute_dims_handler
    for name in ("cumsum", "cumprod"):
        function = getattr(np, name.replace("cum", "cumulative_"), None)
        if callable(function):
            handlers[function] = frontend_lowering(f"array.{name}")(
                partial(_cumulative_alias_handler, function=function, op_name=f"numpy.{name}")
            )
    handlers[np.linalg.cross] = _linalg_cross_handler
    handlers[np.linalg.diagonal] = _linalg_diagonal_handler
    handlers[np.linalg.trace] = _linalg_trace_handler
    for function in (np.linalg.matmul, np.linalg.outer, np.linalg.tensordot, np.linalg.vecdot):
        op_name = f"numpy.{function.__name__}"
        handlers[function] = frontend_lowering(canonicalize_numpy_op(op_name))(
            partial(_linalg_binary_handler, function=function, op_name=op_name)
        )
    for function in (np.linalg.matrix_norm, np.linalg.vector_norm):
        handlers[function] = partial(_linalg_norm_handler, function=function)
