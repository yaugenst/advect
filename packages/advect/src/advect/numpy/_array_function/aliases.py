"""NumPy 2.x aliases for canonical array-family operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as _numpy  # noqa: ICN001 - concrete namespace with dynamic protocol operands

from advect.core._errors import TracingError
from advect.core._protocols import _snapshot_traced
from advect.numpy._array_function.emission import (
    _add_backend_node,
    _get_node,
    _get_value,
    _result_shape_and_dtype,
)
from advect.numpy._op_bindings import canonicalize_numpy_op, frontend_lowering

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike
    from advect.numpy._array_function.emission import ArrayFunctionHandler

np: Any = _numpy

_BINARY_ARITY = 2


def _record_unary(
    *,
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    operand: object,
    result: object,
    op: str,
    attrs: dict[str, Any],
) -> tuple[object, int]:
    shape, dtype = _result_shape_and_dtype(result)
    node_id = _add_backend_node(
        graph=graph,
        op=op,
        inputs=(_get_node(operand, graph, traced_type),),
        value=result,
        attrs=attrs,
        shape=shape,
        dtype=dtype,
    )
    return result, node_id


def _record_binary(
    *,
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    left: object,
    right: object,
    result: object,
    op: str,
    attrs: dict[str, Any],
) -> tuple[object, int]:
    shape, dtype = _result_shape_and_dtype(result)
    node_id = _add_backend_node(
        graph=graph,
        op=op,
        inputs=(
            _get_node(left, graph, traced_type),
            _get_node(right, graph, traced_type),
        ),
        value=result,
        attrs=attrs,
        shape=shape,
        dtype=dtype,
    )
    return result, node_id


def _astype_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    if len(args) != _BINARY_ARITY:
        msg = "numpy.astype expects (x, dtype) during tracing"
        raise TracingError(msg)
    unsupported = set(kwargs) - {"copy", "device"}
    if unsupported:
        msg = f"numpy.astype kwargs not supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)
    device = kwargs.get("device")
    if device not in {None, "cpu"}:
        msg = "numpy.astype device= must be None or 'cpu'"
        raise TracingError(msg)

    x, dtype = args
    copy = bool(kwargs.get("copy", True))
    target_dtype = np.dtype(dtype)
    value = _get_value(x, traced_type)
    if not copy and cast("Any", value).dtype == target_dtype:
        msg = (
            "numpy.astype(copy=False) would create a runtime-dependent alias when the "
            "dtype is unchanged; use x.astype(..., copy=False) to preserve wrapper identity"
        )
        raise TracingError(msg)
    result = np.astype(np.asarray(value), target_dtype, copy=copy)
    attrs: dict[str, Any] = {"dtype": str(target_dtype), "copy": copy}
    return _record_unary(
        graph=graph,
        traced_type=traced_type,
        operand=x,
        result=result,
        op=canonicalize_numpy_op("numpy.astype"),
        attrs=attrs,
    )


@frontend_lowering("array.transpose")
def _matrix_transpose_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    if len(args) != 1 or kwargs:
        msg = "numpy.matrix_transpose expects one array and no kwargs during tracing"
        raise TracingError(msg)
    x = args[0]
    value = _get_value(x, traced_type)
    result = np.matrix_transpose(value)
    rank = cast("Any", value).ndim
    axes = list(range(rank))
    axes[-2], axes[-1] = axes[-1], axes[-2]
    return _record_unary(
        graph=graph,
        traced_type=traced_type,
        operand=x,
        result=result,
        op=canonicalize_numpy_op("numpy.transpose"),
        attrs={"axes": tuple(axes)},
    )


@frontend_lowering("array.transpose")
def _permute_dims_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    if len(args) not in {1, _BINARY_ARITY}:
        msg = "numpy.permute_dims expects (a, axes=None) during tracing"
        raise TracingError(msg)
    unsupported = set(kwargs) - {"axes"}
    if unsupported:
        msg = f"numpy.permute_dims kwargs not supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)
    if len(args) == _BINARY_ARITY and "axes" in kwargs:
        msg = "numpy.permute_dims received axes twice"
        raise TracingError(msg)
    x = args[0]
    axes_raw = args[1] if len(args) == _BINARY_ARITY else kwargs.get("axes")
    axes = None if axes_raw is None else tuple(int(axis) for axis in axes_raw)
    result = np.permute_dims(_get_value(x, traced_type), axes=axes)
    return _record_unary(
        graph=graph,
        traced_type=traced_type,
        operand=x,
        result=result,
        op=canonicalize_numpy_op("numpy.transpose"),
        attrs={"axes": axes},
    )


def _cumulative_alias_handler(
    *,
    function: Callable[..., object],
    op_name: str,
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    if len(args) != 1:
        msg = f"{function.__module__}.{function.__name__} expects one array during tracing"
        raise TracingError(msg)
    unsupported = set(kwargs) - {"axis", "dtype", "include_initial"}
    if unsupported:
        msg = (
            f"{function.__module__}.{function.__name__} kwargs not supported during tracing: "
            f"{sorted(unsupported)}"
        )
        raise TracingError(msg)
    x = args[0]
    axis = kwargs.get("axis")
    dtype = kwargs.get("dtype")
    include_initial = bool(kwargs.get("include_initial", False))
    if include_initial:
        if axis is None and int(x.ndim) != 1:
            msg = "For arrays which have more than one dimension ``axis`` argument is required."
            raise ValueError(msg)
        is_sum = function is np.cumulative_sum
        canonical = cast("Callable[..., Any]", np.cumsum if is_sum else np.cumprod)
        base_kwargs: dict[str, object] = {"axis": axis}
        if dtype is not None:
            base_kwargs["dtype"] = dtype
        base = canonical(x, **base_kwargs)
        if axis is None:
            seed = np.reshape(np.sum(base) * 0, (1,))
        else:
            normalized_axis = int(axis)
            if normalized_axis < 0:
                normalized_axis += int(x.ndim)
            seed = np.expand_dims(np.sum(base * 0, axis=normalized_axis), normalized_axis)
        initial = np.zeros_like(seed) if is_sum else np.ones_like(seed)
        result = np.concatenate((initial, base), axis=0 if axis is None else int(axis))
        node_id, concrete = _snapshot_traced(result)
        return concrete, int(node_id)

    call_kwargs: dict[str, Any] = {"axis": axis, "include_initial": False}
    if dtype is not None:
        call_kwargs["dtype"] = dtype
    result = function(_get_value(x, traced_type), **call_kwargs)
    attrs: dict[str, Any] = {}
    if axis is not None:
        attrs["axis"] = int(axis)
    if dtype is not None:
        attrs["dtype"] = str(np.dtype(dtype))
    return _record_unary(
        graph=graph,
        traced_type=traced_type,
        operand=x,
        result=result,
        op=op_name,
        attrs=attrs,
    )


@frontend_lowering("array.cumsum")
def _cumulative_sum_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    return _cumulative_alias_handler(
        function=np.cumulative_sum,
        op_name=canonicalize_numpy_op("numpy.cumsum"),
        graph=graph,
        traced_type=traced_type,
        args=args,
        kwargs=kwargs,
    )


@frontend_lowering("array.cumprod")
def _cumulative_prod_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    return _cumulative_alias_handler(
        function=np.cumulative_prod,
        op_name=canonicalize_numpy_op("numpy.cumprod"),
        graph=graph,
        traced_type=traced_type,
        args=args,
        kwargs=kwargs,
    )


@frontend_lowering("array.cross")
def _linalg_cross_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    if len(args) != _BINARY_ARITY or set(kwargs) - {"axis"}:
        msg = "numpy.linalg.cross expects (x1, x2, *, axis=-1) during tracing"
        raise TracingError(msg)
    left, right = args
    axis = int(kwargs.get("axis", -1))
    result = np.linalg.cross(
        _get_value(left, traced_type),
        _get_value(right, traced_type),
        axis=axis,
    )
    return _record_binary(
        graph=graph,
        traced_type=traced_type,
        left=left,
        right=right,
        result=result,
        op=canonicalize_numpy_op("numpy.cross"),
        attrs={"axis": axis},
    )


def _linalg_binary_handler(
    *,
    function: Callable[..., object],
    op_name: str,
    allowed_kwargs: frozenset[str],
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    if len(args) != _BINARY_ARITY:
        msg = f"{function.__module__}.{function.__name__} expects two arrays during tracing"
        raise TracingError(msg)
    unsupported = set(kwargs) - allowed_kwargs
    if unsupported:
        msg = (
            f"{function.__module__}.{function.__name__} kwargs not supported during tracing: "
            f"{sorted(unsupported)}"
        )
        raise TracingError(msg)
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
    return _record_binary(
        graph=graph,
        traced_type=traced_type,
        left=left,
        right=right,
        result=result,
        op=op_name,
        attrs=attrs,
    )


def _linalg_binary_frontend(
    function: Callable[..., object],
    op_name: str,
    *,
    allowed_kwargs: frozenset[str] = frozenset(),
) -> Callable[..., tuple[object, int]]:
    @frontend_lowering(op_name)
    def handler(
        graph: DynamicTape,
        traced_type: type[TracedArrayLike],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[object, int]:
        return _linalg_binary_handler(
            function=function,
            op_name=op_name,
            allowed_kwargs=allowed_kwargs,
            graph=graph,
            traced_type=traced_type,
            args=args,
            kwargs=kwargs,
        )

    return handler


@frontend_lowering("array.diagonal")
def _linalg_diagonal_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    if len(args) != 1 or set(kwargs) - {"offset"}:
        msg = "numpy.linalg.diagonal expects (x, *, offset=0) during tracing"
        raise TracingError(msg)
    x = args[0]
    offset = int(kwargs.get("offset", 0))
    result = np.linalg.diagonal(_get_value(x, traced_type), offset=offset)
    return _record_unary(
        graph=graph,
        traced_type=traced_type,
        operand=x,
        result=result,
        op=canonicalize_numpy_op("numpy.diagonal"),
        attrs={"offset": offset, "axis1": -2, "axis2": -1},
    )


def _linalg_norm_handler(
    *,
    function: Callable[..., object],
    op_name: str,
    allowed_kwargs: frozenset[str],
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    if len(args) != 1:
        msg = f"{function.__module__}.{function.__name__} expects one array during tracing"
        raise TracingError(msg)
    unsupported = set(kwargs) - allowed_kwargs
    if unsupported:
        msg = (
            f"{function.__module__}.{function.__name__} kwargs not supported during tracing: "
            f"{sorted(unsupported)}"
        )
        raise TracingError(msg)
    x = args[0]
    attrs = dict(kwargs)
    axis = attrs.get("axis")
    if axis is not None:
        attrs["axis"] = (
            int(axis) if isinstance(axis, (int, np.integer)) else tuple(int(item) for item in axis)
        )
    result = function(_get_value(x, traced_type), **kwargs)
    return _record_unary(
        graph=graph,
        traced_type=traced_type,
        operand=x,
        result=result,
        op=op_name,
        attrs=attrs,
    )


@frontend_lowering("array.trace")
def _linalg_trace_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    if len(args) != 1 or set(kwargs) - {"offset", "dtype"}:
        msg = "numpy.linalg.trace expects (x, *, offset=0, dtype=None) during tracing"
        raise TracingError(msg)
    x = args[0]
    offset = int(kwargs.get("offset", 0))
    dtype = kwargs.get("dtype")
    result = np.linalg.trace(_get_value(x, traced_type), offset=offset, dtype=dtype)
    attrs: dict[str, Any] = {"offset": offset, "axis1": -2, "axis2": -1}
    if dtype is not None:
        attrs["dtype"] = str(np.dtype(dtype))
    return _record_unary(
        graph=graph,
        traced_type=traced_type,
        operand=x,
        result=result,
        op=canonicalize_numpy_op("numpy.trace"),
        attrs=attrs,
    )


def register_alias_handlers(
    handlers: dict[Callable[..., Any], ArrayFunctionHandler],
) -> None:
    """Register NumPy names that lower to existing canonical operations."""
    handlers[np.astype] = _astype_handler
    handlers[np.matrix_transpose] = _matrix_transpose_handler
    handlers[np.permute_dims] = _permute_dims_handler
    cumulative_sum = getattr(np, "cumulative_sum", None)
    if callable(cumulative_sum):
        handlers[cumulative_sum] = _cumulative_sum_handler
    cumulative_prod = getattr(np, "cumulative_prod", None)
    if callable(cumulative_prod):
        handlers[cumulative_prod] = _cumulative_prod_handler
    handlers[np.linalg.cross] = _linalg_cross_handler
    handlers[np.linalg.diagonal] = _linalg_diagonal_handler
    handlers[np.linalg.matmul] = _linalg_binary_frontend(
        np.linalg.matmul,
        canonicalize_numpy_op("numpy.matmul"),
    )
    handlers[np.linalg.outer] = _linalg_binary_frontend(
        np.linalg.outer,
        canonicalize_numpy_op("numpy.outer"),
    )
    handlers[np.linalg.tensordot] = _linalg_binary_frontend(
        np.linalg.tensordot,
        canonicalize_numpy_op("numpy.tensordot"),
        allowed_kwargs=frozenset({"axes"}),
    )
    handlers[np.linalg.vecdot] = _linalg_binary_frontend(
        np.linalg.vecdot,
        canonicalize_numpy_op("numpy.vecdot"),
        allowed_kwargs=frozenset({"axis"}),
    )
    handlers[np.linalg.matrix_transpose] = _matrix_transpose_handler
    handlers[np.linalg.matrix_norm] = lambda graph, traced_type, args, kwargs: (
        _linalg_norm_handler(
            function=np.linalg.matrix_norm,
            op_name=canonicalize_numpy_op("numpy.linalg.matrix_norm"),
            allowed_kwargs=frozenset({"keepdims", "ord"}),
            graph=graph,
            traced_type=traced_type,
            args=args,
            kwargs=kwargs,
        )
    )
    handlers[np.linalg.vector_norm] = lambda graph, traced_type, args, kwargs: (
        _linalg_norm_handler(
            function=np.linalg.vector_norm,
            op_name=canonicalize_numpy_op("numpy.linalg.vector_norm"),
            allowed_kwargs=frozenset({"axis", "keepdims", "ord"}),
            graph=graph,
            traced_type=traced_type,
            args=args,
            kwargs=kwargs,
        )
    )
    handlers[np.linalg.trace] = _linalg_trace_handler
