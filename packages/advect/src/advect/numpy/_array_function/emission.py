"""Common helpers for NumPy array-function dispatch.

This module contains helper utilities and small handler factories used by the
NumPy ``__array_function__`` dispatch layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as _numpy  # noqa: ICN001 - concrete namespace with dynamic protocol operands

from advect.core._array_protocol_helpers import (
    weak_scalar_runtime_value,
)
from advect.core._context import (
    _get_operation_recorder,
    _is_recorder_in_active_trace_stack,
    get_source_location,
)
from advect.core._errors import TracingError
from advect.core._protocols import _snapshot_traced
from advect.numpy._array_function.composite import _finish
from advect.numpy._composite_lowering import NON_SCALAR_INITIAL, lower_controlled_reduction
from advect.numpy._op_bindings import canonicalize_numpy_op, frontend_lowering
from advect.numpy._signature import CLIP_KEYWORDS, bind_clip_bounds
from advect.numpy._static_attr_arrays import decode_static_array_attr, encode_static_array_attr

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike

np: Any = _numpy


# Array-function lowerings may return nested tuple/list result containers
# (for example ``histogramdd`` and NumPy's ``unique_*`` named tuples).  The
# value tree and node-id tree have the same shape; leaves are arrays and ints.
ArrayFunctionResult = tuple[Any, Any]
type ArrayFunctionHandler = Callable[
    [DynamicTape, type[TracedArrayLike], tuple[Any, ...], dict[str, Any]],
    ArrayFunctionResult,
]


@dataclass(frozen=True, slots=True)
class _LiteralOperand:
    value: object


type _Operand = int | _LiteralOperand


def _add_backend_node(
    *,
    graph: DynamicTape,
    op: str,
    inputs: tuple[_Operand, ...],
    value: object,
    attrs: dict[str, Any],
    shape: tuple[int, ...] | None = None,
    dtype: object = None,
) -> int:
    """Record one NumPy node; its shape and dtype default to ``value``'s."""
    if shape is None:
        shape, dtype = _result_shape_and_dtype(value)
    native_attrs = {**attrs, "_advect_backend": "numpy"}
    parents = tuple(item for item in inputs if isinstance(item, int))
    literals = tuple(item.value for item in inputs if isinstance(item, _LiteralOperand))
    return graph.record_operation(
        op,
        parents,
        value,
        native_attrs,
        shape,
        dtype,
        input_positions=(
            tuple(position for position, item in enumerate(inputs) if isinstance(item, int))
            if literals
            else None
        ),
        literals=literals,
        source_location=get_source_location(),
    )


def _emit(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    op: str,
    operands: Iterable[object],
    result: object,
    attrs: dict[str, Any] | None = None,
    *,
    weak_scalars: bool = False,
) -> tuple[Any, int]:
    """Record ``result`` as NumPy operation ``op`` over ``operands``.

    ``weak_scalars`` records exact Python scalars as weak literals, for the
    functions NumPy promotes under NEP 50 rather than after array coercion.
    """
    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op(op),
        inputs=tuple(
            _LiteralOperand(operand)
            if weak_scalars and type(operand) in _PYTHON_SCALARS
            else _get_node(operand, graph, traced_type)
            for operand in operands
        ),
        value=result,
        attrs={} if attrs is None else attrs,
    )
    return result, node_id


def _get_weak_value(arg: object, traced_type: type[TracedArrayLike]) -> object:
    """Return an exact Python scalar unchanged so NumPy promotes it weakly."""
    return arg if type(arg) in _PYTHON_SCALARS else _get_value(arg, traced_type)


def _get_array_value(arg: object, traced_type: type[TracedArrayLike]) -> object:
    """Return an operand's value with a Python scalar as the array NumPy reads it as.

    A weak scalar's value is a Python scalar, which a handler that reads its
    operand's shape or dtype must see as NumPy's strong rank-zero conversion.
    """
    value = _get_value(arg, traced_type)
    return np.asarray(value) if type(value) in _PYTHON_SCALARS else value


def _get_value(arg: object, traced_type: type[TracedArrayLike]) -> object:
    """Extract backend value from TracedArray or convert plain inputs to ndarray."""
    snapshot = getattr(arg, "_advect_snapshot", None)
    if isinstance(arg, traced_type) or callable(snapshot):
        owner = cast("Any", arg).recorder
        operation_recorder = _get_operation_recorder()
        if operation_recorder is not None and owner is not operation_recorder:
            if not _is_recorder_in_active_trace_stack(owner):
                msg = "Cannot evaluate an array operand from an unrelated trace"
                raise TracingError(msg)
            _snapshot_traced(arg)
            return arg
        _node_id, value = _snapshot_traced(arg)
        return weak_scalar_runtime_value(arg, value)
    return np.asarray(arg)


def _get_node(
    arg: object,
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
) -> _Operand:
    """Return an SSA parent or retain a concrete operand as a literal."""
    snapshot = getattr(arg, "_advect_snapshot", None)
    if isinstance(arg, traced_type) or callable(snapshot):
        owner = cast("Any", arg).recorder
        if owner is graph:
            node_id, _value = _snapshot_traced(arg)
            return node_id
        if _is_recorder_in_active_trace_stack(owner):
            _snapshot_traced(arg)
            return _LiteralOperand(arg)
        msg = "Cannot record an array operand from an unrelated trace"
        raise TracingError(msg)
    return _LiteralOperand(np.asarray(arg))


def _result_shape_and_dtype(value: object) -> tuple[tuple[int, ...], object]:
    """Get shape/dtype metadata without forcing traced wrappers to ndarray."""
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        shape = tuple(int(dim) for dim in cast("Any", value).shape)
        dtype = cast("Any", value).dtype
        return shape, dtype
    arr = np.asarray(value)
    return tuple(int(dim) for dim in arr.shape), arr.dtype


_WHERE_NARGS = 3
_INTERP_NARGS = 3
_MULTI_INPUT_MAX_ARGS = 2
_PYTHON_SCALARS = (bool, int, float, complex)


def _is_traced_operand(value: object, traced_type: type[TracedArrayLike]) -> bool:
    return isinstance(value, traced_type) or callable(getattr(value, "_advect_snapshot", None))


def _clip_static_bound_to_attr(
    bound: object | None,
) -> bool | int | float | dict[str, Any] | None:
    # Only exact Python scalars are weak under NEP 50; a NumPy scalar or 0-d
    # array bound keeps its dtype so it promotes the result as in NumPy.
    if bound is None or type(bound) in {bool, int, float}:
        return cast("bool | int | float | None", bound)

    arr = np.asarray(bound)
    if arr.ndim == 0 and not isinstance(arr.item(), (bool, int, float)):
        msg = (
            f"numpy.clip only supports scalar numeric or array bounds during tracing "
            f"(got scalar {type(arr.item()).__name__})"
        )
        raise TracingError(msg)
    return encode_static_array_attr(arr)


def _make_reduction_handler(np_func: Callable[..., Any]) -> Callable[..., tuple[Any, int]]:
    """Create the handler for one NumPy reduction, variance, or cumulative scan."""
    name = np_func.__name__
    op = f"numpy.{name}"
    cumulative = name.startswith("cum")
    variance = name.endswith(("var", "std"))

    def handler(
        graph: DynamicTape,
        traced_type: type[TracedArrayLike],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Any, int]:
        # NumPy's dispatcher already rejected unknown and repeated arguments, and
        # the runtime moved out= and every later positional argument into kwargs.
        values = dict(zip(("axis", "dtype"), args[1:], strict=False)) | kwargs
        source = args[0]
        lowered = lower_controlled_reduction(name, source, values, error=TracingError)
        if lowered is not None:
            return _finish(lowered, traced_type=traced_type)

        axis = values.get("axis")
        call_kwargs: dict[str, Any] = {"axis": axis}
        attrs: dict[str, Any] = {}
        if cumulative:
            if axis is not None:
                attrs["axis"] = int(axis)
        else:
            call_kwargs["keepdims"] = attrs["keepdims"] = bool(values.get("keepdims", False))
            if axis is not None:
                attrs["axis"] = axis if isinstance(axis, tuple) else (axis,)
        dtype = values.get("dtype")
        if dtype is not None:
            call_kwargs["dtype"] = dtype
            attrs["dtype"] = str(np.dtype(dtype))
        if variance:
            call_kwargs["ddof"] = ddof = values.get("correction", values.get("ddof", 0))
            if ddof:
                attrs["ddof"] = float(ddof)
        initial = values.get("initial")
        if initial is not None:
            call_kwargs["initial"] = initial
            initial = np.asarray(initial)
            if initial.ndim != 0:
                raise ValueError(NON_SCALAR_INITIAL)
            attrs["initial"] = initial.item()

        result = np_func(_get_value(source, traced_type), **call_kwargs)
        return _emit(graph, traced_type, op, (source,), result, attrs)

    return handler


def _make_unary_shape_handler(
    np_func: Callable[..., Any],
    op_name: str,
    param_names: tuple[str, ...],
    attr_transform: Callable[[Any], Any] | None = None,
) -> Callable[..., tuple[Any, int]]:
    """Create a handler for unary shape operations (reshape, transpose, etc.)."""

    def handler(
        graph: DynamicTape,
        traced_type: type[TracedArrayLike],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Any, int]:
        if not args or len(args) > len(param_names) + 1:
            msg = (
                f"{op_name} expects one array and at most {len(param_names)} "
                "positional metadata arguments during tracing"
            )
            raise TracingError(msg)
        unsupported = set(kwargs) - set(param_names)
        if unsupported:
            msg = f"{op_name} kwargs not supported during tracing: {sorted(unsupported)}"
            raise TracingError(msg)
        a = args[0]
        # NumPy's dispatcher already rejected a parameter passed twice.
        params: dict[str, Any] = dict(zip(param_names, args[1:], strict=False))
        params.update((name, kwargs[name]) for name in param_names if name in kwargs)

        result = np_func(_get_value(a, traced_type), **params)

        attrs: dict[str, Any] = {}
        for name, val in params.items():
            if val is not None:
                attrs[name] = attr_transform(val) if attr_transform else val

        return _emit(graph, traced_type, op_name, (a,), result, attrs)

    return handler


def _make_binary_handler(
    np_func: Callable[..., Any], op_name: str
) -> Callable[..., tuple[Any, int]]:
    """Create a handler for binary operations (dot, etc.)."""

    def handler(
        graph: DynamicTape,
        traced_type: type[TracedArrayLike],
        args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> tuple[Any, int]:
        a, b = args[0], args[1]
        result = np_func(_get_value(a, traced_type), _get_value(b, traced_type))
        return _emit(graph, traced_type, op_name, (a, b), result)

    return handler


def _make_multi_input_handler(
    np_func: Callable[..., Any],
    op_name: str,
    *,
    weak_scalars: bool,
) -> Callable[..., tuple[Any, int]]:
    """Create a handler for multi-input operations (concatenate, stack).

    With ``weak_scalars``, Python scalar inputs promote weakly instead of being
    coerced to arrays first.
    """

    def handler(
        graph: DynamicTape,
        traced_type: type[TracedArrayLike],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Any, int]:
        arrays = args[0]
        if not isinstance(arrays, (tuple, list)) or not arrays:
            msg = f"{op_name} requires a non-empty tuple or list during tracing"
            raise TracingError(msg)
        axis = args[1] if len(args) == _MULTI_INPUT_MAX_ARGS else kwargs.get("axis", 0)
        call_kwargs: dict[str, Any] = {"axis": axis}
        attrs: dict[str, Any] = {"axis": axis}
        if kwargs.get("dtype") is not None:
            dtype = kwargs["dtype"]
            call_kwargs["dtype"] = dtype
            attrs["dtype"] = str(np.dtype(dtype))
        if "casting" in kwargs:
            casting = str(kwargs["casting"])
            call_kwargs["casting"] = casting
            attrs["casting"] = casting

        get_value = _get_weak_value if weak_scalars else _get_value
        result = np_func([get_value(a, traced_type) for a in arrays], **call_kwargs)
        return _emit(graph, traced_type, op_name, arrays, result, attrs, weak_scalars=weak_scalars)

    return handler


def _make_like_handler(
    np_func: Callable[..., Any],
    op_name: str,
) -> Callable[..., tuple[Any, int]]:
    """Create a handler for *_like operations (zeros_like, ones_like, full_like)."""

    def handler(
        graph: DynamicTape,
        traced_type: type[TracedArrayLike],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Any, int]:
        positional_names = ("dtype", "order", "subok", "shape")
        values = dict(zip(positional_names, args[1:], strict=False)) | kwargs

        a = args[0]
        call_kwargs: dict[str, Any] = {}
        attrs: dict[str, Any] = {}
        dtype = values.get("dtype")
        if dtype is not None:
            call_kwargs["dtype"] = dtype
            attrs["dtype"] = str(np.dtype(dtype))
        if "order" in values:
            order = str(values["order"])
            call_kwargs["order"] = order
            attrs["order"] = order
        if "subok" in values:
            subok = bool(values["subok"])
            call_kwargs["subok"] = subok
            attrs["subok"] = subok
        shape = values.get("shape")
        if shape is not None:
            try:
                normalized_shape = tuple(int(size) for size in shape)
            except TypeError:
                normalized_shape = (int(shape),)
            call_kwargs["shape"] = normalized_shape
            attrs["shape"] = normalized_shape
        if values.get("device") is not None:
            device = values["device"]
            call_kwargs["device"] = device
            attrs["device"] = device

        result = np_func(_get_value(a, traced_type), **call_kwargs)
        return _emit(graph, traced_type, op_name, (a,), result, attrs)

    return handler


def _make_atleast_handler(
    np_func: Callable[..., Any], op_name: str, *, target_ndim: int
) -> Callable[..., tuple[Any, int]]:
    """Create handlers for np.atleast_1d/2d/3d."""

    def handler(
        graph: DynamicTape,
        traced_type: type[TracedArrayLike],
        args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> tuple[Any, int]:
        if len(args) != 1:
            msg = (
                f"{op_name} supports one input during tracing. "
                "Call it separately for each input so every result keeps its alias provenance."
            )
            raise TracingError(msg)

        result = np_func(_get_value(args[0], traced_type))
        return _emit(graph, traced_type, op_name, args, result, {"target_ndim": target_ndim})

    return frontend_lowering(op_name)(handler)


def _make_clip_handler(op_name: str) -> Callable[..., tuple[Any, int]]:
    """Create a handler for ``clip`` with traced/static bounds."""

    def handler(
        graph: DynamicTape,
        traced_type: type[TracedArrayLike],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Any, int]:
        unsupported = set(kwargs) - CLIP_KEYWORDS
        if unsupported:
            msg = f"numpy.clip kwargs are not supported during tracing: {sorted(unsupported)}"
            raise TracingError(msg)

        a, a_min_raw, a_max_raw = bind_clip_bounds(args, kwargs)

        min_is_input = isinstance(a_min_raw, traced_type)
        max_is_input = isinstance(a_max_raw, traced_type)

        a_min_attr = None if min_is_input else _clip_static_bound_to_attr(a_min_raw)
        a_max_attr = None if max_is_input else _clip_static_bound_to_attr(a_max_raw)
        a_min_eval = (
            _get_value(a_min_raw, traced_type)
            if min_is_input
            else decode_static_array_attr(a_min_attr)
        )
        a_max_eval = (
            _get_value(a_max_raw, traced_type)
            if max_is_input
            else decode_static_array_attr(a_max_attr)
        )

        result = np.clip(
            _get_value(a, traced_type),
            cast("Any", a_min_eval),
            cast("Any", a_max_eval),
        )

        operands = [a]
        if min_is_input:
            operands.append(a_min_raw)
        if max_is_input:
            operands.append(a_max_raw)
        attrs = {
            "a_min": a_min_attr,
            "a_max": a_max_attr,
            "_advect_clip_min_is_input": min_is_input,
            "_advect_clip_max_is_input": max_is_input,
        }
        return _emit(graph, traced_type, op_name, operands, result, attrs)

    return handler


def _make_where_handler(op_name: str) -> Callable[..., tuple[Any, int]]:
    """Create a handler for 3-argument ``where(condition, x, y)``."""

    def handler(
        graph: DynamicTape,
        traced_type: type[TracedArrayLike],
        args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> tuple[Any, int]:
        if len(args) != _WHERE_NARGS:
            msg = (
                "numpy.where is only supported during tracing in its "
                "3-argument form "
                "(where(condition, x, y))"
            )
            raise TracingError(msg)

        result = np.where(*(_get_weak_value(item, traced_type) for item in args))
        return _emit(graph, traced_type, op_name, args, result, weak_scalars=True)

    return handler


def _make_interp_handler(op_name: str) -> Callable[..., tuple[Any, int]]:
    """Create a handler for ``interp``."""

    def handler(
        graph: DynamicTape,
        traced_type: type[TracedArrayLike],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Any, int]:
        x, xp, fp = args[:_INTERP_NARGS]
        values = dict(zip(("left", "right", "period"), args[_INTERP_NARGS:], strict=False))
        values |= kwargs
        left = values.get("left")
        right = values.get("right")
        period = values.get("period")
        if _is_traced_operand(period, traced_type):
            msg = "numpy.interp period= must be static because it controls periodic sorting"
            raise TracingError(msg)
        left_is_input = _is_traced_operand(left, traced_type)
        right_is_input = _is_traced_operand(right, traced_type)
        base_left = None if left_is_input else left
        base_right = None if right_is_input else right

        result = np.interp(
            _get_value(x, traced_type),
            _get_value(xp, traced_type),
            _get_value(fp, traced_type),
            left=base_left,
            right=base_right,
            period=period,
        )

        attrs: dict[str, Any] = {}
        if base_left is not None:
            attrs["left"] = base_left
        if base_right is not None:
            attrs["right"] = base_right
        if period is not None:
            attrs["period"] = period

        _, node_id = _emit(graph, traced_type, op_name, (x, xp, fp), result, attrs)
        if period is not None or not (left_is_input or right_is_input):
            return result, node_id

        traced_ctor = cast("Callable[..., TracedArrayLike]", traced_type)
        result_tracer = traced_ctor(
            value=result,
            node_id=node_id,
            recorder=graph,
        )
        if left_is_input:
            result_tracer = np.where(x < xp[0], left, result_tracer)
        if right_is_input:
            result_tracer = np.where(x > xp[-1], right, result_tracer)
        result_node_id, result_value = _snapshot_traced(result_tracer)
        return result_value, result_node_id

    return handler
