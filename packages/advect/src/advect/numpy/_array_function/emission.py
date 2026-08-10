"""Common helpers for NumPy array-function dispatch.

This module contains helper utilities and small handler factories used by the
NumPy ``__array_function__`` dispatch layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as _numpy  # noqa: ICN001 - concrete namespace with dynamic protocol operands

from advect.core._array_protocol_helpers import (
    literals_are_weak,
    weak_scalar_runtime_value,
)
from advect.core._context import (
    _get_operation_recorder,
    _is_recorder_in_active_trace_stack,
    get_source_location,
)
from advect.core._errors import TracingError
from advect.core._protocols import _snapshot_traced
from advect.numpy._op_bindings import canonicalize_numpy_op, frontend_lowering
from advect.numpy._static_attr_arrays import decode_static_array_attr, encode_static_array_attr

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

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


def _canonical_op(op_name: str) -> str:
    return canonicalize_numpy_op(op_name)


def _backend_qualified(name: str) -> str:
    return f"numpy.{name}"


def _with_backend_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    out = dict(attrs)
    out["_advect_backend"] = "numpy"
    return out


def _add_backend_node(  # noqa: PLR0913 - one operation owns complete metadata
    *,
    graph: DynamicTape,
    op: str,
    inputs: tuple[_Operand, ...],
    value: object,
    attrs: dict[str, Any],
    shape: tuple[int, ...],
    dtype: object,
    num_outputs: int = 1,
    output_shapes: tuple[tuple[int, ...], ...] | None = None,
    output_dtypes: tuple[object, ...] | None = None,
) -> int:
    native_attrs = _with_backend_attrs(attrs)
    _ = (num_outputs, output_shapes, output_dtypes)
    parents = tuple(item for item in inputs if isinstance(item, int))
    literals = tuple(item.value for item in inputs if isinstance(item, _LiteralOperand))
    source_location = get_source_location()
    if not literals:
        return graph.record_operation(
            op,
            parents,
            value,
            native_attrs,
            shape,
            dtype,
            source_location=source_location,
        )
    parent_positions = tuple(
        position for position, item in enumerate(inputs) if isinstance(item, int)
    )
    return graph.record_operation_with_literals(
        op,
        parents,
        parent_positions,
        literals,
        value,
        native_attrs,
        shape,
        dtype,
        source_location=source_location,
        literal_weak=literals_are_weak(list(literals)),
    )


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
            return int(node_id)
        if _is_recorder_in_active_trace_stack(owner):
            _snapshot_traced(arg)
            return _LiteralOperand(arg)
        msg = "Cannot record an array operand from an unrelated trace"
        raise TracingError(msg)
    return _LiteralOperand(np.asarray(arg))


def _get_values(
    args: Sequence[Any],
    traced_type: type[TracedArrayLike],
) -> list[object]:
    """Extract values from a sequence of arrays."""
    return [_get_value(a, traced_type) for a in args]


def _get_nodes(
    args: Sequence[Any],
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
) -> tuple[_Operand, ...]:
    """Get node IDs from a sequence of arrays."""
    return tuple(_get_node(a, graph, traced_type) for a in args)


def _result_shape_and_dtype(value: object) -> tuple[tuple[int, ...], object]:
    """Get shape/dtype metadata without forcing traced wrappers to ndarray."""
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        shape = tuple(int(dim) for dim in cast("Any", value).shape)
        dtype = cast("Any", value).dtype
        return shape, dtype
    snapshot = getattr(value, "_advect_snapshot", None)
    if callable(snapshot):
        _node_id, inner = _snapshot_traced(value)
        return _result_shape_and_dtype(inner)
    arr = np.asarray(value)
    return tuple(int(dim) for dim in arr.shape), arr.dtype


_SUPPORTED_REDUCTION_KWARGS = frozenset({"axis", "dtype", "initial", "keepdims", "out", "where"})
_SUPPORTED_AXIS_KEEPDIMS_KWARGS = frozenset({"axis", "initial", "keepdims", "out", "where"})
_SUPPORTED_VARIANCE_KWARGS = frozenset(
    {"axis", "correction", "ddof", "dtype", "keepdims", "mean", "out", "where"}
)
_SUPPORTED_CUMULATIVE_KWARGS = frozenset({"axis", "dtype"})
_SUPPORTED_ARG_REDUCTION_KWARGS = frozenset({"axis", "keepdims", "out"})
_SUPPORTED_INTERP_KWARGS = frozenset({"left", "right", "period"})
_SUPPORTED_CLIP_KWARGS = frozenset({"min", "max", "a_min", "a_max", "out"})
_WHERE_NARGS = 3
_INTERP_NARGS = 3
_CLIP_NARGS = 3
_MULTI_INPUT_MAX_ARGS = 2


def _is_traced_operand(value: object, traced_type: type[TracedArrayLike]) -> bool:
    return isinstance(value, traced_type) or callable(getattr(value, "_advect_snapshot", None))


def _finish_composite_reduction(
    result: object,
    *,
    traced_type: type[TracedArrayLike],
) -> tuple[object, int]:
    if not _is_traced_operand(result, traced_type):
        msg = "Composite reduction lowering did not produce a traced array"
        raise TracingError(msg)
    node_id, value = _snapshot_traced(result)
    return value, int(node_id)


def _reduction_accumulator_dtype(
    source: object,
    *,
    requested: object | None,
    real_result: bool,
) -> object:
    """Match NumPy's mean/variance accumulator dtype without a concrete reduction."""
    if requested is not None:
        return requested
    dtype = getattr(source, "dtype", None)
    if dtype is None:
        dtype = _result_shape_and_dtype(source)[1]
    name = str(dtype)
    if name.startswith(("bool", "int", "uint")):
        return "float64"
    if real_result and name.startswith("complex"):
        return "float32" if "64" in name else "float64"
    return dtype


def _lower_controlled_reduction(
    np_func: Callable[..., Any],
    *,
    function_name: str,
    source: object,
    values: dict[str, Any],
    traced_type: type[TracedArrayLike],
) -> tuple[object, int]:
    axis = values.get("axis")
    keepdims = bool(values.get("keepdims", False))
    dtype = values.get("dtype")
    call_kwargs: dict[str, Any] = {"axis": axis, "keepdims": keepdims}
    if dtype is not None:
        call_kwargs["dtype"] = dtype

    where = values.get("where")
    if where is None:
        selected = source
    else:
        if function_name in {"mean", "nanmean"}:
            valid = np.broadcast_to(where, cast("Any", source).shape)
            if function_name == "nanmean":
                valid = np.where(
                    np.isnan(source),
                    np.zeros_like(valid, dtype=bool),
                    valid,
                )
            numerator = np.sum(
                np.where(valid, source, np.zeros_like(source)),
                axis=axis,
                dtype=dtype,
                keepdims=keepdims,
            )
            count = np.sum(valid, axis=axis, keepdims=keepdims)
            result = numerator / count
            target_dtype = _reduction_accumulator_dtype(
                source,
                requested=dtype,
                real_result=False,
            )
            return _finish_composite_reduction(
                result.astype(target_dtype),
                traced_type=traced_type,
            )
        identity = (
            np.ones_like(source) if function_name in {"prod", "nanprod"} else np.zeros_like(source)
        )
        selected = np.where(where, source, identity)

    initial = values.get("initial")
    dynamic_initial = _is_traced_operand(initial, traced_type)
    if "initial" in values and not dynamic_initial:
        call_kwargs["initial"] = initial
    result = np_func(selected, **call_kwargs)
    if dynamic_initial:
        result = result * initial if function_name in {"prod", "nanprod"} else result + initial
    return _finish_composite_reduction(result, traced_type=traced_type)


def _lower_controlled_extrema(
    np_func: Callable[..., Any],
    *,
    function_name: str,
    source: object,
    values: dict[str, Any],
    traced_type: type[TracedArrayLike],
) -> tuple[object, int]:
    axis = values.get("axis")
    keepdims = bool(values.get("keepdims", False))
    call_kwargs: dict[str, Any] = {"axis": axis, "keepdims": keepdims}
    initial = values.get("initial")
    where = values.get("where")
    if where is not None:
        if initial is None:
            msg = f"numpy.{function_name} with where= requires initial= during tracing"
            raise TracingError(msg)
        base = np_func(np.where(where, source, initial), **call_kwargs)
    elif function_name in {"nanmin", "nanmax"}:
        base = np_func(np.where(np.isnan(source), initial, source), **call_kwargs)
    else:
        base = np_func(source, **call_kwargs)
    combine = np.maximum if function_name in {"max", "amax", "nanmax"} else np.minimum
    result = combine(base, initial)
    return _finish_composite_reduction(result, traced_type=traced_type)


def _lower_variance_controls(
    *,
    function_name: str,
    source: object,
    values: dict[str, Any],
    traced_type: type[TracedArrayLike],
) -> tuple[object, int]:
    array: Any = source
    dtype = values.get("dtype")
    if dtype is not None:
        array = array.astype(dtype)
    axis = values.get("axis")
    keepdims = bool(values.get("keepdims", False))
    where = values.get("where")
    valid = (
        np.ones_like(array, dtype=bool) if where is None else np.broadcast_to(where, array.shape)
    )
    if function_name in {"nanvar", "nanstd"}:
        valid = np.where(
            np.isnan(array),
            np.zeros_like(valid, dtype=bool),
            valid,
        )

    count_with_dims = np.astype(
        np.sum(valid, axis=axis, keepdims=True),
        array.dtype,
    )
    supplied_mean = values.get("mean")
    if supplied_mean is None:
        supplied_mean = (
            np.sum(
                np.where(valid, array, np.zeros_like(array)),
                axis=axis,
                keepdims=True,
            )
            / count_with_dims
        )
    centered = np.where(valid, array, supplied_mean) - supplied_mean
    squared = np.real(np.conjugate(centered) * centered)
    numerator = np.sum(
        np.where(valid, squared, np.zeros_like(squared)),
        axis=axis,
        keepdims=keepdims,
    )
    count = np.astype(
        np.sum(valid, axis=axis, keepdims=keepdims),
        squared.dtype,
    )
    correction_value = values.get("correction", values.get("ddof", 0))
    correction = (
        np.astype(correction_value, squared.dtype)
        if _is_traced_operand(correction_value, traced_type)
        else np.asarray(correction_value, dtype=squared.dtype)
    )
    result = numerator / (count - correction)
    if function_name in {"std", "nanstd"}:
        result = np.sqrt(result)
    result = result.astype(
        _reduction_accumulator_dtype(
            source,
            requested=dtype,
            real_result=True,
        )
    )
    return _finish_composite_reduction(result, traced_type=traced_type)


def _clip_static_bound_to_attr(
    bound: object | None,
) -> bool | int | float | dict[str, Any] | None:
    if bound is None:
        return None

    arr = np.asarray(bound)
    if arr.ndim != 0:
        return cast("dict[str, Any]", encode_static_array_attr(arr))

    py_scalar = arr.item()
    if isinstance(py_scalar, np.generic):
        py_scalar = py_scalar.item()
    if isinstance(py_scalar, (bool, int, float)):
        return py_scalar
    msg = (
        f"{_backend_qualified('clip')} only supports scalar numeric or array bounds during tracing "
        f"(got scalar {type(py_scalar).__name__})"
    )
    raise TracingError(msg)


def _parse_clip_bounds(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, object | None, object | None]:
    if len(args) not in (1, _CLIP_NARGS):
        msg = (
            f"{_backend_qualified('clip')} is only supported during tracing as "
            "clip(a, a_min, a_max) "
            "or clip(a, *, min=..., max=...)"
        )
        raise TracingError(msg)

    if len(args) == _CLIP_NARGS:
        if any(k in kwargs for k in ("min", "max", "a_min", "a_max")):
            msg = (
                f"{_backend_qualified('clip')} bounds must be provided either "
                "positionally or via keywords, not both"
            )
            raise TracingError(msg)
        return args[0], args[1], args[2]

    a = args[0]
    has_minmax = ("min" in kwargs) or ("max" in kwargs)
    has_aminamax = ("a_min" in kwargs) or ("a_max" in kwargs)
    if has_minmax and has_aminamax:
        msg = (
            f"{_backend_qualified('clip')} does not support mixing min/max "
            "with a_min/a_max during tracing"
        )
        raise TracingError(msg)

    if has_minmax:
        return a, kwargs.get("min"), kwargs.get("max")

    if has_aminamax:
        if ("a_min" in kwargs) != ("a_max" in kwargs):
            msg = (
                f"{_backend_qualified('clip')} only supports a_min/a_max during tracing "
                "when both are provided "
                "(use min/max for single-sided bounds)"
            )
            raise TracingError(msg)
        return a, kwargs.get("a_min"), kwargs.get("a_max")

    msg = f"{_backend_qualified('clip')} requires bounds during tracing"
    raise TracingError(msg)


def _make_reduction_handler(
    np_func: Callable[..., Any], op_name: str
) -> Callable[..., tuple[Any, int]]:
    """Create a handler for reduction operations (sum, mean, prod, etc.)."""

    def handler(
        graph: DynamicTape,
        traced_type: type[TracedArrayLike],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Any, int]:
        function_name = str(getattr(np_func, "__name__", "reduction"))
        has_initial = function_name not in {"mean", "nanmean"}
        positional_names = (
            ("axis", "dtype", "out", "keepdims", "initial", "where")
            if has_initial
            else ("axis", "dtype", "out", "keepdims")
        )
        if not args or len(args) > len(positional_names) + 1:
            msg = f"{op_name} received too many positional arguments during tracing"
            raise TracingError(msg)
        unsupported = set(kwargs.keys()) - _SUPPORTED_REDUCTION_KWARGS
        if unsupported:
            msg = (
                f"Reduction kwargs not yet supported during tracing: {sorted(unsupported)}. "
                f"Supported kwargs are: {sorted(_SUPPORTED_REDUCTION_KWARGS)}"
            )
            raise TracingError(msg)

        values = dict(kwargs)
        for name, value in zip(positional_names, args[1:], strict=False):
            if name in values:
                msg = f"{op_name} received {name} twice"
                raise TracingError(msg)
            values[name] = value
        if values.get("out") is not None:
            msg = f"{op_name} out= is not supported by this handler during tracing"
            raise TracingError(msg)

        a = args[0]
        initial = values.get("initial")
        if "where" in values or _is_traced_operand(initial, traced_type):
            return _lower_controlled_reduction(
                np_func,
                function_name=function_name,
                source=a,
                values=values,
                traced_type=traced_type,
            )
        axis = values.get("axis")
        keepdims = bool(values.get("keepdims", False))
        dtype = values.get("dtype")
        call_kwargs: dict[str, Any] = {"axis": axis, "keepdims": keepdims}
        if dtype is not None:
            call_kwargs["dtype"] = dtype
        if has_initial and "initial" in values:
            call_kwargs["initial"] = values["initial"]

        result = np_func(_get_value(a, traced_type), **call_kwargs)

        attrs: dict[str, Any] = {"keepdims": keepdims}
        if axis is not None:
            attrs["axis"] = (axis,) if not isinstance(axis, tuple) else axis
        if dtype is not None:
            attrs["dtype"] = str(np.dtype(dtype))
        if has_initial and "initial" in values:
            initial = np.asarray(values["initial"])
            if initial.ndim != 0:
                msg = f"{op_name} initial= must be scalar during tracing"
                raise TracingError(msg)
            attrs["initial"] = initial.item()

        result_shape, result_dtype = _result_shape_and_dtype(result)
        node_id = _add_backend_node(
            graph=graph,
            op=_canonical_op(op_name),
            inputs=(_get_node(a, graph, traced_type),),
            value=result,
            attrs=attrs,
            shape=result_shape,
            dtype=result_dtype,
        )
        return result, node_id

    return handler


def _make_axis_keepdims_reduction_handler(
    np_func: Callable[..., Any], op_name: str
) -> Callable[..., tuple[Any, int]]:
    """Create a handler for reduction ops that do not accept dtype (nanmin/nanmax)."""

    def handler(
        graph: DynamicTape,
        traced_type: type[TracedArrayLike],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Any, int]:
        positional_names = ("axis", "out", "keepdims", "initial", "where")
        if not args or len(args) > len(positional_names) + 1:
            msg = f"{op_name} received too many positional arguments during tracing"
            raise TracingError(msg)
        unsupported = set(kwargs.keys()) - _SUPPORTED_AXIS_KEEPDIMS_KWARGS
        if unsupported:
            msg = (
                f"Reduction kwargs not yet supported during tracing: {sorted(unsupported)}. "
                f"Supported kwargs are: {sorted(_SUPPORTED_AXIS_KEEPDIMS_KWARGS)}"
            )
            raise TracingError(msg)

        values = dict(kwargs)
        for name, value in zip(positional_names, args[1:], strict=False):
            if name in values:
                msg = f"{op_name} received {name} twice"
                raise TracingError(msg)
            values[name] = value
        if values.get("out") is not None:
            msg = f"{op_name} out= is not supported by this handler during tracing"
            raise TracingError(msg)

        a = args[0]
        function_name = str(getattr(np_func, "__name__", "reduction"))
        initial = values.get("initial")
        if "where" in values or _is_traced_operand(initial, traced_type):
            return _lower_controlled_extrema(
                np_func,
                function_name=function_name,
                source=a,
                values=values,
                traced_type=traced_type,
            )
        axis = values.get("axis")
        keepdims = bool(values.get("keepdims", False))
        call_kwargs: dict[str, Any] = {"axis": axis, "keepdims": keepdims}
        if "initial" in values:
            call_kwargs["initial"] = values["initial"]
        result = np_func(_get_value(a, traced_type), **call_kwargs)

        attrs: dict[str, Any] = {"keepdims": keepdims}
        if axis is not None:
            attrs["axis"] = (axis,) if not isinstance(axis, tuple) else axis
        if "initial" in values:
            initial = np.asarray(values["initial"])
            if initial.ndim != 0:
                msg = f"{op_name} initial= must be scalar during tracing"
                raise TracingError(msg)
            attrs["initial"] = initial.item()

        result_shape, result_dtype = _result_shape_and_dtype(result)
        node_id = _add_backend_node(
            graph=graph,
            op=_canonical_op(op_name),
            inputs=(_get_node(a, graph, traced_type),),
            value=result,
            attrs=attrs,
            shape=result_shape,
            dtype=result_dtype,
        )
        return result, node_id

    return handler


def _make_variance_handler(
    np_func: Callable[..., Any], op_name: str
) -> Callable[..., tuple[Any, int]]:
    """Create a handler for variance-style reductions (nanvar/nanstd)."""

    def handler(
        graph: DynamicTape,
        traced_type: type[TracedArrayLike],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Any, int]:
        positional_names = ("axis", "dtype", "out", "ddof", "keepdims")
        if not args or len(args) > len(positional_names) + 1:
            msg = f"{op_name} received too many positional arguments during tracing"
            raise TracingError(msg)
        unsupported = set(kwargs.keys()) - _SUPPORTED_VARIANCE_KWARGS
        if unsupported:
            msg = (
                f"Reduction kwargs not yet supported during tracing: {sorted(unsupported)}. "
                f"Supported kwargs are: {sorted(_SUPPORTED_VARIANCE_KWARGS)}"
            )
            raise TracingError(msg)

        values = dict(kwargs)
        for name, value in zip(positional_names, args[1:], strict=False):
            if name in values:
                msg = f"{op_name} received {name} twice"
                raise TracingError(msg)
            values[name] = value
        if values.get("out") is not None:
            msg = f"{op_name} out= is not supported by this handler during tracing"
            raise TracingError(msg)
        if "correction" in values and "ddof" in values:
            msg = f"{op_name} accepts only one of correction= and ddof="
            raise TracingError(msg)

        a = args[0]
        function_name = str(getattr(np_func, "__name__", "variance"))
        correction = values.get("correction", values.get("ddof", 0))
        if (
            "where" in values
            or values.get("mean") is not None
            or _is_traced_operand(correction, traced_type)
        ):
            return _lower_variance_controls(
                function_name=function_name,
                source=a,
                values=values,
                traced_type=traced_type,
            )
        axis = values.get("axis")
        keepdims = bool(values.get("keepdims", False))
        dtype = values.get("dtype")
        ddof = values.get("correction", values.get("ddof", 0))
        call_kwargs: dict[str, Any] = {"axis": axis, "keepdims": keepdims, "ddof": ddof}
        if dtype is not None:
            call_kwargs["dtype"] = dtype

        result = np_func(_get_value(a, traced_type), **call_kwargs)

        attrs: dict[str, Any] = {"keepdims": keepdims}
        if axis is not None:
            attrs["axis"] = (axis,) if not isinstance(axis, tuple) else axis
        if dtype is not None:
            attrs["dtype"] = str(np.dtype(dtype))
        if ddof:
            attrs["ddof"] = float(ddof)

        result_shape, result_dtype = _result_shape_and_dtype(result)
        node_id = _add_backend_node(
            graph=graph,
            op=_canonical_op(op_name),
            inputs=(_get_node(a, graph, traced_type),),
            value=result,
            attrs=attrs,
            shape=result_shape,
            dtype=result_dtype,
        )
        return result, node_id

    return handler


def _make_cumulative_handler(
    np_func: Callable[..., Any], op_name: str
) -> Callable[..., tuple[Any, int]]:
    """Create a handler for cumulative ops (cumsum/cumprod)."""

    def handler(
        graph: DynamicTape,
        traced_type: type[TracedArrayLike],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Any, int]:
        positional_names = ("axis", "dtype", "out")
        unsupported = set(kwargs.keys()) - {*_SUPPORTED_CUMULATIVE_KWARGS, "out"}
        if unsupported:
            msg = (
                f"Cumulative kwargs not yet supported during tracing: {sorted(unsupported)}. "
                f"Supported kwargs are: {sorted(_SUPPORTED_CUMULATIVE_KWARGS)}"
            )
            raise TracingError(msg)

        if not args or len(args) > len(positional_names) + 1:
            msg = f"{op_name} expects an array plus optional axis, dtype, and out"
            raise TracingError(msg)
        values = dict(kwargs)
        for name, value in zip(positional_names, args[1:], strict=False):
            if name in values:
                msg = f"{op_name} received {name} twice"
                raise TracingError(msg)
            values[name] = value
        if values.get("out") is not None:
            msg = f"{op_name} positional out= is not supported; pass out= by keyword"
            raise TracingError(msg)

        a = args[0]
        axis = values.get("axis")
        dtype = values.get("dtype")

        call_kwargs: dict[str, Any] = {"axis": axis}
        if dtype is not None:
            call_kwargs["dtype"] = dtype

        result = np_func(_get_value(a, traced_type), **call_kwargs)

        attrs: dict[str, Any] = {}
        if axis is not None:
            attrs["axis"] = int(axis)
        if dtype is not None:
            attrs["dtype"] = str(np.dtype(dtype))

        result_shape, result_dtype = _result_shape_and_dtype(result)
        node_id = _add_backend_node(
            graph=graph,
            op=_canonical_op(op_name),
            inputs=(_get_node(a, graph, traced_type),),
            value=result,
            attrs=attrs,
            shape=result_shape,
            dtype=result_dtype,
        )
        return result, node_id

    return handler


def _make_arg_reduction_handler(
    np_func: Callable[..., Any], op_name: str
) -> Callable[..., tuple[Any, int]]:
    """Create a handler for argmin/argmax-style reductions."""

    def handler(
        graph: DynamicTape,
        traced_type: type[TracedArrayLike],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Any, int]:
        positional_names = ("axis", "out")
        unsupported = set(kwargs.keys()) - _SUPPORTED_ARG_REDUCTION_KWARGS
        if unsupported:
            msg = (
                f"Arg reduction kwargs not yet supported during tracing: {sorted(unsupported)}. "
                f"Supported kwargs are: {sorted(_SUPPORTED_ARG_REDUCTION_KWARGS)}"
            )
            raise TracingError(msg)

        if not args or len(args) > len(positional_names) + 1:
            msg = f"{op_name} expects an array plus optional axis and out"
            raise TracingError(msg)
        values = dict(kwargs)
        for name, value in zip(positional_names, args[1:], strict=False):
            if name in values:
                msg = f"{op_name} received {name} twice"
                raise TracingError(msg)
            values[name] = value

        out = values.get("out")
        if out is not None:
            msg = f"{op_name} out= is not supported during tracing"
            raise TracingError(msg)

        a = args[0]
        axis = values.get("axis")
        keepdims = values.get("keepdims", False)

        result = np_func(_get_value(a, traced_type), axis=axis, keepdims=keepdims)
        result_shape, result_dtype = _result_shape_and_dtype(result)

        attrs: dict[str, Any] = {"keepdims": bool(keepdims)}
        if axis is not None:
            attrs["axis"] = int(axis)

        node_id = _add_backend_node(
            graph=graph,
            op=_canonical_op(op_name),
            inputs=(_get_node(a, graph, traced_type),),
            value=result,
            attrs=attrs,
            shape=result_shape,
            dtype=result_dtype,
        )
        return result, node_id

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
        params: dict[str, Any] = {}
        for i, name in enumerate(param_names):
            if len(args) > i + 1:
                if name in kwargs:
                    msg = f"{op_name} received {name} twice"
                    raise TracingError(msg)
                params[name] = args[i + 1]
            elif name in kwargs:
                params[name] = kwargs[name]

        result = np_func(_get_value(a, traced_type), **params)

        attrs: dict[str, Any] = {}
        for name, val in params.items():
            if val is not None:
                attrs[name] = attr_transform(val) if attr_transform else val

        node_id = _add_backend_node(
            graph=graph,
            op=_canonical_op(op_name),
            inputs=(_get_node(a, graph, traced_type),),
            value=result,
            attrs=attrs,
            shape=result.shape,
            dtype=result.dtype,
        )
        return result, node_id

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

        node_id = _add_backend_node(
            graph=graph,
            op=_canonical_op(op_name),
            inputs=(_get_node(a, graph, traced_type), _get_node(b, graph, traced_type)),
            value=result,
            attrs={},
            shape=result.shape,
            dtype=result.dtype,
        )
        return result, node_id

    return handler


def _make_multi_input_handler(
    np_func: Callable[..., Any], op_name: str
) -> Callable[..., tuple[Any, int]]:
    """Create a handler for multi-input operations (concatenate, stack)."""

    def handler(
        graph: DynamicTape,
        traced_type: type[TracedArrayLike],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Any, int]:
        if len(args) not in {1, _MULTI_INPUT_MAX_ARGS}:
            msg = f"{op_name} expects an array sequence and optional axis during tracing"
            raise TracingError(msg)
        unsupported = set(kwargs) - {"axis", "dtype", "casting"}
        if unsupported:
            msg = f"{op_name} kwargs not supported during tracing: {sorted(unsupported)}"
            raise TracingError(msg)
        if len(args) == _MULTI_INPUT_MAX_ARGS and "axis" in kwargs:
            msg = f"{op_name} received axis twice"
            raise TracingError(msg)
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

        result = np_func(_get_values(arrays, traced_type), **call_kwargs)

        node_id = _add_backend_node(
            graph=graph,
            op=_canonical_op(op_name),
            inputs=_get_nodes(arrays, graph, traced_type),
            value=result,
            attrs=attrs,
            shape=result.shape,
            dtype=result.dtype,
        )
        return result, node_id

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
        if not args or len(args) > len(positional_names) + 1:
            msg = f"{op_name} received an invalid positional signature during tracing"
            raise TracingError(msg)
        unsupported = set(kwargs) - {*positional_names, "device"}
        if unsupported:
            msg = f"{op_name} kwargs not supported during tracing: {sorted(unsupported)}"
            raise TracingError(msg)
        values = dict(kwargs)
        for name, value in zip(positional_names, args[1:], strict=False):
            if name in values:
                msg = f"{op_name} received {name} twice"
                raise TracingError(msg)
            values[name] = value

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

        node_id = _add_backend_node(
            graph=graph,
            op=_canonical_op(op_name),
            inputs=(_get_node(a, graph, traced_type),),
            value=result,
            attrs=attrs,
            shape=result.shape,
            dtype=result.dtype,
        )
        return result, node_id

    return handler


def _make_atleast_handler(
    np_func: Callable[..., Any], op_name: str, *, target_ndim: int
) -> Callable[..., tuple[Any, int]]:
    """Create handlers for np.atleast_1d/2d/3d."""

    def handler(
        graph: DynamicTape,
        traced_type: type[TracedArrayLike],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Any, int]:
        if kwargs:
            msg = f"{op_name} kwargs are not supported during tracing: {sorted(kwargs)}"
            raise TracingError(msg)
        if not args:
            msg = f"{op_name} requires at least one input array"
            raise TracingError(msg)
        if len(args) != 1:
            msg = (
                f"{op_name} supports one input during tracing. "
                "Call it separately for each input so every result keeps its alias provenance."
            )
            raise TracingError(msg)

        value = _get_value(args[0], traced_type)
        result = np_func(value)
        result_shape, result_dtype = _result_shape_and_dtype(result)
        node_id = _add_backend_node(
            graph=graph,
            op=_canonical_op(op_name),
            inputs=(_get_node(args[0], graph, traced_type),),
            value=result,
            attrs={"target_ndim": target_ndim},
            shape=result_shape,
            dtype=result_dtype,
        )
        return result, node_id

    return frontend_lowering(op_name)(handler)


def _make_clip_handler(op_name: str) -> Callable[..., tuple[Any, int]]:
    """Create a handler for ``clip`` with traced/static bounds."""

    def handler(
        graph: DynamicTape,
        traced_type: type[TracedArrayLike],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Any, int]:
        unsupported = set(kwargs.keys()) - _SUPPORTED_CLIP_KWARGS
        if unsupported:
            msg = (
                f"{_backend_qualified('clip')} kwargs are not supported during tracing: "
                f"{sorted(unsupported)}"
            )
            raise TracingError(msg)

        out = kwargs.get("out")
        if out is not None:
            msg = f"{_backend_qualified('clip')} out= is not supported during tracing"
            raise TracingError(msg)

        a, a_min_raw, a_max_raw = _parse_clip_bounds(args, kwargs)

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

        node_inputs = [_get_node(a, graph, traced_type)]
        if min_is_input:
            node_inputs.append(_get_node(a_min_raw, graph, traced_type))
        if max_is_input:
            node_inputs.append(_get_node(a_max_raw, graph, traced_type))

        node_id = _add_backend_node(
            graph=graph,
            op=_canonical_op(op_name),
            inputs=tuple(node_inputs),
            value=result,
            attrs={
                "a_min": a_min_attr,
                "a_max": a_max_attr,
                "_advect_clip_min_is_input": min_is_input,
                "_advect_clip_max_is_input": max_is_input,
            },
            shape=result.shape,
            dtype=result.dtype,
        )
        return result, node_id

    return handler


def _make_where_handler(op_name: str) -> Callable[..., tuple[Any, int]]:
    """Create a handler for 3-argument ``where(condition, x, y)``."""

    def handler(
        graph: DynamicTape,
        traced_type: type[TracedArrayLike],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Any, int]:
        if kwargs:
            msg = (
                f"{_backend_qualified('where')} kwargs are not supported during tracing: "
                f"{sorted(kwargs)}"
            )
            raise TracingError(msg)
        if len(args) != _WHERE_NARGS:
            msg = (
                f"{_backend_qualified('where')} is only supported during tracing in its "
                "3-argument form "
                "(where(condition, x, y))"
            )
            raise TracingError(msg)

        condition, x, y = args
        result = np.where(
            _get_value(condition, traced_type),
            _get_value(x, traced_type),
            _get_value(y, traced_type),
        )

        node_id = _add_backend_node(
            graph=graph,
            op=_canonical_op(op_name),
            inputs=(
                _get_node(condition, graph, traced_type),
                _get_node(x, graph, traced_type),
                _get_node(y, graph, traced_type),
            ),
            value=result,
            attrs={},
            shape=result.shape,
            dtype=result.dtype,
        )
        return result, node_id

    return handler


def _make_interp_handler(op_name: str) -> Callable[..., tuple[Any, int]]:
    """Create a handler for ``interp``."""

    def handler(
        graph: DynamicTape,
        traced_type: type[TracedArrayLike],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Any, int]:
        positional_names = ("left", "right", "period")
        unsupported = set(kwargs.keys()) - _SUPPORTED_INTERP_KWARGS
        if unsupported:
            msg = (
                f"{_backend_qualified('interp')} kwargs not yet supported during tracing: "
                f"{sorted(unsupported)}. "
                f"Supported kwargs are: {sorted(_SUPPORTED_INTERP_KWARGS)}"
            )
            raise TracingError(msg)
        if len(args) < _INTERP_NARGS or len(args) > _INTERP_NARGS + len(positional_names):
            msg = (
                f"{_backend_qualified('interp')} expects "
                "(x, xp, fp, left=None, right=None, period=None)"
            )
            raise TracingError(msg)

        x, xp, fp = args[:_INTERP_NARGS]
        values = dict(kwargs)
        for name, value in zip(positional_names, args[_INTERP_NARGS:], strict=False):
            if name in values:
                msg = f"{_backend_qualified('interp')} received {name} twice"
                raise TracingError(msg)
            values[name] = value
        left = values.get("left")
        right = values.get("right")
        period = values.get("period")
        if _is_traced_operand(period, traced_type):
            msg = (
                f"{_backend_qualified('interp')} period= must be static because "
                "it controls periodic sorting"
            )
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

        node_id = _add_backend_node(
            graph=graph,
            op=_canonical_op(op_name),
            inputs=(
                _get_node(x, graph, traced_type),
                _get_node(xp, graph, traced_type),
                _get_node(fp, graph, traced_type),
            ),
            value=result,
            attrs=attrs,
            shape=result.shape,
            dtype=result.dtype,
        )
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
        return result_value, int(result_node_id)

    return handler
