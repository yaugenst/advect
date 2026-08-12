# ruff: noqa: ANN401
# Composite lowerings intentionally accept both concrete arrays and tracers.
"""Differentiable NumPy algorithms assembled from Advect's primitive surface."""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, Any, cast

import numpy as _numpy  # noqa: ICN001 - typed module and dynamic lowering namespace
from numpy.lib.stride_tricks import sliding_window_view

from advect.core._errors import TracingError
from advect.core._protocols import _snapshot_traced
from advect.numpy._array_function.composite import (
    _finish,
    _first_traced,
    _lift_composite_constant,
    _ndim,
    _normalize_axes,
)
from advect.numpy._array_function.normalization import _bind_optional_positionals

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike
    from advect.numpy._array_function.composite import CompositeResult
    from advect.numpy._array_function.emission import ArrayFunctionHandler


_BINARY_ARITY = 2
_TERNARY_ARITY = 3
# NumPy's stubs cannot express calls intentionally dispatched through
# TracedArray. Keep the private lowering namespace dynamic while its public
# handler boundary remains fully annotated.
np: Any = _numpy
_I0_SMALL_COEFFICIENTS = (
    -4.4153416464793394e-18,
    3.3307945188222384e-17,
    -2.431279846547955e-16,
    1.715391285555133e-15,
    -1.1685332877993451e-14,
    7.676185498604936e-14,
    -4.856446783111929e-13,
    2.95505266312964e-12,
    -1.726826291441556e-11,
    9.675809035373237e-11,
    -5.189795601635263e-10,
    2.6598237246823866e-09,
    -1.300025009986248e-08,
    6.046995022541919e-08,
    -2.670793853940612e-07,
    1.1173875391201037e-06,
    -4.4167383584587505e-06,
    1.6448448070728897e-05,
    -5.754195010082104e-05,
    0.00018850288509584165,
    -0.0005763755745385824,
    0.0016394756169413358,
    -0.004324309995050576,
    0.010546460394594998,
    -0.02373741480589947,
    0.04930528423967071,
    -0.09490109704804764,
    0.17162090152220877,
    -0.3046826723431984,
    0.6767952744094761,
)
_I0_LARGE_COEFFICIENTS = (
    -7.233180487874754e-18,
    -4.830504485944182e-18,
    4.46562142029676e-17,
    3.461222867697461e-17,
    -2.8276239805165836e-16,
    -3.425485619677219e-16,
    1.7725601330565264e-15,
    3.8116806693526224e-15,
    -9.554846698828307e-15,
    -4.150569347287222e-14,
    1.5400862175214098e-14,
    3.8527783827421426e-13,
    7.180124451383666e-13,
    -1.7941785315068062e-12,
    -1.3215811840447713e-11,
    -3.1499165279632416e-11,
    1.1889147107846439e-11,
    4.94060238822497e-10,
    3.3962320257083865e-09,
    2.266668990498178e-08,
    2.0489185894690638e-07,
    2.8913705208347567e-06,
    6.889758346916824e-05,
    0.0033691164782556943,
    0.8044904110141088,
)


def _chebyshev_evaluate(value: Any, coefficients: tuple[float, ...]) -> Any:
    current: Any = coefficients[0]
    previous: Any = 0.0
    before_previous: Any = 0.0
    for coefficient in coefficients[1:]:
        before_previous = previous
        previous = current
        current = value * previous - before_previous + coefficient
    return 0.5 * (current - before_previous)


def _lift_typed_constant(value: object, anchor: TracedArrayLike) -> Any:
    array = np.asarray(value)
    zero = np.astype(np.sum(np.zeros_like(anchor)), array.dtype)
    return zero + array


def _concrete_value(value: Any, traced_type: type[TracedArrayLike]) -> Any:
    if isinstance(value, traced_type):
        return _snapshot_traced(value)[1]
    if isinstance(value, tuple):
        return tuple(_concrete_value(item, traced_type) for item in value)
    if isinstance(value, list):
        return [_concrete_value(item, traced_type) for item in value]
    return value


def _i0_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    if len(args) != 1 or kwargs:
        msg = "numpy.i0 expects one real array during tracing"
        raise TracingError(msg)
    value = args[0]
    dtype = np.dtype(value.dtype)
    if np.issubdtype(dtype, np.complexfloating):
        msg = "numpy.i0 does not support complex values"
        raise TypeError(msg)
    if not np.issubdtype(dtype, np.floating):
        value = np.astype(value, np.float64)
    magnitude = np.absolute(value)
    small_region = np.less_equal(magnitude, 8.0)
    small = np.exp(magnitude) * _chebyshev_evaluate(
        magnitude / 2.0 - 2.0,
        _I0_SMALL_COEFFICIENTS,
    )
    safe_large_magnitude = np.where(small_region, 8.0, magnitude)
    large = (
        np.exp(safe_large_magnitude)
        * _chebyshev_evaluate(
            32.0 / safe_large_magnitude - 2.0,
            _I0_LARGE_COEFFICIENTS,
        )
        / np.sqrt(safe_large_magnitude)
    )
    return _finish(np.where(small_region, small, large), traced_type=traced_type)


def _arange_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    values = _bind_optional_positionals(
        name="arange",
        args=args,
        kwargs=kwargs,
        required=1,
        optional=("stop", "step", "dtype"),
        keyword_only=frozenset({"device", "like"}),
    )
    unsupported = set(values) - {"stop", "step", "dtype", "device", "like"}
    if unsupported:
        msg = f"numpy.arange kwargs not supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)
    if len(args) == 1 and "stop" not in values:
        start: Any = 0
        stop = args[0]
    else:
        start = args[0]
        stop = values.get("stop")
    if stop is None:
        msg = "numpy.arange requires a stop value"
        raise TracingError(msg)
    step = values.get("step", 1)
    anchor = _first_traced(
        (start, stop, step, values.get("like")),
        traced_type=traced_type,
    )
    if anchor is None:
        msg = "numpy.arange requires a traced bound, step, or like= operand"
        raise TracingError(msg)
    call_kwargs: dict[str, Any] = {}
    if values.get("dtype") is not None:
        call_kwargs["dtype"] = values["dtype"]
    if values.get("device") is not None:
        call_kwargs["device"] = values["device"]
    concrete_start = _concrete_value(start, traced_type)
    concrete_stop = _concrete_value(stop, traced_type)
    concrete_step = _concrete_value(step, traced_type)
    expected = np.arange(
        concrete_start,
        concrete_stop,
        concrete_step,
        **call_kwargs,
    )
    result = _lift_typed_constant(expected, anchor)
    positions = np.arange(expected.size)
    if isinstance(start, traced_type):
        result = result + (cast("Any", start) - concrete_start)
    if isinstance(step, traced_type):
        result = result + positions * (cast("Any", step) - concrete_step)
    return _finish(np.astype(result, expected.dtype), traced_type=traced_type)


def _block_depth(value: object) -> int:
    if not isinstance(value, list):
        return 0
    if not value:
        msg = "numpy.block does not accept empty lists"
        raise TracingError(msg)
    depths = tuple(_block_depth(item) for item in value)
    if len(set(depths)) != 1:
        msg = "numpy.block list depths must match"
        raise TracingError(msg)
    return depths[0] + 1


def _block_leaves(value: object) -> tuple[object, ...]:
    if not isinstance(value, list):
        return (value,)
    return tuple(item for child in value for item in _block_leaves(child))


def _block_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    if len(args) != 1 or kwargs:
        msg = "numpy.block expects one nested list during tracing"
        raise TracingError(msg)
    arrays = args[0]
    depth = _block_depth(arrays)
    leaves = _block_leaves(arrays)
    anchor = _first_traced(leaves, traced_type=traced_type)
    if anchor is None:
        msg = "numpy.block requires a traced operand"
        raise TracingError(msg)
    result_ndim = max(depth, *(_ndim(item) for item in leaves))

    def assemble(value: object, level: int) -> Any:
        if not isinstance(value, list):
            array: Any = (
                value if isinstance(value, traced_type) else _lift_composite_constant(value, anchor)
            )
            shape = (1,) * (result_ndim - int(array.ndim)) + tuple(array.shape)
            return np.reshape(array, shape)
        children = tuple(assemble(item, level - 1) for item in value)
        return np.concatenate(children, axis=-level)

    return _finish(assemble(arrays, depth), traced_type=traced_type)


def _apply_along_axis_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    if len(args) < _TERNARY_ARITY:
        msg = "numpy.apply_along_axis expects (func1d, axis, arr, *args) during tracing"
        raise TracingError(msg)
    function, axis_raw, array, *function_args = args
    if not callable(function):
        msg = "numpy.apply_along_axis func1d must be callable"
        raise TracingError(msg)
    axis = _normalize_axes(int(axis_raw), int(array.ndim))[0]
    moved = np.moveaxis(array, axis, -1)
    batch_shape = tuple(int(size) for size in moved.shape[:-1])
    batch_size = math.prod(batch_shape)
    rows = np.reshape(moved, (batch_size, int(moved.shape[-1])))
    outputs: list[Any] = []
    for index in range(batch_size):
        output: Any = function(rows[index], *function_args, **kwargs)
        if not isinstance(output, traced_type):
            output = _lift_composite_constant(output, array)
        outputs.append(output)
    if not outputs:
        msg = "numpy.apply_along_axis cannot iterate an empty batch during tracing"
        raise TracingError(msg)
    output_shape = tuple(int(size) for size in outputs[0].shape)
    if any(tuple(item.shape) != output_shape for item in outputs):
        msg = "numpy.apply_along_axis func1d returned inconsistent shapes"
        raise TracingError(msg)
    stacked = np.reshape(np.stack(tuple(outputs)), (*batch_shape, *output_shape))
    before = axis
    after = int(array.ndim) - axis - 1
    output_rank = len(output_shape)
    permutation = (
        *range(before),
        *range(before + after, before + after + output_rank),
        *range(before, before + after),
    )
    if permutation != tuple(range(stacked.ndim)):
        stacked = np.transpose(stacked, permutation)
    return _finish(stacked, traced_type=traced_type)


def _apply_over_axes_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    if len(args) != _TERNARY_ARITY or kwargs:
        msg = "numpy.apply_over_axes expects (func, a, axes) during tracing"
        raise TracingError(msg)
    function, array, axes_raw = args
    if not callable(function):
        msg = "numpy.apply_over_axes func must be callable"
        raise TracingError(msg)
    axes = _normalize_axes(axes_raw, int(array.ndim))
    result: Any = array
    for axis in axes:
        reduced: Any = function(result, axis)
        if not isinstance(reduced, traced_type):
            reduced = _lift_composite_constant(
                reduced,
                cast("TracedArrayLike", result),
            )
        if reduced.ndim == result.ndim - 1:
            reduced = np.expand_dims(reduced, axis)
        elif reduced.ndim != result.ndim:
            msg = "numpy.apply_over_axes func must preserve rank or remove only its axis"
            raise TracingError(msg)
        result = reduced
    return _finish(result, traced_type=traced_type)


def _logspace_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    values = _bind_optional_positionals(
        name="logspace",
        args=args,
        kwargs=kwargs,
        required=_BINARY_ARITY,
        optional=("num", "endpoint", "base", "dtype", "axis"),
    )
    start, stop = args[:2]
    num = int(values.get("num", 50))
    endpoint = bool(values.get("endpoint", True))
    base = values.get("base", 10.0)
    dtype = values.get("dtype")
    axis = int(values.get("axis", 0))
    exponents = np.linspace(start, stop, num=num, endpoint=endpoint, axis=axis)
    result = np.power(base, exponents)
    if dtype is not None:
        result = np.astype(result, dtype)
    return _finish(result, traced_type=traced_type)


def _geomspace_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    values = _bind_optional_positionals(
        name="geomspace",
        args=args,
        kwargs=kwargs,
        required=_BINARY_ARITY,
        optional=("num", "endpoint", "dtype", "axis"),
    )
    start, stop = args[:2]
    num = int(values.get("num", 50))
    endpoint = bool(values.get("endpoint", True))
    dtype = values.get("dtype")
    axis = int(values.get("axis", 0))
    anchor = _first_traced((start, stop), traced_type=traced_type)
    if anchor is None:
        msg = "numpy.geomspace requires a traced endpoint"
        raise TracingError(msg)
    computation_dtype = np.result_type(start, stop, float(num))
    start_value = np.astype(
        start if isinstance(start, traced_type) else _lift_composite_constant(start, anchor),
        computation_dtype,
    )
    stop_value = np.astype(
        stop if isinstance(stop, traced_type) else _lift_composite_constant(stop, anchor),
        computation_dtype,
    )
    rotation = np.sign(start_value)
    normalized_start = start_value / rotation
    normalized_stop = stop_value / rotation
    result = np.power(
        10.0,
        np.linspace(
            np.log10(normalized_start),
            np.log10(normalized_stop),
            num=num,
            endpoint=endpoint,
            axis=0,
        ),
    )
    if num > 0:
        position_shape = (num,) + (1,) * int(start_value.ndim)
        positions = np.reshape(np.arange(num), position_shape)
        result = np.where(positions == 0, normalized_start, result)
        if num > 1 and endpoint:
            result = np.where(positions == num - 1, normalized_stop, result)
    result = result * rotation
    if axis != 0:
        result = np.moveaxis(result, 0, axis)
    if dtype is not None:
        result = np.astype(result, dtype)
    return _finish(result, traced_type=traced_type)


def _unstack_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    if len(args) != 1 or set(kwargs) - {"axis"}:
        msg = "numpy.unstack expects (x, *, axis=0) during tracing"
        raise TracingError(msg)
    array = args[0]
    axis = _normalize_axes(int(kwargs.get("axis", 0)), int(array.ndim))[0]
    result = tuple(np.take(array, index, axis=axis) for index in range(int(array.shape[axis])))
    return _finish(result, traced_type=traced_type)


def _sliding_window_view_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    if len(args) not in {_BINARY_ARITY, _TERNARY_ARITY}:
        msg = "numpy.lib.stride_tricks.sliding_window_view expects (x, window_shape, axis=None)"
        raise TracingError(msg)
    unsupported = set(kwargs) - {"axis", "subok", "writeable"}
    if unsupported:
        msg = (
            "numpy.lib.stride_tricks.sliding_window_view kwargs not supported during "
            f"tracing: {sorted(unsupported)}"
        )
        raise TracingError(msg)
    if len(args) == _TERNARY_ARITY and "axis" in kwargs:
        msg = "numpy.lib.stride_tricks.sliding_window_view received axis twice"
        raise TracingError(msg)
    if bool(kwargs.get("subok", False)) or bool(kwargs.get("writeable", False)):
        msg = "sliding_window_view tracing returns a non-writeable base-independent array"
        raise TracingError(msg)
    array, window_raw = args[:2]
    axis_raw = args[2] if len(args) == _TERNARY_ARITY else kwargs.get("axis")
    if axis_raw is None:
        axes = tuple(range(int(array.ndim)))
    else:
        axes = _normalize_axes(axis_raw, int(array.ndim))
    if isinstance(window_raw, (int, np.integer)):
        windows = (int(window_raw),)
    else:
        windows = tuple(int(size) for size in window_raw)
    if len(windows) != len(axes):
        msg = "sliding_window_view window_shape and axis must have matching lengths"
        raise TracingError(msg)
    if any(size <= 0 for size in windows):
        msg = "sliding_window_view window dimensions must be positive"
        raise TracingError(msg)
    if any(size > int(array.shape[axis]) for size, axis in zip(windows, axes, strict=True)):
        msg = "sliding_window_view window exceeds an input dimension"
        raise TracingError(msg)

    base_shape = [int(size) for size in array.shape]
    for size, axis in zip(windows, axes, strict=True):
        base_shape[axis] -= size - 1
    slices: list[object] = []
    for offsets in itertools.product(*(range(size) for size in windows)):
        index = [slice(None)] * int(array.ndim)
        for offset, axis in zip(offsets, axes, strict=True):
            index[axis] = slice(offset, offset + base_shape[axis])
        slices.append(array[tuple(index)])
    stacked = np.stack(tuple(slices), axis=-1)
    result = np.reshape(stacked, (*base_shape, *windows))
    return _finish(result, traced_type=traced_type)


def _sort_complex_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    if len(args) != 1 or kwargs:
        msg = "numpy.sort_complex expects one array during tracing"
        raise TracingError(msg)
    dtype = (
        args[0].dtype
        if np.issubdtype(args[0].dtype, np.complexfloating)
        else np.dtype(np.complex128)
    )
    return _finish(np.sort(np.astype(args[0], dtype)), traced_type=traced_type)


def _unwrap_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    period = kwargs.pop("period", 2 * np.pi)
    values = _bind_optional_positionals(
        name="unwrap",
        args=args,
        kwargs=kwargs,
        required=1,
        optional=("discont", "axis"),
    )
    phase = args[0]
    axis = int(values.get("axis", -1))
    discontinuity = values.get("discont")
    threshold = period / 2 if discontinuity is None else np.maximum(discontinuity, period / 2)
    delta = np.diff(phase, axis=axis)
    wrapped = np.remainder(delta + period / 2, period) - period / 2
    wrapped = np.where((wrapped == -period / 2) & (delta > 0), period / 2, wrapped)
    correction = np.where(np.abs(delta) < threshold, 0, wrapped - delta)
    accumulated = np.cumsum(correction, axis=axis)
    leading = [slice(None)] * int(phase.ndim)
    leading[axis] = slice(0, 1)
    offsets = np.concatenate((np.zeros_like(phase[tuple(leading)]), accumulated), axis=axis)
    return _finish(phase + offsets, traced_type=traced_type)


def _real_if_close_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    values = _bind_optional_positionals(
        name="real_if_close",
        args=args,
        kwargs=kwargs,
        required=1,
        optional=("tol",),
    )
    array = args[0]
    _node_id, concrete = _snapshot_traced(array)
    concrete_result = np.real_if_close(np.asarray(concrete), tol=values.get("tol", 100))
    result = np.real(array) if not np.iscomplexobj(concrete_result) else np.copy(array)
    return _finish(result, traced_type=traced_type)


def _bincount_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    values = _bind_optional_positionals(
        name="bincount",
        args=args,
        kwargs=kwargs,
        required=1,
        optional=("weights", "minlength"),
    )
    indices = args[0]
    anchor = _first_traced((indices, values.get("weights")), traced_type=traced_type)
    if anchor is None:
        msg = "numpy.bincount requires a traced indices or weights operand"
        raise TracingError(msg)
    concrete_indices = (
        np.asarray(_snapshot_traced(indices)[1])
        if isinstance(indices, traced_type)
        else np.asarray(indices)
    )
    if concrete_indices.ndim != 1 or not np.issubdtype(
        concrete_indices.dtype,
        np.integer,
    ):
        msg = "numpy.bincount indices must be a one-dimensional integer array"
        raise TracingError(msg)
    if np.any(concrete_indices < 0):
        msg = "numpy.bincount indices must be non-negative"
        raise TracingError(msg)
    minlength_raw = values.get("minlength", 0)
    if isinstance(minlength_raw, traced_type) or not isinstance(
        minlength_raw,
        (int, np.integer),
    ):
        msg = "numpy.bincount minlength must be a static integer"
        raise TracingError(msg)
    minlength = int(minlength_raw)
    if minlength < 0:
        msg = "'minlength' must not be negative"
        raise ValueError(msg)
    weights = values.get("weights")
    if weights is None:
        counts = np.bincount(concrete_indices, minlength=minlength)
        result = _lift_typed_constant(counts, anchor)
        return _finish(result, traced_type=traced_type)
    weight_array = np.atleast_1d(
        weights if isinstance(weights, traced_type) else _lift_composite_constant(weights, anchor)
    )
    if weight_array.ndim != 1 or weight_array.shape[0] != concrete_indices.shape[0]:
        msg = "numpy.bincount weights must be one-dimensional and match indices"
        raise TracingError(msg)
    weight_node_id, concrete_weights = _snapshot_traced(weight_array)
    result = np.bincount(
        concrete_indices,
        weights=np.asarray(concrete_weights),
        minlength=minlength,
    )
    node_id = graph.record_operation_with_literals(
        "array_ext.bincount",
        (weight_node_id,),
        (1,),
        (concrete_indices,),
        result,
        {"minlength": minlength, "_advect_backend": "numpy"},
        result.shape,
        result.dtype,
    )
    return result, node_id


def _insert_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    values = _bind_optional_positionals(
        name="insert",
        args=args,
        kwargs=kwargs,
        required=_TERNARY_ARITY,
        optional=("axis",),
    )
    array_raw, obj, inserted_raw = args[:_TERNARY_ARITY]
    if isinstance(obj, traced_type):
        msg = "numpy.insert obj= must be static because it controls output shape"
        raise TracingError(msg)
    anchor = _first_traced((array_raw, inserted_raw), traced_type=traced_type)
    if anchor is None:
        msg = "numpy.insert requires a traced array or values operand"
        raise TracingError(msg)
    array: Any = (
        array_raw
        if isinstance(array_raw, traced_type)
        else _lift_composite_constant(array_raw, anchor)
    )
    inserted: Any = (
        inserted_raw
        if isinstance(inserted_raw, traced_type)
        else _lift_composite_constant(inserted_raw, anchor)
    )
    axis_raw = values.get("axis")
    if axis_raw is None:
        array = np.ravel(array)
        axis = 0
    else:
        axis = int(axis_raw)
        if axis < 0:
            axis += int(array.ndim)
        if axis < 0 or axis >= int(array.ndim):
            msg = f"numpy.insert axis {axis_raw} is out of bounds"
            raise TracingError(msg)

    inserted = np.astype(inserted, array.dtype)
    if axis_raw is None:
        inserted = np.ravel(inserted)
    array_shape = tuple(int(size) for size in array.shape)
    inserted_shape = tuple(int(size) for size in inserted.shape)
    source_markers = np.arange(int(array.size), dtype=np.int64).reshape(array_shape)
    inserted_markers = np.arange(int(inserted.size), dtype=np.int64).reshape(inserted_shape)
    try:
        source_fill = _numpy.full_like(inserted_markers, -1)
        inserted_fill = _numpy.full_like(source_markers, -1)
        source_map = _numpy.insert(source_markers, obj, source_fill, axis=axis)
        inserted_map = _numpy.insert(inserted_fill, obj, inserted_markers, axis=axis)
    except (IndexError, TypeError, ValueError) as exc:
        raise type(exc)(str(exc)) from exc

    output_shape = tuple(int(size) for size in source_map.shape)
    source_mask = source_map >= 0
    inserted_mask = inserted_map >= 0
    if np.any(source_mask):
        source_values = np.reshape(
            np.take(np.ravel(array), np.maximum(source_map, 0)),
            output_shape,
        )
    else:
        source_values = np.zeros(output_shape, dtype=array.dtype) + np.sum(array) * 0
    if not np.any(inserted_mask):
        return _finish(source_values, traced_type=traced_type)
    inserted_values = np.reshape(
        np.take(np.ravel(inserted), np.maximum(inserted_map, 0)),
        output_shape,
    )
    return _finish(
        np.where(inserted_mask, inserted_values, source_values),
        traced_type=traced_type,
    )


def _histogram_edges(
    samples: Any,
    concrete_samples: np.ndarray[Any, Any],
    bins: Any,
    concrete_edges: np.ndarray[Any, Any],
    histogram_range: Any,
    *,
    anchor: TracedArrayLike,
    traced_type: type[TracedArrayLike],
) -> Any:
    if isinstance(bins, traced_type):
        return bins
    concrete_bins = _concrete_value(bins, traced_type)
    if np.ndim(concrete_bins) == 1:
        return _lift_typed_constant(concrete_edges, anchor)

    bin_count = int(concrete_bins)
    if histogram_range is None:
        if concrete_samples.size == 0:
            minimum = _lift_typed_constant(0.0, anchor)
            maximum = _lift_typed_constant(1.0, anchor)
        else:
            traced_samples = (
                samples
                if isinstance(samples, traced_type)
                else _lift_typed_constant(samples, anchor)
            )
            minimum = np.min(traced_samples)
            maximum = np.max(traced_samples)
            if concrete_samples.min() == concrete_samples.max():
                minimum = minimum - 0.5
                maximum = maximum + 0.5
    else:
        minimum, maximum = histogram_range
        if not isinstance(minimum, traced_type):
            minimum = _lift_typed_constant(minimum, anchor)
        if not isinstance(maximum, traced_type):
            maximum = _lift_typed_constant(maximum, anchor)
    return np.linspace(minimum, maximum, bin_count + 1)


def _histogram_indices(
    concrete_samples: np.ndarray[Any, Any],
    concrete_edges: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    bin_indices = np.searchsorted(concrete_edges, concrete_samples, side="right") - 1
    bin_indices[concrete_samples == concrete_edges[-1]] = concrete_edges.size - 2
    valid = (bin_indices >= 0) & (bin_indices < concrete_edges.size - 1)
    return bin_indices, valid


def _histogram_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    values = _bind_optional_positionals(
        name="histogram",
        args=args,
        kwargs=kwargs,
        required=1,
        optional=("bins", "range", "density", "weights"),
    )
    samples = np.ravel(args[0])
    bins = values.get("bins", 10)
    histogram_range = values.get("range")
    weights = values.get("weights")
    anchor = _first_traced(
        (samples, weights, bins, histogram_range),
        traced_type=traced_type,
    )
    if anchor is None:
        msg = "numpy.histogram requires a traced samples, bins, range, or weights operand"
        raise TracingError(msg)
    concrete_array = np.asarray(_concrete_value(samples, traced_type))
    if isinstance(bins, str):
        msg = "numpy.histogram string bin estimators are data-dependent and not traceable"
        raise TracingError(msg)
    concrete_bins = _concrete_value(bins, traced_type)
    concrete_range = _concrete_value(histogram_range, traced_type)
    concrete_weights = None if weights is None else _concrete_value(weights, traced_type)
    expected_histogram, concrete_edges = np.histogram(
        concrete_array,
        bins=concrete_bins,
        range=concrete_range,
        weights=concrete_weights,
    )
    edges = _histogram_edges(
        samples,
        concrete_array,
        bins,
        concrete_edges,
        histogram_range,
        anchor=anchor,
        traced_type=traced_type,
    )
    if weights is None:
        histogram: Any = _lift_typed_constant(expected_histogram, anchor)
    else:
        weight_array = np.ravel(
            weights if isinstance(weights, traced_type) else _lift_typed_constant(weights, anchor)
        )
        if weight_array.shape != samples.shape:
            msg = "numpy.histogram weights must match the sample shape"
            raise TracingError(msg)
        bin_indices, valid = _histogram_indices(concrete_array, concrete_edges)
        safe_indices = np.where(valid, bin_indices, 0)
        histogram = np.astype(
            np.bincount(
                safe_indices,
                weights=weight_array * valid,
                minlength=int(expected_histogram.size),
            ),
            expected_histogram.dtype,
        )
    if bool(values.get("density", False)):
        histogram = histogram / np.sum(histogram) / np.diff(edges)
    return _finish((histogram, edges), traced_type=traced_type)


def _histogram_bin_edges_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    values = _bind_optional_positionals(
        name="histogram_bin_edges",
        args=args,
        kwargs=kwargs,
        required=1,
        optional=("bins", "range", "weights"),
    )
    samples = np.ravel(args[0])
    bins = values.get("bins", 10)
    histogram_range = values.get("range")
    weights = values.get("weights")
    anchor = _first_traced(
        (samples, bins, histogram_range, weights),
        traced_type=traced_type,
    )
    if anchor is None:
        msg = "numpy.histogram_bin_edges requires a traced operand"
        raise TracingError(msg)
    if isinstance(bins, str):
        msg = "numpy.histogram_bin_edges string estimators are data-dependent and not traceable"
        raise TracingError(msg)
    concrete_samples = np.asarray(_concrete_value(samples, traced_type))
    concrete_edges = np.histogram_bin_edges(
        concrete_samples,
        bins=_concrete_value(bins, traced_type),
        range=_concrete_value(histogram_range, traced_type),
        weights=None if weights is None else _concrete_value(weights, traced_type),
    )
    edges = _histogram_edges(
        samples,
        concrete_samples,
        bins,
        concrete_edges,
        histogram_range,
        anchor=anchor,
        traced_type=traced_type,
    )
    return _finish(edges, traced_type=traced_type)


def _histogram2d_bin_specs(bins: Any) -> tuple[object, object]:
    ndim = getattr(bins, "ndim", None)
    paired = isinstance(bins, (tuple, list)) and len(bins) == _BINARY_ARITY
    paired = paired or int(np.ndim(bins) if ndim is None else ndim) == _BINARY_ARITY
    return (bins[0], bins[1]) if paired else (bins, bins)


def _histogram2d_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    values = _bind_optional_positionals(
        name="histogram2d",
        args=args,
        kwargs=kwargs,
        required=_BINARY_ARITY,
        optional=("bins", "range", "density", "weights"),
    )
    x = np.ravel(args[0])
    y = np.ravel(args[1])
    bins = values.get("bins", 10)
    histogram_range = values.get("range")
    weights = values.get("weights")
    anchor = _first_traced(
        (x, y, bins, histogram_range, weights),
        traced_type=traced_type,
    )
    if anchor is None:
        msg = "numpy.histogram2d requires a traced operand"
        raise TracingError(msg)
    concrete_x = np.asarray(_concrete_value(x, traced_type))
    concrete_y = np.asarray(_concrete_value(y, traced_type))
    concrete_bins = _concrete_value(bins, traced_type)
    concrete_range = _concrete_value(histogram_range, traced_type)
    concrete_weights = None if weights is None else _concrete_value(weights, traced_type)
    expected, concrete_x_edges, concrete_y_edges = np.histogram2d(
        concrete_x,
        concrete_y,
        bins=concrete_bins,
        range=concrete_range,
        weights=concrete_weights,
    )
    x_bins, y_bins = _histogram2d_bin_specs(bins)
    x_range, y_range = (None, None) if histogram_range is None else tuple(histogram_range)
    x_edges = _histogram_edges(
        x,
        concrete_x,
        x_bins,
        concrete_x_edges,
        x_range,
        anchor=anchor,
        traced_type=traced_type,
    )
    y_edges = _histogram_edges(
        y,
        concrete_y,
        y_bins,
        concrete_y_edges,
        y_range,
        anchor=anchor,
        traced_type=traced_type,
    )
    if weights is None:
        histogram: Any = _lift_typed_constant(expected, anchor)
    else:
        weight_array = np.ravel(
            weights if isinstance(weights, traced_type) else _lift_typed_constant(weights, anchor)
        )
        if weight_array.shape != x.shape:
            msg = "numpy.histogram2d weights must match x and y"
            raise TracingError(msg)
        x_indices, x_valid = _histogram_indices(concrete_x, concrete_x_edges)
        y_indices, y_valid = _histogram_indices(concrete_y, concrete_y_edges)
        y_bins = int(expected.shape[1])
        valid = x_valid & y_valid
        linear_indices = x_indices * y_bins + y_indices
        safe_indices = np.where(valid, linear_indices, 0)
        histogram = np.reshape(
            np.astype(
                np.bincount(
                    safe_indices,
                    weights=weight_array * valid,
                    minlength=int(expected.size),
                ),
                expected.dtype,
            ),
            expected.shape,
        )
    if bool(values.get("density", False)):
        histogram = (
            histogram / np.sum(histogram) / np.diff(x_edges)[:, None] / np.diff(y_edges)[None, :]
        )
    return _finish(
        (histogram, x_edges, y_edges),
        traced_type=traced_type,
    )


def _histogramdd_handler(  # noqa: PLR0912, PLR0915
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    values = _bind_optional_positionals(
        name="histogramdd",
        args=args,
        kwargs=kwargs,
        required=1,
        optional=("bins", "range", "density", "weights"),
    )
    sample_arg = args[0]
    if isinstance(sample_arg, (tuple, list)):
        raw_columns = tuple(np.ravel(column) for column in sample_arg)
    else:
        sample_ndim = int(cast("Any", getattr(sample_arg, "ndim", np.ndim(sample_arg))))
        if sample_ndim != _BINARY_ARITY:
            msg = "numpy.histogramdd sample must have shape (N, D) during tracing"
            raise TracingError(msg)
        raw_columns = tuple(
            sample_arg[:, dimension] for dimension in range(int(sample_arg.shape[1]))
        )
    if not raw_columns:
        msg = "numpy.histogramdd requires at least one sample dimension"
        raise TracingError(msg)

    bins = values.get("bins", 10)
    histogram_range = values.get("range")
    weights = values.get("weights")
    anchor = _first_traced(
        (raw_columns, bins, histogram_range, weights),
        traced_type=traced_type,
    )
    if anchor is None:
        msg = "numpy.histogramdd requires a traced operand"
        raise TracingError(msg)

    concrete_columns = tuple(
        np.asarray(_concrete_value(column, traced_type)) for column in raw_columns
    )
    sample_count = int(concrete_columns[0].size)
    if any(int(column.size) != sample_count for column in concrete_columns):
        msg = "numpy.histogramdd sample columns must have equal lengths"
        raise TracingError(msg)
    dimensions = len(concrete_columns)
    concrete_sample = np.stack(concrete_columns, axis=1)
    concrete_bins = _concrete_value(bins, traced_type)
    concrete_range = _concrete_value(histogram_range, traced_type)
    concrete_weights = None if weights is None else _concrete_value(weights, traced_type)
    expected, concrete_edges = np.histogramdd(
        concrete_sample,
        bins=concrete_bins,
        range=concrete_range,
        weights=concrete_weights,
    )

    if np.ndim(concrete_bins) == 0:
        bin_specs = (bins,) * dimensions
    elif dimensions == 1 and isinstance(bins, traced_type):
        bin_specs = (bins,)
    else:
        bin_specs = tuple(bins)
    if len(bin_specs) != dimensions:
        msg = "numpy.histogramdd bins must provide one specification per dimension"
        raise TracingError(msg)
    range_specs = (None,) * dimensions if histogram_range is None else tuple(histogram_range)
    if len(range_specs) != dimensions:
        msg = "numpy.histogramdd range must provide one pair per dimension"
        raise TracingError(msg)

    columns = tuple(
        column if isinstance(column, traced_type) else _lift_composite_constant(column, anchor)
        for column in raw_columns
    )
    edges = [
        _histogram_edges(
            column,
            concrete_column,
            bin_spec,
            np.asarray(concrete_edge),
            range_spec,
            anchor=anchor,
            traced_type=traced_type,
        )
        for column, concrete_column, bin_spec, concrete_edge, range_spec in zip(
            columns,
            concrete_columns,
            bin_specs,
            concrete_edges,
            range_specs,
            strict=True,
        )
    ]
    indexed_columns = [
        _histogram_indices(concrete_column, np.asarray(concrete_edge))
        for concrete_column, concrete_edge in zip(
            concrete_columns,
            concrete_edges,
            strict=True,
        )
    ]
    bin_shape = tuple(int(np.asarray(edge).size - 1) for edge in concrete_edges)

    if weights is None:
        histogram: Any = _lift_typed_constant(expected, anchor)
    else:
        weight_array = np.ravel(
            weights if isinstance(weights, traced_type) else _lift_typed_constant(weights, anchor)
        )
        if tuple(weight_array.shape) != (sample_count,):
            msg = "numpy.histogramdd weights must match the number of samples"
            raise TracingError(msg)
        valid = np.ones(sample_count, dtype=bool)
        linear_indices = np.zeros(sample_count, dtype=np.intp)
        for (indices, dimension_valid), bin_count in zip(
            indexed_columns,
            bin_shape,
            strict=True,
        ):
            valid &= dimension_valid
            linear_indices = linear_indices * bin_count + indices
        safe_indices = np.where(valid, linear_indices, 0)
        histogram = np.reshape(
            np.astype(
                np.bincount(
                    safe_indices,
                    weights=weight_array * valid,
                    minlength=int(expected.size),
                ),
                expected.dtype,
            ),
            expected.shape,
        )
    if bool(values.get("density", False)):
        histogram = histogram / np.sum(histogram)
        for dimension, edge in enumerate(edges):
            width_shape = [1] * dimensions
            width_shape[dimension] = bin_shape[dimension]
            histogram = histogram / np.reshape(np.diff(edge), tuple(width_shape))
    return _finish((histogram, edges), traced_type=traced_type)


def register_algorithm_handlers(
    handlers: dict[Callable[..., Any], ArrayFunctionHandler],
) -> None:
    """Register higher-level algorithms with traceable composite lowerings."""
    handlers[np.apply_along_axis] = _apply_along_axis_handler
    handlers[np.apply_over_axes] = _apply_over_axes_handler
    handlers[np.arange] = _arange_handler
    handlers[np.block] = _block_handler
    handlers[np.logspace] = _logspace_handler
    handlers[np.geomspace] = _geomspace_handler
    unstack = getattr(np, "unstack", None)
    if callable(unstack):
        handlers[unstack] = _unstack_handler
    handlers[sliding_window_view] = _sliding_window_view_handler
    handlers[np.sort_complex] = _sort_complex_handler
    handlers[np.unwrap] = _unwrap_handler
    handlers[np.real_if_close] = _real_if_close_handler
    handlers[np.i0] = _i0_handler
    handlers[np.bincount] = _bincount_handler
    handlers[np.insert] = _insert_handler
    handlers[np.histogram] = _histogram_handler
    handlers[np.histogram_bin_edges] = _histogram_bin_edges_handler
    handlers[np.histogram2d] = _histogram2d_handler
    handlers[np.histogramdd] = _histogramdd_handler
