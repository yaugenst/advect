# ruff: noqa: ANN401
# Composite lowerings intentionally accept both concrete arrays and tracers.
"""Trace NumPy array-creation functions as canonical operations or compositions."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, cast

import numpy as _numpy  # noqa: ICN001 - typed module and dynamic lowering namespace

from advect.core._errors import TracingError
from advect.core._protocols import _snapshot_traced
from advect.numpy._array_function.composite import (
    _find_traced,
    _finish,
    _lift_composite_constant,
)
from advect.numpy._array_function.emission import _emit, _get_value
from advect.numpy._array_function.normalization import (
    _bind_optional_positionals,
    _normalize_constant_values,
    _normalize_pad_width,
    _normalize_shape,
)
from advect.numpy._constructors import (
    array as traced_array,
    asanyarray as traced_asanyarray,
    asarray as traced_asarray,
)

np: Any = _numpy

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike
    from advect.numpy._array_function.composite import CompositeResult
    from advect.numpy._array_function.emission import ArrayFunctionHandler

_MIN_REQUIRED_ARGS = 2

# ``numpy.linspace(start, stop, num, endpoint, retstep, dtype, axis)``
_LINSPACE_TRAILING_PARAMETERS = ("num", "endpoint", "retstep", "dtype", "axis")
# The keyword parameters of every np.pad mode that traces differentiably.
_PAD_MODE_PARAMETERS = {
    "constant": frozenset({"constant_values"}),
    "edge": frozenset(),
    "linear_ramp": frozenset({"end_values"}),
    "reflect": frozenset({"reflect_type"}),
    "symmetric": frozenset({"reflect_type"}),
    "wrap": frozenset(),
    "maximum": frozenset({"stat_length"}),
    "mean": frozenset({"stat_length"}),
    "median": frozenset({"stat_length"}),
    "minimum": frozenset({"stat_length"}),
}
_STATISTICAL_PAD_REDUCERS = {
    "maximum": _numpy.max,
    "mean": _numpy.mean,
    "median": _numpy.median,
    "minimum": _numpy.min,
}
_CONSTRUCTOR_KEYWORD_ONLY = frozenset({"device", "like"})


def _full_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    positional_names = ("dtype", "order")
    values = dict(zip(positional_names, args[_MIN_REQUIRED_ARGS:], strict=False)) | kwargs

    shape = _normalize_shape(args[0])
    fill_value = args[1]
    dtype = values.get("dtype")
    order = str(values.get("order", "C"))
    device = values.get("device")

    # NumPy dispatches full to the tracer only through a traced like= operand.
    call_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "order": order,
        "like": _get_value(values["like"], traced_type),
    }
    if device is not None:
        call_kwargs["device"] = device
    result = cast("Any", np.full)(shape, _get_value(fill_value, traced_type), **call_kwargs)

    attrs: dict[str, Any] = {"shape": shape}
    if dtype is not None:
        attrs["dtype"] = str(np.dtype(dtype))
    if order != "C":
        attrs["order"] = order
    if device is not None:
        attrs["device"] = device

    return _emit(graph, traced_type, "numpy.full", (fill_value,), result, attrs)


# NumPy dispatches its constructors to the tracer only through a traced like=.
def _basic_constructor_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    name: str,
    like_function: Callable[..., Any],
) -> CompositeResult:
    values = _bind_optional_positionals(
        name=name,
        args=args,
        kwargs=kwargs,
        required=1,
        optional=("dtype", "order"),
        keyword_only=_CONSTRUCTOR_KEYWORD_ONLY,
    )
    shape = _normalize_shape(args[0])
    anchor = values["like"]
    dtype = float if values.get("dtype") is None else values["dtype"]
    order = str(values.get("order", "C"))
    device = values.get("device")
    like_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "order": order,
        "shape": shape,
    }
    if device is not None:
        like_kwargs["device"] = device
    return _finish(like_function(anchor, **like_kwargs), traced_type=traced_type)


def _static_constructor_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    function: Callable[..., Any],
    optional: tuple[str, ...],
) -> CompositeResult:
    """Lift a like=-dispatched eye, identity or tri into the anchor's trace."""
    name = function.__name__
    values = _bind_optional_positionals(
        name=name,
        args=args,
        kwargs=kwargs,
        required=1,
        optional=optional,
        keyword_only=_CONSTRUCTOR_KEYWORD_ONLY,
    )
    anchor = values.pop("like")
    device = values.pop("device", None)
    concrete = function(args[0], **values)
    result = np.zeros_like(
        anchor,
        dtype=concrete.dtype,
        order=str(values.get("order", "C")),
        shape=concrete.shape,
        device=device,
    )
    return _finish(result + concrete, traced_type=traced_type)


def _constructor_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    constructor: Callable[..., Any],
) -> CompositeResult:
    return _finish(constructor(*args, **kwargs), traced_type=traced_type)


def _full_like_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    positional_names = ("dtype", "order", "subok", "shape")
    values = dict(zip(positional_names, args[_MIN_REQUIRED_ARGS:], strict=False)) | kwargs

    a = args[0]
    fill_value = args[1]
    dtype = values.get("dtype")
    order = str(values.get("order", "K"))
    subok = bool(values.get("subok", True))
    shape = values.get("shape")
    device = values.get("device")

    call_kwargs: dict[str, Any] = {"order": order, "subok": subok}
    if dtype is not None:
        call_kwargs["dtype"] = dtype
    if shape is not None:
        call_kwargs["shape"] = shape
    if device is not None:
        call_kwargs["device"] = device

    result = np.full_like(
        _get_value(a, traced_type), _get_value(fill_value, traced_type), **call_kwargs
    )

    attrs: dict[str, Any] = {}
    if dtype is not None:
        attrs["dtype"] = str(np.dtype(dtype))
    if shape is not None:
        attrs["shape"] = _normalize_shape(shape)
    if order != "K":
        attrs["order"] = order
    if not subok:
        attrs["subok"] = False
    if device is not None:
        attrs["device"] = device

    return _emit(graph, traced_type, "numpy.full_like", (a, fill_value), result, attrs)


def _linspace_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    start, stop = args[0], args[1]
    # ``num``/``endpoint``/``retstep``/``dtype``/``axis`` may arrive positionally.
    # Reading them from kwargs alone silently substitutes NumPy's defaults into
    # both the recorded value and the derivative attrs.
    bound = dict(zip(_LINSPACE_TRAILING_PARAMETERS, args[_MIN_REQUIRED_ARGS:], strict=False))
    bound.update(kwargs)

    num = int(bound.get("num", 50))
    endpoint = bool(bound.get("endpoint", True))
    retstep = bool(bound.get("retstep", False))
    dtype = bound.get("dtype")
    axis = int(bound.get("axis", 0))

    call_kwargs: dict[str, Any] = {
        "num": num,
        "endpoint": endpoint,
        "retstep": False,
        "dtype": dtype,
        "axis": axis,
    }
    if bound.get("device") is not None:
        call_kwargs["device"] = bound["device"]
    result = np.linspace(
        _get_value(start, traced_type),
        _get_value(stop, traced_type),
        **call_kwargs,
    )

    attrs: dict[str, Any] = {
        "num": num,
        "endpoint": endpoint,
        "axis": axis,
    }
    if dtype is not None:
        attrs["dtype"] = str(np.dtype(dtype))
    if bound.get("device") is not None:
        attrs["device"] = bound["device"]

    _, node_id = _emit(graph, traced_type, "numpy.linspace", (start, stop), result, attrs)
    if not retstep:
        return result, node_id
    traced_ctor = cast("Callable[..., TracedArrayLike]", traced_type)
    result_tracer = traced_ctor(value=result, node_id=node_id, recorder=graph)
    divisor = num - 1 if endpoint else num
    if divisor > 0:
        step = (stop - start) / divisor
    else:
        concrete_step = np.linspace(
            _get_value(start, traced_type),
            _get_value(stop, traced_type),
            num=num,
            endpoint=endpoint,
            retstep=True,
            dtype=dtype,
            axis=axis,
        )[1]
        step = _lift_composite_constant(concrete_step, result_tracer)
    step_node_id, step_value = _snapshot_traced(step)
    return (result, step_value), (node_id, step_node_id)


def _pad_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    mode = str(args[2] if len(args) > _MIN_REQUIRED_ARGS else kwargs.get("mode", "constant"))
    mode_parameters = _PAD_MODE_PARAMETERS.get(mode, frozenset())
    unsupported = set(kwargs) - ({"mode"} | mode_parameters)
    if unsupported:
        msg = f"numpy.pad kwargs not supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)

    x = args[0]
    pad_width = _normalize_pad_width(args[1])
    constant_values = kwargs.get("constant_values", 0)

    if mode not in _PAD_MODE_PARAMETERS:
        msg = (
            f"numpy.pad(mode={mode!r}) has a data-dependent nonlinear padding rule "
            "and is not differentiable through Advect's NumPy frontend"
        )
        raise TracingError(msg)
    if mode in _STATISTICAL_PAD_REDUCERS:
        return _statistical_pad(
            x,
            pad_width,
            mode=mode,
            stat_length=kwargs.get("stat_length"),
            traced_type=traced_type,
        )
    parameter_is_traced = _find_traced((constant_values, kwargs.get("end_values")), traced_type)
    if mode != "constant" or parameter_is_traced is not None:
        return _linear_pad(
            x,
            pad_width,
            mode=mode,
            constant_values=constant_values,
            end_values=kwargs.get("end_values", 0),
            reflect_type=str(kwargs.get("reflect_type", "even")),
            traced_type=traced_type,
        )

    pad_fn = cast("Callable[..., Any]", np.pad)
    result = pad_fn(
        _get_value(x, traced_type), pad_width=pad_width, mode=mode, constant_values=constant_values
    )

    attrs: dict[str, Any] = {
        "pad_width": pad_width,
        "mode": mode,
        "constant_values": _normalize_constant_values(constant_values),
    }

    return _emit(graph, traced_type, "numpy.pad", (x,), result, attrs)


def _normalize_pad_parameter(
    value: object,
    *,
    ndim: int,
    traced_type: type[TracedArrayLike],
    name: str,
) -> tuple[tuple[object, object], ...]:
    if isinstance(value, traced_type):
        shape = tuple(value.shape)
        if not shape:
            return ((value, value),) * ndim
        if shape == (2,):
            pair = (value[0], value[1])
            return (pair,) * ndim
        if shape == (ndim, 2):
            return tuple((value[axis, 0], value[axis, 1]) for axis in range(ndim))
        msg = f"numpy.pad {name} shape {shape} cannot broadcast to ({ndim}, 2)"
        raise TracingError(msg)
    array = np.asarray(value)
    try:
        broadcast = np.broadcast_to(array, (ndim, 2))
    except ValueError as error:
        msg = f"numpy.pad {name} shape {array.shape} cannot broadcast to ({ndim}, 2)"
        raise TracingError(msg) from error
    return tuple((row[0], row[1]) for row in broadcast)


def _pad_axis_slice(
    value: Any,
    *,
    axis: int,
    index: slice,
) -> tuple[slice, ...]:
    indices = [slice(None)] * int(value.ndim)
    indices[axis] = index
    return tuple(indices)


def _pad_axis_shape(value: Any, *, axis: int, length: int) -> tuple[int, ...]:
    shape = [int(dimension) for dimension in value.shape]
    shape[axis] = length
    return tuple(shape)


def _pad_edge_axis(
    value: Any,
    *,
    axis: int,
    width: tuple[int, int],
) -> Any:
    before, after = width
    parts: list[Any] = []
    if before:
        edge = value[_pad_axis_slice(value, axis=axis, index=slice(0, 1))]
        parts.append(np.broadcast_to(edge, _pad_axis_shape(value, axis=axis, length=before)))
    parts.append(value)
    if after:
        edge = value[_pad_axis_slice(value, axis=axis, index=slice(-1, None))]
        parts.append(np.broadcast_to(edge, _pad_axis_shape(value, axis=axis, length=after)))
    return np.concatenate(tuple(parts), axis=axis)


def _pad_reflect_axis(
    value: Any,
    *,
    axis: int,
    width: tuple[int, int],
    mode: str,
    reflect_type: str,
) -> Any:
    before, after = width
    size = int(value.shape[axis])
    period = size if mode == "symmetric" else size - 1
    quotient, remainder = np.divmod(np.arange(-before, size + after), period)
    even_period = quotient % 2 == 0
    reflected = period - 1 - remainder if mode == "symmetric" else period - remainder
    indices = np.where(even_period, remainder, reflected)
    source = np.take(value, indices, axis=axis)
    if reflect_type == "even":
        return source

    coefficient_shape = [1] * int(value.ndim)
    coefficient_shape[axis] = indices.size

    def coefficient(values: Any) -> Any:
        return np.reshape(np.asarray(values, dtype=value.dtype), tuple(coefficient_shape))

    source_sign = coefficient(np.where(even_period, 1, -1))
    left_weight = coefficient(np.where(even_period, -quotient, 1 - quotient))
    right_weight = coefficient(np.where(even_period, quotient, quotient + 1))
    left_edge = value[_pad_axis_slice(value, axis=axis, index=slice(0, 1))]
    right_edge = value[_pad_axis_slice(value, axis=axis, index=slice(-1, None))]
    return np.astype(
        source_sign * source + left_weight * left_edge + right_weight * right_edge,
        value.dtype,
    )


def _pad_parameter_axis(
    value: Any,
    *,
    axis: int,
    width: tuple[int, int],
    mode: str,
    parameters: tuple[object, object],
) -> Any:
    parts: list[Any] = []
    for position, (length, endpoint) in enumerate(zip(width, parameters, strict=True)):
        if position == 1:
            parts.append(value)
        if length == 0:
            continue
        if mode == "constant":
            part = np.broadcast_to(
                endpoint,
                _pad_axis_shape(value, axis=axis, length=length),
            )
        else:
            edge_index = slice(0, 1) if position == 0 else slice(-1, None)
            edge = value[_pad_axis_slice(value, axis=axis, index=edge_index)]
            edge = np.squeeze(edge, axis=axis)
            part = np.linspace(
                endpoint,
                edge,
                num=length,
                endpoint=False,
                dtype=value.dtype,
                axis=axis,
            )
            if position == 1:
                part = np.flip(part, axis=axis)
        parts.append(part)
    return np.astype(np.concatenate(tuple(parts), axis=axis), value.dtype)


def _pad_axis_transform(
    value: Any,
    *,
    axis: int,
    width: tuple[int, int],
    mode: str,
    parameters: tuple[object, object] | None,
    reflect_type: str,
) -> Any:
    before, after = width
    if before == 0 and after == 0:
        return value
    size = int(value.shape[axis])
    if mode != "constant" and size == 0:
        msg = f"can't extend empty axis {axis} using modes other than 'constant' or 'empty'"
        raise ValueError(msg)
    if mode == "wrap":
        indices = np.arange(-before, size + after) % size
        return np.take(value, indices, axis=axis)
    if mode == "edge" or (mode in {"reflect", "symmetric"} and size == 1):
        return _pad_edge_axis(value, axis=axis, width=width)
    if mode in {"reflect", "symmetric"}:
        return _pad_reflect_axis(
            value,
            axis=axis,
            width=width,
            mode=mode,
            reflect_type=reflect_type,
        )
    if parameters is None:
        msg = f"numpy.pad mode {mode!r} requires a pair of boundary parameters"
        raise TracingError(msg)
    return _pad_parameter_axis(
        value,
        axis=axis,
        width=width,
        mode=mode,
        parameters=parameters,
    )


def _linear_pad(
    value: Any,
    pad_width: tuple[tuple[int, int], ...],
    *,
    mode: str,
    constant_values: object,
    end_values: object,
    reflect_type: str,
    traced_type: type[TracedArrayLike],
) -> CompositeResult:
    ndim = int(value.ndim)
    widths = _pad_widths(pad_width, ndim)
    if reflect_type not in {"even", "odd"}:
        msg = "numpy.pad reflect_type must be 'even' or 'odd'"
        raise TracingError(msg)
    parameter_values: tuple[tuple[object, object], ...] | None = None
    if mode == "constant":
        parameter_values = _normalize_pad_parameter(
            constant_values,
            ndim=ndim,
            traced_type=traced_type,
            name="constant_values",
        )
    elif mode == "linear_ramp":
        parameter_values = _normalize_pad_parameter(
            end_values,
            ndim=ndim,
            traced_type=traced_type,
            name="end_values",
        )
    result = value
    for axis, width in enumerate(widths):
        result = _pad_axis_transform(
            result,
            axis=axis,
            width=width,
            mode=mode,
            parameters=None if parameter_values is None else parameter_values[axis],
            reflect_type=reflect_type,
        )
    return _finish(result, traced_type=traced_type)


def _pad_widths(
    pad_width: tuple[tuple[int, int], ...],
    ndim: int,
) -> tuple[tuple[int, int], ...]:
    widths = pad_width if len(pad_width) == ndim else pad_width * ndim
    if len(widths) != ndim:
        msg = f"numpy.pad pad_width cannot broadcast to {ndim} dimensions"
        raise TracingError(msg)
    if any(before < 0 or after < 0 for before, after in widths):
        msg = "index can't contain negative values"
        raise ValueError(msg)
    return widths


def _statistical_pad(
    value: Any,
    pad_width: tuple[tuple[int, int], ...],
    *,
    mode: str,
    stat_length: object,
    traced_type: type[TracedArrayLike],
) -> CompositeResult:
    ndim = int(value.ndim)
    widths = _pad_widths(pad_width, ndim)
    lengths: tuple[tuple[object, object], ...] = ((None, None),) * ndim
    # NumPy returns an empty array before it reads stat_length or takes statistics.
    empty = 0 in tuple(value.shape)
    if stat_length is not None and not empty:
        # NumPy rounds stat_length to non-negative indices.
        rounded = np.round(np.asarray(stat_length)).astype(np.intp)
        if np.any(rounded < 0):
            msg = "index can't contain negative values"
            raise ValueError(msg)
        lengths = _normalize_pad_parameter(
            rounded,
            ndim=ndim,
            traced_type=traced_type,
            name="stat_length",
        )
    reducer = _STATISTICAL_PAD_REDUCERS[mode]
    rounds = np.issubdtype(value.dtype, np.integer)
    result: Any = value
    for axis, (width, length_pair) in enumerate(zip(widths, lengths, strict=True)):
        size = int(result.shape[axis])
        if size == 0 and any(width):
            msg = f"can't extend empty axis {axis} using modes other than 'constant' or 'empty'"
            raise ValueError(msg)
        counts = tuple(
            size if length is None else min(int(cast("Any", length)), size)
            for length in length_pair
        )
        if 0 in counts and mode in {"maximum", "minimum"} and not empty:
            msg = "stat_length of 0 yields no value for padding"
            raise ValueError(msg)
        if not any(width):
            continue
        regions = (slice(0, counts[0]), slice(size - counts[1], size))
        parts: list[Any] = []
        for position, (pad_length, region) in enumerate(zip(width, regions, strict=True)):
            if position == 1:
                parts.append(result)
            if pad_length == 0:
                continue
            chunk = result[_pad_axis_slice(result, axis=axis, index=region)]
            statistic = reducer(chunk, axis=axis, keepdims=True)
            if rounds:
                # NumPy rounds statistics before storing them in an integer array.
                statistic = np.round(statistic)
            shape = _pad_axis_shape(result, axis=axis, length=pad_length)
            parts.append(np.broadcast_to(statistic, shape))
        result = np.astype(np.concatenate(tuple(parts), axis=axis), value.dtype)
    return _finish(result, traced_type=traced_type)


def register_creation_handlers(
    handlers: dict[Callable[..., Any], ArrayFunctionHandler],
) -> None:
    """Register constructors that NumPy dispatches through a traced like= value."""
    for function, constructor in (
        (np.array, traced_array),
        (np.asarray, traced_asarray),
        (np.asanyarray, traced_asanyarray),
    ):
        handlers[function] = partial(_constructor_handler, constructor=constructor)
    for function, like_function in (
        (np.zeros, np.zeros_like),
        (np.ones, np.ones_like),
        (np.empty, np.empty_like),
    ):
        handlers[function] = partial(
            _basic_constructor_handler, name=function.__name__, like_function=like_function
        )
    for function, optional in (
        (np.eye, ("M", "k", "dtype", "order")),
        (np.identity, ("dtype",)),
        (np.tri, ("M", "k", "dtype")),
    ):
        handlers[function] = partial(
            _static_constructor_handler, function=function, optional=optional
        )
    handlers[np.full] = _full_handler
    handlers[np.full_like] = _full_like_handler
    handlers[np.linspace] = _linspace_handler
    handlers[np.pad] = _pad_handler
