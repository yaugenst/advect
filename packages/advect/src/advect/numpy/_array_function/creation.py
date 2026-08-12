# ruff: noqa: ANN401
# Composite lowerings intentionally accept both concrete arrays and tracers.
"""Trace NumPy array-creation functions as canonical operations or compositions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as _numpy  # noqa: ICN001 - typed module and dynamic lowering namespace

from advect.core._errors import TracingError
from advect.core._protocols import _snapshot_traced
from advect.numpy._array_function.composite import (
    _finish,
    _first_traced,
    _lift_composite_constant,
)
from advect.numpy._array_function.emission import _add_backend_node, _get_node, _get_value
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
from advect.numpy._op_bindings import canonicalize_numpy_op
from advect.numpy._static_attr_arrays import encode_static_array_attr

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
_LINEAR_PAD_MODES = frozenset({"constant", "edge", "linear_ramp", "reflect", "symmetric", "wrap"})
_STATISTICAL_PAD_MODES = frozenset({"maximum", "mean", "median", "minimum"})
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
    like = values.get("like")

    full_fn = cast("Callable[..., Any]", np.full)
    call_kwargs: dict[str, Any] = {"dtype": dtype, "order": order}
    if device is not None:
        call_kwargs["device"] = device
    if like is not None:
        like_array = _get_value(like, traced_type)
        call_kwargs["like"] = like_array
    else:
        like_array = None
    result = full_fn(shape, _get_value(fill_value, traced_type), **call_kwargs)

    attrs: dict[str, Any] = {"shape": shape}
    if dtype is not None:
        attrs["dtype"] = str(np.dtype(dtype))
    if order != "C":
        attrs["order"] = order
    if device is not None:
        attrs["device"] = device
    if like_array is not None and not isinstance(like, traced_type):
        attrs["like"] = encode_static_array_attr(like_array)

    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op("numpy.full"),
        inputs=(_get_node(fill_value, graph, traced_type),),
        value=result,
        attrs=attrs,
        shape=result.shape,
        dtype=result.dtype,
    )
    return result, node_id


def _like_anchor(
    values: dict[str, Any],
    *,
    traced_type: type[TracedArrayLike],
    name: str,
) -> TracedArrayLike:
    anchor = _first_traced(values.get("like"), traced_type=traced_type)
    if anchor is None:
        msg = (
            f"numpy.{name} tracing requires like= to be a traced array; "
            "constructor shape metadata must remain static"
        )
        raise TracingError(msg)
    return anchor


def _basic_constructor_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    name: str,
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
    anchor = _like_anchor(values, traced_type=traced_type, name=name)
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
    function = {
        "zeros": np.zeros_like,
        "ones": np.ones_like,
        "empty": np.empty_like,
    }[name]
    return _finish(
        function(anchor, **like_kwargs),
        traced_type=traced_type,
    )


def _eye_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    values = _bind_optional_positionals(
        name="eye",
        args=args,
        kwargs=kwargs,
        required=1,
        optional=("M", "k", "dtype", "order"),
        keyword_only=_CONSTRUCTOR_KEYWORD_ONLY,
    )
    anchor = _like_anchor(values, traced_type=traced_type, name="eye")
    rows = int(args[0])
    columns = rows if values.get("M") is None else int(values["M"])
    diagonal = int(values.get("k", 0))
    dtype = values.get("dtype", float)
    order = str(values.get("order", "C"))
    concrete = np.eye(rows, columns, k=diagonal, dtype=dtype, order=order)
    result = np.zeros_like(
        anchor,
        dtype=concrete.dtype,
        order=order,
        shape=concrete.shape,
        device=values.get("device"),
    )
    return _finish(result + concrete, traced_type=traced_type)


def _identity_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    values = _bind_optional_positionals(
        name="identity",
        args=args,
        kwargs=kwargs,
        required=1,
        optional=("dtype",),
        keyword_only=_CONSTRUCTOR_KEYWORD_ONLY,
    )
    anchor = _like_anchor(values, traced_type=traced_type, name="identity")
    concrete = np.identity(int(args[0]), dtype=values.get("dtype"))
    result = np.zeros_like(
        anchor,
        dtype=concrete.dtype,
        shape=concrete.shape,
        device=values.get("device"),
    )
    return _finish(result + concrete, traced_type=traced_type)


def _tri_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    values = _bind_optional_positionals(
        name="tri",
        args=args,
        kwargs=kwargs,
        required=1,
        optional=("M", "k", "dtype"),
        keyword_only=_CONSTRUCTOR_KEYWORD_ONLY,
    )
    anchor = _like_anchor(values, traced_type=traced_type, name="tri")
    rows = int(args[0])
    columns = rows if values.get("M") is None else int(values["M"])
    concrete = np.tri(
        rows,
        columns,
        k=int(values.get("k", 0)),
        dtype=values.get("dtype", float),
    )
    result = np.zeros_like(
        anchor,
        dtype=concrete.dtype,
        shape=concrete.shape,
        device=values.get("device"),
    )
    return _finish(result + concrete, traced_type=traced_type)


def register_creation_handlers(
    handlers: dict[Callable[..., Any], ArrayFunctionHandler],
) -> None:
    """Register constructors that NumPy dispatches through a traced like= value."""
    handlers[np.array] = lambda _graph, traced_type, args, kwargs: _finish(
        traced_array(*args, **kwargs),
        traced_type=traced_type,
    )
    handlers[np.asarray] = lambda _graph, traced_type, args, kwargs: _finish(
        traced_asarray(*args, **kwargs),
        traced_type=traced_type,
    )
    handlers[np.asanyarray] = lambda _graph, traced_type, args, kwargs: _finish(
        traced_asanyarray(*args, **kwargs),
        traced_type=traced_type,
    )
    handlers[np.zeros] = lambda graph, traced_type, args, kwargs: _basic_constructor_handler(
        graph,
        traced_type,
        args,
        kwargs,
        name="zeros",
    )
    handlers[np.ones] = lambda graph, traced_type, args, kwargs: _basic_constructor_handler(
        graph,
        traced_type,
        args,
        kwargs,
        name="ones",
    )
    handlers[np.empty] = lambda graph, traced_type, args, kwargs: _basic_constructor_handler(
        graph,
        traced_type,
        args,
        kwargs,
        name="empty",
    )
    handlers[np.eye] = _eye_handler
    handlers[np.identity] = _identity_handler
    handlers[np.tri] = _tri_handler


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

    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op("numpy.full_like"),
        inputs=(
            _get_node(a, graph, traced_type),
            _get_node(fill_value, graph, traced_type),
        ),
        value=result,
        attrs=attrs,
        shape=result.shape,
        dtype=result.dtype,
    )
    return result, node_id


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

    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op("numpy.linspace"),
        inputs=(
            _get_node(start, graph, traced_type),
            _get_node(stop, graph, traced_type),
        ),
        value=result,
        attrs=attrs,
        shape=result.shape,
        dtype=result.dtype,
    )
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
    mode_parameters = {
        "constant": frozenset({"constant_values"}),
        "linear_ramp": frozenset({"end_values"}),
        "reflect": frozenset({"reflect_type"}),
        "symmetric": frozenset({"reflect_type"}),
        "maximum": frozenset({"stat_length"}),
        "mean": frozenset({"stat_length"}),
        "median": frozenset({"stat_length"}),
        "minimum": frozenset({"stat_length"}),
    }.get(mode, frozenset())
    unsupported = set(kwargs) - ({"mode"} | set(mode_parameters))
    if unsupported:
        msg = f"numpy.pad kwargs not supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)

    x = args[0]
    pad_width = _normalize_pad_width(args[1])
    constant_values = kwargs.get("constant_values", 0)

    if mode not in _LINEAR_PAD_MODES | _STATISTICAL_PAD_MODES:
        msg = (
            f"numpy.pad(mode={mode!r}) has a data-dependent nonlinear padding rule "
            "and is not differentiable through Advect's NumPy frontend"
        )
        raise TracingError(msg)
    if mode in _STATISTICAL_PAD_MODES:
        return _statistical_pad(
            x,
            pad_width,
            mode=mode,
            stat_length=kwargs.get("stat_length"),
            traced_type=traced_type,
        )
    parameter_is_traced = _first_traced(
        (constant_values, kwargs.get("end_values")),
        traced_type=traced_type,
    )
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

    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op("numpy.pad"),
        inputs=(_get_node(x, graph, traced_type),),
        value=result,
        attrs=attrs,
        shape=result.shape,
        dtype=result.dtype,
    )
    return result, node_id


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
    widths = pad_width if len(pad_width) == ndim else pad_width * ndim
    if len(widths) != ndim:
        msg = f"numpy.pad pad_width cannot broadcast to {ndim} dimensions"
        raise TracingError(msg)
    if any(before < 0 or after < 0 for before, after in widths):
        msg = "numpy.pad pad_width cannot contain negative values"
        raise TracingError(msg)
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


def _normalize_stat_lengths(
    value: object,
    *,
    ndim: int,
) -> tuple[tuple[int | None, int | None], ...]:
    if value is None:
        return ((None, None),) * ndim
    array = np.asarray(value)
    try:
        broadcast = np.broadcast_to(array, (ndim, 2))
    except ValueError as error:
        msg = f"numpy.pad stat_length shape {array.shape} cannot broadcast to ({ndim}, 2)"
        raise TracingError(msg) from error
    return tuple((int(row[0]), int(row[1])) for row in broadcast)


def _statistical_pad(
    value: Any,
    pad_width: tuple[tuple[int, int], ...],
    *,
    mode: str,
    stat_length: object,
    traced_type: type[TracedArrayLike],
) -> CompositeResult:
    ndim = int(value.ndim)
    widths = pad_width if len(pad_width) == ndim else pad_width * ndim
    if len(widths) != ndim:
        msg = f"numpy.pad pad_width cannot broadcast to {ndim} dimensions"
        raise TracingError(msg)
    lengths = _normalize_stat_lengths(stat_length, ndim=ndim)
    reducer = {
        "maximum": np.max,
        "mean": np.mean,
        "median": np.median,
        "minimum": np.min,
    }[mode]
    result: Any = value
    for axis, ((before, after), (before_length, after_length)) in enumerate(
        zip(widths, lengths, strict=True)
    ):
        size = int(result.shape[axis])
        before_count = size if before_length is None else min(before_length, size)
        after_count = size if after_length is None else min(after_length, size)
        before_index = [slice(None)] * ndim
        after_index = [slice(None)] * ndim
        before_index[axis] = slice(0, before_count)
        after_index[axis] = slice(size - after_count, size)
        before_value = reducer(result[tuple(before_index)], axis=axis, keepdims=True)
        after_value = reducer(result[tuple(after_index)], axis=axis, keepdims=True)
        before_shape = [int(length) for length in result.shape]
        after_shape = list(before_shape)
        before_shape[axis] = before
        after_shape[axis] = after
        result = np.concatenate(
            (
                np.broadcast_to(before_value, tuple(before_shape)),
                result,
                np.broadcast_to(after_value, tuple(after_shape)),
            ),
            axis=axis,
        )
        result = np.astype(result, value.dtype)
    return _finish(result, traced_type=traced_type)
