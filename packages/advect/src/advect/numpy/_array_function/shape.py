"""Shape-related ``__array_function__`` handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as _numpy  # noqa: ICN001 - concrete namespace with dynamic protocol operands

from advect.core._errors import TracingError
from advect.numpy._array_function.emission import (
    _add_backend_node,
    _get_node,
    _get_value,
    _make_unary_shape_handler,
)
from advect.numpy._op_bindings import canonicalize_numpy_op, frontend_lowering
from advect.numpy._static_attr_arrays import encode_static_array_attr

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike
    from advect.numpy._array_function.emission import ArrayFunctionHandler

np: Any = _numpy


def _op_name(suffix: str) -> str:
    return f"numpy.{suffix}"


def _with_backend_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    out = dict(attrs)
    out["_advect_backend"] = "numpy"
    return out


_DIAG_MAX_ARGS = 2
_REPEAT_MIN_ARGS = 2
_REPEAT_MAX_ARGS = 3
_REPEAT_AXIS_POSITIONAL_ARGS = 3
_TILE_NARGS = 2
_RESHAPE_MAX_ARGS = 3
_RESHAPE_ORDER_POSITION = 2
_TRACE_MAX_ARGS = 6
_DIFF_MAX_ARGS = 5


def _reshape_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    """Handle reshape with backend compatibility (shape vs newshape)."""
    if not args or len(args) > _RESHAPE_MAX_ARGS:
        msg = "np.reshape expects (a, shape, order='C') during tracing"
        raise TracingError(msg)
    unsupported = set(kwargs) - {"copy", "newshape", "order", "shape"}
    if unsupported:
        msg = f"np.reshape kwargs not supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)
    a = args[0]

    if len(args) > 1 and ("shape" in kwargs or "newshape" in kwargs):
        msg = "np.reshape received shape twice"
        raise TracingError(msg)
    if "shape" in kwargs and "newshape" in kwargs:
        msg = "np.reshape received both shape= and newshape="
        raise TracingError(msg)
    shape = args[1] if len(args) > 1 else kwargs.get("shape", kwargs.get("newshape"))
    if shape is None:
        msg = "np.reshape requires a shape argument"
        raise ValueError(msg)

    if len(args) > _RESHAPE_ORDER_POSITION and "order" in kwargs:
        msg = "np.reshape received order twice"
        raise TracingError(msg)
    order = (
        args[_RESHAPE_ORDER_POSITION]
        if len(args) > _RESHAPE_ORDER_POSITION
        else kwargs.get("order", "C")
    )
    copy = kwargs.get("copy")

    call_kwargs: dict[str, Any] = {"order": order}
    if copy is not None:
        call_kwargs["copy"] = bool(copy)
    result = np.reshape(_get_value(a, traced_type), shape, **call_kwargs)
    resolved_shape = tuple(int(d) for d in result.shape)
    attrs: dict[str, Any] = {"shape": resolved_shape, "order": order}
    if copy is not None:
        attrs["copy"] = bool(copy)

    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op(_op_name("reshape")),
        inputs=(_get_node(a, graph, traced_type),),
        value=result,
        attrs=_with_backend_attrs(attrs),
        shape=resolved_shape,
        dtype=result.dtype,
    )
    return result, node_id


def _normalize_axis_spec(value: object) -> object:
    """Normalize axis/source/destination specs for JSON-safe attrs."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return int(value.item())
        return tuple(int(item) for item in value.tolist())
    if isinstance(value, (tuple, list, range)):
        return tuple(int(item) for item in value)
    return value


def _diag_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    """Handle ``np.diag`` with traced inputs."""
    unsupported = set(kwargs) - {"k"}
    if unsupported:
        msg = f"{_op_name('diag')} kwargs not yet supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)

    if len(args) not in {1, _DIAG_MAX_ARGS}:
        msg = f"{_op_name('diag')} expects one array argument and optional k during tracing"
        raise TracingError(msg)

    a = args[0]
    k = kwargs.get("k", args[1] if len(args) == _DIAG_MAX_ARGS else 0)
    k_int = int(k)

    result = np.diag(_get_value(a, traced_type), k=k_int)
    attrs: dict[str, Any] = {}
    if k_int != 0:
        attrs["k"] = k_int

    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op(_op_name("diag")),
        inputs=(_get_node(a, graph, traced_type),),
        value=result,
        attrs=_with_backend_attrs(attrs),
        shape=result.shape,
        dtype=result.dtype,
    )
    return result, node_id


def _trace_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    """Handle ``np.trace`` with explicit attr capture."""
    unsupported = set(kwargs) - {"offset", "axis1", "axis2", "dtype", "out"}
    if unsupported:
        msg = f"{_op_name('trace')} kwargs not yet supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)

    if not args or len(args) > _TRACE_MAX_ARGS:
        msg = f"{_op_name('trace')} expects (a, offset, axis1, axis2, dtype, out)"
        raise TracingError(msg)

    positional_names = ("offset", "axis1", "axis2", "dtype", "out")
    values = dict(kwargs)
    for name, value in zip(positional_names, args[1:], strict=False):
        if name in values:
            msg = f"{_op_name('trace')} received {name} twice"
            raise TracingError(msg)
        values[name] = value

    if values.get("out") is not None:
        msg = f"{_op_name('trace')} out= is not supported during tracing"
        raise TracingError(msg)

    a = args[0]
    offset = int(values.get("offset", 0))
    axis1 = int(values.get("axis1", 0))
    axis2 = int(values.get("axis2", 1))
    dtype = values.get("dtype")

    result = np.trace(
        _get_value(a, traced_type),
        offset=offset,
        axis1=axis1,
        axis2=axis2,
        dtype=dtype,
    )
    attrs: dict[str, Any] = {}
    if offset != 0:
        attrs["offset"] = offset
    if axis1 != 0:
        attrs["axis1"] = axis1
    if axis2 != 1:
        attrs["axis2"] = axis2
    if dtype is not None:
        attrs["dtype"] = str(np.dtype(dtype))

    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op(_op_name("trace")),
        inputs=(_get_node(a, graph, traced_type),),
        value=result,
        attrs=_with_backend_attrs(attrs),
        shape=result.shape,
        dtype=result.dtype,
    )
    return result, node_id


def _diff_handler(  # noqa: PLR0912 - exact NumPy signature and operand modes
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    """Handle ``np.diff`` with differentiable prepend/append operands."""
    unsupported = set(kwargs) - {"n", "axis", "prepend", "append"}
    if unsupported:
        msg = f"{_op_name('diff')} kwargs not yet supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)

    if not args or len(args) > _DIFF_MAX_ARGS:
        msg = f"{_op_name('diff')} expects (a, n, axis, prepend, append) during tracing"
        raise TracingError(msg)

    values = dict(kwargs)
    for name, value in zip(("n", "axis", "prepend", "append"), args[1:], strict=False):
        if name in values:
            msg = f"{_op_name('diff')} received {name} twice"
            raise TracingError(msg)
        values[name] = value

    n = int(values.get("n", 1))
    if n < 0:
        msg = f"{_op_name('diff')} requires n >= 0 during tracing (got n={n})"
        raise TracingError(msg)

    axis = int(values.get("axis", -1))
    a = args[0]
    prepend_specified = "prepend" in values
    append_specified = "append" in values
    prepend = values["prepend"] if prepend_specified else None
    append = values["append"] if append_specified else None
    prepend_is_input = prepend_specified and isinstance(prepend, traced_type)
    append_is_input = append_specified and isinstance(append, traced_type)

    diff_kwargs: dict[str, Any] = {"n": n, "axis": axis}
    if prepend_specified:
        diff_kwargs["prepend"] = _get_value(prepend, traced_type)
    if append_specified:
        diff_kwargs["append"] = _get_value(append, traced_type)

    result = np.diff(_get_value(a, traced_type), **diff_kwargs)

    attrs: dict[str, Any] = {
        "n": n,
        "_advect_diff_prepend_input": prepend_is_input,
        "_advect_diff_append_input": append_is_input,
    }
    if axis != -1:
        attrs["axis"] = axis
    if prepend_specified and not prepend_is_input:
        prepend_arr = np.asarray(prepend)
        prepend_scalar = prepend_arr.item() if prepend_arr.ndim == 0 else None
        if isinstance(prepend_scalar, (bool, int, float)):
            attrs["prepend"] = prepend_scalar
        else:
            attrs["prepend"] = encode_static_array_attr(prepend_arr)
    if append_specified and not append_is_input:
        append_arr = np.asarray(append)
        append_scalar = append_arr.item() if append_arr.ndim == 0 else None
        if isinstance(append_scalar, (bool, int, float)):
            attrs["append"] = append_scalar
        else:
            attrs["append"] = encode_static_array_attr(append_arr)

    inputs = [_get_node(a, graph, traced_type)]
    if prepend_is_input:
        inputs.append(_get_node(prepend, graph, traced_type))
    if append_is_input:
        inputs.append(_get_node(append, graph, traced_type))
    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op(_op_name("diff")),
        inputs=tuple(inputs),
        value=result,
        attrs=_with_backend_attrs(attrs),
        shape=result.shape,
        dtype=result.dtype,
    )
    return result, node_id


def _repeat_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    """Handle ``np.repeat`` with scalar integer repeats."""
    unsupported = set(kwargs) - {"axis", "repeats"}
    if unsupported:
        msg = f"{_op_name('repeat')} kwargs not yet supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)

    if not args:
        msg = f"{_op_name('repeat')} expects at least one positional argument during tracing"
        raise TracingError(msg)
    if len(args) > _REPEAT_MAX_ARGS:
        msg = f"{_op_name('repeat')} supports at most three positional arguments during tracing"
        raise TracingError(msg)

    if len(args) >= _REPEAT_MIN_ARGS and "repeats" in kwargs:
        msg = f"{_op_name('repeat')} repeats must be provided positionally or via keyword, not both"
        raise TracingError(msg)
    if len(args) >= _REPEAT_AXIS_POSITIONAL_ARGS and "axis" in kwargs:
        msg = f"{_op_name('repeat')} axis must be provided positionally or via keyword, not both"
        raise TracingError(msg)

    a = args[0]
    if len(args) >= _REPEAT_MIN_ARGS:
        repeats_raw = args[1]
    elif "repeats" in kwargs:
        repeats_raw = kwargs["repeats"]
    else:
        msg = f"{_op_name('repeat')} requires a repeats argument during tracing"
        raise TracingError(msg)

    repeats_arr = np.asarray(repeats_raw)
    if repeats_arr.ndim != 0:
        msg = f"{_op_name('repeat')} supports only scalar repeats during tracing"
        raise TracingError(msg)

    repeats = int(repeats_arr.item())
    axis_raw = args[2] if len(args) == _REPEAT_AXIS_POSITIONAL_ARGS else kwargs.get("axis")
    axis = None if axis_raw is None else int(axis_raw)
    result = np.repeat(_get_value(a, traced_type), repeats, axis=axis)

    attrs: dict[str, Any] = {"repeats": repeats}
    if axis is not None:
        attrs["axis"] = axis
    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op(_op_name("repeat")),
        inputs=(_get_node(a, graph, traced_type),),
        value=result,
        attrs=_with_backend_attrs(attrs),
        shape=result.shape,
        dtype=result.dtype,
    )
    return result, node_id


def _tile_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    """Handle ``np.tile`` with normalized reps attrs."""
    unsupported = set(kwargs) - {"reps"}
    if unsupported:
        msg = f"{_op_name('tile')} kwargs not yet supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)

    if len(args) == _TILE_NARGS:
        if "reps" in kwargs:
            msg = f"{_op_name('tile')} reps must be provided positionally or via keyword, not both"
            raise TracingError(msg)
        a, reps_raw = args
    elif len(args) == 1 and "reps" in kwargs:
        a = args[0]
        reps_raw = kwargs["reps"]
    else:
        msg = f"{_op_name('tile')} expects arguments (A, reps) during tracing"
        raise TracingError(msg)

    reps_obj = _normalize_axis_spec(reps_raw)
    reps: int | tuple[int, ...]
    if isinstance(reps_obj, int):
        reps = reps_obj
    elif isinstance(reps_obj, tuple):
        reps = tuple(int(item) for item in reps_obj)
    else:
        msg = (
            f"{_op_name('tile')} reps must be an int or tuple of ints "
            f"(got {type(reps_obj).__name__})"
        )
        raise TracingError(msg)
    result = np.tile(_get_value(a, traced_type), reps)

    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op(_op_name("tile")),
        inputs=(_get_node(a, graph, traced_type),),
        value=result,
        attrs=_with_backend_attrs({"reps": reps}),
        shape=result.shape,
        dtype=result.dtype,
    )
    return result, node_id


def register_shape_handlers(
    handlers: dict[Callable[..., Any], ArrayFunctionHandler],
) -> None:
    """Register shape-related array functions."""
    handlers[np.reshape] = _reshape_handler
    handlers[np.transpose] = _make_unary_shape_handler(
        np.transpose,
        _op_name("transpose"),
        ("axes",),
        _normalize_axis_spec,
    )
    handlers[np.swapaxes] = _make_unary_shape_handler(
        np.swapaxes,
        "array.swapaxes",
        ("axis1", "axis2"),
        _normalize_axis_spec,
    )
    frontend_lowering("array.swapaxes")(handlers[np.swapaxes])
    handlers[np.broadcast_to] = _make_unary_shape_handler(
        np.broadcast_to,
        _op_name("broadcast_to"),
        ("shape",),
        _normalize_axis_spec,
    )
    handlers[np.flip] = _make_unary_shape_handler(
        np.flip, _op_name("flip"), ("axis",), _normalize_axis_spec
    )
    handlers[np.fliplr] = _make_unary_shape_handler(np.fliplr, _op_name("fliplr"), ())
    handlers[np.flipud] = _make_unary_shape_handler(np.flipud, _op_name("flipud"), ())
    handlers[np.roll] = _make_unary_shape_handler(
        np.roll, _op_name("roll"), ("shift", "axis"), _normalize_axis_spec
    )
    handlers[np.rot90] = _make_unary_shape_handler(
        np.rot90, _op_name("rot90"), ("k", "axes"), _normalize_axis_spec
    )
    handlers[np.rollaxis] = _make_unary_shape_handler(
        np.rollaxis, _op_name("rollaxis"), ("axis", "start"), _normalize_axis_spec
    )
    handlers[np.real] = _make_unary_shape_handler(np.real, _op_name("real"), ())
    handlers[np.imag] = _make_unary_shape_handler(np.imag, _op_name("imag"), ())
    handlers[np.triu] = _make_unary_shape_handler(np.triu, _op_name("triu"), ("k",), int)
    handlers[np.tril] = _make_unary_shape_handler(np.tril, _op_name("tril"), ("k",), int)
    handlers[np.diagonal] = _make_unary_shape_handler(
        np.diagonal,
        _op_name("diagonal"),
        ("offset", "axis1", "axis2"),
        _normalize_axis_spec,
    )
    handlers[np.trace] = _trace_handler
    handlers[np.diag] = _diag_handler
    handlers[np.diff] = _diff_handler
    handlers[np.repeat] = _repeat_handler
    handlers[np.tile] = _tile_handler
    handlers[np.ravel] = _make_unary_shape_handler(np.ravel, _op_name("ravel"), ("order",), str)
    handlers[np.squeeze] = _make_unary_shape_handler(
        np.squeeze, _op_name("squeeze"), ("axis",), _normalize_axis_spec
    )
    handlers[np.expand_dims] = _make_unary_shape_handler(
        np.expand_dims, _op_name("expand_dims"), ("axis",), _normalize_axis_spec
    )
    handlers[np.moveaxis] = _make_unary_shape_handler(
        np.moveaxis,
        _op_name("moveaxis"),
        ("source", "destination"),
        _normalize_axis_spec,
    )
