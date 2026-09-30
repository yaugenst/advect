# ruff: noqa: ANN401
# Composite lowerings intentionally accept both concrete arrays and tracers.
"""Trace miscellaneous NumPy array functions through canonical operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as _numpy  # noqa: ICN001 - typed module and dynamic lowering namespace

from advect.core._errors import TracingError
from advect.core._protocols import _snapshot_traced
from advect.numpy._array_function.composite import _finish
from advect.numpy._array_function.emission import _emit, _get_value, _is_traced_operand
from advect.numpy._array_function.normalization import (
    _normalize_gradient_axes,
    _normalize_kth,
)
from advect.numpy._composite_lowering import lower_gradient, operand_ndim
from advect.numpy._op_bindings import frontend_lowering
from advect.numpy._signature import ascending_sort_kwargs, take_mode

np: Any = _numpy

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike
    from advect.numpy._array_function.emission import ArrayFunctionHandler, ArrayFunctionResult

_ANGLE_POSITIONAL_ARGS = 2
_TAKE_ARGS = 2
_TAKE_ALONG_AXIS_ARGS = 3


def _clean_nonfinite_component(
    component: Any,
    *,
    nan: Any,
    posinf: Any,
    neginf: Any,
) -> Any:
    without_nan = np.where(np.isnan(component), nan, component)
    infinity_value = np.where(component > 0, posinf, neginf)
    return np.where(np.isinf(component), infinity_value, without_nan)


def _differentiable_nan_to_num(
    value: Any,
    *,
    nan: Any | None,
    posinf: Any | None,
    neginf: Any | None,
) -> Any:
    dtype = np.dtype(value.dtype)
    if not np.issubdtype(dtype, np.inexact):
        # NumPy returns non-inexact input unchanged; replacements only join the trace.
        result = value
        for replacement in (nan, posinf, neginf):
            if replacement is not None:
                result = result + np.zeros_like(replacement, dtype=dtype, shape=())
        return np.astype(result, dtype)
    real_dtype = np.empty((), dtype=dtype).real.dtype
    limit = np.finfo(real_dtype).max
    nan_value = 0.0 if nan is None else nan
    positive_value = limit if posinf is None else posinf
    negative_value = -limit if neginf is None else neginf
    if np.issubdtype(dtype, np.complexfloating):
        real = _clean_nonfinite_component(
            np.real(value),
            nan=nan_value,
            posinf=positive_value,
            neginf=negative_value,
        )
        imaginary = _clean_nonfinite_component(
            np.imag(value),
            nan=nan_value,
            posinf=positive_value,
            neginf=negative_value,
        )
        return real + imaginary * 1j
    return _clean_nonfinite_component(
        value,
        nan=nan_value,
        posinf=positive_value,
        neginf=negative_value,
    )


def _angle_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    x = args[0]
    deg = bool(args[1] if len(args) == _ANGLE_POSITIONAL_ARGS else kwargs.get("deg", False))
    result = np.angle(_get_value(x, traced_type), deg=deg)
    attrs: dict[str, Any] = {}
    if deg:
        attrs["deg"] = True
    return _emit(graph, traced_type, "numpy.angle", (x,), result, attrs)


def _nan_to_num_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    positional_names = ("copy", "nan", "posinf", "neginf")
    values = dict(zip(positional_names, args[1:], strict=False)) | kwargs

    x = args[0]
    copy = bool(values.get("copy", True))
    if not copy:
        msg = (
            "numpy.nan_to_num(copy=False) mutates its input and is not supported during "
            "tracing. Use copy=True and rebind the returned value."
        )
        raise TracingError(msg)
    nan = values.get("nan")
    posinf = values.get("posinf")
    neginf = values.get("neginf")
    if any(_is_traced_operand(value, traced_type) for value in (nan, posinf, neginf)):
        result = _differentiable_nan_to_num(
            x,
            nan=nan,
            posinf=posinf,
            neginf=neginf,
        )
        node_id, result_value = _snapshot_traced(result)
        return result_value, node_id

    call_kwargs: dict[str, Any] = {"copy": copy}
    if nan is not None:
        call_kwargs["nan"] = nan
    if posinf is not None:
        call_kwargs["posinf"] = posinf
    if neginf is not None:
        call_kwargs["neginf"] = neginf
    result = np.nan_to_num(_get_value(x, traced_type), **call_kwargs)

    attrs: dict[str, Any] = {"copy": copy}
    if nan is not None:
        attrs["nan"] = nan
    if posinf is not None:
        attrs["posinf"] = posinf
    if neginf is not None:
        attrs["neginf"] = neginf

    return _emit(graph, traced_type, "numpy.nan_to_num", (x,), result, attrs)


def _sinc_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    _kwargs: dict[str, Any],
) -> tuple[Any, int]:
    x = args[0]
    result = np.sinc(_get_value(x, traced_type))
    return _emit(graph, traced_type, "numpy.sinc", (x,), result)


@frontend_lowering("advect.copy")
def _copy_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    positional_names = ("order", "subok")
    values = dict(zip(positional_names, args[1:], strict=False)) | kwargs
    order = str(values.get("order", "K"))
    subok = bool(values.get("subok", False))
    if subok:
        msg = (
            "numpy.copy(subok=True) is not supported during tracing because "
            "durable programs do not preserve ndarray subclass identity"
        )
        raise TracingError(msg)

    x = args[0]
    result = np.copy(_get_value(x, traced_type), order=order, subok=subok)
    return _emit(
        graph,
        traced_type,
        "advect.copy",
        (x,),
        result,
        {"order": order},
    )


def _take_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    values = dict(zip(("axis",), args[_TAKE_ARGS:], strict=False)) | kwargs
    mode = take_mode(values.get("mode", "raise"))

    source, indices = args[:2]
    axis_value = values.get("axis")
    axis = None if axis_value is None else int(axis_value)
    result = np.take(
        _get_value(source, traced_type),
        _get_value(indices, traced_type),
        axis=axis,
        mode=mode,
    )
    return _emit(
        graph, traced_type, "numpy.take", (source, indices), result, {"axis": axis, "mode": mode}
    )


def _take_along_axis_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    if len(args) == _TAKE_ALONG_AXIS_ARGS:
        axis_value = args[2]
    elif "axis" in kwargs:
        axis_value = kwargs["axis"]
    else:
        msg = "numpy.take_along_axis requires axis during tracing"
        raise TracingError(msg)
    if axis_value is None:
        msg = "numpy.take_along_axis axis=None is not supported during tracing"
        raise TracingError(msg)
    axis = int(axis_value)

    source, indices = args[:2]
    result = np.take_along_axis(
        _get_value(source, traced_type),
        _get_value(indices, traced_type),
        axis=axis,
    )
    return _emit(
        graph, traced_type, "numpy.take_along_axis", (source, indices), result, {"axis": axis}
    )


def _flatten_for_axis_none(x: Any, axis: Any) -> tuple[Any, int]:
    """Apply NumPy's ``axis=None``: order the flattened array along its only axis."""
    # reshape, unlike ravel, also stages inside a staged derivative program.
    return (np.reshape(x, (-1,)), -1) if axis is None else (x, int(axis))


def _sort_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    positional_names = ("axis", "kind", "order")
    kwargs = ascending_sort_kwargs("sort", kwargs)
    unsupported = set(kwargs) - {*positional_names, "stable"}
    if unsupported:
        msg = f"numpy.sort kwargs not supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)
    values = dict(zip(positional_names, args[1:], strict=False)) | kwargs

    x, axis = _flatten_for_axis_none(args[0], values.get("axis", -1))
    kind = values.get("kind")
    order = values.get("order")
    stable = values.get("stable")
    result = np.sort(
        _get_value(x, traced_type),
        axis=axis,
        kind=kind,
        order=order,
        stable=stable,
    )

    attrs: dict[str, Any] = {"axis": axis}
    if kind is not None:
        attrs["kind"] = str(kind)
    if order is not None:
        attrs["order"] = order
    if stable is not None:
        attrs["stable"] = bool(stable)

    return _emit(graph, traced_type, "numpy.sort", (x,), result, attrs)


def _partition_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    positional_names = ("kth", "axis", "kind", "order")
    values = dict(zip(positional_names, args[1:], strict=False)) | kwargs
    x, axis = _flatten_for_axis_none(args[0], values.get("axis", -1))
    kth = _normalize_kth(values["kth"])
    kind = values.get("kind", "introselect")
    order = values.get("order")

    result = np.partition(_get_value(x, traced_type), kth=kth, axis=axis, kind=kind, order=order)

    attrs: dict[str, Any] = {"kth": kth, "axis": axis}
    if kind is not None:
        attrs["kind"] = str(kind)
    if order is not None:
        attrs["order"] = order

    return _emit(graph, traced_type, "numpy.partition", (x,), result, attrs)


def _gradient_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ArrayFunctionResult:
    a = args[0]
    axis = kwargs.get("axis")
    edge_order = cast("Literal[1, 2]", int(kwargs.get("edge_order", 1)))
    if edge_order not in {1, 2}:
        msg = f"numpy.gradient edge_order must be 1 or 2, got {edge_order}"
        raise TracingError(msg)

    a_value = _get_value(a, traced_type)
    axes = _normalize_gradient_axes(axis=axis, ndim=operand_ndim(a_value))
    if len(args) > 1:
        return _finish(
            lower_gradient(a, args[1:], axes=axes, edge_order=edge_order, error=TracingError),
            traced_type=traced_type,
        )

    axis_arg: int | tuple[int, ...] = axes[0] if len(axes) == 1 else axes

    result = np.gradient(a_value, axis=axis_arg, edge_order=edge_order)
    outputs = tuple(result) if isinstance(result, (list, tuple)) else (result,)

    node_ids = [
        _emit(
            graph,
            traced_type,
            "numpy.gradient",
            (a,),
            output,
            {"axis": out_axis, "edge_order": edge_order},
        )[1]
        for out_axis, output in zip(axes, outputs, strict=True)
    ]
    if len(outputs) == 1:
        return outputs[0], node_ids[0]
    return outputs, tuple(node_ids)


def register_misc_handlers(
    handlers: dict[Callable[..., Any], ArrayFunctionHandler],
) -> None:
    """Register the remaining single-purpose array functions."""
    handlers[np.angle] = _angle_handler
    handlers[np.nan_to_num] = _nan_to_num_handler
    handlers[np.sinc] = _sinc_handler
    handlers[np.copy] = _copy_handler
    handlers[np.take] = _take_handler
    handlers[np.take_along_axis] = _take_along_axis_handler
    handlers[np.sort] = _sort_handler
    handlers[np.partition] = _partition_handler
    handlers[np.gradient] = _gradient_handler
