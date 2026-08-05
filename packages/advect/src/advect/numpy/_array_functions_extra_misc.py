# ruff: noqa: ANN401
# Composite lowerings intentionally accept both concrete arrays and tracers.
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as _numpy  # noqa: ICN001 - typed module and dynamic lowering namespace

from advect.core._errors import TracingError
from advect.core._protocols import _snapshot_traced
from advect.numpy._array_functions_extra_common import (
    _normalize_gradient_axes,
    _normalize_kth,
)
from advect.numpy._array_functions_extra_composite import _finish
from advect.numpy._gradient_lowering import lower_gradient_axis, operand_ndim
from advect.numpy._op_bindings import canonicalize_numpy_op, frontend_lowering
from advect.numpy._protocol_array_function_common import (
    _add_backend_node,
    _get_node,
    _get_value,
    _result_shape_and_dtype,
)

np: Any = _numpy

if TYPE_CHECKING:
    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike
    from advect.numpy._protocol_array_function_common import ArrayFunctionResult

_ANGLE_POSITIONAL_ARGS = 2
_TAKE_ARGS = 2
_TAKE_ALONG_AXIS_ARGS = 3


def _is_traced(value: object, traced_type: type[TracedArrayLike]) -> bool:
    return isinstance(value, traced_type) or callable(getattr(value, "_advect_snapshot", None))


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
        result = value
        for replacement in (nan, posinf, neginf):
            if replacement is not None:
                result = result + replacement * 0
        return result
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
    if len(args) not in {1, 2}:
        msg = "numpy.angle expects (z, deg=False) during tracing"
        raise TracingError(msg)
    unsupported = set(kwargs) - {"deg"}
    if unsupported:
        msg = f"numpy.angle kwargs not supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)
    if len(args) == _ANGLE_POSITIONAL_ARGS and "deg" in kwargs:
        msg = "numpy.angle received deg twice"
        raise TracingError(msg)

    x = args[0]
    deg = bool(args[1] if len(args) == _ANGLE_POSITIONAL_ARGS else kwargs.get("deg", False))
    result = np.angle(_get_value(x, traced_type), deg=deg)
    attrs: dict[str, Any] = {}
    if deg:
        attrs["deg"] = True
    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op("numpy.angle"),
        inputs=(_get_node(x, graph, traced_type),),
        value=result,
        attrs=attrs,
        shape=result.shape,
        dtype=result.dtype,
    )
    return result, node_id


def _nan_to_num_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    positional_names = ("copy", "nan", "posinf", "neginf")
    if not args or len(args) > len(positional_names) + 1:
        msg = "numpy.nan_to_num expects (x, copy=True, nan=0, posinf=None, neginf=None)"
        raise TracingError(msg)
    unsupported = set(kwargs) - set(positional_names)
    if unsupported:
        msg = f"numpy.nan_to_num kwargs not supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)
    values = dict(kwargs)
    for name, value in zip(positional_names, args[1:], strict=False):
        if name in values:
            msg = f"numpy.nan_to_num received {name} twice"
            raise TracingError(msg)
        values[name] = value

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
    if any(_is_traced(value, traced_type) for value in (nan, posinf, neginf) if value is not None):
        result = _differentiable_nan_to_num(
            x,
            nan=nan,
            posinf=posinf,
            neginf=neginf,
        )
        node_id, result_value = _snapshot_traced(result)
        return result_value, int(node_id)

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

    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op("numpy.nan_to_num"),
        inputs=(_get_node(x, graph, traced_type),),
        value=result,
        attrs=attrs,
        shape=result.shape,
        dtype=result.dtype,
    )
    return result, node_id


def _sinc_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    if len(args) != 1:
        msg = "numpy.sinc expects one input array during tracing"
        raise TracingError(msg)
    if kwargs:
        msg = f"numpy.sinc kwargs not supported during tracing: {sorted(kwargs)}"
        raise TracingError(msg)

    x = args[0]
    result = np.sinc(_get_value(x, traced_type))
    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op("numpy.sinc"),
        inputs=(_get_node(x, graph, traced_type),),
        value=result,
        attrs={},
        shape=result.shape,
        dtype=result.dtype,
    )
    return result, node_id


@frontend_lowering("advect.copy")
def _copy_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    positional_names = ("order", "subok")
    if not args or len(args) > len(positional_names) + 1:
        msg = "numpy.copy expects (a, order='K', subok=False) during tracing"
        raise TracingError(msg)
    unsupported = set(kwargs) - set(positional_names)
    if unsupported:
        msg = f"numpy.copy kwargs not supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)
    values = dict(kwargs)
    for name, value in zip(positional_names, args[1:], strict=False):
        if name in values:
            msg = f"numpy.copy received {name} twice"
            raise TracingError(msg)
        values[name] = value
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
    result_shape, result_dtype = _result_shape_and_dtype(result)
    node_id = _add_backend_node(
        graph=graph,
        op="advect.copy",
        inputs=(_get_node(x, graph, traced_type),),
        value=result,
        attrs={"order": order, "_advect_backend": "numpy"},
        shape=result_shape,
        dtype=result_dtype,
    )
    return result, node_id


def _take_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    positional_names = ("axis", "out", "mode")
    if len(args) < _TAKE_ARGS or len(args) > _TAKE_ARGS + len(positional_names):
        msg = "numpy.take expects (a, indices, axis=None, out=None, mode='raise')"
        raise TracingError(msg)
    unsupported = set(kwargs) - set(positional_names)
    if unsupported:
        msg = f"numpy.take kwargs not supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)
    values = dict(kwargs)
    for name, value in zip(positional_names, args[_TAKE_ARGS:], strict=False):
        if name in values:
            msg = f"numpy.take received {name} twice"
            raise TracingError(msg)
        values[name] = value
    if values.get("out") is not None:
        msg = "numpy.take positional out= is not supported; pass out= by keyword"
        raise TracingError(msg)
    mode = str(values.get("mode", "raise"))
    if mode not in {"raise", "wrap", "clip"}:
        msg = "numpy.take mode must be raise, wrap, or clip"
        raise TracingError(msg)

    source, indices = args[:2]
    axis_value = values.get("axis")
    axis = None if axis_value is None else int(axis_value)
    result = np.take(
        _get_value(source, traced_type),
        _get_value(indices, traced_type),
        axis=axis,
        mode=mode,
    )
    shape, dtype = _result_shape_and_dtype(result)
    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op("numpy.take"),
        inputs=(
            _get_node(source, graph, traced_type),
            _get_node(indices, graph, traced_type),
        ),
        value=result,
        attrs={"axis": axis, "mode": mode},
        shape=shape,
        dtype=dtype,
    )
    return result, node_id


def _take_along_axis_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    if len(args) not in {2, 3}:
        msg = "numpy.take_along_axis expects (arr, indices, axis) during tracing"
        raise TracingError(msg)
    unsupported = set(kwargs) - {"axis"}
    if unsupported:
        msg = f"numpy.take_along_axis kwargs not supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)
    if len(args) == _TAKE_ALONG_AXIS_ARGS:
        if "axis" in kwargs:
            msg = "numpy.take_along_axis axis was provided twice"
            raise TracingError(msg)
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
    shape, dtype = _result_shape_and_dtype(result)
    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op("numpy.take_along_axis"),
        inputs=(
            _get_node(source, graph, traced_type),
            _get_node(indices, graph, traced_type),
        ),
        value=result,
        attrs={"axis": axis},
        shape=shape,
        dtype=dtype,
    )
    return result, node_id


def _sort_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    positional_names = ("axis", "kind", "order")
    if not args or len(args) > len(positional_names) + 1:
        msg = "numpy.sort expects (a, axis=-1, kind=None, order=None, *, stable=None)"
        raise TracingError(msg)
    unsupported = set(kwargs) - {*positional_names, "stable"}
    if unsupported:
        msg = f"numpy.sort kwargs not supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)
    values = dict(kwargs)
    for name, value in zip(positional_names, args[1:], strict=False):
        if name in values:
            msg = f"numpy.sort received {name} twice"
            raise TracingError(msg)
        values[name] = value

    x = args[0]
    axis = int(values.get("axis", -1))
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

    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op("numpy.sort"),
        inputs=(_get_node(x, graph, traced_type),),
        value=result,
        attrs=attrs,
        shape=result.shape,
        dtype=result.dtype,
    )
    return result, node_id


def _partition_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    positional_names = ("kth", "axis", "kind", "order")
    if not args or len(args) > len(positional_names) + 1:
        msg = "numpy.partition expects (a, kth) during tracing"
        raise TracingError(msg)
    unsupported = set(kwargs) - set(positional_names)
    if unsupported:
        msg = f"numpy.partition kwargs not supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)

    values = dict(kwargs)
    for name, value in zip(positional_names, args[1:], strict=False):
        if name in values:
            msg = f"numpy.partition received {name} twice"
            raise TracingError(msg)
        values[name] = value
    x = args[0]
    if "kth" in values:
        kth = _normalize_kth(values["kth"])
    else:
        msg = "numpy.partition requires kth during tracing"
        raise TracingError(msg)
    axis = int(values.get("axis", -1))
    kind = values.get("kind", "introselect")
    order = values.get("order")

    result = np.partition(_get_value(x, traced_type), kth=kth, axis=axis, kind=kind, order=order)

    attrs: dict[str, Any] = {"kth": kth, "axis": axis}
    if kind is not None:
        attrs["kind"] = str(kind)
    if order is not None:
        attrs["order"] = order

    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op("numpy.partition"),
        inputs=(_get_node(x, graph, traced_type),),
        value=result,
        attrs=attrs,
        shape=result.shape,
        dtype=result.dtype,
    )
    return result, node_id


def _gradient_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ArrayFunctionResult:
    if not args:
        msg = "numpy.gradient requires an input array during tracing"
        raise TracingError(msg)
    unsupported = set(kwargs) - {"axis", "edge_order"}
    if unsupported:
        msg = f"numpy.gradient kwargs not supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)

    a = args[0]
    axis = kwargs.get("axis")
    edge_order = cast("Literal[1, 2]", int(kwargs.get("edge_order", 1)))
    if edge_order not in {1, 2}:
        msg = f"numpy.gradient edge_order must be 1 or 2, got {edge_order}"
        raise TracingError(msg)

    a_value = cast("Any", _get_value(a, traced_type))
    axes = _normalize_gradient_axes(axis=axis, ndim=a_value.ndim)
    spacings = args[1:]
    if spacings:
        if len(spacings) == 1 and operand_ndim(spacings[0]) == 0:
            normalized_spacings = spacings * len(axes)
        elif len(spacings) == len(axes):
            normalized_spacings = spacings
        else:
            msg = (
                "numpy.gradient requires one scalar spacing or one spacing per "
                f"gradient axis; got {len(spacings)} spacings for {len(axes)} axes"
            )
            raise TracingError(msg)
        outputs = tuple(
            lower_gradient_axis(
                np,
                a,
                spacing,
                axis=out_axis,
                edge_order=edge_order,
            )
            for out_axis, spacing in zip(axes, normalized_spacings, strict=True)
        )
        composite = outputs[0] if len(outputs) == 1 else outputs
        return _finish(composite, traced_type=traced_type)

    axis_arg: int | tuple[int, ...] = axes[0] if len(axes) == 1 else axes

    result = np.gradient(a_value, axis=axis_arg, edge_order=edge_order)
    outputs = tuple(result) if isinstance(result, (list, tuple)) else (result,)
    if len(outputs) != len(axes):
        msg = (
            "numpy.gradient tracing expected output count to match axis count "
            f"(got {len(outputs)} outputs for axes={axes})"
        )
        raise TracingError(msg)

    input_id = _get_node(a, graph, traced_type)
    node_ids: list[int] = []
    for out_axis, output in zip(axes, outputs, strict=True):
        output_shape, output_dtype = _result_shape_and_dtype(output)
        node_id = _add_backend_node(
            graph=graph,
            op=canonicalize_numpy_op("numpy.gradient"),
            inputs=(input_id,),
            value=output,
            attrs={"axis": out_axis, "edge_order": edge_order},
            shape=output_shape,
            dtype=output_dtype,
        )
        node_ids.append(node_id)

    if len(outputs) == 1:
        return outputs[0], node_ids[0]
    return outputs, tuple(node_ids)
