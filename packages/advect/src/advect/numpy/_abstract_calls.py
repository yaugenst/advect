# ruff: noqa: C901, PLR0911, PLR0912, PLR0915, PLR2004
"""NumPy calling conventions for payload-free staged arrays."""

from __future__ import annotations

import math
import operator
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from advect.core._abstract import AbstractArray, _lift, _record_abstract_op
from advect.core._abstract_helpers import dtype_name, normalize_axis, shape_tuple
from advect.core._array_api.results import restore_array_api_result
from advect.core._errors import MutationError, TracingError
from advect.core._registry import get_registry
from advect.numpy._composite_lowering import (
    NON_SCALAR_INITIAL,
    REDUCTIONS,
    lower_average,
    lower_compress,
    lower_controlled_reduction,
    lower_cumulative_initial,
    lower_gradient,
    lower_matrix_power,
    operand_dtype,
)
from advect.numpy._constructors import _normalize_order
from advect.numpy._op_bindings import staged_numpy_op
from advect.numpy._signature import (
    CLIP_KEYWORDS,
    ascending_sort_kwargs,
    bind_clip_bounds,
    take_mode,
)
from advect.numpy._staged_out import validate_staged_out

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from advect.core._abstract import AbstractTrace

_CASTING_RULES = frozenset({"no", "equiv", "safe", "same_kind", "unsafe"})


def _empty_out(value: object) -> bool:
    return value is None or (isinstance(value, tuple) and len(value) == 1 and value[0] is None)


def can_cast_dtype(source: object, target: object, *, casting: str) -> bool:
    """Apply NumPy's casting relation at the frontend boundary."""
    if casting not in _CASTING_RULES:
        raise ValueError(f"Unknown NumPy casting rule {casting!r}")
    return np.can_cast(cast("Any", source), cast("Any", target), casting=cast("Any", casting))


def _record_numpy(
    trace: AbstractTrace,
    raw_name: str,
    operands: Sequence[object],
    attrs: Mapping[str, object],
) -> AbstractArray | tuple[AbstractArray, ...]:
    """Record an already-bound NumPy call through core's canonical boundary."""
    return _record_abstract_op(
        trace,
        staged_numpy_op(raw_name),
        operands,
        attrs,
        graph_attrs={"_advect_backend": "numpy"},
    )


def _numpy_array(
    trace: AbstractTrace,
    raw_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> AbstractArray:
    result = apply_numpy(trace, raw_name, args, kwargs)
    if not isinstance(result, AbstractArray):
        raise TypeError(f"Single-output abstract operation {raw_name!r} returned a tuple")
    return result


def _functionalize_out(
    trace: AbstractTrace,
    raw_name: str,
    raw_args: tuple[Any, ...],
    kwargs: dict[str, Any],
    out: object,
) -> AbstractArray:
    destinations = out if isinstance(out, tuple) else (out,)
    tuple_out = isinstance(out, tuple)
    tuple_allowed = bool(kwargs.get("_advect_ufunc_call")) or raw_name == "clip"
    if tuple_out and not tuple_allowed:
        raise MutationError(f"numpy.{raw_name} does not accept a tuple destination for staged out=")
    if len(destinations) != 1 or not isinstance(destinations[0], AbstractArray):
        raise MutationError("Staged NumPy out= requires one owned staged array destination")
    destination = destinations[0]
    if destination._trace is not trace:  # noqa: SLF001 - frontend owns staged mutation
        raise TracingError("Staged NumPy out= cannot target an array from another trace")
    destination._require_mutable("NumPy out=")  # noqa: SLF001

    validate_staged_out(
        raw_name,
        raw_args,
        dict(kwargs),
        destination,
        tuple_out=tuple_out and (bool(kwargs.get("_advect_ufunc_call")) or raw_name == "clip"),
    )
    ufunc_call = bool(kwargs.pop("_advect_ufunc_call", False))
    result_mask = ufunc_call or raw_name == "clip"
    where = kwargs.pop("where", None) if result_mask else None
    if result_mask:
        if kwargs.get("dtype") is not None:
            raise TracingError(
                f"numpy.{raw_name} dtype= is not supported with staged out=; "
                "ufunc dtype selects a computation loop rather than only an output dtype"
            )
        for signature_name in ("signature", "sig"):
            if kwargs.get(signature_name) is not None:
                raise TracingError(
                    f"numpy.{raw_name} {signature_name}= is not supported with staged out="
                )
        for control in ("dtype", "signature", "sig", "casting", "order", "subok"):
            kwargs.pop(control, None)
    replacement = _numpy_array(trace, raw_name, raw_args, kwargs)
    if where is not None:
        replacement = _numpy_array(trace, "where", (where, replacement, destination), {})
    if replacement.shape != destination.shape:
        raise MutationError(
            f"NumPy out= result shape {replacement.shape!r} does not match "
            f"destination shape {destination.shape!r}"
        )
    if replacement.spec.dtype != destination.spec.dtype:
        # validate_staged_out applied NumPy's casting rule, so the cast back is
        # a neutral astype that any provider replays.
        replacement = replacement.astype(destination.spec.dtype)
    destination._commit(replacement)  # noqa: SLF001 - frontend owns staged mutation
    return destination


def _gradient(
    trace: AbstractTrace,
    raw_args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> AbstractArray | tuple[AbstractArray, ...]:
    source = _lift(trace, raw_args[0])
    axis_value = kwargs.get("axis")
    if axis_value is None:
        raw_axes = tuple(range(source.ndim))
    elif isinstance(axis_value, int):
        raw_axes = (axis_value,)
    else:
        raw_axes = tuple(axis_value)
    axes = tuple(axis if axis >= 0 else axis + source.ndim for axis in raw_axes)
    if any(axis < 0 or axis >= source.ndim for axis in axes) or len(set(axes)) != len(axes):
        raise ValueError(f"gradient() received invalid axes {axis_value!r}")
    edge_order = int(kwargs.get("edge_order", 1))
    if edge_order not in {1, 2}:
        raise ValueError("gradient() edge_order must be 1 or 2")
    return lower_gradient(
        source,
        raw_args[1:] or (1.0,),
        axes=axes,
        edge_order=edge_order,
        error=TypeError,
    )


def _matrix_power(trace: AbstractTrace, raw_args: tuple[Any, ...]) -> AbstractArray:
    matrix = _lift(trace, raw_args[0])
    try:
        # NumPy reads the exponent with operator.index, so bools and NumPy integers count.
        exponent = operator.index(raw_args[1])
    except TypeError as error:
        raise TypeError("matrix_power exponent must be a static integer") from error
    if matrix.ndim < 2 or matrix.shape[-2] != matrix.shape[-1]:
        raise ValueError("matrix_power requires square matrices")
    return lower_matrix_power(matrix, exponent)


def _compress(
    trace: AbstractTrace,
    raw_args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> AbstractArray:
    condition, source = raw_args[:2]
    if isinstance(condition, AbstractArray):
        raise TracingError(
            "Staged numpy.compress requires a captured concrete condition; "
            "a live traced condition has a data-dependent output shape"
        )
    return lower_compress(condition, _lift(trace, source), kwargs.get("axis"))


def _diff(
    trace: AbstractTrace,
    raw_args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> AbstractArray:
    source_raw = raw_args[0]
    n = kwargs.get("n", 1)
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("diff n must be a non-negative integer")
    source = _lift(trace, source_raw)
    axis = normalize_axis(kwargs.get("axis", -1), source.ndim)

    def emit_diff(value: AbstractArray) -> AbstractArray:
        return cast(
            "AbstractArray",
            _record_numpy(trace, "diff", (value,), {"axis": axis, "n": n}),
        )

    if n == 0 or (kwargs.get("prepend") is None and kwargs.get("append") is None):
        return emit_diff(source)
    boundary_shape = list(source.shape)
    boundary_shape[axis] = 1

    def lift_boundary(raw_value: object) -> AbstractArray:
        boundary = _lift(trace, raw_value)
        if boundary.shape == ():
            return _numpy_array(
                trace,
                "broadcast_to",
                (boundary, tuple(boundary_shape)),
                {},
            )
        return boundary

    parts = [lift_boundary(kwargs["prepend"])] if kwargs.get("prepend") is not None else []
    parts.append(source)
    if kwargs.get("append") is not None:
        parts.append(lift_boundary(kwargs["append"]))
    return emit_diff(_numpy_array(trace, "concatenate", (tuple(parts),), {"axis": axis}))


def _pinv(
    trace: AbstractTrace,
    raw_args: tuple[Any, ...],
    raw_kwargs: dict[str, Any],
) -> AbstractArray:
    value = _lift(trace, raw_args[0])
    kwargs = dict(raw_kwargs)
    tolerance_names = tuple(name for name in ("rcond", "rtol") if kwargs.get(name) is not None)
    if len(tolerance_names) > 1:
        raise TypeError("pinv() accepts only one of rcond= and rtol=")
    operands: list[object] = [value]
    attrs: dict[str, object] = {}
    if tolerance_names:
        tolerance_name = tolerance_names[0]
        operands.append(kwargs.pop(tolerance_name))
        attrs["_advect_pinv_tolerance"] = tolerance_name
    if "hermitian" in kwargs:
        attrs["hermitian"] = bool(kwargs.pop("hermitian"))
    return cast(
        "AbstractArray",
        _record_abstract_op(
            trace,
            "array_ext.linalg.pinv",
            operands,
            attrs,
            graph_attrs={"_advect_backend": "numpy"},
        ),
    )


def _full(
    trace: AbstractTrace,
    raw_args: tuple[Any, ...],
    raw_kwargs: dict[str, Any],
) -> AbstractArray:
    shape, fill_value = raw_args[:2]
    kwargs = dict(raw_kwargs)
    # ``like=`` only selects the NumPy frontend.  Once this binder is active it
    # has no graph semantics and, in particular, must not become a data
    # dependency of the fill operation.
    kwargs.pop("like", None)
    attrs = {**kwargs, "shape": shape_tuple(shape)}
    return cast(
        "AbstractArray",
        _record_abstract_op(
            trace,
            "array.full",
            (fill_value,),
            attrs,
            graph_attrs={"_advect_backend": "numpy"},
        ),
    )


def apply_numpy(
    trace: AbstractTrace,
    raw_name: str,
    raw_args: tuple[Any, ...],
    raw_kwargs: dict[str, Any],
) -> AbstractArray | tuple[AbstractArray, ...] | int:
    """Bind one staged NumPy call and emit canonical operations.

    The array-function protocol passes optional metadata by name, so only
    operands and required metadata arrive positionally.
    """
    trace.require_open()
    args = list(raw_args)
    kwargs: dict[str, Any] = dict(raw_kwargs)
    if raw_name == "linalg.qr" and kwargs.get("mode") == "r":
        raw_name = "linalg.qr_r"
    if "out" in kwargs:
        out = kwargs.pop("out")
        if not _empty_out(out):
            return _functionalize_out(trace, raw_name, raw_args, kwargs, out)

    if raw_name == "gradient":
        return _gradient(trace, raw_args, kwargs)
    if raw_name == "size":
        value = _lift(trace, args[0])
        axis_value = kwargs.get("axis")
        return (
            math.prod(value.shape)
            if axis_value is None
            else value.shape[normalize_axis(axis_value, value.ndim)]
        )
    if raw_name == "average":
        return lower_average(
            _lift(trace, raw_args[0]),
            kwargs.get("weights"),
            axis=kwargs.get("axis"),
            keepdims=bool(kwargs.get("keepdims", False)),
            returned=bool(kwargs.get("returned", False)),
        )
    if raw_name == "compress":
        return _compress(trace, raw_args, kwargs)
    if raw_name == "linalg.matrix_power":
        return _matrix_power(trace, raw_args)
    if raw_name in {"cumulative_prod", "cumulative_sum"} and kwargs.get("include_initial", False):
        return lower_cumulative_initial(
            raw_name,
            _lift(trace, raw_args[0]),
            axis=kwargs.get("axis"),
            dtype=kwargs.get("dtype"),
        )
    if raw_name in {"cumprod", "cumsum", "cumulative_prod", "cumulative_sum"}:
        # NumPy scans a 0-d input as a one-element vector, and cumsum and
        # cumprod scan the flattened array when axis is None.
        source = _lift(trace, args[0])
        flattens = kwargs.get("axis") is None and raw_name in {"cumprod", "cumsum"}
        if source.ndim == 0 or (flattens and source.ndim != 1):
            args[0] = _numpy_array(trace, "reshape", (source, (math.prod(source.shape),)), {})
    if raw_name == "round":
        # np.round returns integers unchanged; only other data rounds with rint.
        source = _lift(trace, args[0])
        if kwargs.get("decimals", 0) == 0 and np.dtype(source.spec.dtype).kind in "iu":
            return source.copy()
    if raw_name in {"empty", "ones", "zeros"} and kwargs.get("dtype") is None:
        kwargs["dtype"] = "float64"
    if raw_name == "eye":
        order = kwargs.pop("order", "C")
        if order != "C":
            raise TypeError("Abstract staging of numpy.eye supports only order='C'")
        device = kwargs.pop("device", None)
        if device not in {None, "cpu"}:
            raise TypeError("Abstract staging of numpy.eye supports only device='cpu'")
        columns = kwargs.pop("M", None)
        if columns is not None:
            kwargs["n_cols"] = columns
        if kwargs.get("dtype") is float:
            kwargs["dtype"] = "float64"
    if raw_name == "clip":
        unsupported = tuple(sorted(set(kwargs) - CLIP_KEYWORDS))
        if unsupported:
            raise TypeError(
                f"Abstract staging of clip() does not support attributes {unsupported!r}"
            )
        value, lower, upper = bind_clip_bounds(tuple(args), kwargs)
        operands = [value]
        attrs = {
            "_advect_clip_min_is_input": lower is not None,
            "_advect_clip_max_is_input": upper is not None,
        }
        if lower is not None:
            operands.append(lower)
        if upper is not None:
            operands.append(upper)
        return cast(
            "AbstractArray",
            _record_abstract_op(
                trace,
                "array.clip",
                operands,
                attrs,
                graph_attrs={"_advect_backend": "numpy"},
            ),
        )
    if raw_name == "diff":
        return _diff(trace, raw_args, kwargs)
    if raw_name in {"argsort", "sort"}:
        kwargs = ascending_sort_kwargs(raw_name, kwargs)
        # axis=None sorts the flattened array.
        if kwargs.get("axis", -1) is None:
            source = _lift(trace, args[0])
            args = [_numpy_array(trace, "reshape", (source, (math.prod(source.shape),)), {})]
            kwargs["axis"] = -1
    # NumPy reads svd's compute_uv by truthiness.
    if raw_name == "linalg.svd" and not kwargs.pop("compute_uv", True):
        return _numpy_array(trace, "linalg.svdvals", (args[0],), {})
    if raw_name in {"linalg.pinv", "pinv"}:
        return _pinv(trace, raw_args, kwargs)
    if raw_name == "full":
        return _full(trace, raw_args, kwargs)

    op = staged_numpy_op(raw_name)
    rule = get_registry().get(op).abstract_schema
    if rule is None:
        raise AssertionError(f"Operation {op!r} has no abstract schema")
    if rule.sequence_operand:
        if not args or not isinstance(args[0], (tuple, list)) or not args[0]:
            raise TypeError(f"{raw_name}() requires a non-empty list or tuple of arrays")
        operands = list(args.pop(0))
    else:
        if len(args) < rule.operands:
            raise TypeError(f"{raw_name}() requires {rule.operands} array operands")
        operands = [args.pop(0) for _ in range(rule.operands)]
    if raw_name.endswith("matrix_transpose"):
        rank = len(getattr(operands[0], "shape", ()))
        if rank < 2:
            raise ValueError("matrix_transpose requires an array with at least two dimensions")
        axes = list(range(rank))
        axes[-2], axes[-1] = axes[-1], axes[-2]
        kwargs["axes"] = tuple(axes)
    if raw_name in {"linalg.diagonal", "linalg.trace"}:
        kwargs["axis1"] = -2
        kwargs["axis2"] = -1
    if raw_name in {"convolve", "correlate"}:
        kwargs.setdefault("mode", "full" if raw_name == "convolve" else "valid")
    if raw_name == "take_along_axis" and args:
        if "axis" in kwargs:
            raise TypeError("take_along_axis() received 'axis' twice")
        kwargs["axis"] = args.pop(0)
    for name in rule.positional_attrs:
        if not args:
            break
        if name in kwargs:
            raise TypeError(f"{raw_name}() received {name!r} twice")
        kwargs[name] = args.pop(0)
    if args:
        raise TypeError(f"Cannot stage positional metadata for {raw_name}: {tuple(args)!r}")
    if raw_name == "take" and "mode" in kwargs:
        kwargs["mode"] = take_mode(kwargs["mode"])
    if raw_name in REDUCTIONS:
        source = _lift(trace, operands[0])
        controlled = lower_controlled_reduction(raw_name, source, kwargs, error=TypeError)
        if controlled is not None:
            return controlled
        initial = kwargs.get("initial")
        if isinstance(initial, (list, tuple)) or getattr(initial, "ndim", 0) != 0:
            raise ValueError(NON_SCALAR_INITIAL)
        if callable(getattr(initial, "item", None)):
            # A static NumPy scalar or 0-d array initial= is an ordinary graph attribute.
            kwargs["initial"] = cast("Any", initial).item()
    if raw_name == "astype":
        if "order" in kwargs:
            kwargs["order"] = _normalize_order(kwargs["order"], default="K")
        casting = kwargs.get("casting", "unsafe")
        subok = kwargs.get("subok", False)
        copy = kwargs.get("copy", True)
        if not isinstance(casting, str) or casting not in _CASTING_RULES:
            raise ValueError(f"astype() received invalid casting rule {casting!r}")
        if type(subok) is not bool:
            raise TypeError("astype() subok must be a bool")
        if type(copy) is not bool:
            raise TypeError("astype() copy must be a bool")
        target_dtype = kwargs.get("dtype")
        source_dtype = operand_dtype(operands[0])
        if target_dtype is not None and not can_cast_dtype(
            source_dtype,
            target_dtype,
            casting=casting,
        ):
            raise TypeError(
                f"Cannot cast array data from {dtype_name(source_dtype)!r} to "
                f"{dtype_name(target_dtype)!r} according to the {casting!r} rule"
            )
    result = _record_abstract_op(
        trace,
        op,
        operands,
        kwargs,
        graph_attrs={"_advect_backend": "numpy"},
    )
    return restore_array_api_result(raw_name, result) if isinstance(result, tuple) else result


__all__ = ["apply_numpy", "can_cast_dtype"]
