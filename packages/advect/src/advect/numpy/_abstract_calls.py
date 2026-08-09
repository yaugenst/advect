# ruff: noqa: C901, EM101, EM102, PLR0911, PLR0912, PLR0915, PLR2004, TRY003
"""NumPy calling conventions for payload-free staged arrays."""

from __future__ import annotations

import math
from itertools import product
from typing import TYPE_CHECKING, Any, cast

from advect.core._abstract import AbstractArray, _lift, _new_abstract_array, _record_abstract_op
from advect.core._abstract_helpers import (
    dtype_name,
    normalize_axes,
    normalize_axis,
    shape_tuple,
)
from advect.core._abstract_model import ArraySpec
from advect.core._array_api.results import restore_array_api_result
from advect.core._errors import MutationError, TracingError
from advect.core._registry import get_registry
from advect.numpy._gradient_lowering import lower_gradient_axis, operand_ndim
from advect.numpy._op_bindings import staged_numpy_op
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
    import numpy as np  # noqa: PLC0415 - frontend-local provider policy

    return bool(np.can_cast(cast("Any", source), cast("Any", target), casting=cast("Any", casting)))


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


def _cast_abstract(
    trace: AbstractTrace,
    value: AbstractArray,
    dtype: object,
    *,
    casting: str = "unsafe",
) -> AbstractArray:
    target_dtype = dtype_name(dtype)
    if not can_cast_dtype(value.dtype, target_dtype, casting=casting):
        raise MutationError(
            f"Cannot cast NumPy out= result from {dtype_name(value.dtype)!r} to "
            f"{target_dtype!r} according to the {casting!r} casting rule"
        )
    return cast(
        "AbstractArray",
        _record_abstract_op(
            trace,
            "array.astype",
            (value,),
            {"casting": casting, "copy": False, "dtype": target_dtype},
            graph_attrs={"_advect_backend": "numpy"},
        ),
    )


def _functionalize_out(
    trace: AbstractTrace,
    raw_name: str,
    raw_args: tuple[Any, ...],
    kwargs: dict[str, Any],
    out: object,
) -> AbstractArray:
    destinations = out if isinstance(out, tuple) else (out,)
    tuple_out = isinstance(out, tuple)
    tuple_allowed = (
        bool(kwargs.get("_advect_ufunc_out_tuple"))
        or bool(kwargs.get("_advect_ufunc_call"))
        or raw_name == "clip"
    )
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
    kwargs.pop("_advect_ufunc_out_tuple", None)
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
    if dtype_name(replacement.dtype) != dtype_name(destination.dtype):
        replacement = _cast_abstract(trace, replacement, destination.dtype)
    destination._commit(replacement)  # noqa: SLF001 - frontend owns staged mutation
    return destination


def _reduction(
    trace: AbstractTrace,
    name: str,
    source: object,
    *,
    axis: object,
    dtype: object,
    keepdims: bool,
    initial: object | None = None,
) -> AbstractArray:
    attrs: dict[str, object] = {"axis": axis, "keepdims": keepdims}
    if dtype is not None:
        attrs["dtype"] = dtype
    if initial is not None:
        item = getattr(initial, "item", None)
        attrs["initial"] = item() if callable(item) else initial
    return _numpy_array(trace, name, (source,), attrs)


class _GradientNamespace:
    __slots__ = ("trace",)

    def __init__(self, trace: AbstractTrace) -> None:
        self.trace = trace

    def concatenate(self, arrays: tuple[Any, ...], *, axis: int) -> AbstractArray:
        return _numpy_array(self.trace, "concatenate", (arrays,), {"axis": axis})

    def diff(self, value: object) -> AbstractArray:
        return _numpy_array(self.trace, "diff", (value,), {})

    def reshape(self, value: object, shape: tuple[int, ...]) -> AbstractArray:
        return _numpy_array(self.trace, "reshape", (value, shape), {})


def _gradient(
    trace: AbstractTrace,
    raw_args: tuple[Any, ...],
    raw_kwargs: dict[str, Any],
) -> AbstractArray | tuple[AbstractArray, ...]:
    if not raw_args:
        raise TypeError("gradient() requires an input array")
    kwargs = dict(raw_kwargs)
    unexpected = set(kwargs) - {"axis", "edge_order"}
    if unexpected:
        raise TypeError(f"Cannot stage gradient() attributes {tuple(sorted(unexpected))!r}")
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
    spacings = raw_args[1:]
    if not spacings:
        normalized_spacings: tuple[object, ...] = (1.0,) * len(axes)
    elif len(spacings) == 1 and operand_ndim(spacings[0]) == 0:
        normalized_spacings = spacings * len(axes)
    elif len(spacings) == len(axes):
        normalized_spacings = spacings
    else:
        raise TypeError("gradient() requires one scalar spacing or one spacing per gradient axis")
    namespace = _GradientNamespace(trace)
    outputs = tuple(
        cast(
            "AbstractArray",
            lower_gradient_axis(
                namespace,
                source,
                spacing,
                axis=axis,
                edge_order=edge_order,
            ),
        )
        for axis, spacing in zip(axes, normalized_spacings, strict=True)
    )
    return outputs[0] if len(outputs) == 1 else outputs


def _controlled_reduction(
    trace: AbstractTrace,
    raw_name: str,
    source: AbstractArray,
    kwargs: dict[str, Any],
) -> AbstractArray | None:
    additive = {"sum", "nansum"}
    multiplicative = {"prod", "nanprod"}
    means = {"mean", "nanmean"}
    extrema = {"amax", "amin", "max", "min", "nanmax", "nanmin"}
    variances = {"nanstd", "nanvar", "std", "var"}
    initial = kwargs.get("initial")
    has_dynamic_initial = isinstance(initial, AbstractArray)
    correction = kwargs.get("correction", kwargs.get("ddof", 0))

    if raw_name in additive | multiplicative | means:
        if "where" not in kwargs and not has_dynamic_initial:
            return None
        axis = kwargs.get("axis")
        dtype = kwargs.get("dtype")
        keepdims = bool(kwargs.get("keepdims", False))
        where = kwargs.get("where")
        selected: object = source
        if where is not None:
            valid = _numpy_array(trace, "broadcast_to", (where, source.shape), {})
            if raw_name == "nanmean":
                isnan = _numpy_array(trace, "isnan", (source,), {})
                valid = _numpy_array(trace, "where", (isnan, False, valid), {})
            if raw_name in means:
                zero = _numpy_array(trace, "zeros_like", (source,), {})
                numerator_source = _numpy_array(
                    trace,
                    "where",
                    (valid, source, zero),
                    {},
                )
                numerator = _reduction(
                    trace,
                    "sum",
                    numerator_source,
                    axis=axis,
                    dtype=dtype,
                    keepdims=keepdims,
                )
                count = _reduction(
                    trace,
                    "sum",
                    _numpy_array(trace, "astype", (valid, "int64"), {}),
                    axis=axis,
                    dtype=None,
                    keepdims=keepdims,
                )
                count = _numpy_array(trace, "astype", (count, source.dtype), {})
                return _numpy_array(trace, "divide", (numerator, count), {})
            identity = _numpy_array(
                trace,
                "ones_like" if raw_name in multiplicative else "zeros_like",
                (source,),
                {},
            )
            selected = _numpy_array(trace, "where", (valid, source, identity), {})
        result = _reduction(
            trace,
            raw_name,
            selected,
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
            initial=None if has_dynamic_initial else initial,
        )
        if not has_dynamic_initial:
            return result
        combine = "multiply" if raw_name in multiplicative else "add"
        return _numpy_array(trace, combine, (result, initial), {})

    if raw_name in extrema:
        if "where" not in kwargs and not has_dynamic_initial:
            return None
        axis = kwargs.get("axis")
        keepdims = bool(kwargs.get("keepdims", False))
        where = kwargs.get("where")
        if initial is None:
            raise TypeError(f"{raw_name}() with where= requires initial=")
        if where is not None:
            selected = _numpy_array(trace, "where", (where, source, initial), {})
        elif raw_name in {"nanmax", "nanmin"}:
            isnan = _numpy_array(trace, "isnan", (source,), {})
            selected = _numpy_array(trace, "where", (isnan, initial, source), {})
        else:
            selected = source
        base = _reduction(
            trace,
            raw_name,
            selected,
            axis=axis,
            dtype=None,
            keepdims=keepdims,
        )
        combine = "maximum" if raw_name in {"amax", "max", "nanmax"} else "minimum"
        return _numpy_array(trace, combine, (base, initial), {})

    if raw_name not in variances or (
        "where" not in kwargs
        and kwargs.get("mean") is None
        and not isinstance(correction, AbstractArray)
    ):
        return None

    dtype = kwargs.get("dtype")
    if dtype is not None:
        source = _numpy_array(trace, "astype", (source, dtype), {})
    axis = kwargs.get("axis")
    keepdims = bool(kwargs.get("keepdims", False))
    where = kwargs.get("where")
    valid = (
        _numpy_array(trace, "ones_like", (source,), {"dtype": "bool"})
        if where is None
        else _numpy_array(trace, "broadcast_to", (where, source.shape), {})
    )
    if raw_name in {"nanstd", "nanvar"}:
        isnan = _numpy_array(trace, "isnan", (source,), {})
        valid = _numpy_array(trace, "where", (isnan, False, valid), {})
    valid_int = _numpy_array(trace, "astype", (valid, "int64"), {})
    count_with_dims = _reduction(
        trace,
        "sum",
        valid_int,
        axis=axis,
        dtype=None,
        keepdims=True,
    )
    count_with_dims = _numpy_array(trace, "astype", (count_with_dims, source.dtype), {})
    supplied_mean = kwargs.get("mean")
    if supplied_mean is None:
        zero = _numpy_array(trace, "zeros_like", (source,), {})
        selected = _numpy_array(trace, "where", (valid, source, zero), {})
        numerator = _reduction(
            trace,
            "sum",
            selected,
            axis=axis,
            dtype=None,
            keepdims=True,
        )
        supplied_mean = _numpy_array(trace, "divide", (numerator, count_with_dims), {})
    safe_source = _numpy_array(trace, "where", (valid, source, supplied_mean), {})
    centered = _numpy_array(trace, "subtract", (safe_source, supplied_mean), {})
    conjugate = _numpy_array(trace, "conjugate", (centered,), {})
    magnitude_squared = _numpy_array(trace, "multiply", (conjugate, centered), {})
    squared = _numpy_array(trace, "real", (magnitude_squared,), {})
    squared_zero = _numpy_array(trace, "zeros_like", (squared,), {})
    selected_squared = _numpy_array(trace, "where", (valid, squared, squared_zero), {})
    numerator = _reduction(
        trace,
        "sum",
        selected_squared,
        axis=axis,
        dtype=None,
        keepdims=keepdims,
    )
    count = _reduction(
        trace,
        "sum",
        valid_int,
        axis=axis,
        dtype=None,
        keepdims=keepdims,
    )
    count = _numpy_array(trace, "astype", (count, squared.dtype), {})
    typed_correction = _numpy_array(trace, "astype", (_lift(trace, correction), squared.dtype), {})
    denominator = _numpy_array(trace, "subtract", (count, typed_correction), {})
    result = _numpy_array(trace, "divide", (numerator, denominator), {})
    return _numpy_array(trace, "sqrt", (result,), {}) if raw_name in {"nanstd", "std"} else result


def _average(
    trace: AbstractTrace,
    raw_args: tuple[Any, ...],
    raw_kwargs: dict[str, Any],
) -> AbstractArray | tuple[AbstractArray, AbstractArray]:
    if not raw_args or len(raw_args) > 4:
        raise TypeError("average() expects (a, axis, weights, returned)")
    values = dict(raw_kwargs)
    unexpected = set(values) - {"axis", "keepdims", "returned", "weights"}
    if unexpected:
        raise TypeError(f"Abstract staging of average() does not support {tuple(unexpected)!r}")
    for name, value in zip(("axis", "weights", "returned"), raw_args[1:], strict=False):
        if name in values:
            raise TypeError(f"average() received {name!r} twice")
        values[name] = value
    array = _lift(trace, raw_args[0])
    axis = values.get("axis")
    keepdims = bool(values.get("keepdims", False))
    returned = bool(values.get("returned", False))
    weights_raw = values.get("weights")
    if weights_raw is None:
        result = _numpy_array(
            trace,
            "mean",
            (array,),
            {"axis": axis, "keepdims": keepdims},
        )
        if not returned:
            return result
        count = (
            math.prod(array.shape)
            if axis is None
            else math.prod(array.shape[item] for item in normalize_axes(axis, array.ndim))
        )
        weight_sum = _numpy_array(
            trace,
            "multiply",
            (_numpy_array(trace, "ones_like", (result,), {}), count),
            {},
        )
        return result, weight_sum
    weights = _lift(trace, weights_raw)
    if weights.shape != array.shape:
        if axis is None:
            raise TypeError("Axis must be specified when shapes of a and weights differ.")
        axes = normalize_axes(axis, array.ndim)
        expected_shape = tuple(array.shape[item] for item in axes)
        if weights.shape != expected_shape:
            raise ValueError(
                "Shape of weights must be consistent with shape of a along specified axis."
            )
        expanded_shape = [1] * array.ndim
        for weight_axis, array_axis in enumerate(axes):
            expanded_shape[array_axis] = weights.shape[weight_axis]
        weights = _numpy_array(trace, "reshape", (weights, tuple(expanded_shape)), {})
    weighted = _numpy_array(trace, "multiply", (array, weights), {})
    reduction_attrs = {"axis": axis, "keepdims": keepdims}
    numerator = _numpy_array(trace, "sum", (weighted,), reduction_attrs)
    denominator = _numpy_array(trace, "sum", (weights,), reduction_attrs)
    result = _numpy_array(trace, "divide", (numerator, denominator), {})
    if not returned:
        return result
    weight_sum = _numpy_array(
        trace,
        "multiply",
        (_numpy_array(trace, "ones_like", (result,), {}), denominator),
        {},
    )
    return result, weight_sum


def _matrix_power(
    trace: AbstractTrace,
    raw_args: tuple[Any, ...],
    raw_kwargs: dict[str, Any],
) -> AbstractArray:
    if len(raw_args) != 2 or raw_kwargs:
        raise TypeError("matrix_power() expects a matrix and a static integer exponent")
    matrix = _lift(trace, raw_args[0])
    exponent = raw_args[1]
    if isinstance(exponent, bool) or not isinstance(exponent, int):
        raise TypeError("matrix_power exponent must be a static integer")
    if matrix.ndim < 2 or matrix.shape[-2] != matrix.shape[-1]:
        raise ValueError("matrix_power requires square matrices")
    if exponent == 0:
        identity = _numpy_array(
            trace,
            "eye",
            (matrix.shape[-1],),
            {"dtype": matrix.dtype},
        )
        zeros = _numpy_array(trace, "zeros_like", (matrix,), {})
        return _numpy_array(trace, "add", (zeros, identity), {})
    base = _numpy_array(trace, "linalg.inv", (matrix,), {}) if exponent < 0 else matrix
    remaining = abs(exponent)
    result: AbstractArray | None = None
    while remaining:
        if remaining & 1:
            result = base if result is None else _numpy_array(trace, "matmul", (result, base), {})
        remaining >>= 1
        if remaining:
            base = _numpy_array(trace, "matmul", (base, base), {})
    if result is None:  # pragma: no cover - exponent zero returns above
        raise AssertionError("matrix_power failed to produce a result")
    return result


def _cumulative_with_initial(
    trace: AbstractTrace,
    raw_name: str,
    raw_args: tuple[Any, ...],
    raw_kwargs: dict[str, Any],
) -> AbstractArray:
    if not raw_args or len(raw_args) > 2:
        raise TypeError(f"{raw_name}() expects an array and optional axis")
    if len(raw_args) == 2 and "axis" in raw_kwargs:
        raise TypeError(f"{raw_name}() received 'axis' twice")
    source = _lift(trace, raw_args[0])
    axis_value = raw_args[1] if len(raw_args) == 2 else raw_kwargs.get("axis")
    if axis_value is None:
        if source.ndim != 1:
            raise ValueError(
                "cumulative operations require axis= for inputs with more than one dimension"
            )
        axis = 0
    else:
        axis = normalize_axis(axis_value, source.ndim)
    options = dict(raw_kwargs)
    options["axis"] = axis
    options.pop("include_initial", None)
    base = _numpy_array(trace, raw_name, (source,), options)
    seed_shape = list(base.shape)
    seed_shape[axis] = 1
    fill_value = 1 if raw_name in {"cumprod", "cumulative_prod"} else 0
    seed = _numpy_array(
        trace,
        "full",
        (tuple(seed_shape), fill_value),
        {"dtype": base.dtype},
    )
    return _numpy_array(trace, "concatenate", ((seed, base),), {"axis": axis})


def _compress(
    trace: AbstractTrace,
    raw_args: tuple[Any, ...],
    raw_kwargs: dict[str, Any],
) -> AbstractArray:
    if len(raw_args) not in {2, 3} or set(raw_kwargs) - {"axis"}:
        raise TypeError("compress() expects (condition, a, axis=None)")
    if len(raw_args) == 3 and "axis" in raw_kwargs:
        raise TypeError("compress() received 'axis' twice")
    condition, source_raw = raw_args[:2]

    def condition_values(value: object) -> tuple[bool, ...]:
        if isinstance(value, AbstractArray):
            raise TracingError(
                "Staged numpy.compress requires a captured concrete condition; "
                "a live traced condition has a data-dependent output shape"
            )
        shape = getattr(value, "shape", None)
        if shape is not None:
            normalized_shape = tuple(int(size) for size in shape)
            if not normalized_shape:
                return (bool(value),)
            return tuple(
                bool(cast("Any", value)[index])
                for index in product(*(range(size) for size in normalized_shape))
            )
        if isinstance(value, (tuple, list)):
            return tuple(item for child in value for item in condition_values(child))
        return (bool(value),)

    source = _lift(trace, source_raw)
    axis_raw = raw_args[2] if len(raw_args) == 3 else raw_kwargs.get("axis")
    if axis_raw is None:
        source = _numpy_array(trace, "reshape", (source, (math.prod(source.shape),)), {})
        axis = 0
    else:
        axis = normalize_axis(axis_raw, source.ndim)
    limit = source.shape[axis]
    positions = tuple(
        index for index, selected in enumerate(condition_values(condition)[:limit]) if selected
    )
    position_spec = ArraySpec((len(positions),), "int64")
    position_array = _new_abstract_array(
        trace,
        int(trace.add_constant(positions, position_spec)),
        position_spec,
        owned=False,
    )
    return _numpy_array(trace, "take", (source, position_array), {"axis": axis})


def _diff(
    trace: AbstractTrace,
    raw_args: tuple[Any, ...],
    raw_kwargs: dict[str, Any],
) -> AbstractArray:
    if not raw_args or len(raw_args) > 5:
        raise TypeError("diff() expects (a, n, axis, prepend, append)")
    values = dict(raw_kwargs)
    unexpected = set(values) - {"append", "axis", "n", "prepend"}
    if unexpected:
        raise TypeError(
            f"Abstract staging of diff() does not support {tuple(sorted(unexpected))!r}"
        )
    source_raw = raw_args[0]
    for name, value in zip(("n", "axis", "prepend", "append"), raw_args[1:], strict=False):
        if name in values:
            raise TypeError(f"diff() received {name!r} twice")
        values[name] = value
    n = values.get("n", 1)
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("diff n must be a non-negative integer")
    source = _lift(trace, source_raw)
    axis = normalize_axis(values.get("axis", -1), source.ndim)

    def emit_diff(value: AbstractArray) -> AbstractArray:
        return cast(
            "AbstractArray",
            _record_numpy(trace, "diff", (value,), {"axis": axis, "n": n}),
        )

    if n == 0 or (values.get("prepend") is None and values.get("append") is None):
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

    parts = [lift_boundary(values["prepend"])] if values.get("prepend") is not None else []
    parts.append(source)
    if values.get("append") is not None:
        parts.append(lift_boundary(values["append"]))
    return emit_diff(_numpy_array(trace, "concatenate", (tuple(parts),), {"axis": axis}))


def _pinv(
    trace: AbstractTrace,
    raw_args: tuple[Any, ...],
    raw_kwargs: dict[str, Any],
) -> AbstractArray:
    args = list(raw_args)
    kwargs = dict(raw_kwargs)
    if not args:
        raise TypeError("pinv() requires an input")
    value = _lift(trace, args.pop(0))
    if len(args) > 1:
        raise TypeError("pinv() accepts at most one positional tolerance")
    unexpected = set(kwargs) - {"hermitian", "rcond", "rtol"}
    if unexpected:
        raise TypeError(
            f"Abstract staging of pinv() does not support attributes {tuple(sorted(unexpected))!r}"
        )
    if args:
        if "rcond" in kwargs or "rtol" in kwargs:
            raise TypeError("pinv() received its tolerance twice")
        kwargs["rcond"] = args.pop()
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
        hermitian = kwargs.pop("hermitian")
        if type(hermitian) is not bool:
            raise TypeError("pinv hermitian must be a bool")
        attrs["hermitian"] = hermitian
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
    args = list(raw_args)
    kwargs = dict(raw_kwargs)
    if len(args) < 2:
        raise TypeError("full() requires shape and fill_value")
    shape = args.pop(0)
    fill_value = args.pop(0)
    for name in ("dtype", "order"):
        if not args:
            break
        if name in kwargs:
            raise TypeError(f"full() received {name!r} twice")
        kwargs[name] = args.pop(0)
    if args:
        raise TypeError(f"Cannot stage positional metadata for full: {tuple(args)!r}")
    unexpected = set(kwargs) - {"device", "dtype", "like", "order"}
    if unexpected:
        raise TypeError(
            f"Abstract staging of full() does not support attributes {tuple(sorted(unexpected))!r}"
        )
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
    """Bind one staged NumPy call and emit canonical operations."""
    trace.require_open()
    if raw_name in {"concatenate", "dot"} and len(raw_args) > 2:
        if len(raw_args) > 3:
            raise TypeError(f"{raw_name}() accepts at most three positional arguments")
        if "out" in raw_kwargs:
            raise TypeError(f"{raw_name}() received 'out' twice")
        raw_kwargs = {**raw_kwargs, "out": raw_args[2]}
        raw_args = raw_args[:2]
    args = list(raw_args)
    kwargs = dict(raw_kwargs)
    if raw_name == "linalg.qr":
        positional_mode = args[1] if len(args) > 1 else None
        mode = kwargs.get("mode", positional_mode if positional_mode is not None else "reduced")
        if mode == "r":
            raw_name = "linalg.qr_r"
    if "out" in kwargs:
        out = kwargs.pop("out")
        if not _empty_out(out):
            return _functionalize_out(trace, raw_name, raw_args, kwargs, out)

    if raw_name == "gradient":
        return _gradient(trace, raw_args, kwargs)
    if raw_name == "size":
        if not args or len(args) > 2 or set(kwargs) - {"axis"}:
            raise TypeError("size() expects an array and optional axis")
        if len(args) == 2 and "axis" in kwargs:
            raise TypeError("size() received 'axis' twice")
        value = _lift(trace, args[0])
        axis_value = args[1] if len(args) == 2 else kwargs.get("axis")
        return (
            math.prod(value.shape)
            if axis_value is None
            else value.shape[normalize_axis(axis_value, value.ndim)]
        )
    if raw_name == "average":
        return _average(trace, raw_args, raw_kwargs)
    if raw_name == "compress":
        return _compress(trace, raw_args, raw_kwargs)
    if raw_name == "linalg.matrix_power":
        return _matrix_power(trace, raw_args, raw_kwargs)
    if raw_name in {"cumulative_prod", "cumulative_sum"} and bool(
        kwargs.get("include_initial", False)
    ):
        return _cumulative_with_initial(trace, raw_name, raw_args, raw_kwargs)
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
        if not args:
            raise TypeError("clip() requires an input")
        value = args.pop(0)
        lower = args.pop(0) if args else kwargs.pop("a_min", None)
        upper = args.pop(0) if args else kwargs.pop("a_max", None)
        if args or kwargs:
            raise TypeError(
                f"Abstract staging of clip() does not support attributes {tuple(sorted(kwargs))!r}"
            )
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
        return _diff(trace, raw_args, raw_kwargs)
    if raw_name == "linalg.svd" and kwargs.get("compute_uv", True) is False:
        if len(args) != 1:
            raise TypeError("linalg.svd() expects one input array")
        unexpected = set(kwargs) - {"compute_uv", "full_matrices", "hermitian"}
        if unexpected:
            raise TypeError(
                f"Abstract staging of linalg.svd() does not support {tuple(sorted(unexpected))!r}"
            )
        return _numpy_array(trace, "linalg.svdvals", (args[0],), {})
    if raw_name in {"linalg.pinv", "pinv"}:
        return _pinv(trace, raw_args, raw_kwargs)
    if raw_name == "full":
        return _full(trace, raw_args, raw_kwargs)

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
    if raw_name == "convolve" and not args and "mode" not in kwargs:
        kwargs["mode"] = "full"
    if raw_name == "correlate" and not args and "mode" not in kwargs:
        kwargs["mode"] = "valid"
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
    if raw_name in {
        "amax",
        "amin",
        "max",
        "mean",
        "min",
        "nanmax",
        "nanmean",
        "nanmin",
        "nanprod",
        "nanstd",
        "nansum",
        "nanvar",
        "prod",
        "std",
        "sum",
        "var",
    }:
        source = _lift(trace, operands[0])
        controlled = _controlled_reduction(trace, raw_name, source, kwargs)
        if controlled is not None:
            return controlled
    if raw_name == "astype":
        order = kwargs.get("order", "K")
        casting = kwargs.get("casting", "unsafe")
        subok = kwargs.get("subok", False)
        copy = kwargs.get("copy", True)
        if not isinstance(order, str) or order not in {"A", "C", "F", "K"}:
            raise ValueError(f"astype() received invalid order {order!r}")
        if not isinstance(casting, str) or casting not in _CASTING_RULES:
            raise ValueError(f"astype() received invalid casting rule {casting!r}")
        if type(subok) is not bool:
            raise TypeError("astype() subok must be a bool")
        if type(copy) is not bool:
            raise TypeError("astype() copy must be a bool")
        target_dtype = kwargs.get("dtype")
        source_dtype = getattr(operands[0], "dtype", None)
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
    if isinstance(result, tuple) and raw_name == "linalg.slogdet":
        return restore_array_api_result(raw_name, result)
    return result


__all__ = ["apply_numpy", "can_cast_dtype"]
