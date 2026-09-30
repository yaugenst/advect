# ruff: noqa: ANN401 - lowerings accept tracers, staged arrays, and constants alike
"""NumPy compositions shared by dynamic tracing and abstract staging.

Each lowering calls NumPy's public API on its operands.  NumPy's protocols route
every call on a ``TracedArray`` to its dynamic handler and every call on a
staged array to ``apply_numpy``, so one formula serves both lifetimes.  Callers
keep only their lifetime-specific validation and error types.
"""

from __future__ import annotations

import math
import operator
from typing import TYPE_CHECKING, Any, cast

import numpy as _numpy  # noqa: ICN001 - dynamic protocol namespace
from numpy.lib.array_utils import normalize_axis_index, normalize_axis_tuple

from advect.core._array_protocol_helpers import _staged_value
from advect.core._errors import TracingError
from advect.core._protocols import _is_traced

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

np: Any = _numpy
_NUMPY_VALUES = (_numpy.ndarray, _numpy.generic)

_PRODUCTS = frozenset({"nanprod", "prod"})
_MEANS = frozenset({"mean", "nanmean"})
_MAXIMA = frozenset({"amax", "max", "nanmax"})
_EXTREMA = _MAXIMA | {"amin", "min", "nanmin"}
_VARIANCES = frozenset({"nanstd", "nanvar", "std", "var"})
REDUCTIONS = _PRODUCTS | _MEANS | _EXTREMA | _VARIANCES | {"nansum", "sum"}
_SECOND_EDGE_ORDER = 2
# NumPy's error for a reduction initial= that is not a scalar.
NON_SCALAR_INITIAL = "setting an array element with a sequence."
UFUNC_METHODS = {
    ("add", "accumulate"): "cumsum",
    ("add", "reduce"): "sum",
    ("multiply", "accumulate"): "cumprod",
    ("multiply", "reduce"): "prod",
}


def operand_ndim(value: object) -> int:
    """Read rank without coercing a tracer to a concrete array."""
    ndim = getattr(value, "ndim", None)
    return int(ndim) if ndim is not None else int(np.ndim(value))


def operand_dtype(value: object) -> Any:
    """Read the NumPy dtype of an operand, a staged value's from its canonical spec.

    Staged code presents its array provider's dtype objects, which NumPy need
    not interpret, so the frontend reads the dtype that staging recorded for a
    staged value or a tracer that wraps one.
    """
    if isinstance(value, _NUMPY_VALUES):
        return value.dtype
    staged = _staged_value(value)
    if staged is not None:
        return np.dtype(staged.spec.dtype)
    return np.dtype(cast("Any", value).dtype)


def operand_shape(value: object) -> tuple[int, ...]:
    """Read shape without coercing a tracer; a Python scalar or sequence as NumPy does."""
    shape = getattr(value, "shape", None)
    return tuple(int(size) for size in (np.shape(value) if shape is None else shape))


def _cast(value: Any, dtype: object) -> Any:
    return value if operand_dtype(value) == np.dtype(dtype) else np.astype(value, dtype)


def _as_dtype(value: Any, dtype: object) -> Any:
    """Convert ``initial=`` to the result dtype as NumPy's reductions do.

    A live value is cast unsafely, like NumPy's 0-d array ``initial=``.  A static
    value goes through NumPy's own conversion, which truncates a float for
    integer data and rejects an out-of-range or complex scalar.  It stays a NumPy
    scalar of exactly the result dtype, so it cannot promote the result, and it
    keeps a value only that dtype holds, such as a ``uint64`` above ``2**63``.
    """
    if _is_traced(value):
        return _cast(value, dtype)
    return np.maximum.reduce(np.empty(0, dtype=dtype), initial=value)


def _exclude_nan(valid: Any, values: Any) -> Any:
    return np.where(np.isnan(values), False, valid)  # noqa: FBT003 - boolean fill value


def _accumulator_dtype(source: Any, requested: object | None, *, real: bool) -> object:
    """Match NumPy's mean and variance accumulator dtype."""
    if requested is not None:
        return requested
    dtype = operand_dtype(source)
    if dtype.kind in "biu":
        return np.dtype(np.float64)
    return np.finfo(dtype).dtype if real and dtype.kind == "c" else dtype


def lower_controlled_reduction(
    name: str,
    source: Any,
    values: Mapping[str, Any],
    *,
    error: type[Exception],
) -> Any | None:
    """Lower controls a canonical reduction cannot record, or return ``None``.

    ``where=``, a traced ``initial=``, ``mean=``, and a traced correction become
    masks and elementwise combinations around plain reductions.
    """
    if name in _VARIANCES:
        correction = _variance_correction(name, values, error=error)
        if "where" in values or values.get("mean") is not None or _is_traced(correction):
            return _lower_variance(name, source, values, correction)
        return None
    initial = values.get("initial")
    traced_initial = _is_traced(initial)
    if "where" not in values and not traced_initial:
        return None
    function = getattr(np, name)
    axis = values.get("axis")
    keepdims = bool(values.get("keepdims", False))
    where = values.get("where")
    if name in _EXTREMA:
        if initial is None:
            msg = f"numpy.{name} with where= requires initial="
            raise error(msg)
        initial = _as_dtype(initial, operand_dtype(source))
        selected = source if where is None else np.where(where, source, initial)
        nan_ignoring = name.startswith("nan")
        if nan_ignoring:
            # NumPy's fmax/fmin reduction lets initial= win over every selected NaN.
            selected = np.where(np.isnan(selected), initial, selected)
        reduced = function(selected, axis=axis, keepdims=keepdims)
        combined = (np.maximum if name in _MAXIMA else np.minimum)(reduced, initial)
        if nan_ignoring and (_is_traced(initial) or np.isnan(initial)):
            # It also lets every selected number win over a NaN initial=.
            selected = np.where(np.isnan(initial), reduced, combined)
            # np.where returns a 0-d array where the reduction returns a scalar.
            return selected if keepdims or operand_ndim(selected) else selected[()]
        return combined

    options: dict[str, Any] = {"axis": axis, "keepdims": keepdims}
    dtype = values.get("dtype")
    if dtype is not None:
        options["dtype"] = dtype
    if where is not None and name in _MEANS:
        valid = np.broadcast_to(where, source.shape)
        if name == "nanmean":
            valid = _exclude_nan(valid, source)
        numerator = np.sum(np.where(valid, source, np.zeros_like(source)), **options)
        count = np.sum(valid, axis=axis, keepdims=keepdims)
        # NumPy divides the sum by an integer count, then casts to the mean dtype.
        return _cast(numerator / count, _accumulator_dtype(source, dtype, real=False))
    if where is not None:
        identity = np.ones_like(source) if name in _PRODUCTS else np.zeros_like(source)
        source = np.where(where, source, identity)
    if initial is not None and not traced_initial:
        options["initial"] = initial
    result = function(source, **options)
    if not traced_initial:
        return result
    initial = _as_dtype(initial, operand_dtype(result))
    return result * initial if name in _PRODUCTS else result + initial


def _variance_correction(name: str, values: Mapping[str, Any], *, error: type[Exception]) -> Any:
    """Apply NumPy's rule that ``correction=`` admits only a zero ``ddof=``."""
    ddof = values.get("ddof", 0)
    if "correction" not in values:
        return ddof
    if _is_traced(ddof):
        msg = f"numpy.{name} requires a static ddof= beside correction="
        raise error(msg)
    if ddof != 0:
        msg = "ddof and correction can't be provided simultaneously."
        raise ValueError(msg)
    return values["correction"]


def _lower_variance(name: str, source: Any, values: Mapping[str, Any], correction: Any) -> Any:
    dtype = values.get("dtype")
    array = source if dtype is None else np.astype(source, dtype)
    axis = values.get("axis")
    keepdims = bool(values.get("keepdims", False))
    where = values.get("where")
    valid = (
        np.ones_like(array, dtype=bool) if where is None else np.broadcast_to(where, array.shape)
    )
    if name.startswith("nan"):
        valid = _exclude_nan(valid, array)
    mean = values.get("mean")
    if mean is None:
        count = np.astype(np.sum(valid, axis=axis, keepdims=True), operand_dtype(array))
        mean = np.sum(np.where(valid, array, np.zeros_like(array)), axis=axis, keepdims=True)
        mean = mean / count
    centered = np.where(valid, array, mean) - mean
    squared = np.real(np.conjugate(centered) * centered)
    numerator = np.sum(
        np.where(valid, squared, np.zeros_like(squared)),
        axis=axis,
        keepdims=keepdims,
    )
    squared_dtype = operand_dtype(squared)
    # A static mask counts to a NumPy scalar, which np.astype rejects on NumPy 2.0.
    count = np.sum(valid, axis=axis, keepdims=keepdims).astype(squared_dtype)
    if _is_traced(correction):
        correction = _cast(correction, squared_dtype)
    else:
        correction = np.asarray(correction, dtype=squared_dtype)
    degrees = count - correction
    if name.startswith("nan") and operand_dtype(source).kind in "fc":
        # NumPy's nanvar marks slices without positive degrees of freedom as NaN.
        denominator = np.where(degrees > 0, degrees, np.nan)
    else:
        # NumPy's var clamps negative degrees of freedom to zero.
        denominator = np.maximum(degrees, 0)
    result = numerator / denominator
    if name.endswith("std"):
        result = np.sqrt(result)
    return _cast(result, _accumulator_dtype(source, dtype, real=True))


def lower_average(
    array: Any,
    weights: Any,
    *,
    axis: Any,
    keepdims: bool,
    returned: bool,
    check_weight_sum: Callable[[Any], None] | None = None,
) -> Any:
    """Lower ``numpy.average`` to sums, or ``mean`` without weights."""
    if weights is None:
        result = np.mean(array, axis=axis, keepdims=keepdims)
        axes = range(array.ndim) if axis is None else normalize_axis_tuple(axis, array.ndim)
        weight_sum: Any = math.prod(array.shape[item] for item in axes)
    else:
        weight_shape = operand_shape(weights)
        if weight_shape != tuple(array.shape):
            if axis is None:
                msg = "Axis must be specified when shapes of a and weights differ."
                raise TypeError(msg)
            axes = normalize_axis_tuple(axis, array.ndim)
            if weight_shape != tuple(array.shape[item] for item in axes):
                msg = "Shape of weights must be consistent with shape of a along specified axis."
                raise ValueError(msg)
            expanded = [1] * array.ndim
            for weight_axis, array_axis in enumerate(axes):
                expanded[array_axis] = weight_shape[weight_axis]
            weights = np.reshape(weights, tuple(expanded))
        weight_sum = np.sum(weights, axis=axis, keepdims=keepdims)
        if check_weight_sum is not None:
            check_weight_sum(weight_sum)
        result = np.sum(array * weights, axis=axis, keepdims=keepdims) / weight_sum
    return (result, np.ones_like(result) * weight_sum) if returned else result


def lower_matrix_power(matrix: Any, exponent: int) -> Any:
    """Lower an integer matrix power to inverses and binary exponentiation."""
    if exponent == 0:
        return np.zeros_like(matrix) + np.eye(matrix.shape[-1], dtype=operand_dtype(matrix))
    base = np.linalg.inv(matrix) if exponent < 0 else matrix
    remaining = abs(exponent)
    result = None
    while remaining:
        if remaining & 1:
            result = base if result is None else np.matmul(result, base)
        remaining >>= 1
        if remaining:
            base = np.matmul(base, base)
    return result


def lower_cumulative_initial(name: str, source: Any, *, axis: Any, dtype: object) -> Any:
    """Prepend the identity slice to ``cumulative_sum`` or ``cumulative_prod``."""
    if operand_ndim(source) == 0:
        # NumPy scans a 0-d input as a one-element vector.
        source = np.reshape(source, (1,))
    if axis is None:
        if operand_ndim(source) != 1:
            msg = "For arrays which have more than one dimension ``axis`` argument is required."
            raise ValueError(msg)
        axis = 0
    scan = np.cumsum if name == "cumulative_sum" else np.cumprod
    base = scan(source, axis=axis) if dtype is None else scan(source, axis=axis, dtype=dtype)
    seed_shape = list(base.shape)
    seed_shape[axis] = 1
    seed = (np.zeros_like if name == "cumulative_sum" else np.ones_like)(
        base,
        shape=tuple(seed_shape),
    )
    return np.concatenate((seed, base), axis=axis)


def lower_compress(condition: object, array: Any, axis: Any) -> Any:
    """Gather the positions a concrete ``compress`` condition selects."""
    if axis is None:
        array = np.reshape(array, (math.prod(array.shape),))
        axis = 0
    else:
        axis = normalize_axis_index(operator.index(axis), array.ndim)
    return np.take(array, _compress_positions(condition, array.shape[axis], axis), axis=axis)


def _compress_positions(condition: object, length: int, axis: int) -> Any:
    """Return the positions NumPy's compress selects along one axis of ``length``."""
    condition_array = np.asarray(condition)
    if condition_array.ndim != 1:
        msg = "condition must be a 1-d array"
        raise ValueError(msg)
    positions = np.flatnonzero(condition_array)
    beyond = positions[positions >= length]
    if beyond.size:
        msg = f"index {beyond[0]} is out of bounds for axis {axis} with size {length}"
        raise IndexError(msg)
    return positions


def lower_ufunc_method(
    ufunc: Any,
    method: str,
    inputs: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Lower the supported ufunc methods to the equivalent NumPy calls."""
    qualified = f"numpy.{ufunc.__name__}.{method}"
    function = UFUNC_METHODS.get((ufunc.__name__, method))
    if function is not None:
        if len(inputs) != 1:
            msg = f"{qualified} expects one input array"
            raise TracingError(msg)
        if method == "accumulate" and operand_ndim(inputs[0]) == 0:
            # np.cumsum scans a 0-d input as a vector; the ufunc method rejects it.
            msg = "cannot accumulate on a scalar"
            raise TypeError(msg)
        # A reduction's single destination arrives as NumPy's one-tuple out=.
        out = kwargs.pop("out", None)
        destination = out[0] if isinstance(out, tuple) else out
        if destination is not None:
            kwargs["out"] = destination
        kwargs.setdefault("axis", 0)
        return getattr(np, function)(inputs[0], **kwargs)
    if method != "outer":
        msg = f"{qualified} is not supported by Advect's NumPy frontend"
        raise TracingError(msg)
    if len(inputs) != 2 or ufunc.nin != 2 or ufunc.nout != 1 or ufunc.signature is not None:  # noqa: PLR2004
        msg = (
            f"{qualified} requires an ordinary binary, single-output ufunc; "
            "generalized ufunc signatures are unsupported"
        )
        raise TracingError(msg)
    left, right = inputs
    left_ndim, right_ndim = operand_ndim(left), operand_ndim(right)
    return ufunc(
        np.expand_dims(left, axis=tuple(range(left_ndim, left_ndim + right_ndim))),
        np.expand_dims(right, axis=tuple(range(left_ndim))),
        **kwargs,
    )


def lower_gradient(
    source: Any,
    spacings: tuple[Any, ...],
    *,
    axes: tuple[int, ...],
    edge_order: int,
    error: type[Exception],
) -> Any:
    """Lower NumPy's gradient along normalized ``axes`` with NumPy's formulas."""
    if len(spacings) == 1 and operand_ndim(spacings[0]) == 0:
        spacings *= len(axes)
    elif len(spacings) != len(axes):
        msg = (
            "numpy.gradient requires one scalar spacing or one spacing per gradient axis; "
            f"got {len(spacings)} spacings for {len(axes)} axes"
        )
        raise error(msg)
    if operand_dtype(source).kind not in "fc":
        source = np.astype(source, np.float64)
    dtype = operand_dtype(source)
    outputs = tuple(
        # NumPy writes every gradient into an array of the input's inexact dtype.
        _cast(_gradient_axis(source, spacing, axis=axis, edge_order=edge_order), dtype)
        for axis, spacing in zip(axes, spacings, strict=True)
    )
    return outputs[0] if len(outputs) == 1 else outputs


def _gradient_axis(source: Any, spacing: Any, *, axis: int, edge_order: int) -> Any:
    rank = int(source.ndim)
    length = int(source.shape[axis])
    minimum = 3 if edge_order == _SECOND_EDGE_ORDER else 2
    if length < minimum:
        msg = (
            f"gradient edge_order={edge_order} requires at least {minimum} points along axis {axis}"
        )
        raise ValueError(msg)

    def along(start: int | None, stop: int | None) -> Any:
        index = [slice(None)] * rank
        index[axis] = slice(start, stop)
        return source[tuple(index)]

    if operand_ndim(spacing) != 0:
        if operand_ndim(spacing) != 1 or operand_shape(spacing)[0] != length:
            msg = (
                "gradient coordinate spacing must be one-dimensional and match "
                f"axis {axis} length {length}"
            )
            raise ValueError(msg)
        if not _is_traced(spacing):
            spacing = np.asarray(spacing)
        if operand_dtype(spacing).kind in "biu":
            spacing = np.astype(spacing, np.float64)
        spacing = np.diff(spacing)
        if not _is_traced(spacing) and bool(np.all(spacing == spacing[0])):
            # NumPy reduces evenly spaced concrete coordinates to one spacing.
            spacing = spacing[0]

    if operand_ndim(spacing) == 0:
        interior = (along(2, None) - along(0, -2)) / (2.0 * spacing)
        if edge_order == 1:
            left = (along(1, 2) - along(0, 1)) / spacing
            right = (along(-1, None) - along(-2, -1)) / spacing
        else:
            left = (-1.5 / spacing) * along(0, 1) + (2.0 / spacing) * along(1, 2)
            left = left + (-0.5 / spacing) * along(2, 3)
            right = (0.5 / spacing) * along(-3, -2) + (-2.0 / spacing) * along(-2, -1)
            right = right + (1.5 / spacing) * along(-1, None)
        return np.concatenate((left, interior, right), axis=axis)

    dx1 = spacing[:-1]
    dx2 = spacing[1:]
    coefficient_shape = [1] * rank
    coefficient_shape[axis] = length - 2

    def along_axis(coefficients: Any) -> Any:
        return np.reshape(coefficients, tuple(coefficient_shape))

    interior = (
        along_axis(-dx2 / (dx1 * (dx1 + dx2))) * along(0, -2)
        + along_axis((dx2 - dx1) / (dx1 * dx2)) * along(1, -1)
        + along_axis(dx1 / (dx2 * (dx1 + dx2))) * along(2, None)
    )
    if edge_order == 1:
        left = (along(1, 2) - along(0, 1)) / spacing[0]
        right = (along(-1, None) - along(-2, -1)) / spacing[-1]
    else:
        dx1, dx2 = spacing[0], spacing[1]
        left = (
            -(2.0 * dx1 + dx2) / (dx1 * (dx1 + dx2)) * along(0, 1)
            + (dx1 + dx2) / (dx1 * dx2) * along(1, 2)
            - dx1 / (dx2 * (dx1 + dx2)) * along(2, 3)
        )
        dx1, dx2 = spacing[-2], spacing[-1]
        right = (
            dx2 / (dx1 * (dx1 + dx2)) * along(-3, -2)
            - (dx2 + dx1) / (dx1 * dx2) * along(-2, -1)
            + (2.0 * dx2 + dx1) / (dx2 * (dx1 + dx2)) * along(-1, None)
        )
    return np.concatenate((left, interior, right), axis=axis)


__all__ = [
    "REDUCTIONS",
    "UFUNC_METHODS",
    "lower_average",
    "lower_compress",
    "lower_controlled_reduction",
    "lower_cumulative_initial",
    "lower_gradient",
    "lower_matrix_power",
    "lower_ufunc_method",
    "operand_dtype",
    "operand_ndim",
    "operand_shape",
]
