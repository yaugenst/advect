# ruff: noqa: ANN401
# Composite lowerings intentionally accept both concrete arrays and tracers.
"""Order statistics lowered to sorting and gather primitives."""

from __future__ import annotations

import math
import warnings
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as _numpy  # noqa: ICN001 - typed module and dynamic lowering namespace

from advect.core._array_protocol_helpers import literal_is_weak
from advect.core._errors import TracingError
from advect.numpy._array_function.composite import (
    _concrete_array,
    _finish,
    _lift_composite_constant,
    _normalize_axes,
)

np: Any = _numpy

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike
    from advect.numpy._array_function.composite import CompositeResult
    from advect.numpy._array_function.emission import ArrayFunctionHandler


_SUPPORTED_METHODS = frozenset(
    {
        "averaged_inverted_cdf",
        "closest_observation",
        "hazen",
        "higher",
        "interpolated_inverted_cdf",
        "inverted_cdf",
        "linear",
        "lower",
        "median_unbiased",
        "midpoint",
        "nearest",
        "normal_unbiased",
        "weibull",
    }
)
# These methods select an observation, so NumPy keeps the data dtype.
_SELECTION_METHODS = frozenset(
    {"closest_observation", "higher", "inverted_cdf", "lower", "nearest"}
)
# NumPy's virtual index for these methods keeps an integer quantile's dtype.
_INTEGER_INDEX_METHODS = frozenset(
    {"averaged_inverted_cdf", "interpolated_inverted_cdf", "weibull"}
)
# NumPy's Hyndman and Fan (alpha, beta) parameters; see _virtual_index.
_CONTINUOUS_METHOD_PARAMETERS = {
    "hazen": (0.5, 0.5),
    "interpolated_inverted_cdf": (0, 1),
    "median_unbiased": (1 / 3.0, 1 / 3.0),
    "normal_unbiased": (3 / 8.0, 3 / 8.0),
    "weibull": (0, 0),
}
_DISCRETE_RANK_ROUNDING = {"higher": np.ceil, "lower": np.floor, "nearest": np.rint}
# From this weight on, NumPy's lerp interpolates back from the upper observation.
_UPPER_LERP_WEIGHT = 0.5


def _prepare_reduction(
    value: Any,
    *,
    axis: object,
) -> tuple[Any, tuple[int, ...], tuple[int, ...]]:
    ndim = int(value.ndim)
    if axis is None:
        axes = tuple(range(ndim))
        return np.ravel(value), (), axes
    axes = _normalize_axes(axis, ndim)
    remaining = tuple(index for index in range(ndim) if index not in set(axes))
    moved = np.moveaxis(
        value,
        axes,
        tuple(range(ndim - len(axes), ndim)),
    )
    batch_shape = tuple(int(value.shape[index]) for index in remaining)
    reduction_size = math.prod(int(value.shape[index]) for index in axes)
    return np.reshape(moved, (*batch_shape, reduction_size)), batch_shape, axes


def _selected_index(method: str, count: int, quantile: Any) -> int | None:
    """Return the observation NumPy takes without interpolating, if any."""
    rank = (count - 1) * quantile
    if method in _DISCRETE_RANK_ROUNDING:
        return int(_DISCRETE_RANK_ROUNDING[method](rank))
    if method == "linear" and isinstance(rank, np.integer):
        # NumPy takes an integer quantile's observation directly.
        return int(rank)
    if method in {"closest_observation", "inverted_cdf"}:
        index = count * quantile - 1
        if method == "closest_observation":
            index = index - 0.5
        previous = math.floor(index)
        # closest_observation keeps an exact index only on NumPy's odd order statistic.
        keep = index == previous and (method == "inverted_cdf" or previous % 2 == 1)
        return max(0, previous if keep else previous + 1)
    return None


def _virtual_index(method: str, count: int, quantile: Any) -> Any:
    """Return NumPy's fractional index of an interpolating quantile method."""
    if method == "linear":
        # NumPy prefers this to the equivalent alpha = beta = 1 form to limit rounding.
        return (count - 1) * quantile
    if method == "midpoint":
        rank = (count - 1) * quantile
        return 0.5 * (np.floor(rank) + np.ceil(rank))
    if method == "averaged_inverted_cdf":
        return count * quantile - 1
    alpha, beta = _CONTINUOUS_METHOD_PARAMETERS[method]
    return count * quantile + (alpha + quantile * (1 - alpha - beta)) - 1


def _one_quantile(
    sorted_values: Any,
    quantile: Any,
    *,
    quantile_value: Any,
    method: str,
    weak: bool,
) -> Any:
    """Follow NumPy's quantile method for one quantile.

    ``quantile_value`` is the concrete quantile in NumPy's dtype, so indexes
    and interpolation weights round exactly as in NumPy. ``quantile`` is the
    operand that carries its derivative, and ``weak`` says whether NumPy
    interpolates with it as a Python float (NEP 50).
    """
    count = int(sorted_values.shape[-1])
    if count == 0:
        msg = "quantile cannot reduce an empty axis during tracing"
        raise TracingError(msg)
    if method == "median":
        # np.median takes the mean of the middle observations rather than
        # interpolating, and that mean accumulates integers in float64 and
        # float16 in float32. For floating data, NumPy's nanmedian along an
        # axis shorter than 600 instead sums the pair, doubled for an odd
        # count, in the data dtype through np.ma.median and so overflows to inf
        # near the dtype's limit; the traced nanmedian deliberately keeps
        # np.median's finite mean.
        middle = sorted_values[..., (count - 1) // 2]
        if count % 2:
            return middle
        dtype = sorted_values.dtype
        accumulator = np.float32 if dtype == np.float16 else np.result_type(dtype, 0.0)
        pair = (middle, sorted_values[..., count // 2])
        lower, upper = (
            item if item.dtype == accumulator else np.astype(item, accumulator) for item in pair
        )
        return (lower + upper) * 0.5
    selected = _selected_index(method, count, quantile_value)
    if selected is not None:
        return sorted_values[..., selected]
    virtual = _virtual_index(method, count, quantile_value)
    previous = math.floor(virtual)
    if virtual >= count - 1:
        lower_index = upper_index = count - 1
    elif virtual < 0:
        lower_index = upper_index = 0
    else:
        lower_index, upper_index = previous, previous + 1
    gamma_value = virtual - previous
    if method in {"averaged_inverted_cdf", "midpoint"}:
        # NumPy fixes these weights and stores them in the index dtype.
        if method == "midpoint":
            fixed = 0.0 if virtual % 1 == 0 else 0.5
        else:
            fixed = 0.5 if gamma_value == 0 else 1.0
        gamma_value = np.asarray(fixed, dtype=np.result_type(virtual))[()]
        gamma = float(gamma_value) if weak else gamma_value
    else:
        gamma = _virtual_index(method, count, quantile) - previous
    lower = sorted_values[..., lower_index]
    upper = sorted_values[..., upper_index]
    difference = upper - lower
    if gamma_value >= _UPPER_LERP_WEIGHT:
        return upper - difference * (1 - gamma)
    return lower + difference * gamma


def _quantile_dtype(
    source_dtype: Any,
    quantile_dtype: Any,
    *,
    method: str,
    weak_quantile: bool,
) -> Any:
    """Return NumPy's quantile result dtype for one method.

    ``quantile_dtype`` is the dtype NumPy's quantile sees, after percentile's
    division by 100.
    """
    if method in _SELECTION_METHODS or (method == "linear" and quantile_dtype.kind in "biu"):
        return source_dtype
    # Interpolation promotes the data with the quantile; Python scalars stay weak.
    if weak_quantile:
        return np.result_type(source_dtype, 0.0)
    # NumPy interpolates with its virtual index's dtype.
    index_dtype = np.result_type(quantile_dtype, 0 if method in _INTEGER_INDEX_METHODS else 0.0)
    return np.result_type(source_dtype, index_dtype)


def _propagate_nan_slices(result: Any, sorted_values: Any) -> Any:
    """Return the NaN that sorts last for every slice containing NaN, as NumPy does."""
    if not np.issubdtype(sorted_values.dtype, np.inexact):
        return result
    last = sorted_values[..., -1]
    if result.ndim == 0 and result.dtype != last.dtype:
        # NumPy returns a 0-d result as that NaN itself, keeping the data dtype.
        return last if np.isnan(_concrete_array(last)) else result
    # A traced selection needs no concrete payload, so abstract staging keeps it.
    return np.where(np.isnan(last), last, result)


def _keep_reduced_dims(result: Any, source: Any, axes: tuple[int, ...]) -> Any:
    """Restore each reduced axis with length one after the quantile axes."""
    kept = tuple(1 if index in axes else int(size) for index, size in enumerate(source.shape))
    leading = int(result.ndim) - (int(source.ndim) - len(axes))
    return np.reshape(result, (*result.shape[:leading], *kept))


def _scalar_if_full(result: Any, *, keepdims: bool) -> Any:
    """Return a 0-d result without kept dimensions as the scalar NumPy returns."""
    return result[()] if result.ndim == 0 and not keepdims else result


def _checked_quantiles(quantile: Any, scale: float) -> Any:
    """Return the concrete quantiles NumPy sees, after percentile's division by 100."""
    numpy_quantile = _concrete_array(quantile)
    if scale != 1:
        numpy_quantile = numpy_quantile / scale
    if np.any((numpy_quantile < 0) | (numpy_quantile > 1)):
        msg = "quantiles must lie in the closed interval [0, 1]"
        raise TracingError(msg)
    return numpy_quantile


def _sorted_quantile(
    sorted_values: Any,
    quantile: Any,
    *,
    method: str,
    scale: float,
    traced_type: type[TracedArrayLike],
) -> Any:
    """Evaluate every quantile along the last axis of ``sorted_values``.

    The result has the quantile's shape followed by the batch shape.
    """
    numpy_quantile = _checked_quantiles(quantile, scale)
    quantile_shape = tuple(int(size) for size in numpy_quantile.shape)
    flat_values = numpy_quantile.reshape(-1)
    # NumPy interpolates a Python int or float quantile weakly, and so a weak
    # tracer that stands for one; np.float64 subclasses float but is strong.
    weak_quantile = literal_is_weak(quantile) and _concrete_array(quantile).dtype.kind in "if"
    if isinstance(quantile, traced_type):
        # Python arithmetic keeps a weak quantile weak, where a NumPy function
        # such as ravel would return a strong array.
        traced_quantiles: Any = (quantile,) if weak_quantile else np.ravel(quantile)
        operands = tuple(
            traced_quantiles[index] / scale if scale != 1 else traced_quantiles[index]
            for index in range(flat_values.size)
        )
    else:
        # A Python scalar quantile interpolates with weak NumPy promotion.
        operands = tuple(float(item) if weak_quantile else item for item in flat_values)
    results = tuple(
        _one_quantile(
            sorted_values,
            operand,
            quantile_value=concrete_value,
            method=method,
            weak=weak_quantile,
        )
        for operand, concrete_value in zip(operands, flat_values, strict=True)
    )
    if quantile_shape:
        batch_shape = tuple(int(size) for size in sorted_values.shape[:-1])
        # An empty quantile array selects nothing from each slice, as in NumPy.
        stacked = np.stack(results) if results else np.moveaxis(sorted_values[..., :0], -1, 0)
        result = np.reshape(stacked, (*quantile_shape, *batch_shape))
    else:
        result = results[0]
    dtype = _quantile_dtype(
        sorted_values.dtype,
        numpy_quantile.dtype,
        method=method,
        weak_quantile=weak_quantile,
    )
    if result.dtype != dtype:
        result = np.astype(result, dtype)
    return _propagate_nan_slices(result, sorted_values)


def _quantile_result(
    value: Any,
    quantile: Any,
    *,
    axis: object,
    keepdims: bool,
    method: str,
    scale: float,
    traced_type: type[TracedArrayLike],
) -> Any:
    source = value if isinstance(value, traced_type) else np.asarray(value)
    prepared, _batch_shape, axes = _prepare_reduction(source, axis=axis)
    sorted_values = np.sort(prepared, axis=-1)
    result = _sorted_quantile(
        sorted_values,
        quantile,
        method=method,
        scale=scale,
        traced_type=traced_type,
    )
    if keepdims:
        result = _keep_reduced_dims(result, source, axes)
    if not isinstance(result, traced_type) and isinstance(quantile, traced_type):
        result = _lift_composite_constant(result, quantile)
    return _scalar_if_full(result, keepdims=keepdims)


def _nan_quantile_result(
    value: Any,
    quantile: Any,
    *,
    axis: object,
    keepdims: bool,
    method: str,
    scale: float,
    traced_type: type[TracedArrayLike],
) -> Any:
    source = value if isinstance(value, traced_type) else np.asarray(value)
    if np.issubdtype(source.dtype, np.complexfloating):
        msg = "nanquantile does not support complex inputs"
        raise TracingError(msg)
    prepared, batch_shape, axes = _prepare_reduction(source, axis=axis)
    reduction_size = int(prepared.shape[-1])
    batch_size = math.prod(batch_shape)
    flat_source = np.reshape(prepared, (batch_size, reduction_size))
    # NaN sorts last, so a slice's valid values lead its sorted row.
    sorted_rows = np.sort(flat_source, axis=-1)
    counts = reduction_size - np.count_nonzero(np.isnan(_concrete_array(flat_source)), axis=-1)
    quantile_shape = tuple(int(size) for size in np.shape(_checked_quantiles(quantile, scale)))
    # Slices with the same valid count share every index and weight, so each
    # count is evaluated once for all of its slices.
    parts: list[Any] = []
    slices: list[Any] = []
    unique_counts = np.unique(counts)
    for count in unique_counts:
        rows = np.flatnonzero(counts == count)
        if count == 0:
            # An all-NaN slice keeps the data dtype, as in NumPy's nanquantile.
            nan_dtype = source.dtype if np.issubdtype(source.dtype, np.inexact) else np.float64
            part = np.full((*quantile_shape, rows.size), np.nan, nan_dtype)
        else:
            group = sorted_rows if rows.size == batch_size else np.take(sorted_rows, rows, axis=0)
            part = _sorted_quantile(
                group if count == reduction_size else group[:, :count],
                quantile,
                method=method,
                scale=scale,
                traced_type=traced_type,
            )
        parts.append(part)
        slices.append(rows)
    result = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=-1)
    if len(parts) > 1:
        result = np.take(result, np.argsort(np.concatenate(slices)), axis=-1)
    # NumPy's apply_along_axis casts every slice to the first slice's dtype.
    first_dtype = parts[int(np.searchsorted(unique_counts, counts[0]))].dtype
    if result.dtype != first_dtype:
        result = np.astype(result, first_dtype)
    result = np.reshape(result, (*quantile_shape, *batch_shape))
    if keepdims:
        result = _keep_reduced_dims(result, source, axes)
    if not isinstance(result, traced_type):
        anchor = source if isinstance(source, traced_type) else quantile
        result = _lift_composite_constant(result, anchor)
    return _scalar_if_full(result, keepdims=keepdims)


def _weighted_quantile_result(
    value: Any,
    quantile: Any,
    weights: Any,
    *,
    axis: object,
    keepdims: bool,
    scale: float,
    traced_type: type[TracedArrayLike],
) -> Any:
    source = value if isinstance(value, traced_type) else np.asarray(value)
    prepared, batch_shape, axes = _prepare_reduction(source, axis=axis)
    weight_array = weights if isinstance(weights, traced_type) else np.asarray(weights)
    if int(weight_array.ndim) == 1:
        if len(axes) != 1 or int(weight_array.shape[0]) != int(source.shape[axes[0]]):
            msg = "One-dimensional quantile weights must match one reduction axis"
            raise TracingError(msg)
        weight_shape = [1] * int(source.ndim)
        weight_shape[axes[0]] = int(weight_array.shape[0])
        weight_array = np.broadcast_to(np.reshape(weight_array, tuple(weight_shape)), source.shape)
    elif tuple(weight_array.shape) != tuple(source.shape):
        msg = "Quantile weights must be one-dimensional or match the input shape"
        raise TracingError(msg)
    prepared_weights, _, _ = _prepare_reduction(weight_array, axis=axis)

    reduction_size = int(prepared.shape[-1])
    batch_size = math.prod(batch_shape)
    flat_source = np.reshape(prepared, (batch_size, reduction_size))
    concrete_source = _concrete_array(flat_source)
    concrete_weights = _concrete_array(np.reshape(prepared_weights, (batch_size, reduction_size)))
    concrete_quantile = _checked_quantiles(quantile, scale)
    quantile_shape = tuple(int(size) for size in concrete_quantile.shape)

    order = np.argsort(concrete_source, axis=-1, kind="stable")
    sorted_weights = np.take_along_axis(concrete_weights, order, axis=-1)
    if np.any(sorted_weights < 0):
        msg = "quantile weights must be non-negative"
        raise TracingError(msg)
    # NumPy normalizes float64 cumulative weights, compares them in a float
    # quantile's dtype, and lowers zeros to -1 so q=0 skips leading zero weights.
    cumulative = np.cumsum(sorted_weights, axis=-1, dtype=np.float64)
    totals = cumulative[:, -1:]
    if not np.all(np.isfinite(totals) & (totals > 0)):
        msg = "quantile weights must have a positive finite sum"
        raise TracingError(msg)
    cumulative = cumulative / totals
    if concrete_quantile.dtype.kind == "f":
        cumulative = cumulative.astype(concrete_quantile.dtype)
    cumulative[cumulative == 0] = -1
    # The inverted CDF takes the first observation whose cumulative weight
    # reaches the quantile, and the NaN that sorts last for a slice with NaN.
    positions = np.minimum(
        np.sum(cumulative[:, None, :] < concrete_quantile.reshape(-1)[:, None], axis=-1),
        reduction_size - 1,
    )
    has_nan = np.isnan(np.take_along_axis(concrete_source, order[:, -1:], axis=-1))
    positions = np.where(has_nan, reduction_size - 1, positions)
    selected = np.take_along_axis(order, positions, axis=-1)
    result = np.take_along_axis(flat_source, selected, axis=-1)
    result = np.reshape(np.moveaxis(result, -1, 0), (*quantile_shape, *batch_shape))
    for anchor in (weights, quantile):
        if isinstance(anchor, traced_type):
            result = result + np.zeros_like(anchor, dtype=result.dtype, shape=())
    if keepdims:
        result = _keep_reduced_dims(result, source, axes)
    return result


def _quantile_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    function: Callable[..., object],
) -> CompositeResult:
    name = function.__name__
    ignore_nan = name.startswith("nan")
    percentile = name.endswith("percentile")
    # NumPy's dispatcher bound the call, and the runtime moved out= and later arguments.
    values = dict(zip(("axis",), args[2:], strict=False)) | kwargs
    if bool(values.get("overwrite_input", False)):
        msg = f"numpy.{name}(overwrite_input=True) would mutate its input during tracing"
        raise TracingError(msg)
    interpolation = values.get("interpolation")
    method = str(values.get("method", "linear"))
    if interpolation is not None:
        if method != "linear":
            msg = f"numpy.{name} cannot receive both method= and interpolation="
            raise TracingError(msg)
        warnings.warn(
            f"numpy.{name}(interpolation=...) is deprecated; use method=...",
            DeprecationWarning,
            stacklevel=3,
        )
        method = str(interpolation)
    if method not in _SUPPORTED_METHODS:
        msg = (
            f"quantile method={method!r} is not supported during tracing; "
            f"supported methods are {sorted(_SUPPORTED_METHODS)}"
        )
        raise TracingError(msg)
    weights = values.get("weights")
    if weights is not None and method != "inverted_cdf":
        msg = f"numpy.{name} weights= requires method='inverted_cdf'"
        raise TracingError(msg)
    if weights is not None and ignore_nan:
        msg = f"numpy.{name} weighted NaN filtering is not supported during tracing"
        raise TracingError(msg)
    if weights is not None:
        result = _weighted_quantile_result(
            args[0],
            args[1],
            weights,
            axis=values.get("axis"),
            keepdims=bool(values.get("keepdims", False)),
            scale=100.0 if percentile else 1.0,
            traced_type=traced_type,
        )
        return _finish(result, traced_type=traced_type)
    result_fn = _nan_quantile_result if ignore_nan else _quantile_result
    result = result_fn(
        args[0],
        args[1],
        axis=values.get("axis"),
        keepdims=bool(values.get("keepdims", False)),
        method=method,
        scale=100.0 if percentile else 1.0,
        traced_type=traced_type,
    )
    return _finish(result, traced_type=traced_type)


def _median_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    function: Callable[..., object],
) -> CompositeResult:
    ignore_nan = function is np.nanmedian
    # NumPy's dispatcher bound the call, and the runtime moved out= and later arguments.
    values = dict(zip(("axis",), args[1:], strict=False)) | kwargs
    if bool(values.get("overwrite_input", False)):
        msg = (
            f"numpy.{function.__name__}(overwrite_input=True) would mutate its input during tracing"
        )
        raise TracingError(msg)
    result_fn = _nan_quantile_result if ignore_nan else _quantile_result
    result = result_fn(
        args[0],
        0.5,
        axis=values.get("axis"),
        keepdims=bool(values.get("keepdims", False)),
        method="median",
        scale=1.0,
        traced_type=traced_type,
    )
    return _finish(result, traced_type=traced_type)


def register_statistics_handlers(
    handlers: dict[Callable[..., Any], ArrayFunctionHandler],
) -> None:
    """Register differentiable order statistics."""
    for function in (np.quantile, np.percentile, np.nanquantile, np.nanpercentile):
        handlers[function] = partial(_quantile_handler, function=function)
    for function in (np.median, np.nanmedian):
        handlers[function] = partial(_median_handler, function=function)
