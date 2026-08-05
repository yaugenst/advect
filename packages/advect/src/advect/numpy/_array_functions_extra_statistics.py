# ruff: noqa: ANN401
# Composite lowerings intentionally accept both concrete arrays and tracers.
"""Order statistics lowered to sorting and gather primitives."""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any

import numpy as _numpy  # noqa: ICN001 - typed module and dynamic lowering namespace

from advect.core._errors import TracingError
from advect.core._protocols import _snapshot_traced
from advect.numpy._array_functions_extra_composite import (
    _finish,
    _normalize_axes,
)

np: Any = _numpy

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike
    from advect.numpy._array_functions_extra_composite import CompositeResult


_BINARY_ARITY = 2
_MAX_MEDIAN_POSITIONAL = 5
_MAX_QUANTILE_POSITIONAL = 7
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
_CONTINUOUS_METHOD_PARAMETERS = {
    "hazen": (0.5, 0.5),
    "interpolated_inverted_cdf": (0.0, 1.0),
    "linear": (1.0, 1.0),
    "median_unbiased": (1.0 / 3.0, 1.0 / 3.0),
    "normal_unbiased": (3.0 / 8.0, 3.0 / 8.0),
    "weibull": (0.0, 0.0),
}


def _concrete_array(value: Any, traced_type: type[TracedArrayLike]) -> np.ndarray[Any, Any]:
    if isinstance(value, traced_type):
        return np.asarray(_snapshot_traced(value)[1])
    return np.asarray(value)


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


def _one_quantile(  # noqa: PLR0911 - each NumPy method has distinct selection semantics
    sorted_values: Any,
    quantile: Any,
    *,
    quantile_value: float,
    method: str,
) -> Any:
    count = int(sorted_values.shape[-1])
    if count == 0:
        msg = "quantile cannot reduce an empty axis during tracing"
        raise TracingError(msg)
    linear_rank = quantile_value * (count - 1)
    if method == "lower":
        return sorted_values[..., math.floor(linear_rank)]
    if method == "higher":
        return sorted_values[..., math.ceil(linear_rank)]
    if method == "nearest":
        return sorted_values[..., int(np.rint(linear_rank))]
    if method == "midpoint":
        lower_index = math.floor(linear_rank)
        upper_index = math.ceil(linear_rank)
        lower = sorted_values[..., lower_index]
        upper = sorted_values[..., upper_index]
        return (lower + upper) * 0.5

    if method == "inverted_cdf":
        index = max(0, math.ceil(count * quantile_value) - 1)
        return sorted_values[..., index]
    if method == "averaged_inverted_cdf":
        virtual_index = count * quantile_value - 1
        lower_unclipped = math.floor(virtual_index)
        if virtual_index < 0:
            lower_index = upper_index = 0
        elif virtual_index >= count - 1:
            lower_index = upper_index = count - 1
        else:
            lower_index = lower_unclipped
            upper_index = lower_unclipped + 1
        if virtual_index == math.floor(virtual_index):
            return (sorted_values[..., lower_index] + sorted_values[..., upper_index]) * 0.5
        return sorted_values[..., upper_index]
    if method == "closest_observation":
        virtual_index = count * quantile_value - 1.5
        lower_index = math.floor(virtual_index)
        fractional = virtual_index - lower_index
        choose_lower = fractional == 0 and lower_index % 2 == 1
        index = lower_index if choose_lower else lower_index + 1
        return sorted_values[..., max(0, min(count - 1, index))]

    alpha, beta = _CONTINUOUS_METHOD_PARAMETERS[method]
    virtual_value = count * quantile_value + alpha + quantile_value * (1 - alpha - beta) - 1
    lower_unclipped = math.floor(virtual_value)
    if virtual_value < 0:
        lower_index = upper_index = 0
    elif virtual_value >= count - 1:
        lower_index = upper_index = count - 1
    else:
        lower_index = lower_unclipped
        upper_index = lower_unclipped + 1
    lower = sorted_values[..., lower_index]
    upper = sorted_values[..., upper_index]
    virtual_index = count * quantile + alpha + quantile * (1 - alpha - beta) - 1
    return lower + (upper - lower) * (virtual_index - lower_unclipped)


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
    if method not in _SUPPORTED_METHODS:
        msg = (
            f"quantile method={method!r} is not supported during tracing; "
            f"supported methods are {sorted(_SUPPORTED_METHODS)}"
        )
        raise TracingError(msg)
    source = value if isinstance(value, traced_type) else np.asarray(value)
    prepared, batch_shape, axes = _prepare_reduction(source, axis=axis)
    sorted_values = np.sort(prepared, axis=-1)
    concrete_quantile = _concrete_array(quantile, traced_type) / scale
    if np.any((concrete_quantile < 0) | (concrete_quantile > 1)):
        msg = "quantiles must lie in the closed interval [0, 1]"
        raise TracingError(msg)
    quantile_shape = tuple(int(size) for size in concrete_quantile.shape)
    flat_values = concrete_quantile.reshape(-1)
    traced_quantiles = (
        np.ravel(quantile) if isinstance(quantile, traced_type) else np.ravel(concrete_quantile)
    )
    results = tuple(
        _one_quantile(
            sorted_values,
            traced_quantiles[index] / scale
            if isinstance(quantile, traced_type)
            else concrete_value,
            quantile_value=float(concrete_value),
            method=method,
        )
        for index, concrete_value in enumerate(flat_values)
    )
    if quantile_shape:
        result = np.reshape(np.stack(results, axis=0), (*quantile_shape, *batch_shape))
    else:
        result = results[0]
    if keepdims:
        reduced = set(axes)
        kept_shape = tuple(
            1 if index in reduced else int(source.shape[index]) for index in range(int(source.ndim))
        )
        result = np.reshape(result, (*quantile_shape, *kept_shape))
    if not isinstance(result, traced_type) and isinstance(quantile, traced_type):
        result = result + np.sum(quantile) * 0
    return result


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
    batch_size = math.prod(batch_shape) if batch_shape else 1
    flat_source = np.reshape(prepared, (batch_size, reduction_size))
    concrete_source = _concrete_array(flat_source, traced_type)
    concrete_quantile = _concrete_array(quantile, traced_type) / scale
    quantile_shape = tuple(int(size) for size in concrete_quantile.shape)
    rows: list[Any] = []
    for batch_index in range(batch_size):
        valid_indices = np.flatnonzero(~np.isnan(concrete_source[batch_index]))
        row = flat_source[batch_index]
        if valid_indices.size:
            rows.append(
                _quantile_result(
                    np.take(row, valid_indices),
                    quantile,
                    axis=None,
                    keepdims=False,
                    method=method,
                    scale=scale,
                    traced_type=traced_type,
                )
            )
            continue
        anchor = np.sum(np.nan_to_num(row))
        if isinstance(quantile, traced_type):
            anchor = anchor + np.sum(quantile) * 0
        rows.append(np.zeros(quantile_shape) + anchor * 0 + np.nan)

    result = np.reshape(np.stack(tuple(rows), axis=-1), (*quantile_shape, *batch_shape))
    if keepdims:
        reduced = set(axes)
        kept_shape = tuple(
            1 if index in reduced else int(source.shape[index]) for index in range(int(source.ndim))
        )
        result = np.reshape(result, (*quantile_shape, *kept_shape))
    return result


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
    batch_size = math.prod(batch_shape) if batch_shape else 1
    flat_source = np.reshape(prepared, (batch_size, reduction_size))
    flat_weights = np.reshape(prepared_weights, (batch_size, reduction_size))
    concrete_source = _concrete_array(flat_source, traced_type)
    concrete_weights = _concrete_array(flat_weights, traced_type)
    concrete_quantile = _concrete_array(quantile, traced_type) / scale
    if np.any((concrete_quantile < 0) | (concrete_quantile > 1)):
        msg = "quantiles must lie in the closed interval [0, 1]"
        raise TracingError(msg)
    quantile_shape = tuple(int(size) for size in concrete_quantile.shape)
    flat_quantiles = concrete_quantile.reshape(-1)

    rows: list[Any] = []
    for batch_index in range(batch_size):
        order = np.argsort(concrete_source[batch_index], kind="stable")
        sorted_source = np.take(flat_source[batch_index], order)
        sorted_weights = concrete_weights[batch_index, order]
        if np.any(sorted_weights < 0):
            msg = "quantile weights must be non-negative"
            raise TracingError(msg)
        total = np.sum(sorted_weights)
        if not np.isfinite(total) or total <= 0:
            msg = "quantile weights must have a positive finite sum"
            raise TracingError(msg)
        cumulative = np.cumsum(sorted_weights) / total
        selected = tuple(
            sorted_source[
                min(
                    int(np.searchsorted(cumulative, quantile_value, side="left")),
                    reduction_size - 1,
                )
            ]
            for quantile_value in flat_quantiles
        )
        row_result = np.reshape(np.stack(selected), quantile_shape)
        rows.append(row_result)

    result = np.reshape(np.stack(tuple(rows), axis=-1), (*quantile_shape, *batch_shape))
    if isinstance(weights, traced_type):
        result = result + np.sum(weights) * 0
    if isinstance(quantile, traced_type):
        result = result + np.sum(quantile) * 0
    if keepdims:
        reduced = set(axes)
        kept_shape = tuple(
            1 if index in reduced else int(source.shape[index]) for index in range(int(source.ndim))
        )
        result = np.reshape(result, (*quantile_shape, *kept_shape))
    return result


def _quantile_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    percentile: bool,
    ignore_nan: bool = False,
) -> CompositeResult:
    name = f"{'nan' if ignore_nan else ''}{'percentile' if percentile else 'quantile'}"
    if len(args) < _BINARY_ARITY or len(args) > _MAX_QUANTILE_POSITIONAL:
        msg = f"numpy.{name} received an invalid positional signature during tracing"
        raise TracingError(msg)
    positional_names = ("axis", "out", "overwrite_input", "method", "keepdims")
    supported_keywords = {*positional_names, "interpolation", "weights"}
    unsupported = set(kwargs) - supported_keywords
    if unsupported:
        msg = f"numpy.{name} kwargs not supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)
    values = dict(kwargs)
    for attr_name, attr_value in zip(positional_names, args[2:], strict=False):
        if attr_name in values:
            msg = f"numpy.{name} received {attr_name} twice"
            raise TracingError(msg)
        values[attr_name] = attr_value
    if values.get("out") is not None:
        msg = f"numpy.{name} out= is not supported during tracing"
        raise TracingError(msg)
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
    ignore_nan: bool = False,
) -> CompositeResult:
    if not args or len(args) > _MAX_MEDIAN_POSITIONAL:
        name = "nanmedian" if ignore_nan else "median"
        msg = (
            f"numpy.{name} expects (a, axis=None, out=None, overwrite_input=False, keepdims=False)"
        )
        raise TracingError(msg)
    positional_names = ("axis", "out", "overwrite_input", "keepdims")
    unsupported = set(kwargs) - set(positional_names)
    if unsupported:
        name = "nanmedian" if ignore_nan else "median"
        msg = f"numpy.{name} kwargs not supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)
    values = dict(kwargs)
    for name, value in zip(positional_names, args[1:], strict=False):
        if name in values:
            function_name = "nanmedian" if ignore_nan else "median"
            msg = f"numpy.{function_name} received {name} twice"
            raise TracingError(msg)
        values[name] = value
    if values.get("out") is not None:
        function_name = "nanmedian" if ignore_nan else "median"
        msg = f"numpy.{function_name} out= is not supported during tracing"
        raise TracingError(msg)
    if bool(values.get("overwrite_input", False)):
        function_name = "nanmedian" if ignore_nan else "median"
        msg = f"numpy.{function_name}(overwrite_input=True) would mutate its input during tracing"
        raise TracingError(msg)
    result_fn = _nan_quantile_result if ignore_nan else _quantile_result
    result = result_fn(
        args[0],
        0.5,
        axis=values.get("axis"),
        keepdims=bool(values.get("keepdims", False)),
        method="midpoint",
        scale=1.0,
        traced_type=traced_type,
    )
    return _finish(result, traced_type=traced_type)


def register_statistics_handlers(
    handlers: dict[Callable[..., Any], Callable[..., Any]],
) -> None:
    """Register differentiable order statistics."""
    handlers[np.quantile] = lambda graph, traced_type, args, kwargs: _quantile_handler(
        graph,
        traced_type,
        args,
        kwargs,
        percentile=False,
    )
    handlers[np.percentile] = lambda graph, traced_type, args, kwargs: _quantile_handler(
        graph,
        traced_type,
        args,
        kwargs,
        percentile=True,
    )
    handlers[np.median] = _median_handler
    handlers[np.nanquantile] = lambda graph, traced_type, args, kwargs: _quantile_handler(
        graph,
        traced_type,
        args,
        kwargs,
        percentile=False,
        ignore_nan=True,
    )
    handlers[np.nanpercentile] = lambda graph, traced_type, args, kwargs: _quantile_handler(
        graph,
        traced_type,
        args,
        kwargs,
        percentile=True,
        ignore_nan=True,
    )
    handlers[np.nanmedian] = lambda graph, traced_type, args, kwargs: _median_handler(
        graph,
        traced_type,
        args,
        kwargs,
        ignore_nan=True,
    )
