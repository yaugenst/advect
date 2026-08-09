"""Reductions JVP rules."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, cast

from advect.autodiff.rules.array_family._backend_runtime import _moveaxis, _scalar_like, xp
from advect.autodiff.rules.array_family.jvp.common import (
    _asarray_preserving_trace,
    _asarray_unwrapped,
    _astype_preserving_trace,
    _flatten_reduction_axes,
    _maxmin_tangent,
    _nan_maxmin_tangent,
    _normalize_axis_tuple,
    _normalize_cumulative_axis,
    _prod_jvp_last_axis,
    _reshape_reduction_result,
    _shape_unwrapped,
    _zeros_output_tangent,
)


def _real_ddof(value: object, *, operation: str) -> float:
    if not isinstance(value, Real):
        msg = f"{operation} JVP requires a real-valued ddof"
        raise NotImplementedError(msg)
    return float(value)


def _nanmean_keepdims(
    x: xp.ndarray,
    mask: xp.ndarray,
    *,
    axis: int | tuple[int, ...] | None,
) -> xp.ndarray:
    valid_count = xp.sum(xp.logical_not(mask), axis=axis, keepdims=True)
    safe_count = xp.where(valid_count == 0, xp.ones_like(valid_count), valid_count)
    total = xp.sum(
        xp.where(mask, xp.zeros_like(x), x),
        axis=axis,
        keepdims=True,
    )
    return cast("xp.ndarray", total / safe_count)


def _jvp_cumsum(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axis: int | None = None,
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.cumsum."""
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return xp.cumsum(_asarray_preserving_trace(tangent), axis=axis)


def _jvp_max(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    initial: object | None = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    result = cast(
        "xp.ndarray[Any, Any]",
        _maxmin_tangent(
            x,
            tangent,
            axis=axis,
            keepdims=keepdims,
            reduce_kind="max",
        ),
    )
    if initial is None:
        return result
    base = xp.max(x, axis=axis, keepdims=keepdims)
    return cast(
        "xp.ndarray[Any, Any]",
        xp.where(base < initial, xp.zeros_like(result), result),
    )


def _jvp_min(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    initial: object | None = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    result = cast(
        "xp.ndarray[Any, Any]",
        _maxmin_tangent(
            x,
            tangent,
            axis=axis,
            keepdims=keepdims,
            reduce_kind="min",
        ),
    )
    if initial is None:
        return result
    base = xp.min(x, axis=axis, keepdims=keepdims)
    return cast(
        "xp.ndarray[Any, Any]",
        xp.where(base > initial, xp.zeros_like(result), result),
    )


def _jvp_cumprod(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axis: object = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)

    x_shape = _shape_unwrapped(x)
    axis_norm = _normalize_cumulative_axis(axis, ndim=len(x_shape))
    out_dtype = xp.result_type(x, tangent, xp.float64)

    if axis_norm is None:
        flat_size = math.prod(x_shape)
        x_last = xp.reshape(x, (1, flat_size))
        dx_last = xp.reshape(tangent, (1, flat_size))
        y_last = xp.cumprod(x_last, axis=-1)
        output_count = int(_shape_unwrapped(y_last)[-1])
        if output_count == 0:
            return xp.astype(tangent, out_dtype) * 0
        terms: list[Any] = [dx_last[..., 0]]
        for index in range(1, output_count):
            terms.append(
                terms[index - 1] * x_last[..., index] + y_last[..., index - 1] * dx_last[..., index]
            )
        return xp.reshape(xp.stack(terms, axis=-1), _shape_unwrapped(ans))

    x_last = _moveaxis(x, axis_norm, -1)
    dx_last = _moveaxis(tangent, axis_norm, -1)
    y_last = xp.cumprod(x_last, axis=-1)
    output_count = int(_shape_unwrapped(y_last)[-1])
    if output_count == 0:
        return xp.astype(tangent, out_dtype) * 0
    terms = [dx_last[..., 0]]
    for index in range(1, output_count):
        terms.append(
            terms[index - 1] * x_last[..., index] + y_last[..., index - 1] * dx_last[..., index]
        )
    out_last = xp.stack(terms, axis=-1)
    return _moveaxis(out_last, -1, axis_norm)


def _jvp_nanmean(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    mask = xp.isnan(_asarray_preserving_trace(x))
    dx_eff = xp.where(mask, xp.zeros_like(tangent), tangent)
    count = xp.sum(xp.logical_not(mask), axis=axis, keepdims=keepdims)
    safe_count = xp.where(count == 0, xp.ones_like(count), count)
    quotient = xp.sum(dx_eff, axis=axis, keepdims=keepdims) / safe_count
    out = xp.where(
        count == 0,
        xp.zeros_like(quotient),
        quotient,
    )
    return cast(
        "xp.ndarray[Any, Any]",
        _astype_preserving_trace(out, dtype=_asarray_unwrapped(ans).dtype),
    )


def _jvp_nanmin(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    initial: object | None = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    result = cast(
        "xp.ndarray[Any, Any]",
        _nan_maxmin_tangent(
            x,
            tangent,
            axis=axis,
            keepdims=keepdims,
            reduce_kind="min",
        ),
    )
    if initial is None:
        return result
    base = xp.nanmin(x, axis=axis, keepdims=keepdims)
    return cast(
        "xp.ndarray[Any, Any]",
        xp.where(base > initial, xp.zeros_like(result), result),
    )


def _jvp_nanmax(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    initial: object | None = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    result = cast(
        "xp.ndarray[Any, Any]",
        _nan_maxmin_tangent(
            x,
            tangent,
            axis=axis,
            keepdims=keepdims,
            reduce_kind="max",
        ),
    )
    if initial is None:
        return result
    base = xp.nanmax(x, axis=axis, keepdims=keepdims)
    return cast(
        "xp.ndarray[Any, Any]",
        xp.where(base < initial, xp.zeros_like(result), result),
    )


def _jvp_nansum(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    dtype: Any = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    mask = xp.logical_not(xp.isnan(_asarray_preserving_trace(x)))
    out = xp.sum(
        xp.where(mask, tangent, xp.zeros_like(tangent)),
        axis=axis,
        keepdims=keepdims,
        dtype=dtype,
    )
    return cast(
        "xp.ndarray[Any, Any]",
        _astype_preserving_trace(out, dtype=_asarray_unwrapped(ans).dtype),
    )


def _jvp_nanprod(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    initial: object | None = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    x_shape = _shape_unwrapped(x)
    mask = xp.isnan(_asarray_preserving_trace(x))
    x_eff = xp.where(mask, xp.ones_like(x), x)
    dx_eff = xp.where(mask, xp.zeros_like(tangent), tangent)
    axes = _normalize_axis_tuple(axis, ndim=len(x_shape))
    x_flat, _ = _flatten_reduction_axes(x_eff, axes=axes)
    dx_flat, _ = _flatten_reduction_axes(dx_eff, axes=axes)
    reduced = _prod_jvp_last_axis(x_flat, dx_flat)
    result = cast(
        "xp.ndarray[Any, Any]",
        _reshape_reduction_result(
            reduced,
            input_shape=x_shape,
            axes=axes,
            keepdims=keepdims,
        ),
    )
    return result if initial is None else cast("xp.ndarray[Any, Any]", result * initial)


def _jvp_var(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    ddof: object = 0,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, rest
    ddof = _real_ddof(attrs.get("correction", ddof), operation="var")
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    x_shape = _shape_unwrapped(x)
    axes = _normalize_axis_tuple(axis, ndim=len(x_shape))
    n = math.prod(x_shape[index] for index in axes)
    den = n - ddof
    if den <= 0:
        msg = "var JVP requires count > ddof for reduced slices"
        raise NotImplementedError(msg)
    mean = xp.mean(x, axis=axis, keepdims=True)
    centered = x - mean
    numerator = xp.sum(
        xp.real(xp.conjugate(centered) * tangent),
        axis=axis,
        keepdims=keepdims,
    )
    return cast(
        "xp.ndarray[Any, Any]",
        _astype_preserving_trace(
            _scalar_like(2.0 / den, ans) * numerator,
            dtype=_asarray_unwrapped(ans).dtype,
        ),
    )


def _jvp_std(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    ddof: object = 0,
    **attrs: Any,
) -> xp.ndarray:
    _ = rest
    ddof = _real_ddof(attrs.get("correction", ddof), operation="std")
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    x_shape = _shape_unwrapped(x)
    axes = _normalize_axis_tuple(axis, ndim=len(x_shape))
    n = math.prod(x_shape[index] for index in axes)
    den = n - ddof
    if den <= 0:
        msg = "std JVP requires count > ddof for reduced slices"
        raise NotImplementedError(msg)
    mean = xp.mean(x, axis=axis, keepdims=True)
    centered = x - mean
    numerator = xp.sum(
        xp.real(xp.conjugate(centered) * tangent),
        axis=axis,
        keepdims=keepdims,
    )
    zero_ans = xp.zeros_like(ans)
    safe_ans = xp.where(ans == zero_ans, xp.ones_like(ans), ans)
    quotient = numerator / (_scalar_like(den, safe_ans) * safe_ans)
    out = xp.where(ans == zero_ans, xp.zeros_like(quotient), quotient)
    return cast(
        "xp.ndarray[Any, Any]",
        _astype_preserving_trace(out, dtype=_asarray_unwrapped(ans).dtype),
    )


def _jvp_nanvar(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    ddof: object = 0,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, rest
    ddof = _real_ddof(attrs.get("correction", ddof), operation="nanvar")
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    mask = xp.isnan(_asarray_preserving_trace(x))
    count = xp.sum(xp.logical_not(mask), axis=axis, keepdims=keepdims)
    mean = _nanmean_keepdims(x, mask, axis=axis)
    centered_raw = x - mean
    centered = xp.where(mask, xp.zeros_like(centered_raw), centered_raw)
    dx_eff = xp.where(mask, xp.zeros_like(tangent), tangent)
    numerator = xp.sum(
        xp.real(xp.conjugate(centered) * dx_eff),
        axis=axis,
        keepdims=keepdims,
    )
    out = (_scalar_like(2.0, count) * numerator) / (count - _scalar_like(ddof, count))
    return cast(
        "xp.ndarray[Any, Any]",
        _astype_preserving_trace(out, dtype=_asarray_unwrapped(ans).dtype),
    )


def _jvp_nanstd(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    ddof: object = 0,
    **attrs: Any,
) -> xp.ndarray:
    _ = rest
    ddof = _real_ddof(attrs.get("correction", ddof), operation="nanstd")
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    mask = xp.isnan(_asarray_preserving_trace(x))
    count = xp.sum(xp.logical_not(mask), axis=axis, keepdims=keepdims)
    mean = _nanmean_keepdims(x, mask, axis=axis)
    centered_raw = x - mean
    centered = xp.where(mask, xp.zeros_like(centered_raw), centered_raw)
    dx_eff = xp.where(mask, xp.zeros_like(tangent), tangent)
    numerator = xp.sum(
        xp.real(xp.conjugate(centered) * dx_eff),
        axis=axis,
        keepdims=keepdims,
    )
    zero_ans = xp.zeros_like(ans)
    safe_ans = xp.where(ans == zero_ans, xp.ones_like(ans), ans)
    quotient = numerator / ((count - _scalar_like(ddof, count)) * safe_ans)
    out = xp.where(ans == zero_ans, xp.zeros_like(quotient), quotient)
    return cast(
        "xp.ndarray[Any, Any]",
        _astype_preserving_trace(out, dtype=_asarray_unwrapped(ans).dtype),
    )


def _jvp_prod(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    initial: object | None = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    x_shape = _shape_unwrapped(x)
    axes = _normalize_axis_tuple(axis, ndim=len(x_shape))
    x_flat, _ = _flatten_reduction_axes(x, axes=axes)
    dx_flat, _ = _flatten_reduction_axes(tangent, axes=axes)
    reduced = _prod_jvp_last_axis(x_flat, dx_flat)
    result = cast(
        "xp.ndarray[Any, Any]",
        _reshape_reduction_result(
            reduced,
            input_shape=x_shape,
            axes=axes,
            keepdims=keepdims,
        ),
    )
    return result if initial is None else cast("xp.ndarray[Any, Any]", result * initial)
