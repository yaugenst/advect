"""Elementwise JVP rules."""

from __future__ import annotations

from typing import Any, cast

from advect.autodiff.rules.array_family._backend_runtime import _scalar_like, xp
from advect.autodiff.rules.array_family._transpose_utils import dtype_is_inexact
from advect.autodiff.rules.array_family.jvp.common import (
    _WHERE_INPUT_ARITY,
    _asarray_preserving_trace,
    _asarray_unwrapped,
    _astype_preserving_trace,
    _coerce_tangent_or_zeros,
    _infer_tangent_dtype,
    _iscomplex_unwrapped,
    _validate_tangent_arity,
    _zeros_output_tangent,
    linear_jvp,
)
from advect.core._protocols import _innermost


@linear_jvp
def _jvp_conjugate(t: Any, **_: Any) -> Any:
    """Conjugation is real-linear, so it conjugates the tangent."""
    return xp.conjugate(t)


def _jvp_where(
    ans: xp.ndarray,
    condition: xp.ndarray,
    x: xp.ndarray,
    y: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, x, y, rest, attrs
    _validate_tangent_arity(
        op_name="numpy.where",
        inputs=(condition, x, y),
        tangents=tangents[:_WHERE_INPUT_ARITY],
    )
    if len(tangents) < _WHERE_INPUT_ARITY:
        msg = "numpy.where JVP requires tangents for (condition, x, y) slots"
        raise RuntimeError(msg)
    dx = tangents[1]
    dy = tangents[2]
    if dx is None and dy is None:
        return _zeros_output_tangent(ans, tangents)
    dtype = _infer_tangent_dtype(ans, tangents)
    dx_arr = _coerce_tangent_or_zeros(dx, primal=ans, dtype=dtype)
    dy_arr = _coerce_tangent_or_zeros(dy, primal=ans, dtype=dtype)
    condition_mask = xp.asarray(_innermost(condition))
    return xp.where(condition_mask, dx_arr, dy_arr)


def _jvp_clip(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    a_min: object | None = None,
    a_max: object | None = None,
    **attrs: Any,
) -> xp.ndarray:
    min_is_input = bool(attrs.get("_advect_clip_min_is_input", False))
    max_is_input = bool(attrs.get("_advect_clip_max_is_input", False))

    if not tangents:
        msg = "numpy.clip JVP requires at least one tangent slot"
        raise RuntimeError(msg)
    tx = tangents[0]
    rest_values = list(rest)
    min_value = a_min
    max_value = a_max
    tmin: xp.ndarray | None = None
    tmax: xp.ndarray | None = None
    cursor = 0
    tangent_cursor = 1
    if min_is_input:
        if cursor >= len(rest_values):
            msg = "numpy.clip JVP expected traced a_min primal input"
            raise RuntimeError(msg)
        min_value = rest_values[cursor]
        cursor += 1
        tmin = tangents[tangent_cursor] if tangent_cursor < len(tangents) else None
        tangent_cursor += 1
    if max_is_input:
        if cursor >= len(rest_values):
            msg = "numpy.clip JVP expected traced a_max primal input"
            raise RuntimeError(msg)
        max_value = rest_values[cursor]
        cursor += 1
        tmax = tangents[tangent_cursor] if tangent_cursor < len(tangents) else None
        tangent_cursor += 1
    if cursor != len(rest_values):
        msg = (
            "numpy.clip JVP received unexpected primal inputs "
            f"(expected {cursor}, got {len(rest_values)})"
        )
        raise RuntimeError(msg)

    # clip is minimum(maximum(x, min), max), and ties select x. Each mask
    # combines every present operand, so it has the broadcast result shape
    # even where x or a bound is smaller; crossed bounds select max. Bounds
    # are compared in the result dtype, as clip rounds a weak Python bound.
    dtype = _asarray_unwrapped(ans).dtype
    x_arr = _asarray_unwrapped(x)
    min_arr, max_arr = (
        None if bound is None else xp.asarray(_innermost(bound), dtype=dtype)
        for bound in (min_value, max_value)
    )
    interior_mask = xp.ones_like(x_arr, dtype=xp.bool)
    if min_arr is not None:
        interior_mask = xp.logical_and(interior_mask, x_arr >= min_arr)
    if max_arr is not None:
        interior_mask = xp.logical_and(interior_mask, x_arr <= max_arr)
    if min_arr is not None and max_arr is not None:
        crossed = min_arr > max_arr
        below_mask = xp.logical_and(x_arr < min_arr, xp.logical_not(crossed))
        above_mask = xp.logical_or(x_arr > max_arr, crossed)
    else:
        below_mask = None if min_arr is None else x_arr < min_arr
        above_mask = None if max_arr is None else x_arr > max_arr

    out: Any | None = None
    if tx is not None:
        out = xp.where(interior_mask, tx, xp.zeros_like(tx))
    for tangent, mask in ((tmin, below_mask), (tmax, above_mask)):
        if tangent is not None and mask is not None:
            contribution = xp.where(mask, tangent, xp.zeros_like(tangent))
            out = contribution if out is None else xp.add(out, contribution)
    if out is None:
        return _zeros_output_tangent(ans, tangents)
    return cast("xp.ndarray[Any, Any]", out)


def _jvp_real(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray, ...],
    **attrs: Any,
) -> xp.ndarray:
    _ = rest, attrs
    tangent = tangents[0]
    ans_dtype = _asarray_unwrapped(ans).dtype
    if _iscomplex_unwrapped(x):
        return cast(
            "xp.ndarray[Any, Any]", _astype_preserving_trace(xp.real(tangent), dtype=ans_dtype)
        )
    return cast("xp.ndarray[Any, Any]", _astype_preserving_trace(tangent, dtype=ans_dtype))


def _jvp_imag(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray, ...],
    **attrs: Any,
) -> xp.ndarray:
    _ = rest, attrs
    tangent = tangents[0]
    tangent_arr = _asarray_preserving_trace(tangent)
    ans_dtype = _asarray_unwrapped(ans).dtype
    if _iscomplex_unwrapped(x):
        return cast(
            "xp.ndarray[Any, Any]",
            _astype_preserving_trace(xp.imag(tangent_arr), dtype=ans_dtype),
        )
    return xp.zeros_like(_asarray_unwrapped(ans), dtype=ans_dtype)


def _jvp_absolute(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray, ...],
    **attrs: Any,
) -> xp.ndarray:
    _ = rest, attrs
    tangent = tangents[0]
    if _iscomplex_unwrapped(x):
        # Keep the complex direction traced so nested derivatives see it.
        x_arr = _asarray_preserving_trace(x)
        at_zero = xp.abs(x_arr) == 0
        safe = xp.where(at_zero, _scalar_like(1.0, ans), ans)
        return xp.where(
            at_zero,
            _scalar_like(0.0, ans),
            xp.real(xp.multiply(xp.conjugate(x_arr), tangent)) / safe,
        )
    return cast("xp.ndarray[Any, Any]", xp.multiply(xp.sign(_asarray_unwrapped(x)), tangent))


def _jvp_sign(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray, ...],
    **attrs: Any,
) -> xp.ndarray:
    """Differentiate NumPy's complex unit-phase sign away from zero."""
    _ = rest, attrs
    tangent = tangents[0]
    concrete_x = _asarray_unwrapped(x)
    if not xp.iscomplexobj(concrete_x):
        return _zeros_output_tangent(ans, tangents)
    x_arr = _asarray_preserving_trace(x)
    tangent_arr = _asarray_preserving_trace(tangent)
    magnitude = xp.abs(x_arr)
    safe = xp.where(magnitude == 0, 1.0, magnitude)
    magnitude_tangent = xp.real(xp.multiply(xp.conjugate(x_arr), tangent_arr)) / safe
    result = xp.divide(tangent_arr, safe) - xp.multiply(x_arr, magnitude_tangent) / (safe * safe)
    return xp.where(magnitude == 0, 0.0, result)


def _jvp_astype(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray, ...],
    dtype: str | None = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = x, rest, dtype, attrs
    if not dtype_is_inexact(_asarray_unwrapped(ans).dtype):
        return _zeros_output_tangent(ans, tangents)
    return cast(
        "xp.ndarray[Any, Any]",
        _astype_preserving_trace(tangents[0], dtype=_asarray_unwrapped(ans).dtype),
    )


def _jvp_nan_to_num(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray, ...],
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, rest, attrs
    tangent = tangents[0]
    mask = xp.isfinite(_asarray_unwrapped(x))
    return cast("xp.ndarray[Any, Any]", xp.where(mask, tangent, 0.0))


def _jvp_angle(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray, ...],
    deg: bool = False,
    **attrs: Any,
) -> xp.ndarray:
    """Differentiate phase away from the origin under the real-linear convention."""
    _ = rest, attrs
    tangent = tangents[0]
    ans_dtype = _asarray_unwrapped(ans).dtype
    if not _iscomplex_unwrapped(x):
        return xp.zeros_like(_asarray_unwrapped(ans), dtype=ans_dtype)
    out = xp.imag(_asarray_preserving_trace(tangent) / x)
    if deg:
        out = out * (180.0 / xp.pi)
    return cast(
        "xp.ndarray[Any, Any]",
        _astype_preserving_trace(out, dtype=ans_dtype),
    )
