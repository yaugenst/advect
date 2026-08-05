"""Elementwise JVP rules."""

from __future__ import annotations

from typing import Any, cast

from advect.autodiff.rules.array_family._backend_runtime import xp
from advect.autodiff.rules.array_family._impl.jvp.common import (
    _WHERE_INPUT_ARITY,
    _asarray_preserving_trace,
    _asarray_unwrapped,
    _astype_preserving_trace,
    _coerce_tangent_or_zeros,
    _infer_tangent_dtype,
    _iscomplex_unwrapped,
    _normalize_output_tangent,
    _unwrap_traced_leaf,
    _validate_tangent_arity,
    _zeros_output_tangent,
)
from advect.autodiff.rules.array_family._transpose_utils import dtype_is_inexact


def _active_tangent_operand(tangent: Any | None) -> Any | None:
    """Keep array/traced operands intact and coerce only container/scalar inputs."""
    if (
        tangent is None
        or hasattr(tangent, "shape")
        or callable(getattr(tangent, "_advect_snapshot", None))
    ):
        return tangent
    return xp.asarray(tangent)


def _prepare_signed_fanout_tangents(
    ans: Any,
    tangents: tuple[Any | None, ...],
) -> tuple[Any | None, Any | None, xp.dtype[Any]]:
    """Coerce active signed-fanout operands only when dtype promotion requires it."""
    tx = _active_tangent_operand(tangents[0] if len(tangents) > 0 else None)
    ty = _active_tangent_operand(tangents[1] if len(tangents) > 1 else None)
    ans_dtype = _asarray_unwrapped(ans).dtype
    tx_dtype = None if tx is None else _asarray_unwrapped(tx).dtype
    ty_dtype = None if ty is None else _asarray_unwrapped(ty).dtype

    if tx_dtype is None:
        target_dtype = (
            ans_dtype
            if ty_dtype is None or ty_dtype == ans_dtype
            else xp.dtype(xp.result_type(ans_dtype, ty_dtype))
        )
    elif ty_dtype is None:
        target_dtype = (
            ans_dtype if tx_dtype == ans_dtype else xp.dtype(xp.result_type(ans_dtype, tx_dtype))
        )
    elif tx_dtype == ans_dtype and ty_dtype == ans_dtype:
        target_dtype = ans_dtype
    else:
        target_dtype = xp.dtype(xp.result_type(ans_dtype, tx_dtype, ty_dtype))

    if tx is not None and tx_dtype != target_dtype:
        tx = _astype_preserving_trace(tx, dtype=target_dtype)
    if ty is not None and ty_dtype != target_dtype:
        ty = _astype_preserving_trace(ty, dtype=target_dtype)
    return tx, ty, target_dtype


def _jvp_add(
    ans: xp.ndarray,
    x: xp.ndarray,
    y: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.add."""
    _ = rest, attrs
    _validate_tangent_arity(op_name="numpy.add", inputs=(x, y), tangents=tangents[:2])
    tx, ty, target_dtype = _prepare_signed_fanout_tangents(ans, tangents)
    if tx is None:
        contribution = ty
    elif ty is None:
        contribution = tx
    else:
        contribution = tx + ty
    normalized = (
        None
        if contribution is None
        else _normalize_output_tangent(
            ans,
            tangents,
            contribution,
            target_dtype=target_dtype,
        )
    )
    return cast(
        "xp.ndarray[Any, Any]",
        _zeros_output_tangent(ans, tangents) if normalized is None else normalized,
    )


def _jvp_subtract(
    ans: xp.ndarray,
    x: xp.ndarray,
    y: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.subtract."""
    _ = rest, attrs
    _validate_tangent_arity(op_name="numpy.subtract", inputs=(x, y), tangents=tangents[:2])
    tx, ty, target_dtype = _prepare_signed_fanout_tangents(ans, tangents)
    if tx is None:
        contribution = None if ty is None else -ty
    elif ty is None:
        contribution = tx
    else:
        contribution = tx - ty
    normalized = (
        None
        if contribution is None
        else _normalize_output_tangent(
            ans,
            tangents,
            contribution,
            target_dtype=target_dtype,
        )
    )
    return cast(
        "xp.ndarray[Any, Any]",
        _zeros_output_tangent(ans, tangents) if normalized is None else normalized,
    )


def _jvp_conjugate(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.conjugate."""
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return cast("xp.ndarray[Any, Any]", xp.conjugate(tangent))


def _jvp_multiply(
    ans: xp.ndarray,
    x: xp.ndarray,
    y: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.multiply."""
    _ = rest, attrs
    _validate_tangent_arity(op_name="numpy.multiply", inputs=(x, y), tangents=tangents[:2])
    dx = _active_tangent_operand(tangents[0] if len(tangents) > 0 else None)
    dy = _active_tangent_operand(tangents[1] if len(tangents) > 1 else None)

    if dx is None:
        contribution = None if dy is None else x * dy
    elif dy is None:
        contribution = dx * y
    else:
        contribution = dx * y + x * dy
    normalized = (
        None if contribution is None else _normalize_output_tangent(ans, tangents, contribution)
    )
    return cast(
        "xp.ndarray[Any, Any]",
        _zeros_output_tangent(ans, tangents) if normalized is None else normalized,
    )


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
    condition_mask = xp.asarray(_unwrap_traced_leaf(condition))
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

    if tx is None and tmin is None and tmax is None:
        return _zeros_output_tangent(ans, tangents)

    x_arr = _asarray_unwrapped(x)
    min_arr = _asarray_unwrapped(min_value) if min_value is not None else None
    max_arr = _asarray_unwrapped(max_value) if max_value is not None else None
    interior_mask = xp.ones_like(x_arr, dtype=xp.bool)
    if min_arr is not None:
        interior_mask &= x_arr >= min_arr
    if max_arr is not None:
        interior_mask &= x_arr <= max_arr

    out: Any | None = None
    if tx is not None:
        out = xp.where(interior_mask, tx, xp.zeros_like(tx))
    if tmin is not None and min_arr is not None:
        contribution = xp.where(x_arr < min_arr, tmin, xp.zeros_like(tmin))
        out = contribution if out is None else out + contribution
    if tmax is not None and max_arr is not None:
        contribution = xp.where(x_arr > max_arr, tmax, xp.zeros_like(tmax))
        out = contribution if out is None else out + contribution
    if out is None:
        return _zeros_output_tangent(ans, tangents)
    return cast("xp.ndarray[Any, Any]", out)


def _jvp_real(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    _ = rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
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
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    _ = rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
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
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    _ = rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    x_arr = _asarray_unwrapped(x)
    if xp.iscomplexobj(x_arr):
        abs_x = xp.abs(x_arr)
        safe = xp.where(abs_x == 0, 1.0, abs_x)
        return xp.where(abs_x == 0, 0.0, xp.real(xp.conjugate(x_arr) * tangent / safe))
    return cast("xp.ndarray[Any, Any]", xp.sign(x_arr) * tangent)


def _jvp_fabs(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    _ = rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return cast("xp.ndarray[Any, Any]", xp.sign(_asarray_unwrapped(x)) * tangent)


def _jvp_sign(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    """Differentiate NumPy's complex unit-phase sign away from zero."""
    _ = rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    concrete_x = _asarray_unwrapped(x)
    if not xp.iscomplexobj(concrete_x):
        return _zeros_output_tangent(ans, tangents)
    x_arr = _asarray_preserving_trace(x)
    tangent_arr = _asarray_preserving_trace(tangent)
    magnitude = xp.abs(x_arr)
    safe = xp.where(magnitude == 0, 1.0, magnitude)
    magnitude_tangent = xp.real(xp.conjugate(x_arr) * tangent_arr) / safe
    result = tangent_arr / safe - x_arr * magnitude_tangent / (safe * safe)
    return cast("xp.ndarray", xp.where(magnitude == 0, 0.0, result))


def _jvp_astype(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    dtype: str | None = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = x, rest, dtype, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None or not dtype_is_inexact(_asarray_unwrapped(ans).dtype):
        return _zeros_output_tangent(ans, tangents)
    return cast(
        "xp.ndarray[Any, Any]",
        _astype_preserving_trace(tangent, dtype=_asarray_unwrapped(ans).dtype),
    )


def _jvp_nan_to_num(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    mask = xp.isfinite(_asarray_unwrapped(x))
    return cast("xp.ndarray[Any, Any]", xp.where(mask, tangent, 0.0))


def _jvp_sinc(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    x_arr = _asarray_unwrapped(x)
    pix = xp.pi * x_arr
    numerator = pix * xp.cos(pix) - xp.sin(pix)
    denom = xp.where(x_arr == 0, 1.0, xp.pi * x_arr * x_arr)
    deriv = xp.where(x_arr == 0, 0.0, numerator / denom)
    return cast("xp.ndarray[Any, Any]", deriv * tangent)


def _jvp_angle(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    deg: bool = False,
    **attrs: Any,
) -> xp.ndarray:
    """Differentiate phase away from the origin under the real-linear convention."""
    _ = rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
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
