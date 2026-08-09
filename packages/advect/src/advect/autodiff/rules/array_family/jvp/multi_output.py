"""Multi Output JVP rules."""

from __future__ import annotations

from typing import Any, cast

from advect.autodiff.rules.array_family._backend_runtime import xp
from advect.autodiff.rules.array_family.jvp.common import (
    _asarray_unwrapped,
    _infer_output_tangent_dtype,
    _zeros_output_tangent_structure,
)


def _jvp_modf(
    ans: tuple[xp.ndarray, xp.ndarray],
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray]:
    _ = x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return cast("tuple[xp.ndarray, xp.ndarray]", _zeros_output_tangent_structure(ans, tangents))
    dtype = _infer_output_tangent_dtype(ans, tangents)
    integral = xp.zeros_like(_asarray_unwrapped(ans[1]), dtype=dtype)
    return (tangent, integral)


def _jvp_frexp(
    ans: tuple[xp.ndarray, xp.ndarray],
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray]:
    _ = x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return cast("tuple[xp.ndarray, xp.ndarray]", _zeros_output_tangent_structure(ans, tangents))
    mantissa, exponent = ans
    mantissa_arr = _asarray_unwrapped(mantissa)
    exponent_arr = _asarray_unwrapped(exponent)
    scale = xp.ldexp(xp.ones_like(mantissa_arr), -exponent_arr)
    d_mantissa = tangent * scale
    d_exponent = xp.zeros_like(exponent_arr, dtype=_infer_output_tangent_dtype(ans, tangents))
    return (d_mantissa, d_exponent)


def _jvp_divmod(
    ans: tuple[xp.ndarray, xp.ndarray],
    x: xp.ndarray,
    y: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray]:
    _ = x, y, rest, attrs
    dx = tangents[0] if len(tangents) > 0 else None
    dy = tangents[1] if len(tangents) > 1 else None
    if dx is None and dy is None:
        return cast("tuple[xp.ndarray, xp.ndarray]", _zeros_output_tangent_structure(ans, tangents))
    quotient, _ = ans
    dtype = _infer_output_tangent_dtype(ans, tangents)
    d_quotient = xp.zeros_like(_asarray_unwrapped(quotient), dtype=dtype)
    d_remainder = xp.zeros_like(_asarray_unwrapped(ans[1]), dtype=dtype)
    if dx is not None:
        d_remainder = d_remainder + dx
    if dy is not None:
        d_remainder = d_remainder - quotient * dy
    return (d_quotient, d_remainder)
