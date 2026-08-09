"""Creation JVP rules."""

from __future__ import annotations

from typing import Any, cast

from advect.autodiff.rules.array_family._backend_runtime import xp
from advect.autodiff.rules.array_family.jvp.common import (
    _astype_preserving_trace,
    _copy_if_untraced_array,
    _shape_unwrapped,
    _zeros_output_tangent,
)


def _jvp_zeros_like(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.zeros_like."""
    _ = inputs, attrs
    return _zeros_output_tangent(ans, tangents)


def _jvp_ones_like(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.ones_like."""
    _ = inputs, attrs
    return _zeros_output_tangent(ans, tangents)


def _jvp_empty_like(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.empty_like."""
    _ = inputs, attrs
    return _zeros_output_tangent(ans, tangents)


def _jvp_full(
    ans: xp.ndarray,
    fill_value: xp.ndarray | complex,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.full."""
    _ = fill_value, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    broadcasted = xp.broadcast_to(tangent, _shape_unwrapped(ans))
    casted = _astype_preserving_trace(broadcasted, dtype=ans.dtype)
    return cast("xp.ndarray[Any, Any]", _copy_if_untraced_array(casted))


def _jvp_full_like(
    ans: xp.ndarray,
    a: xp.ndarray,
    fill_value: xp.ndarray | complex,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.full_like."""
    _ = a, fill_value, rest, attrs
    fill_tangent = tangents[1] if len(tangents) > 1 else None
    if fill_tangent is None:
        return _zeros_output_tangent(ans, tangents)
    broadcasted = xp.broadcast_to(fill_tangent, _shape_unwrapped(ans))
    casted = _astype_preserving_trace(broadcasted, dtype=ans.dtype)
    return cast("xp.ndarray[Any, Any]", _copy_if_untraced_array(casted))
