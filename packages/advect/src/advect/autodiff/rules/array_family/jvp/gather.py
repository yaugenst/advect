"""JVP rules for portable gather operations."""

from __future__ import annotations

from typing import Any, cast

from advect.autodiff.rules.array_family._backend_runtime import (
    _clip_indices,
    _int64_indices,
    _take_along_axis,
    xp,
)
from advect.autodiff.rules.array_family.jvp.common import (
    _zeros_output_tangent,
)


def _jvp_take(
    ans: xp.ndarray,
    x: xp.ndarray,
    indices: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axis: int | None = None,
    mode: str = "raise",
    **attrs: Any,
) -> xp.ndarray:
    _ = x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    if not tangent.shape:
        # NumPy takes along the axis 0 or -1 of a rank-0 source from its
        # one-element flattening.
        axis = None
    if mode == "raise":
        return xp.take(tangent, indices, axis=axis)
    axis_size = tangent.size if axis is None else tangent.shape[axis]
    positions = _int64_indices(indices)
    if mode == "wrap":
        positions = xp.remainder(positions, axis_size)
    else:
        positions = _clip_indices(positions, axis_size)
    return xp.take(tangent, positions, axis=axis)


def _jvp_take_along_axis(
    ans: xp.ndarray,
    x: xp.ndarray,
    indices: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axis: int = -1,
    **attrs: Any,
) -> xp.ndarray:
    _ = x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return _take_along_axis(tangent, indices, axis=axis)


def _jvp_bincount(
    ans: xp.ndarray,
    indices: xp.ndarray,
    weights: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    minlength: int = 0,
    **attrs: Any,
) -> xp.ndarray:
    """Differentiate only the continuous weights of a discrete bincount."""
    _ = indices, weights, rest, attrs
    tangent = tangents[1] if len(tangents) > 1 else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return cast(
        "xp.ndarray",
        xp.bincount(indices, weights=tangent, minlength=minlength),
    )
