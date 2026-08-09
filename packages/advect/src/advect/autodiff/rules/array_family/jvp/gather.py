"""JVP rules for portable gather operations."""

from __future__ import annotations

from typing import Any, cast

from advect.autodiff.rules.array_family._backend_runtime import _take_along_axis, xp
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
    axis_size = int(tangent.size) if axis is None else int(tangent.shape[int(axis)])
    normalized = indices
    if mode == "wrap":
        normalized = xp.remainder(indices, axis_size)
    elif mode == "clip":
        normalized = xp.clip(indices, 0, axis_size - 1)
    return cast("xp.ndarray", xp.take(tangent, normalized, axis=axis))


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
    return cast(
        "xp.ndarray",
        _take_along_axis(tangent, indices, axis=int(axis)),
    )


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
        xp.bincount(indices, weights=tangent, minlength=int(minlength)),
    )
