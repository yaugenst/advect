"""Traceable real adjoints for portable gather operations."""

from __future__ import annotations

from math import prod
from typing import Any, cast

from advect.autodiff.rules.array_family._backend_runtime import (
    _along_axis_positions,
    _clip_indices,
    _int64_indices,
    _moveaxis,
    xp,
)
from advect.autodiff.rules.array_family._transpose_utils import (
    _normalize_axis,
    _shape_of,
)
from advect.autodiff.rules.array_family.jvp.common import _astype_preserving_trace
from advect.core._scatter_add import scatter_add


def _normalized_indices(indices: object, axis_size: int, *, mode: str) -> xp.ndarray:
    index_array = cast("Any", _int64_indices(indices))
    if mode == "clip":
        return _clip_indices(index_array, axis_size)
    zero = xp.zeros_like(index_array)
    size = xp.full_like(index_array, axis_size)
    if mode == "wrap":
        return cast("xp.ndarray", xp.remainder(index_array, size))
    return cast(
        "xp.ndarray",
        xp.where(
            index_array < zero,
            index_array + size,
            index_array,
        ),
    )


def _reshaped(value: Any, shape: tuple[int, ...]) -> xp.ndarray:
    return value if _shape_of(value) == shape else xp.reshape(value, shape)


def _vjp_take(
    ans: xp.ndarray,
    x: xp.ndarray,
    indices: xp.ndarray,
    *rest: object,
    g: xp.ndarray,
    axis: int | None = None,
    mode: str = "raise",
    **attrs: Any,
) -> tuple[xp.ndarray, None]:
    """Scatter-add the cotangent, including repeated indices.

    ``axis=None`` takes from the flattened source, so it scatters along the
    only axis of that flattening. NumPy takes along the axis 0 or -1 of a
    rank-0 source from that one-element flattening too.
    """
    _ = ans, rest, attrs
    source_shape = _shape_of(x)
    if not source_shape:
        axis = None
    shape = (prod(source_shape),) if axis is None else source_shape
    normalized_axis = (
        0 if axis is None else _normalize_axis(axis, ndim=len(shape), op_name="gather")
    )
    size = shape[normalized_axis]
    count = prod(_shape_of(indices))
    cotangent = _astype_preserving_trace(g, dtype=cast("Any", x).dtype)
    scattered = scatter_add(
        _reshaped(cotangent, (*shape[:normalized_axis], count, *shape[normalized_axis + 1 :])),
        _reshaped(_normalized_indices(indices, size, mode=mode), (count,)),
        axis=normalized_axis,
        size=size,
    )
    return _reshaped(scattered, source_shape), None


def _vjp_take_along_axis(
    ans: xp.ndarray,
    x: xp.ndarray,
    indices: xp.ndarray,
    *rest: object,
    g: xp.ndarray,
    axis: int = -1,
    **attrs: Any,
) -> tuple[xp.ndarray, None]:
    """Scatter-add the flat positions the gather read, then undo broadcasting.

    This transposes the flat ``take`` that the JVP lowers to, so a source axis
    of length one accumulates the cotangent of every row it broadcast to.
    """
    _ = ans, rest, attrs
    source_shape = _shape_of(x)
    if len(_shape_of(indices)) != len(source_shape):
        msg = "take_along_axis derivative requires indices with the source rank"
        raise ValueError(msg)
    normalized_axis = _normalize_axis(axis, ndim=len(source_shape), op_name="gather")
    axis_size = source_shape[normalized_axis]
    positions, leading = _along_axis_positions(indices, source_shape, normalized_axis)
    cotangent = _moveaxis(
        _astype_preserving_trace(g, dtype=cast("Any", x).dtype), normalized_axis, -1
    )
    scattered = xp.reshape(
        scatter_add(
            xp.reshape(cotangent, (-1,)), positions, axis=0, size=prod(leading) * axis_size
        ),
        (*leading, axis_size),
    )
    source_leading = (*source_shape[:normalized_axis], *source_shape[normalized_axis + 1 :])
    broadcast_axes = tuple(
        dimension
        for dimension, (source_size, size) in enumerate(zip(source_leading, leading, strict=True))
        if source_size != size
    )
    if broadcast_axes:
        scattered = xp.sum(scattered, axis=broadcast_axes, keepdims=True, dtype=scattered.dtype)
    return _moveaxis(scattered, -1, normalized_axis), None


def _vjp_scatter_add(
    ans: xp.ndarray,
    values: xp.ndarray,
    indices: xp.ndarray,
    *rest: object,
    g: xp.ndarray,
    axis: int,
    **attrs: Any,
) -> tuple[xp.ndarray, None]:
    """Gather each position's cotangent back to every entry scattered there."""
    _ = ans, values, rest, attrs
    return xp.take(g, indices, axis=axis), None


def _vjp_bincount(
    ans: xp.ndarray,
    indices: xp.ndarray,
    weights: xp.ndarray,
    *rest: object,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[None, xp.ndarray]:
    """Gather each bin cotangent back to its continuous input weight."""
    _ = ans, weights, rest, attrs
    return None, xp.take(g, indices)
