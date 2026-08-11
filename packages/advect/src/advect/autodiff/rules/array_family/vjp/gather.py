"""Traceable real adjoints for portable gather operations."""

from __future__ import annotations

from math import prod
from typing import Any, cast

from advect.autodiff.rules.array_family._backend_runtime import (
    _moveaxis,
    current_array_backend_provider,
    xp,
)
from advect.autodiff.rules.array_family.jvp.common import (
    _astype_preserving_trace,
    _is_traced_leaf,
)


def _shape(value: object) -> tuple[int, ...]:
    return tuple(int(dimension) for dimension in cast("Any", value).shape)


def _normalize_axis(axis: int, ndim: int) -> int:
    normalized = axis
    if normalized < 0:
        normalized += ndim
    if normalized < 0 or normalized >= ndim:
        msg = f"gather axis {axis} is out of range for rank {ndim}"
        raise ValueError(msg)
    return normalized


def _normalized_indices(indices: object, axis_size: int, *, mode: str) -> xp.ndarray:
    index_array = cast("Any", indices)
    zero = xp.zeros_like(index_array)
    size = xp.full_like(index_array, axis_size)
    if mode == "wrap":
        return cast("xp.ndarray", xp.remainder(index_array, size))
    if mode == "clip":
        return cast(
            "xp.ndarray",
            xp.clip(index_array, zero, xp.full_like(index_array, axis_size - 1)),
        )
    return cast(
        "xp.ndarray",
        xp.where(
            index_array < zero,
            index_array + size,
            index_array,
        ),
    )


def _positions(indices: object, axis_size: int, *, leading_rank: int) -> xp.ndarray:
    dtype = cast("Any", indices).dtype
    positions = xp.arange(axis_size, dtype=dtype)
    return cast(
        "xp.ndarray",
        xp.reshape(positions, (1,) * leading_rank + (axis_size,)),
    )


def _sum_axes(value: xp.ndarray, axes: tuple[int, ...]) -> xp.ndarray:
    return value if not axes else xp.sum(value, axis=axes, dtype=value.dtype)


def _concrete_add_at() -> Any | None:
    """Return a provider-native indexed-add implementation when one exists."""
    provider = current_array_backend_provider()
    if provider is None:
        return None
    add = getattr(provider.namespace, "add", None)
    add_at = getattr(add, "at", None)
    return add_at if callable(add_at) else None


def _concrete_take_pullback(
    cotangent: xp.ndarray,
    normalized: xp.ndarray,
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
    add_at: Any,
) -> xp.ndarray:
    """Scatter a concrete take cotangent without a dense index basis."""
    result = xp.zeros(source_shape, dtype=cotangent.dtype)
    if axis is None:
        add_at(xp.reshape(result, (-1,)), normalized, cotangent)
        return cast("xp.ndarray", result)

    output_rank = len(source_shape) - 1 + _ndim(normalized)
    coordinates: list[xp.ndarray] = []
    for source_axis, size in enumerate(source_shape):
        if source_axis < axis:
            shape = [1] * output_rank
            shape[source_axis] = size
            coordinates.append(xp.reshape(xp.arange(size, dtype=xp.int64), tuple(shape)))
        elif source_axis == axis:
            shape = (1,) * axis + _shape(normalized) + (1,) * (len(source_shape) - axis - 1)
            coordinates.append(xp.reshape(normalized, shape))
        else:
            output_axis = source_axis - 1 + _ndim(normalized)
            shape = [1] * output_rank
            shape[output_axis] = size
            coordinates.append(xp.reshape(xp.arange(size, dtype=xp.int64), tuple(shape)))
    add_at(result, tuple(coordinates), cotangent)
    return cast("xp.ndarray", result)


def _concrete_take_along_axis_pullback(
    cotangent: xp.ndarray,
    normalized: xp.ndarray,
    *,
    source_shape: tuple[int, ...],
    axis: int,
    add_at: Any,
) -> xp.ndarray:
    """Scatter a concrete take_along_axis cotangent, including duplicates."""
    result = xp.zeros(source_shape, dtype=cotangent.dtype)
    index_shape = _shape(normalized)
    coordinates: list[xp.ndarray] = []
    for dimension, (source_size, index_size) in enumerate(
        zip(source_shape, index_shape, strict=True)
    ):
        if dimension == axis:
            coordinates.append(normalized)
            continue
        shape = [1] * len(source_shape)
        shape[dimension] = index_size
        coordinate = xp.reshape(xp.arange(index_size, dtype=xp.int64), tuple(shape))
        if source_size == 1:
            coordinate = xp.zeros_like(coordinate)
        coordinates.append(coordinate)
    add_at(result, tuple(coordinates), cotangent)
    return cast("xp.ndarray", result)


def _ndim(value: object) -> int:
    return len(_shape(value))


def _restore_dtype(value: xp.ndarray, dtype: object) -> xp.ndarray:
    if getattr(value, "dtype", None) == dtype:
        return value
    return cast(
        "xp.ndarray",
        _astype_preserving_trace(value, dtype=cast("xp.dtype[Any]", dtype)),
    )


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
    """Scatter-add the cotangent, including repeated indices."""
    _ = ans, rest, attrs
    source_shape = _shape(x)
    index_shape = _shape(indices)
    source_dtype = cast("Any", x).dtype
    cotangent = _astype_preserving_trace(g, dtype=source_dtype)

    if axis is None:
        axis_size = prod(source_shape)
        normalized = _normalized_indices(indices, axis_size, mode=mode)
        add_at = _concrete_add_at()
        if (
            add_at is not None
            and not _is_traced_leaf(cotangent)
            and not _is_traced_leaf(normalized)
        ):
            return (
                _restore_dtype(
                    _concrete_take_pullback(
                        cotangent,
                        normalized,
                        source_shape=source_shape,
                        axis=None,
                        add_at=add_at,
                    ),
                    source_dtype,
                ),
                None,
            )
        positions = _positions(indices, axis_size, leading_rank=len(index_shape))
        mask = xp.astype(
            xp.expand_dims(normalized, axis=-1) == positions,
            source_dtype,
        )
        weighted = cast("xp.ndarray", xp.expand_dims(cotangent, axis=-1) * mask)
        flattened = _sum_axes(weighted, tuple(range(len(index_shape))))
        return (
            _restore_dtype(
                xp.reshape(flattened, source_shape),
                source_dtype,
            ),
            None,
        )

    normalized_axis = _normalize_axis(axis, len(source_shape))
    axis_size = source_shape[normalized_axis]
    output_rank = len(source_shape) - 1 + len(index_shape)
    prefix_axes = tuple(range(normalized_axis))
    suffix_axes = tuple(range(normalized_axis + len(index_shape), output_rank))
    index_axes = tuple(range(normalized_axis, normalized_axis + len(index_shape)))
    permutation = (*prefix_axes, *suffix_axes, *index_axes)
    transposed = (
        cotangent
        if permutation == tuple(range(output_rank))
        else xp.permute_dims(cotangent, permutation)
    )

    normalized = _normalized_indices(indices, axis_size, mode=mode)
    add_at = _concrete_add_at()
    if add_at is not None and not _is_traced_leaf(cotangent) and not _is_traced_leaf(normalized):
        return (
            _restore_dtype(
                _concrete_take_pullback(
                    cotangent,
                    normalized,
                    source_shape=source_shape,
                    axis=normalized_axis,
                    add_at=add_at,
                ),
                source_dtype,
            ),
            None,
        )
    positions = _positions(indices, axis_size, leading_rank=len(index_shape))
    mask = xp.astype(
        xp.expand_dims(normalized, axis=-1) == positions,
        source_dtype,
    )
    weighted = cast("xp.ndarray", xp.expand_dims(transposed, axis=-1) * mask)
    prefix_rank = len(source_shape) - 1
    scattered = _sum_axes(
        weighted,
        tuple(range(prefix_rank, prefix_rank + len(index_shape))),
    )
    return (
        _restore_dtype(
            _moveaxis(scattered, -1, normalized_axis),
            source_dtype,
        ),
        None,
    )


def _vjp_take_along_axis(
    ans: xp.ndarray,
    x: xp.ndarray,
    indices: xp.ndarray,
    *rest: object,
    g: xp.ndarray,
    axis: int = -1,
    **attrs: Any,
) -> tuple[xp.ndarray, None]:
    """Scatter-add along one axis and undo non-axis broadcasting."""
    _ = ans, rest, attrs
    source_shape = _shape(x)
    index_shape = _shape(indices)
    if len(index_shape) != len(source_shape):
        msg = "take_along_axis derivative requires indices with the source rank"
        raise ValueError(msg)
    normalized_axis = _normalize_axis(axis, len(source_shape))
    axis_size = source_shape[normalized_axis]
    source_dtype = cast("Any", x).dtype
    cotangent = _astype_preserving_trace(g, dtype=source_dtype)

    normalized = _normalized_indices(indices, axis_size, mode="raise")
    add_at = _concrete_add_at()
    if add_at is not None and not _is_traced_leaf(cotangent) and not _is_traced_leaf(normalized):
        return (
            _concrete_take_along_axis_pullback(
                cotangent,
                normalized,
                source_shape=source_shape,
                axis=normalized_axis,
                add_at=add_at,
            ),
            None,
        )
    positions = _positions(indices, axis_size, leading_rank=len(source_shape))
    mask = xp.astype(
        xp.expand_dims(normalized, axis=-1) == positions,
        source_dtype,
    )
    weighted = cast("xp.ndarray", xp.expand_dims(cotangent, axis=-1) * mask)
    scattered = xp.sum(weighted, axis=normalized_axis, dtype=weighted.dtype)

    target_moved_shape = (
        *source_shape[:normalized_axis],
        *source_shape[normalized_axis + 1 :],
        axis_size,
    )
    scattered_shape = _shape(scattered)
    reduce_axes = tuple(
        dimension
        for dimension, (actual, target) in enumerate(
            zip(scattered_shape, target_moved_shape, strict=True)
        )
        if target == 1 and actual != 1
    )
    if reduce_axes:
        scattered = xp.sum(
            scattered,
            axis=reduce_axes,
            dtype=scattered.dtype,
            keepdims=True,
        )
    scattered = xp.reshape(scattered, target_moved_shape)
    return (
        _moveaxis(scattered, -1, normalized_axis),
        None,
    )


def _vjp_bincount(
    ans: xp.ndarray,
    indices: xp.ndarray,
    weights: xp.ndarray,
    *rest: object,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[None, xp.ndarray]:
    """Gather each bin cotangent back to its continuous input weight."""
    _ = ans, rest, attrs
    gathered = xp.take(g, indices)
    return None, cast(
        "xp.ndarray",
        _astype_preserving_trace(gathered, dtype=cast("Any", weights).dtype),
    )
