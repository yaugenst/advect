"""Finite-difference lowering for NumPy's ``gradient`` contract."""

from __future__ import annotations

from typing import Any, cast

_SECOND_EDGE_ORDER = 2


def operand_ndim(value: object) -> int:
    """Read rank without coercing a tracer to a concrete array."""
    ndim = getattr(value, "ndim", None)
    if ndim is not None:
        return int(ndim)
    shape = getattr(value, "shape", None)
    return len(shape) if shape is not None else 0


def _axis_slice(
    *,
    rank: int,
    axis: int,
    start: int | None,
    stop: int | None,
) -> tuple[slice, ...]:
    index = [slice(None)] * rank
    index[axis] = slice(start, stop)
    return tuple(index)


def _reshape_axis_coefficients(
    namespace: object,
    coefficients: object,
    *,
    rank: int,
    axis: int,
) -> object:
    coefficients_value = cast("Any", coefficients)
    shape = [1] * rank
    shape[axis] = int(coefficients_value.shape[0])
    return cast("Any", namespace).reshape(coefficients, tuple(shape))


def lower_gradient_axis(
    namespace: object,
    source: object,
    spacing: object,
    *,
    axis: int,
    edge_order: int,
) -> object:
    """Lower one NumPy-gradient output to slices and elementary array operations."""
    namespace_value = cast("Any", namespace)
    source_value = cast("Any", source)
    spacing_value = cast("Any", spacing)
    rank = int(source_value.ndim)
    length = int(source_value.shape[axis])
    minimum = 3 if edge_order == _SECOND_EDGE_ORDER else 2
    if length < minimum:
        msg = (
            f"gradient edge_order={edge_order} requires at least {minimum} points along axis {axis}"
        )
        raise ValueError(msg)

    first = _axis_slice(rank=rank, axis=axis, start=0, stop=1)
    second = _axis_slice(rank=rank, axis=axis, start=1, stop=2)
    third = _axis_slice(rank=rank, axis=axis, start=2, stop=3)
    before = _axis_slice(rank=rank, axis=axis, start=0, stop=-2)
    center = _axis_slice(rank=rank, axis=axis, start=1, stop=-1)
    after = _axis_slice(rank=rank, axis=axis, start=2, stop=None)
    antepenultimate = _axis_slice(rank=rank, axis=axis, start=-3, stop=-2)
    penultimate = _axis_slice(rank=rank, axis=axis, start=-2, stop=-1)
    last = _axis_slice(rank=rank, axis=axis, start=-1, stop=None)

    if operand_ndim(spacing) == 0:
        interior = (source_value[after] - source_value[before]) / (2 * spacing_value)
        if edge_order == 1:
            left = (source_value[second] - source_value[first]) / spacing_value
            right = (source_value[last] - source_value[penultimate]) / spacing_value
        else:
            left = (
                -1.5 * source_value[first] + 2.0 * source_value[second] - 0.5 * source_value[third]
            ) / spacing_value
            right = (
                0.5 * source_value[antepenultimate]
                - 2.0 * source_value[penultimate]
                + 1.5 * source_value[last]
            ) / spacing_value
        return namespace_value.concatenate((left, interior, right), axis=axis)

    if operand_ndim(spacing) != 1 or int(spacing_value.shape[0]) != length:
        msg = (
            "gradient coordinate spacing must be one-dimensional and match "
            f"axis {axis} length {length}"
        )
        raise ValueError(msg)
    deltas = namespace_value.diff(spacing)
    dx1 = deltas[:-1]
    dx2 = deltas[1:]
    coefficient_before = _reshape_axis_coefficients(
        namespace,
        -dx2 / (dx1 * (dx1 + dx2)),
        rank=rank,
        axis=axis,
    )
    coefficient_center = _reshape_axis_coefficients(
        namespace,
        (dx2 - dx1) / (dx1 * dx2),
        rank=rank,
        axis=axis,
    )
    coefficient_after = _reshape_axis_coefficients(
        namespace,
        dx1 / (dx2 * (dx1 + dx2)),
        rank=rank,
        axis=axis,
    )
    interior = (
        coefficient_before * source_value[before]
        + coefficient_center * source_value[center]
        + coefficient_after * source_value[after]
    )
    if edge_order == 1:
        left = (source_value[second] - source_value[first]) / deltas[0]
        right = (source_value[last] - source_value[penultimate]) / deltas[-1]
    else:
        left_dx1 = deltas[0]
        left_dx2 = deltas[1]
        left = (
            -(2 * left_dx1 + left_dx2) / (left_dx1 * (left_dx1 + left_dx2)) * source_value[first]
            + (left_dx1 + left_dx2) / (left_dx1 * left_dx2) * source_value[second]
            - left_dx1 / (left_dx2 * (left_dx1 + left_dx2)) * source_value[third]
        )
        right_dx1 = deltas[-2]
        right_dx2 = deltas[-1]
        right = (
            right_dx2 / (right_dx1 * (right_dx1 + right_dx2)) * source_value[antepenultimate]
            - (right_dx2 + right_dx1) / (right_dx1 * right_dx2) * source_value[penultimate]
            + (2 * right_dx2 + right_dx1)
            / (right_dx2 * (right_dx1 + right_dx2))
            * source_value[last]
        )
    return namespace_value.concatenate((left, interior, right), axis=axis)


__all__ = ["lower_gradient_axis", "operand_ndim"]
