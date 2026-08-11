# ruff: noqa: PLR2004
"""Abstract registrations and evaluators for indexing-shaped operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from advect.core._abstract_helpers import (
    accumulation_dtype,
    broadcast_shape,
    diagonal_size,
    dtype_name,
    normalize_axis,
)
from advect.core._abstract_model import ArraySpec, rule

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from advect.core._abstract_model import AbstractRule, ResultEvaluator


RULES: dict[str, AbstractRule] = {
    "array.argsort": rule(
        "argsort",
        1,
        allowed=("axis", "descending", "kind", "order", "stable"),
    ),
    "array.diagonal": rule(
        "diagonal",
        1,
        allowed=("axis1", "axis2", "offset"),
    ),
    "array.searchsorted": rule(
        "searchsorted",
        2,
        allowed=("side", "sorter"),
    ),
    "array.take": rule("take", 2, allowed=("axis", "mode")),
    "array.take_along_axis": rule("take_along_axis", 2, allowed=("axis",)),
    "array.trace": rule(
        "trace",
        1,
        allowed=("axis1", "axis2", "dtype", "offset"),
    ),
}


def _argsort(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    first = specs[0]
    normalize_axis(attrs.get("axis", -1), len(first.shape))
    return (ArraySpec(first.shape, "int64"),)


def _searchsorted(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    if len(specs[0].shape) != 1:
        raise ValueError("searchsorted sorted input must be one-dimensional")
    side = attrs.get("side", "left")
    if side not in {"left", "right"}:
        raise ValueError("searchsorted side must be 'left' or 'right'")
    return (ArraySpec(specs[1].shape, "int64"),)


def _take(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    first = specs[0]
    axis_value = attrs.get("axis")
    if axis_value is None:
        shape = specs[1].shape
    else:
        axis = normalize_axis(axis_value, len(first.shape))
        shape = (*first.shape[:axis], *specs[1].shape, *first.shape[axis + 1 :])
    return (ArraySpec(shape, dtype_name(first.dtype)),)


def _take_along_axis(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    first = specs[0]
    if len(first.shape) != len(specs[1].shape):
        raise ValueError("take_along_axis inputs must have the same rank")
    axis = normalize_axis(attrs.get("axis", -1), len(first.shape))
    source_shape = list(first.shape)
    source_shape[axis] = 1
    return (
        ArraySpec(
            broadcast_shape(tuple(source_shape), specs[1].shape),
            dtype_name(first.dtype),
        ),
    )


def _diagonal(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    first = specs[0]
    if len(first.shape) < 2:
        raise ValueError("linalg.diagonal input must have at least two dimensions")
    offset = attrs.get("offset", 0)
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise TypeError("linalg.diagonal offset must be an integer")
    first_axis = normalize_axis(attrs.get("axis1", 0), len(first.shape))
    second_axis = normalize_axis(attrs.get("axis2", 1), len(first.shape))
    if first_axis == second_axis:
        raise ValueError("diagonal axes must be distinct")
    return (
        ArraySpec(
            (
                *(
                    size
                    for axis, size in enumerate(first.shape)
                    if axis not in {first_axis, second_axis}
                ),
                diagonal_size(
                    first.shape[first_axis],
                    first.shape[second_axis],
                    offset,
                ),
            ),
            dtype_name(first.dtype),
        ),
    )


def _trace(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    first = specs[0]
    if len(first.shape) < 2:
        raise ValueError("linalg.trace input must have at least two dimensions")
    first_axis = normalize_axis(attrs.get("axis1", 0), len(first.shape))
    second_axis = normalize_axis(attrs.get("axis2", 1), len(first.shape))
    if first_axis == second_axis:
        raise ValueError("trace axes must be distinct")
    dtype = (
        dtype_name(attrs["dtype"])
        if attrs.get("dtype") is not None
        else accumulation_dtype(
            first.dtype,
            array_api_version=attrs.get("_advect_array_api_version"),
        )
    )
    return (
        ArraySpec(
            tuple(
                size
                for axis, size in enumerate(first.shape)
                if axis not in {first_axis, second_axis}
            ),
            dtype,
        ),
    )


EVALUATORS: dict[str, ResultEvaluator] = {
    "argsort": _argsort,
    "searchsorted": _searchsorted,
    "take": _take,
    "take_along_axis": _take_along_axis,
    "diagonal": _diagonal,
    "trace": _trace,
}
