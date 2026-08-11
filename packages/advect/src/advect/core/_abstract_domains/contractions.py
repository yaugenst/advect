# ruff: noqa: PLR2004
"""Abstract registrations and evaluators for array contractions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from advect.core._abstract_helpers import (
    broadcast_shape,
    matmul_shape,
    normalize_axis,
    promote_dtype,
    tensordot_shape,
)
from advect.core._abstract_model import ArraySpec, rule

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from advect.core._abstract_model import AbstractRule, ResultEvaluator


RULES: dict[str, AbstractRule] = {
    "array.cross": rule("cross", 2, allowed=("axis",)),
    "array.matmul": rule("matmul", 2),
    "array.outer": rule("outer", 2),
    "array.tensordot": rule("tensordot", 2, allowed=("axes",)),
    "array.vecdot": rule("vecdot", 2, allowed=("axis",)),
    "array_ext.matvec": rule("matmul", 2),
    "array_ext.vecmat": rule("matmul", 2),
    "array_ext.dot": rule("dot", 2),
}


def _matmul(
    specs: Sequence[ArraySpec],
    _attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    return (
        ArraySpec(
            matmul_shape(specs[0].shape, specs[1].shape),
            promote_dtype(specs),
        ),
    )


def _outer(
    specs: Sequence[ArraySpec],
    _attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    if len(specs[0].shape) != 1 or len(specs[1].shape) != 1:
        raise ValueError("linalg.outer inputs must be one-dimensional")
    return (
        ArraySpec(
            (specs[0].shape[0], specs[1].shape[0]),
            promote_dtype(specs),
        ),
    )


def _cross(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    axis_value = attrs.get("axis", -1)
    left_axis = normalize_axis(axis_value, len(specs[0].shape))
    right_axis = normalize_axis(axis_value, len(specs[1].shape))
    if specs[0].shape[left_axis] != 3 or specs[1].shape[right_axis] != 3:
        raise ValueError("linalg.cross requires three-component vectors")
    batch = broadcast_shape(
        tuple(size for axis, size in enumerate(specs[0].shape) if axis != left_axis),
        tuple(size for axis, size in enumerate(specs[1].shape) if axis != right_axis),
    )
    output_axis = normalize_axis(axis_value, len(batch), insertion=True)
    shape = list(batch)
    shape.insert(output_axis, 3)
    return (ArraySpec(tuple(shape), promote_dtype(specs)),)


def _tensordot(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    return (
        ArraySpec(
            tensordot_shape(
                specs[0].shape,
                specs[1].shape,
                attrs.get("axes", 2),
            ),
            promote_dtype(specs),
        ),
    )


def _vecdot(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    axis_value = attrs.get("axis", -1)
    left_axis = normalize_axis(axis_value, len(specs[0].shape))
    right_axis = normalize_axis(axis_value, len(specs[1].shape))
    if specs[0].shape[left_axis] != specs[1].shape[right_axis]:
        raise ValueError("vecdot vector dimensions must have equal length")
    left_batch = tuple(size for axis, size in enumerate(specs[0].shape) if axis != left_axis)
    right_batch = tuple(size for axis, size in enumerate(specs[1].shape) if axis != right_axis)
    return (ArraySpec(broadcast_shape(left_batch, right_batch), promote_dtype(specs)),)


def _dot(
    specs: Sequence[ArraySpec],
    _attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    left = specs[0].shape
    right = specs[1].shape
    if not left or not right:
        shape = broadcast_shape(left, right)
    else:
        contracted_right_axis = 0 if len(right) == 1 else len(right) - 2
        if left[-1] != right[contracted_right_axis]:
            raise ValueError("dot() contracted dimensions must have equal lengths")
        shape = (
            *left[:-1],
            *right[:contracted_right_axis],
            *right[contracted_right_axis + 1 :],
        )
    return (ArraySpec(shape, promote_dtype(specs)),)


EVALUATORS: dict[str, ResultEvaluator] = {
    "matmul": _matmul,
    "outer": _outer,
    "cross": _cross,
    "tensordot": _tensordot,
    "vecdot": _vecdot,
    "dot": _dot,
}
