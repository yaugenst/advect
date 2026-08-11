"""Abstract registrations and evaluators for shape transformations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from advect.core._abstract_helpers import (
    broadcast_shape,
    dtype_name,
    moveaxis_shape,
    normalize_axes,
    normalize_axis,
    promote_dtype,
    reshape_shape,
    shape_tuple,
)
from advect.core._abstract_model import ArraySpec, rule

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from advect.core._abstract_model import AbstractRule, ResultEvaluator


RULES: dict[str, AbstractRule] = {
    "advect.copy": rule("same", 1, allowed=("order",)),
    "array.broadcast_to": rule(
        "broadcast_to",
        1,
        positional=("shape",),
        allowed=("shape",),
        required=("shape",),
    ),
    "array.concatenate": rule(
        "concatenate",
        1,
        positional=("axis",),
        allowed=("axis", "dtype", "casting"),
        sequence=True,
    ),
    "array.expand_dims": rule(
        "expand_dims",
        1,
        positional=("axis",),
        allowed=("axis",),
        required=("axis",),
    ),
    "array.flip": rule("same", 1, allowed=("axis",)),
    "array.moveaxis": rule(
        "moveaxis",
        1,
        positional=("source", "destination"),
        allowed=("destination", "source"),
        required=("destination", "source"),
    ),
    "array.repeat": rule(
        "repeat",
        1,
        positional=("repeats",),
        allowed=("axis", "repeats"),
        required=("repeats",),
    ),
    "array.reshape": rule(
        "reshape",
        1,
        positional=("shape",),
        allowed=("shape", "order", "copy"),
        required=("shape",),
    ),
    "array.roll": rule(
        "same",
        1,
        positional=("shift",),
        allowed=("axis", "shift"),
        required=("shift",),
    ),
    "array.sort": rule(
        "same",
        1,
        allowed=("axis", "descending", "kind", "order", "stable"),
    ),
    "array.squeeze": rule("squeeze", 1, positional=("axis",), allowed=("axis",)),
    "array.stack": rule(
        "stack",
        1,
        positional=("axis",),
        allowed=("axis", "dtype", "casting"),
        sequence=True,
    ),
    "array.tile": rule(
        "tile",
        1,
        positional=("reps",),
        allowed=("reps",),
        required=("reps",),
    ),
    "array.transpose": rule("transpose", 1, positional=("axes",), allowed=("axes",)),
}

for _op in ("array.tril", "array.triu"):
    RULES[_op] = rule("same", 1, allowed=("k",))


def _reshape(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    first = specs[0]
    return (ArraySpec(reshape_shape(first.shape, attrs["shape"]), dtype_name(first.dtype)),)


def _transpose(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    first = specs[0]
    axes_value = attrs.get("axes")
    axes = (
        tuple(reversed(range(len(first.shape))))
        if axes_value is None
        else normalize_axes(axes_value, len(first.shape))
    )
    if len(axes) != len(first.shape):
        raise ValueError("transpose axes must contain every input axis exactly once")
    return (ArraySpec(tuple(first.shape[axis] for axis in axes), dtype_name(first.dtype)),)


def _broadcast_to(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    first = specs[0]
    shape = shape_tuple(attrs["shape"])
    if broadcast_shape(first.shape, shape) != shape:
        raise ValueError(f"Cannot broadcast shape {first.shape!r} to {shape!r}")
    return (ArraySpec(shape, dtype_name(first.dtype)),)


def _expand_dims(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    first = specs[0]
    raw_axes = attrs["axis"]
    axes_values = (raw_axes,) if isinstance(raw_axes, int) else tuple(raw_axes)
    output_rank = len(first.shape) + len(axes_values)
    axes = tuple(normalize_axis(axis, output_rank - 1, insertion=True) for axis in axes_values)
    if len(set(axes)) != len(axes):
        raise ValueError(f"Repeated expansion axis in {raw_axes!r}")
    shape = list(first.shape)
    for axis in sorted(axes):
        shape.insert(axis, 1)
    return (ArraySpec(tuple(shape), dtype_name(first.dtype)),)


def _squeeze(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    first = specs[0]
    axes = (
        tuple(index for index, size in enumerate(first.shape) if size == 1)
        if attrs.get("axis") is None
        else normalize_axes(attrs["axis"], len(first.shape))
    )
    if any(first.shape[axis] != 1 for axis in axes):
        raise ValueError(f"Cannot squeeze non-unit axes {axes!r} from {first.shape!r}")
    axis_set = set(axes)
    return (
        ArraySpec(
            tuple(size for index, size in enumerate(first.shape) if index not in axis_set),
            dtype_name(first.dtype),
        ),
    )


def _moveaxis(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    first = specs[0]
    return (
        ArraySpec(
            moveaxis_shape(first.shape, attrs["source"], attrs["destination"]),
            dtype_name(first.dtype),
        ),
    )


def _repeat(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    first = specs[0]
    repeats = attrs["repeats"]
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 0:
        raise NotImplementedError(
            "Staging repeat() currently requires one non-negative integer repeat count"
        )
    axis_value = attrs.get("axis")
    if axis_value is None:
        return (ArraySpec((math.prod(first.shape) * repeats,), dtype_name(first.dtype)),)
    axis = normalize_axis(axis_value, len(first.shape))
    shape = list(first.shape)
    shape[axis] *= repeats
    return (ArraySpec(tuple(shape), dtype_name(first.dtype)),)


def _tile(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    first = specs[0]
    repetitions = shape_tuple(attrs["reps"])
    if any(repetition < 0 for repetition in repetitions):
        raise ValueError("tile repetitions must be non-negative")
    rank = max(len(first.shape), len(repetitions))
    padded_shape = (1,) * (rank - len(first.shape)) + first.shape
    padded_repetitions = (1,) * (rank - len(repetitions)) + repetitions
    return (
        ArraySpec(
            tuple(
                size * repetition
                for size, repetition in zip(
                    padded_shape,
                    padded_repetitions,
                    strict=True,
                )
            ),
            dtype_name(first.dtype),
        ),
    )


def _concatenate(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    first = specs[0]
    dtype = dtype_name(attrs["dtype"]) if attrs.get("dtype") is not None else None
    if attrs.get("axis", 0) is None:
        return (
            ArraySpec(
                (sum(math.prod(spec.shape) for spec in specs),),
                dtype or promote_dtype(specs),
            ),
        )
    rank = len(first.shape)
    axis = normalize_axis(attrs.get("axis", 0), rank)
    if any(len(spec.shape) != rank for spec in specs):
        raise ValueError("concatenate inputs must have equal rank")
    for spec in specs[1:]:
        if any(
            left != right
            for index, (left, right) in enumerate(zip(first.shape, spec.shape, strict=True))
            if index != axis
        ):
            raise ValueError("concatenate input shapes disagree outside the joined axis")
    shape = list(first.shape)
    shape[axis] = sum(spec.shape[axis] for spec in specs)
    return (ArraySpec(tuple(shape), dtype or promote_dtype(specs)),)


def _stack(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    first = specs[0]
    if any(spec.shape != first.shape for spec in specs[1:]):
        raise ValueError("stack inputs must have identical shapes")
    axis = normalize_axis(attrs.get("axis", 0), len(first.shape), insertion=True)
    shape = list(first.shape)
    shape.insert(axis, len(specs))
    dtype = dtype_name(attrs["dtype"]) if attrs.get("dtype") is not None else None
    return (ArraySpec(tuple(shape), dtype or promote_dtype(specs)),)


EVALUATORS: dict[str, ResultEvaluator] = {
    "reshape": _reshape,
    "transpose": _transpose,
    "broadcast_to": _broadcast_to,
    "expand_dims": _expand_dims,
    "squeeze": _squeeze,
    "moveaxis": _moveaxis,
    "repeat": _repeat,
    "tile": _tile,
    "concatenate": _concatenate,
    "stack": _stack,
}
