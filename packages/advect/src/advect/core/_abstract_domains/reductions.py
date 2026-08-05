# ruff: noqa: EM101, TRY003
"""Abstract registrations and evaluators for reductions and scans."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from advect.core._abstract_helpers import (
    accumulation_dtype,
    dtype_name,
    normalize_axis,
    real_dtype,
    reduction_shape,
)
from advect.core._abstract_model import ArraySpec, rule

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from advect.core._abstract_model import AbstractRule, ResultEvaluator


RULES: dict[str, AbstractRule] = {
    "array.all": rule("bool_reduction", 1, allowed=("axis", "keepdims")),
    "array.any": rule("bool_reduction", 1, allowed=("axis", "keepdims")),
    "array.count_nonzero": rule("index_reduction", 1, allowed=("axis", "keepdims")),
    "array.cumprod": rule(
        "cumulative",
        1,
        allowed=("axis", "dtype", "include_initial"),
    ),
    "array.cumsum": rule(
        "cumulative",
        1,
        allowed=("axis", "dtype", "include_initial"),
    ),
    "array.diff": rule(
        "diff",
        1,
        allowed=("append", "axis", "n", "prepend"),
    ),
}

for _op in ("array.sum", "array.prod"):
    RULES[_op] = rule(
        "accumulation_reduction",
        1,
        positional=("axis",),
        allowed=("axis", "dtype", "keepdims", "initial"),
    )
for _op in ("array.mean", "array.max", "array.min"):
    RULES[_op] = rule(
        "reduction",
        1,
        positional=("axis",),
        allowed=("axis", "dtype", "keepdims", "initial"),
    )
for _op in ("array.std", "array.var"):
    RULES[_op] = rule(
        "real_reduction",
        1,
        positional=("axis",),
        allowed=("axis", "dtype", "keepdims", "correction", "ddof"),
    )
for _op in ("array_ext.nansum", "array_ext.nanprod"):
    RULES[_op] = rule(
        "accumulation_reduction",
        1,
        positional=("axis",),
        allowed=("axis", "dtype", "keepdims", "initial"),
    )
for _op in ("array_ext.nanmean", "array_ext.nanmin", "array_ext.nanmax"):
    RULES[_op] = rule(
        "reduction",
        1,
        positional=("axis",),
        allowed=("axis", "dtype", "keepdims", "initial"),
    )
for _op in ("array_ext.nanstd", "array_ext.nanvar"):
    RULES[_op] = rule(
        "real_reduction",
        1,
        positional=("axis",),
        allowed=("axis", "dtype", "keepdims", "correction", "ddof"),
    )
for _op in ("array.argmax", "array.argmin"):
    RULES[_op] = rule(
        "index_reduction",
        1,
        positional=("axis",),
        allowed=("axis", "keepdims"),
    )


def _reduction(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
    *,
    kind: str,
) -> tuple[ArraySpec, ...]:
    first = specs[0]
    dtype = dtype_name(attrs["dtype"]) if attrs.get("dtype") is not None else None
    shape = reduction_shape(
        first.shape,
        attrs.get("axis"),
        keepdims=bool(attrs.get("keepdims", False)),
    )
    if kind == "bool_reduction":
        result_dtype = "bool"
    elif kind == "index_reduction":
        result_dtype = "int64"
    elif kind == "real_reduction":
        result_dtype = dtype or real_dtype(first.dtype)
    elif kind == "accumulation_reduction":
        result_dtype = dtype or accumulation_dtype(
            first.dtype,
            array_api_version=attrs.get("_advect_array_api_version"),
        )
    else:
        result_dtype = dtype or dtype_name(first.dtype)
    return (ArraySpec(shape, result_dtype),)


def _cumulative(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    first = specs[0]
    axis = attrs.get("axis")
    if axis is None:
        if len(first.shape) != 1:
            raise ValueError(
                "cumulative operations require axis= for inputs with more than one dimension"
            )
    else:
        normalize_axis(axis, len(first.shape))
    dtype = dtype_name(attrs["dtype"]) if attrs.get("dtype") is not None else None
    return (
        ArraySpec(
            first.shape,
            dtype
            or accumulation_dtype(
                first.dtype,
                array_api_version=attrs.get("_advect_array_api_version"),
            ),
        ),
    )


def _diff(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    n = attrs.get("n", 1)
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("diff n must be a non-negative integer")
    first = specs[0]
    axis = normalize_axis(attrs.get("axis", -1), len(first.shape))
    shape = list(first.shape)
    shape[axis] = max(shape[axis] - n, 0)
    return (ArraySpec(tuple(shape), dtype_name(first.dtype)),)


EVALUATORS: dict[str, ResultEvaluator] = {
    kind: partial(_reduction, kind=kind)
    for kind in (
        "accumulation_reduction",
        "bool_reduction",
        "reduction",
        "real_reduction",
        "index_reduction",
    )
}
EVALUATORS.update(
    {
        "cumulative": _cumulative,
        "diff": _diff,
    }
)
