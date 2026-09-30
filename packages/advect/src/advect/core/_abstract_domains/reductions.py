"""Abstract registrations and evaluators for reductions and scans."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from advect.core._abstract_helpers import (
    accumulation_dtype,
    division_dtype,
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
    "array.cumprod": rule("cumulative", 1, allowed=("axis", "dtype", "include_initial")),
    "array.cumsum": rule("cumulative", 1, allowed=("axis", "dtype", "include_initial")),
    "array.diff": rule("diff", 1, allowed=("axis", "n")),
}

# Reductions with a positional axis, keyed by result kind, with their other attributes.
for _kind, _ops, _extra in (
    (
        "accumulation_reduction",
        "array.sum array.prod array_ext.nansum array_ext.nanprod",
        "initial",
    ),
    ("reduction", "array.max array.min array_ext.nanmax array_ext.nanmin", "initial"),
    ("mean_reduction", "array.mean array_ext.nanmean", "initial"),
    ("real_reduction", "array.std array.var array_ext.nanstd array_ext.nanvar", "correction ddof"),
):
    for _op in _ops.split():
        RULES[_op] = rule(
            _kind,
            1,
            positional=("axis",),
            allowed=("axis", "dtype", "keepdims", *_extra.split()),
        )
for _op in ("array.argmax", "array.argmin"):
    RULES[_op] = rule("index_reduction", 1, positional=("axis",), allowed=("axis", "keepdims"))


def _reduction(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
    *,
    kind: str,
) -> tuple[ArraySpec, ...]:
    first = specs[0]
    dtype = attrs.get("dtype")
    shape = reduction_shape(
        first.shape,
        attrs.get("axis"),
        keepdims=bool(attrs.get("keepdims", False)),
    )
    if kind == "bool_reduction":
        result_dtype = "bool"
    elif kind == "index_reduction":
        result_dtype = "int64"
    elif kind == "mean_reduction":
        result_dtype = dtype or division_dtype(first.dtype)
    elif kind == "real_reduction":
        result_dtype = dtype or real_dtype(division_dtype(first.dtype))
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
    dtype = attrs.get("dtype") or accumulation_dtype(
        first.dtype,
        array_api_version=attrs.get("_advect_array_api_version"),
    )
    return (ArraySpec(first.shape, dtype),)


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
        "mean_reduction",
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
