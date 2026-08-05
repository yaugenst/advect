# ruff: noqa: EM101, TRY003
"""Abstract registrations and evaluators for array creation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from advect.core._abstract_helpers import arange_length, dtype_name, shape_tuple
from advect.core._abstract_model import ArraySpec, rule

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from advect.core._abstract_model import AbstractRule, ResultEvaluator


RULES: dict[str, AbstractRule] = {
    "array.arange": rule(
        "arange",
        0,
        positional=("start", "stop", "step"),
        allowed=("dtype", "start", "step", "stop"),
        required=("dtype", "start"),
    ),
    "array.empty": rule(
        "creation",
        0,
        positional=("shape",),
        allowed=("device", "dtype", "order", "shape"),
        required=("dtype", "shape"),
    ),
    "array.empty_like": rule(
        "like",
        1,
        allowed=("device", "dtype", "order", "shape", "subok"),
    ),
    "array.eye": rule(
        "eye",
        0,
        positional=("n_rows", "n_cols"),
        allowed=("dtype", "k", "n_cols", "n_rows"),
        required=("dtype", "n_rows"),
    ),
    "array.full": rule(
        "creation_full",
        1,
        allowed=("device", "dtype", "order", "shape"),
        required=("shape",),
    ),
    "array.full_like": rule(
        "full_like",
        2,
        positional=("dtype", "order", "subok", "shape"),
        allowed=("device", "dtype", "order", "shape", "subok"),
    ),
    "array.linspace": rule(
        "linspace",
        0,
        positional=("start", "stop", "num"),
        allowed=("dtype", "endpoint", "num", "start", "stop"),
        required=("dtype", "num", "start", "stop"),
    ),
    "array.ones": rule(
        "creation",
        0,
        positional=("shape",),
        allowed=("device", "dtype", "order", "shape"),
        required=("dtype", "shape"),
    ),
    "array.ones_like": rule(
        "like",
        1,
        allowed=("device", "dtype", "order", "shape", "subok"),
    ),
    "array.zeros": rule(
        "creation",
        0,
        positional=("shape",),
        allowed=("device", "dtype", "order", "shape"),
        required=("dtype", "shape"),
    ),
    "array.zeros_like": rule(
        "like",
        1,
        allowed=("device", "dtype", "order", "shape", "subok"),
    ),
}


def _creation(
    _specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    return (ArraySpec(shape_tuple(attrs["shape"]), dtype_name(attrs["dtype"])),)


def _eye(
    _specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    n_rows = attrs["n_rows"]
    n_cols = attrs.get("n_cols")
    if (
        isinstance(n_rows, bool)
        or not isinstance(n_rows, int)
        or n_rows < 0
        or (
            n_cols is not None
            and (isinstance(n_cols, bool) or not isinstance(n_cols, int) or n_cols < 0)
        )
    ):
        raise ValueError("eye dimensions must be non-negative integers")
    return (
        ArraySpec(
            (n_rows, n_rows if n_cols is None else n_cols),
            dtype_name(attrs["dtype"]),
        ),
    )


def _arange(
    _specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    start = attrs["start"]
    stop = attrs.get("stop")
    if stop is None:
        start, stop = 0, start
    return (
        ArraySpec(
            (arange_length(start, stop, attrs.get("step", 1)),),
            dtype_name(attrs["dtype"]),
        ),
    )


def _linspace(
    _specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    num = attrs["num"]
    if isinstance(num, bool) or not isinstance(num, int) or num < 0:
        raise ValueError("linspace num must be a non-negative integer")
    if type(attrs.get("endpoint", True)) is not bool:
        raise TypeError("linspace endpoint must be a bool")
    return (ArraySpec((num,), dtype_name(attrs["dtype"])),)


def _creation_full(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    dtype = dtype_name(attrs["dtype"]) if attrs.get("dtype") is not None else None
    return (
        ArraySpec(
            shape_tuple(attrs["shape"]),
            dtype or dtype_name(specs[0].dtype),
        ),
    )


def _full_like(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    dtype = dtype_name(attrs["dtype"]) if attrs.get("dtype") is not None else None
    shape = specs[0].shape if attrs.get("shape") is None else shape_tuple(attrs["shape"])
    return (ArraySpec(shape, dtype or dtype_name(specs[0].dtype)),)


def _like(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    dtype = dtype_name(attrs["dtype"]) if attrs.get("dtype") is not None else None
    shape = specs[0].shape if attrs.get("shape") is None else shape_tuple(attrs["shape"])
    return (ArraySpec(shape, dtype or dtype_name(specs[0].dtype)),)


EVALUATORS: dict[str, ResultEvaluator] = {
    "creation": _creation,
    "eye": _eye,
    "arange": _arange,
    "linspace": _linspace,
    "creation_full": _creation_full,
    "full_like": _full_like,
    "like": _like,
}
