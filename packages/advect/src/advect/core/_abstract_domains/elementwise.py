"""Abstract registrations and evaluators for elementwise operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from advect.core._abstract_helpers import (
    broadcast_shape,
    dtype_kind_bits,
    dtype_name,
    promote_dtype,
    real_dtype,
)
from advect.core._abstract_model import ArraySpec, rule

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from advect.core._abstract_model import AbstractRule, ResultEvaluator


RULES: dict[str, AbstractRule] = {}

for _op in (
    "array.arccos",
    "array.arccosh",
    "array.arcsin",
    "array.arcsinh",
    "array.arctan",
    "array.arctanh",
    "array.ceil",
    "array.conjugate",
    "array.cos",
    "array.cosh",
    "array.exp",
    "array.expm1",
    "array.floor",
    "array.log",
    "array.log1p",
    "array.log2",
    "array.log10",
    "array.negative",
    "array.positive",
    "array.reciprocal",
    "array.rint",
    "array.sign",
    "array.sin",
    "array.sinh",
    "array.square",
    "array.sqrt",
    "array.tan",
    "array.tanh",
    "array.trunc",
    "array.invert",
    "array_ext.spacing",
):
    RULES[_op] = rule("same", 1)

for _op in ("array.absolute", "array.imag", "array.real"):
    RULES[_op] = rule("real", 1)

RULES["array_ext.angle"] = rule(
    "real",
    1,
    positional=("deg",),
    allowed=("deg",),
)

for _op in (
    "array.isfinite",
    "array.isinf",
    "array.isnan",
    "array.logical_not",
    "array.signbit",
):
    RULES[_op] = rule("bool", 1)

for _op in (
    "array.add",
    "array.arctan2",
    "array.bitwise_and",
    "array.bitwise_or",
    "array.bitwise_xor",
    "array.copysign",
    "array.floor_divide",
    "array.hypot",
    "array.logaddexp",
    "array.left_shift",
    "array.logical_and",
    "array.logical_or",
    "array.logical_xor",
    "array.maximum",
    "array.minimum",
    "array.multiply",
    "array.nextafter",
    "array.power",
    "array.remainder",
    "array.right_shift",
    "array.subtract",
    "array_ext.heaviside",
    "array_ext.ldexp",
):
    RULES[_op] = rule("broadcast", 2)

RULES["array.divide"] = rule("true_divide", 2)

for _op in (
    "array.equal",
    "array.greater",
    "array.greater_equal",
    "array.less",
    "array.less_equal",
    "array.not_equal",
):
    RULES[_op] = rule("broadcast_bool", 2)

RULES.update(
    {
        "array.astype": rule(
            "astype",
            1,
            positional=("dtype",),
            allowed=("casting", "copy", "dtype", "order", "subok"),
            required=("dtype",),
        ),
        "array.clip": rule("clip", 1),
        "array.where": rule("where", 3),
    }
)


def _same(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    dtype = dtype_name(attrs["dtype"]) if attrs.get("dtype") is not None else None
    return (ArraySpec(specs[0].shape, dtype or dtype_name(specs[0].dtype)),)


def _real(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    dtype = dtype_name(attrs["dtype"]) if attrs.get("dtype") is not None else None
    return (ArraySpec(specs[0].shape, dtype or real_dtype(specs[0].dtype)),)


def _bool(
    specs: Sequence[ArraySpec],
    _attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    return (ArraySpec(specs[0].shape, "bool"),)


def _broadcast(
    specs: Sequence[ArraySpec],
    _attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    return (
        ArraySpec(
            broadcast_shape(*(spec.shape for spec in specs)),
            promote_dtype(specs),
        ),
    )


def _broadcast_bool(
    specs: Sequence[ArraySpec],
    _attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    return (ArraySpec(broadcast_shape(*(spec.shape for spec in specs)), "bool"),)


def _true_divide(
    specs: Sequence[ArraySpec],
    _attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    result_dtype = promote_dtype(specs)
    kind, _bits = dtype_kind_bits(result_dtype)
    if kind in {"bool", "int", "uint"}:
        result_dtype = "float64"
    return (
        ArraySpec(
            broadcast_shape(*(spec.shape for spec in specs)),
            result_dtype,
        ),
    )


def _where(
    specs: Sequence[ArraySpec],
    _attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    return (
        ArraySpec(
            broadcast_shape(*(spec.shape for spec in specs)),
            promote_dtype(specs[1:]),
        ),
    )


def _astype(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    dtype = attrs.get("dtype")
    return (
        ArraySpec(
            specs[0].shape,
            dtype_name(specs[0].dtype if dtype is None else dtype),
            device=specs[0].device,
        ),
    )


def _clip(
    specs: Sequence[ArraySpec],
    _attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    return (
        ArraySpec(
            broadcast_shape(*(spec.shape for spec in specs)),
            promote_dtype(specs),
        ),
    )


EVALUATORS: dict[str, ResultEvaluator] = {
    "same": _same,
    "real": _real,
    "bool": _bool,
    "broadcast": _broadcast,
    "broadcast_bool": _broadcast_bool,
    "true_divide": _true_divide,
    "where": _where,
    "astype": _astype,
    "clip": _clip,
}
