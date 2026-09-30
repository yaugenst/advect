"""Abstract registrations and evaluators for elementwise operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from advect.core._abstract_helpers import (
    broadcast_shape,
    division_dtype,
    dtype_name,
    inexact_dtype,
    promote_dtype,
    real_dtype,
)
from advect.core._abstract_model import ArraySpec, rule

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import Any

    from advect.core._abstract_model import AbstractRule, ResultEvaluator


RULES: dict[str, AbstractRule] = {}

for _op in (
    "array.ceil",
    "array.floor",
    "array.negative",
    "array.positive",
    "array.sign",
    "array.trunc",
    "array.invert",
):
    RULES[_op] = rule("same", 1)

# NumPy has no boolean loop for these, so booleans compute in int8.
for _op in ("array.conjugate", "array.reciprocal", "array.square"):
    RULES[_op] = rule("numeric", 1)

# NumPy defines these only for floating-point and complex loops.
for _op in (
    "array.arccos",
    "array.arccosh",
    "array.arcsin",
    "array.arcsinh",
    "array.arctan",
    "array.arctanh",
    "array.cos",
    "array.cosh",
    "array.exp",
    "array.expm1",
    "array.log",
    "array.log1p",
    "array.log2",
    "array.log10",
    "array.rint",
    "array.sin",
    "array.sinh",
    "array.sqrt",
    "array.tan",
    "array.tanh",
    "array_ext.spacing",
):
    RULES[_op] = rule("inexact", 1)

for _op in ("array.absolute", "array.imag", "array.real"):
    RULES[_op] = rule("real", 1)

RULES["array_ext.angle"] = rule(
    "angle",
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
    RULES[_op] = rule("broadcast_bool", 1)

for _op in (
    "array.add",
    "array.bitwise_and",
    "array.bitwise_or",
    "array.bitwise_xor",
    "array.maximum",
    "array.minimum",
    "array.multiply",
    "array.subtract",
):
    RULES[_op] = rule("broadcast", 2)

for _op in (
    "array.floor_divide",
    "array.left_shift",
    "array.power",
    "array.remainder",
    "array.right_shift",
):
    RULES[_op] = rule("numeric", 2)

for _op in (
    "array.arctan2",
    "array.copysign",
    "array.hypot",
    "array.logaddexp",
    "array.nextafter",
    "array_ext.heaviside",
):
    RULES[_op] = rule("inexact", 2)

RULES["array.divide"] = rule("true_divide", 2)
RULES["array_ext.ldexp"] = rule("ldexp", 2)

for _op in (
    "array.equal",
    "array.greater",
    "array.greater_equal",
    "array.less",
    "array.less_equal",
    "array.logical_and",
    "array.logical_or",
    "array.logical_xor",
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
        "array.clip": rule(
            "broadcast",
            1,
            optional=("_advect_clip_min_is_input", "_advect_clip_max_is_input"),
        ),
        "array.where": rule("where", 3),
    }
)


def _same(
    specs: Sequence[ArraySpec],
    _attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    return (ArraySpec(specs[0].shape, dtype_name(specs[0].dtype)),)


def _real(
    specs: Sequence[ArraySpec],
    _attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    return (ArraySpec(specs[0].shape, real_dtype(specs[0].dtype)),)


def _broadcast_with(dtype_of: Callable[[Sequence[ArraySpec]], str]) -> ResultEvaluator:
    """Return an evaluator that broadcasts every operand shape to one result dtype."""

    def evaluate(
        specs: Sequence[ArraySpec],
        _attrs: Mapping[str, Any],
    ) -> tuple[ArraySpec, ...]:
        return (ArraySpec(broadcast_shape(*(spec.shape for spec in specs)), dtype_of(specs)),)

    return evaluate


def _numeric_dtype(specs: Sequence[ArraySpec]) -> str:
    """Promote like NumPy's numeric-only loops, which compute booleans in int8."""
    dtype = promote_dtype(specs)
    return "int8" if dtype == "bool" else dtype


def _angle(
    specs: Sequence[ArraySpec],
    _attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    # NumPy evaluates angle(x) as arctan2(x.imag, x.real) with a Python 0 as the
    # imaginary part of real input, so booleans promote to int64 first.
    dtype = inexact_dtype([*specs, ArraySpec((), "int64", weak=True)])
    return (ArraySpec(specs[0].shape, real_dtype(dtype)),)


def _astype(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    return (ArraySpec(specs[0].shape, attrs["dtype"], device=specs[0].device),)


EVALUATORS: dict[str, ResultEvaluator] = {
    "same": _same,
    "real": _real,
    "broadcast": _broadcast_with(promote_dtype),
    "broadcast_bool": _broadcast_with(lambda _specs: "bool"),
    "numeric": _broadcast_with(_numeric_dtype),
    "inexact": _broadcast_with(inexact_dtype),
    "angle": _angle,
    "true_divide": _broadcast_with(lambda specs: division_dtype(promote_dtype(specs))),
    # NumPy's ldexp loops take the result dtype from the mantissa alone.
    "ldexp": _broadcast_with(lambda specs: inexact_dtype(specs[:1])),
    "where": _broadcast_with(lambda specs: promote_dtype(specs[1:])),
    "astype": _astype,
}
