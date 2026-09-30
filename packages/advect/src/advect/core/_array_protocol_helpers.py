"""Shared helpers for backend-neutral array-protocol runtime modules."""

from __future__ import annotations

import math
import operator
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from advect.core._abstract_domains import elementwise
from advect.core._abstract_helpers import (
    PYTHON_SCALAR_TYPES,
    dtype_name,
    promote_dtype,
    value_spec,
)
from advect.core._protocols import _innermost

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from advect.core._abstract import AbstractArray
    from advect.core._abstract_model import ArraySpec

_BINARY_ARITY = 2
# Arity of each operation whose weak operands compute as arrays: the unary and
# binary elementwise operations, and creation from one prototype's metadata.
# Their derivative rules also receive weak operands as Python scalars, which NEP
# 50 promotes; every other built-in rule receives them as rank-zero arrays.
WEAK_SCALAR_OPS = {
    **{
        name: schema.operands
        for name, schema in elementwise.RULES.items()
        if schema.operands in {1, _BINARY_ARITY}
        and schema.kind != "astype"
        and not schema.optional_operands
    },
    **dict.fromkeys(("array.empty_like", "array.ones_like", "array.zeros_like"), 1),
}
# Canonical array leaves of Python's operators. Applied to operands that are
# all weak, an operator computes as Python does and its result stays a weak
# Python scalar (NEP 50); every other operation returns a strong value.
PYTHON_BINARY_OPERATORS: dict[str, Callable[[Any, Any], Any]] = {
    "add": operator.add,
    "bitwise_and": operator.and_,
    "bitwise_or": operator.or_,
    "bitwise_xor": operator.xor,
    "divide": operator.truediv,
    "equal": operator.eq,
    "floor_divide": operator.floordiv,
    "greater": operator.gt,
    "greater_equal": operator.ge,
    "left_shift": operator.lshift,
    "less": operator.lt,
    "less_equal": operator.le,
    "matmul": operator.matmul,
    "multiply": operator.mul,
    "not_equal": operator.ne,
    "power": operator.pow,
    "remainder": operator.mod,
    "right_shift": operator.rshift,
    "subtract": operator.sub,
}
PYTHON_UNARY_OPERATORS: dict[str, Callable[[Any], Any]] = {
    "absolute": abs,
    "conjugate": lambda value: value.conjugate(),
    "imag": lambda value: value.imag,
    "invert": operator.invert,
    "negative": operator.neg,
    "positive": operator.pos,
    "real": lambda value: value.real,
}
_PYTHON_OPERATORS: dict[int, dict[str, Any]] = {
    1: PYTHON_UNARY_OPERATORS,
    _BINARY_ARITY: PYTHON_BINARY_OPERATORS,
}
# Python's bitwise operators keep bools; its arithmetic reads them as ints.
_PYTHON_BOOL_OPERATORS = frozenset({"bitwise_and", "bitwise_or", "bitwise_xor"})
# NumPy's real() and imag() return a Python scalar's own attribute, so every
# spelling of them computes as Python does on a weak operand.
PYTHON_ATTRIBUTE_OPS = frozenset({"array.imag", "array.real"})
# Graph attribute of an operation that computes as Python does on weak
# operands; replay applies the same Python operator, so its result stays weak.
PYTHON_OPERATOR_ATTR = "_advect_python_operator"


def python_operator(op: str, arity: int) -> Callable[..., Any] | None:
    """Return Python's operator that canonical *op* spells at *arity*, if any."""
    return _PYTHON_OPERATORS.get(arity, {}).get(op.rsplit(".", 1)[-1])


def python_scalar_operands(
    op: str,
    specs: Sequence[ArraySpec],
    *,
    by_operator: bool,
) -> list[ArraySpec] | None:
    """Return the operands Python reads when *op* computes as Python on weak scalars.

    A Python operator (*by_operator*), or a ``real`` or ``imag`` read, applied
    only to weak operands computes as Python does, so its result is a weak
    Python scalar (NEP 50). Any other call, such as ``np.sin(s)`` or
    ``xp.multiply(s, s)``, returns a strong value and yields ``None``.
    """
    if not (
        specs
        and all(spec.weak for spec in specs)
        and (by_operator or op in PYTHON_ATTRIBUTE_OPS)
        and python_operator(op, len(specs)) is not None
    ):
        return None
    if op.rsplit(".", 1)[-1] in _PYTHON_BOOL_OPERATORS:
        return list(specs)
    return [
        replace(spec, dtype="int64") if dtype_name(spec.dtype) == "bool" else spec for spec in specs
    ]


def _staged_value(value: object) -> AbstractArray | None:
    """Return the abstract value that an operand is, or that its tracers wrap."""
    staged = _innermost(value)
    if getattr(type(staged), "__advect_abstract_array__", False):
        return cast("AbstractArray", staged)
    return None


def literal_is_weak(value: object) -> bool:
    """Return whether one concrete operand has Python weak-scalar semantics."""
    return type(value) in PYTHON_SCALAR_TYPES or bool(getattr(value, "_advect_weak", False))


def materialize_weak_scalar_operands(
    op: str,
    operands: tuple[object, ...],
    *,
    namespace: object,
) -> tuple[object, ...]:
    """Represent weak scalars as arrays without delegating provider-specific promotion.

    A strong operand fixes each weak scalar's dtype by NEP 50. Weak scalars
    alone compute on arrays of their promoted dtype, as a NumPy function converts
    Python scalars, since a provider's function may require an array.
    """
    if (
        WEAK_SCALAR_OPS.get(op) != len(operands)
        or any(bool(getattr(type(value), "__advect_abstract_array__", False)) for value in operands)
        or not any(literal_is_weak(value) for value in operands)
    ):
        return operands
    asarray = getattr(namespace, "asarray", None)
    if not callable(asarray):
        msg = "The runtime array namespace does not provide asarray()"
        raise TypeError(msg)
    # A tracer of a staged value promotes by its staged weak-scalar category.
    staged = [_staged_value(value) for value in operands]
    promoted = promote_dtype(
        [
            value_spec(value) if spec is None else spec.spec
            for value, spec in zip(operands, staged, strict=True)
        ]
    )
    dtype = getattr(namespace, promoted, None)
    if dtype is None:
        msg = f"The runtime array namespace does not provide dtype {promoted!r}"
        raise TypeError(msg)
    return tuple(
        asarray(value, dtype=dtype) if literal_is_weak(value) else value for value in operands
    )


def weak_scalar_runtime_value(tracer: object, value: object) -> object:
    """Expose a weak concrete scalar without detaching an abstract tracer."""
    shape = getattr(value, "shape", None)
    if (
        (shape is not None and tuple(shape) != ())
        or not bool(getattr(tracer, "_advect_weak", False))
        or callable(getattr(value, "_advect_snapshot", None))
        or bool(getattr(type(value), "__advect_abstract_array__", False))
    ):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    dtype = str(getattr(value, "dtype", "")).lower()
    if "bool" in dtype:
        return bool(value)
    if "complex" in dtype:
        return complex(cast("Any", value))
    if "float" in dtype:
        return float(cast("Any", value))
    if "int" in dtype:
        return int(cast("Any", value))
    return value


def select_item(array: Any, args: tuple[object, ...]) -> Any:  # noqa: ANN401
    """Return NumPy's ``item(*args)`` element as a rank-zero array of ``array``'s kind.

    ``args`` takes the no-index, flat-index, and coordinate item forms.
    """
    shape = tuple(array.shape)
    if not args:
        if math.prod(shape) != 1:
            msg = "can only convert an array of size 1 to a scalar"
            raise ValueError(msg)
        return array[(0,) * len(shape)] if shape else array
    components = args[0] if len(args) == 1 and isinstance(args[0], tuple) else args
    index = tuple(operator.index(cast("Any", component)) for component in components)
    if len(index) == 1:
        return array.reshape((-1,))[index[0]]
    if len(index) != len(shape):
        msg = "incorrect number of indices for array"
        raise ValueError(msg)
    return array[index]
