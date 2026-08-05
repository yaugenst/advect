"""Shared helpers for backend-neutral array-protocol runtime modules."""

from __future__ import annotations

import operator
from typing import Any, cast

from advect.core._abstract_domains import operation_semantics
from advect.core._abstract_helpers import dtype_name, promote_dtype
from advect.core._abstract_model import ArraySpec

_BINARY_ARITY = 2
_WEAK_SCALAR_OPS = frozenset(
    name
    for name, schema, _evaluator in operation_semantics()
    if schema.kind in {"broadcast", "broadcast_bool", "true_divide"}
)


def literal_is_weak(value: object) -> bool:
    """Return whether one concrete operand has Python weak-scalar semantics."""
    return type(value) in {bool, complex, float, int} or bool(getattr(value, "_advect_weak", False))


def literals_are_weak(values: tuple[object, ...] | list[object]) -> bool:
    """Return whether every literal operand is a weak scalar."""
    return bool(values) and all(literal_is_weak(value) for value in values)


def _runtime_array_spec(value: object) -> ArraySpec:
    if type(value) is bool:
        return ArraySpec((), "bool", weak=True)
    if type(value) is int:
        return ArraySpec((), "int64", weak=True)
    if type(value) is float:
        return ArraySpec((), "float64", weak=True)
    if type(value) is complex:
        return ArraySpec((), "complex128", weak=True)
    runtime_value = cast("Any", value)
    return ArraySpec(
        tuple(int(size) for size in runtime_value.shape), dtype_name(runtime_value.dtype)
    )


def materialize_weak_scalar_operands(
    op: str,
    operands: tuple[object, ...],
    *,
    namespace: object,
) -> tuple[object, ...]:
    """Represent weak scalars without delegating provider-specific promotion."""
    if (
        op not in _WEAK_SCALAR_OPS
        or len(operands) != _BINARY_ARITY
        or any(bool(getattr(type(value), "__advect_abstract_array__", False)) for value in operands)
        or not any(literal_is_weak(value) for value in operands)
        or all(literal_is_weak(value) for value in operands)
    ):
        return operands
    asarray = getattr(namespace, "asarray", None)
    if not callable(asarray):
        msg = "The runtime array namespace does not provide asarray()"
        raise TypeError(msg)
    promoted = promote_dtype([_runtime_array_spec(value) for value in operands])
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


def normalize_item_index(
    args: tuple[object, ...],
    *,
    ndim: int,
) -> int | tuple[int, ...] | None:
    """Normalize NumPy's no-index, flat-index, and coordinate item forms."""
    if not args:
        return None
    if len(args) == 1 and not isinstance(args[0], tuple):
        return operator.index(cast("Any", args[0]))
    components = cast("tuple[object, ...]", args[0]) if len(args) == 1 else args
    normalized = tuple(operator.index(cast("Any", component)) for component in components)
    if len(normalized) == 1:
        return normalized[0]
    if len(normalized) != ndim:
        msg = "incorrect number of indices for array"
        raise ValueError(msg)
    return normalized
