# ruff: noqa: ANN401
# Composite lowerings intentionally accept both concrete arrays and tracers.
"""Dynamic differentiable lowerings for NumPy's complex-domain math helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as _numpy  # noqa: ICN001 - typed module and dynamic lowering namespace
from numpy.lib import scimath

from advect.core._errors import TracingError
from advect.core._protocols import _snapshot_traced
from advect.numpy._array_function.composite import (
    _finish,
    _first_traced,
    _lift_composite_constant,
)

np: Any = _numpy

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike
    from advect.numpy._array_function.composite import CompositeResult


_BINARY_ARITY = 2


def _concrete(value: object, traced_type: type[TracedArrayLike]) -> np.ndarray[Any, Any]:
    if isinstance(value, traced_type):
        return np.asarray(_snapshot_traced(value)[1])
    return np.asarray(value)


def _promote_for_result(value: Any, result: object) -> Any:
    result_dtype = np.asarray(result).dtype
    value_dtype = np.dtype(value.dtype)
    return np.astype(value, result_dtype) if result_dtype != value_dtype else value


def _unary_handler(
    *,
    function_name: str,
    scimath_function: Callable[..., Any],
    base_function: Callable[..., Any],
) -> Callable[..., Any]:
    def handler(
        _graph: DynamicTape,
        traced_type: type[TracedArrayLike],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> CompositeResult:
        if len(args) != 1 or kwargs:
            msg = f"numpy.lib.scimath.{function_name} expects one input during tracing"
            raise TracingError(msg)
        concrete = _concrete(args[0], traced_type)
        expected = scimath_function(concrete)
        operand = _promote_for_result(args[0], expected)
        return _finish(base_function(operand), traced_type=traced_type)

    return handler


def _logn_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    if len(args) != _BINARY_ARITY or kwargs:
        msg = "numpy.lib.scimath.logn expects (n, x) during tracing"
        raise TracingError(msg)
    anchor = _first_traced(args, traced_type=traced_type)
    if anchor is None:  # pragma: no cover - NumPy dispatch requires a tracer
        msg = "numpy.lib.scimath.logn requires a traced operand"
        raise TracingError(msg)
    base = (
        args[0] if isinstance(args[0], traced_type) else _lift_composite_constant(args[0], anchor)
    )
    value = (
        args[1] if isinstance(args[1], traced_type) else _lift_composite_constant(args[1], anchor)
    )
    promoted_base = _promote_for_result(
        base,
        scimath.log(_concrete(args[0], traced_type)),
    )
    promoted_value = _promote_for_result(
        value,
        scimath.log(_concrete(args[1], traced_type)),
    )
    return _finish(
        np.log(promoted_value) / np.log(promoted_base),
        traced_type=traced_type,
    )


def _power_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    if len(args) != _BINARY_ARITY or kwargs:
        msg = "numpy.lib.scimath.power expects (x, p) during tracing"
        raise TracingError(msg)
    anchor = _first_traced(args, traced_type=traced_type)
    if anchor is None:  # pragma: no cover - NumPy dispatch requires a tracer
        msg = "numpy.lib.scimath.power requires a traced operand"
        raise TracingError(msg)
    base = (
        args[0] if isinstance(args[0], traced_type) else _lift_composite_constant(args[0], anchor)
    )
    exponent = (
        args[1] if isinstance(args[1], traced_type) else _lift_composite_constant(args[1], anchor)
    )
    expected = scimath.power(
        _concrete(args[0], traced_type),
        _concrete(args[1], traced_type),
    )
    return _finish(
        np.power(_promote_for_result(base, expected), exponent),
        traced_type=traced_type,
    )


def register_scimath_handlers(
    handlers: dict[Callable[..., Any], Callable[..., Any]],
) -> None:
    """Register complex-domain continuations with ordinary traceable ufuncs."""
    unary = (
        ("arccos", scimath.arccos, np.arccos),
        ("arcsin", scimath.arcsin, np.arcsin),
        ("arctanh", scimath.arctanh, np.arctanh),
        ("log", scimath.log, np.log),
        ("log10", scimath.log10, np.log10),
        ("log2", scimath.log2, np.log2),
        ("sqrt", scimath.sqrt, np.sqrt),
    )
    for name, scimath_function, base_function in unary:
        handlers[scimath_function] = _unary_handler(
            function_name=name,
            scimath_function=scimath_function,
            base_function=base_function,
        )
    handlers[scimath.logn] = _logn_handler
    handlers[scimath.power] = _power_handler


__all__ = ["register_scimath_handlers"]
