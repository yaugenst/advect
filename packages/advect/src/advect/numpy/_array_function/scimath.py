# ruff: noqa: ANN401
# Composite lowerings intentionally accept both concrete arrays and tracers.
"""Dynamic differentiable lowerings for NumPy's complex-domain math helpers."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as _numpy  # noqa: ICN001 - typed module and dynamic lowering namespace
from numpy.lib import scimath

from advect.numpy._array_function.composite import _concrete_array, _finish

np: Any = _numpy

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike
    from advect.numpy._array_function.composite import CompositeResult
    from advect.numpy._array_function.emission import ArrayFunctionHandler


def _promote_for_result(value: Any, result: object) -> Any:
    result_dtype = np.asarray(result).dtype
    value_dtype = np.dtype(value.dtype)
    return np.astype(value, result_dtype) if result_dtype != value_dtype else value


def _unary_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    _kwargs: dict[str, Any],
    *,
    name: str,
) -> CompositeResult:
    """Apply NumPy's ufunc of the same name after scimath's complex promotion."""
    expected = getattr(scimath, name)(_concrete_array(args[0]))
    operand = _promote_for_result(args[0], expected)
    return _finish(getattr(np, name)(operand), traced_type=traced_type)


def _logn_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    _kwargs: dict[str, Any],
) -> CompositeResult:
    # One operand is traced, so the result joins the trace without lifting the other.
    base, value = (item if isinstance(item, traced_type) else np.asarray(item) for item in args)
    promoted_base = _promote_for_result(
        base,
        scimath.log(_concrete_array(args[0])),
    )
    promoted_value = _promote_for_result(
        value,
        scimath.log(_concrete_array(args[1])),
    )
    return _finish(
        np.log(promoted_value) / np.log(promoted_base),
        traced_type=traced_type,
    )


def _power_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    _kwargs: dict[str, Any],
) -> CompositeResult:
    base, exponent = (item if isinstance(item, traced_type) else np.asarray(item) for item in args)
    expected = scimath.power(
        _concrete_array(args[0]),
        _concrete_array(args[1]),
    )
    return _finish(
        np.power(_promote_for_result(base, expected), exponent),
        traced_type=traced_type,
    )


def register_scimath_handlers(
    handlers: dict[Callable[..., Any], ArrayFunctionHandler],
) -> None:
    """Register complex-domain continuations with ordinary traceable ufuncs."""
    for name in ("arccos", "arcsin", "arctanh", "log", "log10", "log2", "sqrt"):
        handlers[getattr(scimath, name)] = partial(_unary_handler, name=name)
    handlers[scimath.logn] = _logn_handler
    handlers[scimath.power] = _power_handler


__all__ = ["register_scimath_handlers"]
