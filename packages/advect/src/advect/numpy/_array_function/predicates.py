"""Predicate and counting functions lowered through differentiable array ops."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as _numpy  # noqa: ICN001 - typed module and dynamic lowering namespace

from advect.core._errors import TracingError
from advect.numpy._array_function.composite import _finish, _first_traced

np: Any = _numpy

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike
    from advect.numpy._array_function.composite import CompositeResult


_BINARY_ARITY = 2
_NO_VALUE = getattr(np, "_NoValue", object())


def _shape(value: object) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    return tuple(int(size) for size in shape) if shape is not None else tuple(np.shape(value))


def _bind_optional_positionals(
    *,
    name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    required: int,
    optional: tuple[str, ...],
    keyword_only: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if len(args) < required or len(args) > required + len(optional):
        msg = f"numpy.{name} received an invalid positional signature during tracing"
        raise TracingError(msg)
    allowed = set(optional) | set(keyword_only)
    unsupported = set(kwargs) - allowed
    if unsupported:
        msg = f"numpy.{name} kwargs not supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)
    values = dict(kwargs)
    for parameter, value in zip(optional, args[required:], strict=False):
        if parameter in values:
            msg = f"numpy.{name} received {parameter} twice"
            raise TracingError(msg)
        values[parameter] = value
    return values


def _truth_reduction_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    any_value: bool,
) -> CompositeResult:
    name = "any" if any_value else "all"
    values = _bind_optional_positionals(
        name=name,
        args=args,
        kwargs=kwargs,
        required=1,
        optional=("axis", "out", "keepdims"),
        keyword_only=frozenset({"where"}),
    )
    array = args[0]
    axis = values.get("axis")
    keepdims_raw = values.get("keepdims", False)
    keepdims = False if keepdims_raw is _NO_VALUE else bool(keepdims_raw)
    where = values.get("where", _NO_VALUE)
    truth = np.not_equal(array, 0)
    reduction_kwargs: dict[str, Any] = {
        "axis": axis,
        "keepdims": keepdims,
    }
    if where is not _NO_VALUE:
        reduction_kwargs["where"] = where
    counted = truth if any_value else np.logical_not(truth)
    count = np.sum(counted, **reduction_kwargs)
    result = np.greater(count, 0) if any_value else np.equal(count, 0)
    return _finish(result, traced_type=traced_type)


def _all_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    return _truth_reduction_handler(
        graph,
        traced_type,
        args,
        kwargs,
        any_value=False,
    )


def _any_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    return _truth_reduction_handler(
        graph,
        traced_type,
        args,
        kwargs,
        any_value=True,
    )


def _isclose_result(
    a: object,
    b: object,
    *,
    rtol: object,
    atol: object,
    equal_nan: bool,
) -> object:
    equal = np.equal(a, b)
    safe_a = np.where(equal, 0, a)
    safe_b = np.where(equal, 0, b)
    close = np.less_equal(
        np.absolute(safe_a - safe_b),
        atol + rtol * np.absolute(safe_b),
    )
    close = np.logical_or(close, equal)
    if equal_nan:
        close = np.logical_or(close, np.logical_and(np.isnan(a), np.isnan(b)))
    return close


def _isclose_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    values = _bind_optional_positionals(
        name="isclose",
        args=args,
        kwargs=kwargs,
        required=_BINARY_ARITY,
        optional=("rtol", "atol", "equal_nan"),
    )
    result = _isclose_result(
        args[0],
        args[1],
        rtol=values.get("rtol", 1e-5),
        atol=values.get("atol", 1e-8),
        equal_nan=bool(values.get("equal_nan", False)),
    )
    return _finish(result, traced_type=traced_type)


def _allclose_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    values = _bind_optional_positionals(
        name="allclose",
        args=args,
        kwargs=kwargs,
        required=_BINARY_ARITY,
        optional=("rtol", "atol", "equal_nan"),
    )
    close = _isclose_result(
        args[0],
        args[1],
        rtol=values.get("rtol", 1e-5),
        atol=values.get("atol", 1e-8),
        equal_nan=bool(values.get("equal_nan", False)),
    )
    return _finish(np.all(close), traced_type=traced_type)


def _array_equal_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    values = _bind_optional_positionals(
        name="array_equal",
        args=args,
        kwargs=kwargs,
        required=_BINARY_ARITY,
        optional=("equal_nan",),
    )
    anchor = _first_traced(args[:2], traced_type=traced_type)
    if anchor is None:  # pragma: no cover - NumPy dispatch requires a tracer
        msg = "numpy.array_equal requires a traced operand"
        raise TracingError(msg)
    if _shape(args[0]) != _shape(args[1]):
        result = np.equal(np.sum(anchor) * 0, 1)
        return _finish(result, traced_type=traced_type)
    equal = np.equal(args[0], args[1])
    if bool(values.get("equal_nan", False)):
        equal = np.logical_or(
            equal,
            np.logical_and(np.isnan(args[0]), np.isnan(args[1])),
        )
    return _finish(np.all(equal), traced_type=traced_type)


def _array_equiv_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    if len(args) != _BINARY_ARITY or kwargs:
        msg = "numpy.array_equiv expects two arrays during tracing"
        raise TracingError(msg)
    anchor = _first_traced(args, traced_type=traced_type)
    if anchor is None:  # pragma: no cover - NumPy dispatch requires a tracer
        msg = "numpy.array_equiv requires a traced operand"
        raise TracingError(msg)
    try:
        np.broadcast_shapes(_shape(args[0]), _shape(args[1]))
    except ValueError:
        return _finish(np.equal(np.sum(anchor) * 0, 1), traced_type=traced_type)
    return _finish(np.all(np.equal(args[0], args[1])), traced_type=traced_type)


def _count_nonzero_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    values = _bind_optional_positionals(
        name="count_nonzero",
        args=args,
        kwargs=kwargs,
        required=1,
        optional=("axis",),
        keyword_only=frozenset({"keepdims"}),
    )
    result = np.sum(
        np.not_equal(args[0], 0),
        axis=values.get("axis"),
        keepdims=bool(values.get("keepdims", False)),
    )
    return _finish(result, traced_type=traced_type)


def _unary_predicate_handler(
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    name: str,
    operation: Callable[[object], object],
) -> CompositeResult:
    if len(args) != 1 or kwargs:
        msg = f"numpy.{name} expects one array during tracing"
        raise TracingError(msg)
    return _finish(operation(args[0]), traced_type=traced_type)


def register_predicate_handlers(
    handlers: dict[Callable[..., Any], Callable[..., Any]],
) -> None:
    """Register boolean-valued functions with exact a.e. zero derivatives."""
    handlers[np.all] = _all_handler
    handlers[np.any] = _any_handler
    handlers[np.isclose] = _isclose_handler
    handlers[np.allclose] = _allclose_handler
    handlers[np.array_equal] = _array_equal_handler
    handlers[np.array_equiv] = _array_equiv_handler
    handlers[np.count_nonzero] = _count_nonzero_handler
    handlers[np.iscomplex] = lambda _graph, traced_type, args, kwargs: _unary_predicate_handler(
        traced_type,
        args,
        kwargs,
        name="iscomplex",
        operation=lambda value: np.not_equal(np.imag(value), 0),
    )
    handlers[np.isreal] = lambda _graph, traced_type, args, kwargs: _unary_predicate_handler(
        traced_type,
        args,
        kwargs,
        name="isreal",
        operation=lambda value: np.equal(np.imag(value), 0),
    )
    handlers[np.isposinf] = lambda _graph, traced_type, args, kwargs: _unary_predicate_handler(
        traced_type,
        args,
        kwargs,
        name="isposinf",
        operation=lambda value: np.logical_and(np.isinf(value), np.greater(value, 0)),
    )
    handlers[np.isneginf] = lambda _graph, traced_type, args, kwargs: _unary_predicate_handler(
        traced_type,
        args,
        kwargs,
        name="isneginf",
        operation=lambda value: np.logical_and(np.isinf(value), np.less(value, 0)),
    )
