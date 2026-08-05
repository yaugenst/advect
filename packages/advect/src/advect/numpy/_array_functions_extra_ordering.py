"""Piecewise-constant ordering, index, and membership functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as _numpy  # noqa: ICN001 - typed module and dynamic lowering namespace

from advect.core._errors import TracingError
from advect.core._protocols import _snapshot_traced
from advect.numpy._array_functions_extra_composite import _finish, _first_traced

np: Any = _numpy

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike
    from advect.numpy._array_functions_extra_composite import CompositeResult


_BINARY_ARITY = 2
_NO_VALUE = getattr(np, "_NoValue", object())


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
    unsupported = set(kwargs) - (set(optional) | set(keyword_only))
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


def _concrete(value: object, traced_type: type[TracedArrayLike]) -> object:
    if isinstance(value, traced_type):
        return _snapshot_traced(value)[1]
    if isinstance(value, tuple):
        return tuple(_concrete(item, traced_type) for item in value)
    if isinstance(value, list):
        return [_concrete(item, traced_type) for item in value]
    return value


def _lift_discrete(value: object, anchor: TracedArrayLike) -> object:
    array = np.asarray(value)
    zero = np.astype(np.sum(np.zeros_like(anchor)), array.dtype)
    return zero + array


def _finish_discrete(
    value: object,
    *,
    anchor: TracedArrayLike,
    traced_type: type[TracedArrayLike],
) -> CompositeResult:
    if isinstance(value, tuple):
        lifted = tuple(_lift_discrete(item, anchor) for item in value)
        return _finish(lifted, traced_type=traced_type)
    return _finish(_lift_discrete(value, anchor), traced_type=traced_type)


def _argsort_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    values = _bind_optional_positionals(
        name="argsort",
        args=args,
        kwargs=kwargs,
        required=1,
        optional=("axis", "kind", "order"),
        keyword_only=frozenset({"stable"}),
    )
    array = args[0]
    call_kwargs = {
        key: value for key, value in values.items() if value is not None and value is not _NO_VALUE
    }
    result = np.argsort(_concrete(array, traced_type), **call_kwargs)
    return _finish_discrete(result, anchor=array, traced_type=traced_type)


def _argpartition_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    values = _bind_optional_positionals(
        name="argpartition",
        args=args,
        kwargs=kwargs,
        required=_BINARY_ARITY,
        optional=("axis", "kind", "order"),
    )
    array = args[0]
    call_kwargs = {key: value for key, value in values.items() if value is not None}
    result = np.argpartition(
        _concrete(array, traced_type),
        args[1],
        **call_kwargs,
    )
    return _finish_discrete(result, anchor=array, traced_type=traced_type)


def _nanarg_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    name: str,
    function: Callable[..., Any],
) -> CompositeResult:
    values = _bind_optional_positionals(
        name=name,
        args=args,
        kwargs=kwargs,
        required=1,
        optional=("axis", "out"),
        keyword_only=frozenset({"keepdims"}),
    )
    keepdims_raw = values.get("keepdims", False)
    result = function(
        _concrete(args[0], traced_type),
        axis=values.get("axis"),
        keepdims=False if keepdims_raw is _NO_VALUE else bool(keepdims_raw),
    )
    return _finish_discrete(result, anchor=args[0], traced_type=traced_type)


def _searchsorted_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    values = _bind_optional_positionals(
        name="searchsorted",
        args=args,
        kwargs=kwargs,
        required=_BINARY_ARITY,
        optional=("side", "sorter"),
    )
    anchor = _first_traced((args[:2], values.get("sorter")), traced_type=traced_type)
    if anchor is None:  # pragma: no cover - NumPy dispatch requires a tracer
        msg = "numpy.searchsorted requires a traced operand"
        raise TracingError(msg)
    sorter = values.get("sorter")
    result = np.searchsorted(
        _concrete(args[0], traced_type),
        _concrete(args[1], traced_type),
        side=str(values.get("side", "left")),
        sorter=None if sorter is None else _concrete(sorter, traced_type),
    )
    return _finish_discrete(result, anchor=anchor, traced_type=traced_type)


def _digitize_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    values = _bind_optional_positionals(
        name="digitize",
        args=args,
        kwargs=kwargs,
        required=_BINARY_ARITY,
        optional=("right",),
    )
    anchor = _first_traced(args[:2], traced_type=traced_type)
    if anchor is None:  # pragma: no cover - NumPy dispatch requires a tracer
        msg = "numpy.digitize requires a traced operand"
        raise TracingError(msg)
    result = np.digitize(
        _concrete(args[0], traced_type),
        _concrete(args[1], traced_type),
        right=bool(values.get("right", False)),
    )
    return _finish_discrete(result, anchor=anchor, traced_type=traced_type)


def _nonzero_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    if len(args) != 1 or kwargs:
        msg = "numpy.nonzero expects one array during tracing"
        raise TracingError(msg)
    result = np.nonzero(_concrete(args[0], traced_type))
    return _finish_discrete(result, anchor=args[0], traced_type=traced_type)


def _single_discrete_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    name: str,
    function: Callable[[object], object],
) -> CompositeResult:
    if len(args) != 1 or kwargs:
        msg = f"numpy.{name} expects one array during tracing"
        raise TracingError(msg)
    result = function(_concrete(args[0], traced_type))
    return _finish_discrete(result, anchor=args[0], traced_type=traced_type)


def _lexsort_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    if not args or len(args) > _BINARY_ARITY or set(kwargs) - {"axis"}:
        msg = "numpy.lexsort expects (keys, axis=-1) during tracing"
        raise TracingError(msg)
    if len(args) == _BINARY_ARITY and "axis" in kwargs:
        msg = "numpy.lexsort received axis twice"
        raise TracingError(msg)
    keys = args[0]
    if not isinstance(keys, (tuple, list)) or not keys:
        msg = "numpy.lexsort keys must be a non-empty sequence during tracing"
        raise TracingError(msg)
    anchor = _first_traced(keys, traced_type=traced_type)
    if anchor is None:
        msg = "numpy.lexsort requires at least one traced key"
        raise TracingError(msg)
    concrete_keys = tuple(_concrete(key, traced_type) for key in keys)
    axis = int(args[1] if len(args) == _BINARY_ARITY else kwargs.get("axis", -1))
    return _finish_discrete(
        np.lexsort(concrete_keys, axis=axis),
        anchor=anchor,
        traced_type=traced_type,
    )


def _membership_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    name: str,
    function: Callable[..., Any],
) -> CompositeResult:
    values = _bind_optional_positionals(
        name=name,
        args=args,
        kwargs=kwargs,
        required=_BINARY_ARITY,
        optional=("assume_unique", "invert"),
        keyword_only=frozenset({"kind"}),
    )
    anchor = _first_traced(args[:2], traced_type=traced_type)
    if anchor is None:  # pragma: no cover - NumPy dispatch requires a tracer
        msg = f"numpy.{name} requires a traced operand"
        raise TracingError(msg)
    call_kwargs = {
        "assume_unique": bool(values.get("assume_unique", False)),
        "invert": bool(values.get("invert", False)),
    }
    if values.get("kind") is not None:
        call_kwargs["kind"] = values["kind"]
    result = function(
        _concrete(args[0], traced_type),
        _concrete(args[1], traced_type),
        **call_kwargs,
    )
    return _finish_discrete(result, anchor=anchor, traced_type=traced_type)


def _in1d_without_deprecation(
    values: object,
    test_values: object,
    *,
    assume_unique: bool,
    invert: bool,
    kind: object = None,
) -> object:
    """Evaluate legacy ``in1d`` semantics through its non-deprecated replacement."""
    return np.isin(
        np.ravel(values),
        np.ravel(test_values),
        assume_unique=assume_unique,
        invert=invert,
        kind=kind,
    )


def _ix_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    if not args or kwargs:
        msg = "numpy.ix_ expects one or more one-dimensional arrays during tracing"
        raise TracingError(msg)
    anchor = _first_traced(args, traced_type=traced_type)
    if anchor is None:  # pragma: no cover - NumPy dispatch requires a tracer
        msg = "numpy.ix_ requires a traced operand"
        raise TracingError(msg)
    result = np.ix_(*(_concrete(item, traced_type) for item in args))
    return _finish_discrete(result, anchor=anchor, traced_type=traced_type)


def _matching_indices(
    source: np.ndarray[Any, Any],
    selected: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    flattened = np.ravel(source)
    result: list[int] = []
    for value in np.ravel(selected):
        matches = np.equal(flattened, value)
        if np.issubdtype(flattened.dtype, np.inexact) and np.isnan(value):
            matches = np.logical_or(matches, np.isnan(flattened))
        positions = np.flatnonzero(matches)
        if positions.size == 0:  # pragma: no cover - set operation invariant
            msg = "set operation produced a value absent from its inputs"
            raise TracingError(msg)
        result.append(int(positions[0]))
    return np.asarray(result, dtype=np.intp)


def _selected_set_values(
    sources: tuple[object, ...],
    concrete_result: np.ndarray[Any, Any],
    *,
    traced_type: type[TracedArrayLike],
) -> object:
    concrete_source = np.concatenate(
        tuple(np.ravel(_concrete(source, traced_type)) for source in sources)
    )
    indices = _matching_indices(concrete_source, concrete_result)
    traced_source = np.concatenate(tuple(np.ravel(source) for source in sources))
    return np.astype(np.take(traced_source, indices), concrete_result.dtype)


def _set_operation_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    name: str,
    function: Callable[..., Any],
    supports_indices: bool = False,
) -> CompositeResult:
    optional = ("assume_unique", "return_indices") if supports_indices else ("assume_unique",)
    values = _bind_optional_positionals(
        name=name,
        args=args,
        kwargs=kwargs,
        required=_BINARY_ARITY,
        optional=optional,
    )
    anchor = _first_traced(args[:2], traced_type=traced_type)
    if anchor is None:  # pragma: no cover - NumPy dispatch requires a tracer
        msg = f"numpy.{name} requires a traced operand"
        raise TracingError(msg)
    call_kwargs = {"assume_unique": bool(values.get("assume_unique", False))}
    if supports_indices:
        call_kwargs["return_indices"] = bool(values.get("return_indices", False))
    concrete_result = function(
        _concrete(args[0], traced_type),
        _concrete(args[1], traced_type),
        **call_kwargs,
    )
    if supports_indices and bool(values.get("return_indices", False)):
        concrete_values, first_indices, second_indices = concrete_result
        selected = _selected_set_values(
            args[:2],
            np.asarray(concrete_values),
            traced_type=traced_type,
        )
        return _finish(
            (
                selected,
                _lift_discrete(first_indices, anchor),
                _lift_discrete(second_indices, anchor),
            ),
            traced_type=traced_type,
        )
    concrete_values = np.asarray(concrete_result)
    return _finish(
        _selected_set_values(args[:2], concrete_values, traced_type=traced_type),
        traced_type=traced_type,
    )


def _union_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    if len(args) != _BINARY_ARITY or kwargs:
        msg = "numpy.union1d expects two arrays during tracing"
        raise TracingError(msg)
    concrete = np.union1d(
        _concrete(args[0], traced_type),
        _concrete(args[1], traced_type),
    )
    return _finish(
        _selected_set_values(args[:2], np.asarray(concrete), traced_type=traced_type),
        traced_type=traced_type,
    )


def _trim_zeros_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    values = _bind_optional_positionals(
        name="trim_zeros",
        args=args,
        kwargs=kwargs,
        required=1,
        optional=("trim", "axis"),
    )
    array = args[0]
    concrete = np.asarray(_concrete(array, traced_type))
    axis_raw = values.get("axis")
    if axis_raw is None:
        if concrete.ndim != 1:
            msg = "numpy.trim_zeros without axis requires a one-dimensional array"
            raise TracingError(msg)
        active = np.flatnonzero(concrete)
        axis = 0
    else:
        axis = int(axis_raw)
        if axis < 0:
            axis += concrete.ndim
        other_axes = tuple(index for index in range(concrete.ndim) if index != axis)
        active = np.flatnonzero(np.any(concrete != 0, axis=other_axes))
    trim = str(values.get("trim", "fb")).upper()
    if any(character not in {"F", "B"} for character in trim):
        msg = "numpy.trim_zeros trim must contain only 'f' and/or 'b'"
        raise TracingError(msg)
    start = int(active[0]) if active.size and "F" in trim else 0
    stop = int(active[-1]) + 1 if active.size and "B" in trim else concrete.shape[axis]
    if not active.size and "F" in trim and "B" in trim:
        start = stop = 0
    index = [slice(None)] * concrete.ndim
    index[axis] = slice(start, stop)
    return _finish(array[tuple(index)], traced_type=traced_type)


def _indices_from_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    name: str,
    function: Callable[..., object],
) -> CompositeResult:
    values = _bind_optional_positionals(
        name=name,
        args=args,
        kwargs=kwargs,
        required=1,
        optional=("k",) if name != "diag_indices_from" else (),
    )
    result = function(
        _concrete(args[0], traced_type),
        **values,
    )
    return _finish_discrete(result, anchor=args[0], traced_type=traced_type)


def _multi_index_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    unravel: bool,
) -> CompositeResult:
    name = "unravel_index" if unravel else "ravel_multi_index"
    values = _bind_optional_positionals(
        name=name,
        args=args,
        kwargs=kwargs,
        required=_BINARY_ARITY,
        optional=("order",) if unravel else ("mode", "order"),
    )
    anchor = _first_traced(args[:2], traced_type=traced_type)
    if anchor is None:  # pragma: no cover - NumPy dispatch requires a tracer
        msg = f"numpy.{name} requires a traced operand"
        raise TracingError(msg)
    first = _concrete(args[0], traced_type)
    second = _concrete(args[1], traced_type)
    if unravel:
        result = np.unravel_index(first, second, order=str(values.get("order", "C")))
    else:
        result = np.ravel_multi_index(
            first,
            second,
            mode=values.get("mode", "raise"),
            order=str(values.get("order", "C")),
        )
    return _finish_discrete(result, anchor=anchor, traced_type=traced_type)


def register_ordering_handlers(
    handlers: dict[Callable[..., Any], Callable[..., Any]],
) -> None:
    """Register discrete algorithms with their exact a.e. zero derivatives."""
    handlers[np.argsort] = _argsort_handler
    handlers[np.argpartition] = _argpartition_handler
    handlers[np.argmin] = lambda graph, traced_type, args, kwargs: _nanarg_handler(
        graph,
        traced_type,
        args,
        kwargs,
        name="argmin",
        function=np.argmin,
    )
    handlers[np.argmax] = lambda graph, traced_type, args, kwargs: _nanarg_handler(
        graph,
        traced_type,
        args,
        kwargs,
        name="argmax",
        function=np.argmax,
    )
    handlers[np.nanargmin] = lambda graph, traced_type, args, kwargs: _nanarg_handler(
        graph,
        traced_type,
        args,
        kwargs,
        name="nanargmin",
        function=np.nanargmin,
    )
    handlers[np.nanargmax] = lambda graph, traced_type, args, kwargs: _nanarg_handler(
        graph,
        traced_type,
        args,
        kwargs,
        name="nanargmax",
        function=np.nanargmax,
    )
    handlers[np.searchsorted] = _searchsorted_handler
    handlers[np.digitize] = _digitize_handler
    handlers[np.nonzero] = _nonzero_handler
    handlers[np.argwhere] = lambda graph, traced_type, args, kwargs: _single_discrete_handler(
        graph,
        traced_type,
        args,
        kwargs,
        name="argwhere",
        function=np.argwhere,
    )
    handlers[np.flatnonzero] = lambda graph, traced_type, args, kwargs: _single_discrete_handler(
        graph,
        traced_type,
        args,
        kwargs,
        name="flatnonzero",
        function=np.flatnonzero,
    )
    handlers[np.lexsort] = _lexsort_handler
    handlers[np.isin] = lambda graph, traced_type, args, kwargs: _membership_handler(
        graph,
        traced_type,
        args,
        kwargs,
        name="isin",
        function=np.isin,
    )
    in1d = getattr(np, "in1d", None)
    if callable(in1d):
        handlers[in1d] = lambda graph, traced_type, args, kwargs: _membership_handler(
            graph,
            traced_type,
            args,
            kwargs,
            name="in1d",
            function=_in1d_without_deprecation,
        )
    handlers[np.ix_] = _ix_handler
    handlers[np.setdiff1d] = lambda graph, traced_type, args, kwargs: _set_operation_handler(
        graph,
        traced_type,
        args,
        kwargs,
        name="setdiff1d",
        function=np.setdiff1d,
    )
    handlers[np.intersect1d] = lambda graph, traced_type, args, kwargs: _set_operation_handler(
        graph,
        traced_type,
        args,
        kwargs,
        name="intersect1d",
        function=np.intersect1d,
        supports_indices=True,
    )
    handlers[np.setxor1d] = lambda graph, traced_type, args, kwargs: _set_operation_handler(
        graph,
        traced_type,
        args,
        kwargs,
        name="setxor1d",
        function=np.setxor1d,
    )
    handlers[np.union1d] = _union_handler
    handlers[np.trim_zeros] = _trim_zeros_handler
    handlers[np.diag_indices_from] = lambda graph, traced_type, args, kwargs: _indices_from_handler(
        graph,
        traced_type,
        args,
        kwargs,
        name="diag_indices_from",
        function=np.diag_indices_from,
    )
    handlers[np.tril_indices_from] = lambda graph, traced_type, args, kwargs: _indices_from_handler(
        graph,
        traced_type,
        args,
        kwargs,
        name="tril_indices_from",
        function=np.tril_indices_from,
    )
    handlers[np.triu_indices_from] = lambda graph, traced_type, args, kwargs: _indices_from_handler(
        graph,
        traced_type,
        args,
        kwargs,
        name="triu_indices_from",
        function=np.triu_indices_from,
    )
    handlers[np.ravel_multi_index] = lambda graph, traced_type, args, kwargs: _multi_index_handler(
        graph,
        traced_type,
        args,
        kwargs,
        unravel=False,
    )
    handlers[np.unravel_index] = lambda graph, traced_type, args, kwargs: _multi_index_handler(
        graph,
        traced_type,
        args,
        kwargs,
        unravel=True,
    )
