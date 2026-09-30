"""Piecewise-constant ordering, index, and membership functions."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as _numpy  # noqa: ICN001 - typed module and dynamic lowering namespace

from advect.core._errors import TracingError
from advect.numpy._array_function.composite import (
    _concrete,
    _finish,
    _first_traced,
    _lift_composite_constant,
    _normalize_axes,
)
from advect.numpy._array_function.normalization import _bind_optional_positionals
from advect.numpy._signature import ascending_sort_kwargs

np: Any = _numpy

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike
    from advect.numpy._array_function.composite import CompositeResult
    from advect.numpy._array_function.emission import ArrayFunctionHandler


_BINARY_ARITY = 2
_NO_VALUE = getattr(np, "_NoValue", object())


def _finish_discrete(
    value: object,
    *,
    anchor: TracedArrayLike,
    traced_type: type[TracedArrayLike],
) -> CompositeResult:
    if isinstance(value, tuple):
        lifted = tuple(_lift_composite_constant(item, anchor) for item in value)
        return _finish(lifted, traced_type=traced_type)
    return _finish(_lift_composite_constant(value, anchor), traced_type=traced_type)


def _argsort_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    values = _bind_optional_positionals(
        name="argsort",
        args=args,
        kwargs=ascending_sort_kwargs("argsort", kwargs),
        required=1,
        optional=("axis", "kind", "order"),
        keyword_only=frozenset({"stable"}),
    )
    array = args[0]
    # axis=None is meaningful: NumPy argsorts the flattened array.
    call_kwargs = {key: value for key, value in values.items() if value is not _NO_VALUE}
    result = np.argsort(_concrete(array), **call_kwargs)
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
    result = np.argpartition(_concrete(array), args[1], **values)
    return _finish_discrete(result, anchor=array, traced_type=traced_type)


def _nanarg_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    function: Callable[..., Any],
) -> CompositeResult:
    values = _bind_optional_positionals(
        name=function.__name__,
        args=args,
        kwargs=kwargs,
        required=1,
        optional=("axis", "out"),
        keyword_only=frozenset({"keepdims"}),
    )
    keepdims_raw = values.get("keepdims", False)
    result = function(
        _concrete(args[0]),
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
    sorter = values.get("sorter")
    result = np.searchsorted(
        _concrete(args[0]),
        _concrete(args[1]),
        side=str(values.get("side", "left")),
        sorter=None if sorter is None else _concrete(sorter),
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
    result = np.digitize(
        _concrete(args[0]),
        _concrete(args[1]),
        right=bool(values.get("right", False)),
    )
    return _finish_discrete(result, anchor=anchor, traced_type=traced_type)


def _single_discrete_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    _kwargs: dict[str, Any],
    *,
    function: Callable[[object], object],
) -> CompositeResult:
    result = function(_concrete(args[0]))
    return _finish_discrete(result, anchor=args[0], traced_type=traced_type)


def _lexsort_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    keys = args[0]
    anchor = _first_traced(keys, traced_type=traced_type)
    concrete_keys = _concrete(keys)
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
    call_kwargs = {
        "assume_unique": bool(values.get("assume_unique", False)),
        "invert": bool(values.get("invert", False)),
    }
    if values.get("kind") is not None:
        call_kwargs["kind"] = values["kind"]
    result = function(
        _concrete(args[0]),
        _concrete(args[1]),
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
    _kwargs: dict[str, Any],
) -> CompositeResult:
    anchor = _first_traced(args, traced_type=traced_type)
    result = np.ix_(*(_concrete(item) for item in args))
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
        if positions.size == 0:
            msg = "set operation produced a value absent from its inputs"
            raise TracingError(msg)
        result.append(int(positions[0]))
    return np.asarray(result, dtype=np.intp)


def _selected_set_values(
    sources: tuple[object, ...],
    concrete_result: np.ndarray[Any, Any],
) -> object:
    concrete_source = np.concatenate(tuple(np.ravel(_concrete(source)) for source in sources))
    indices = _matching_indices(concrete_source, concrete_result)
    traced_source = np.concatenate(tuple(np.ravel(source) for source in sources))
    return np.astype(np.take(traced_source, indices), concrete_result.dtype)


def _set_operation_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    function: Callable[..., Any],
) -> CompositeResult:
    name = function.__name__
    supports_indices = function is np.intersect1d
    optional = ("assume_unique", "return_indices") if supports_indices else ("assume_unique",)
    values = _bind_optional_positionals(
        name=name,
        args=args,
        kwargs=kwargs,
        required=_BINARY_ARITY,
        optional=optional,
    )
    anchor = _first_traced(args[:2], traced_type=traced_type)
    call_kwargs = {"assume_unique": bool(values.get("assume_unique", False))}
    if supports_indices:
        call_kwargs["return_indices"] = bool(values.get("return_indices", False))
    concrete_result = function(
        _concrete(args[0]),
        _concrete(args[1]),
        **call_kwargs,
    )
    if supports_indices and bool(values.get("return_indices", False)):
        concrete_values, first_indices, second_indices = concrete_result
        selected = _selected_set_values(args[:2], np.asarray(concrete_values))
        return _finish(
            (
                selected,
                _lift_composite_constant(first_indices, anchor),
                _lift_composite_constant(second_indices, anchor),
            ),
            traced_type=traced_type,
        )
    concrete_values = np.asarray(concrete_result)
    return _finish(
        _selected_set_values(args[:2], concrete_values),
        traced_type=traced_type,
    )


def _union_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    _kwargs: dict[str, Any],
) -> CompositeResult:
    concrete = np.union1d(
        _concrete(args[0]),
        _concrete(args[1]),
    )
    return _finish(
        _selected_set_values(args[:2], np.asarray(concrete)),
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
    concrete = np.asarray(_concrete(array))
    trim = str(values.get("trim", "fb")).upper()
    if any(character not in {"F", "B"} for character in trim):
        msg = "numpy.trim_zeros trim must contain only 'f' and/or 'b'"
        raise TracingError(msg)
    axis = values.get("axis")
    axes = range(concrete.ndim) if axis is None else _normalize_axes(axis, concrete.ndim)
    # NumPy trims each requested axis to the bounding box of the nonzero entries
    # and empties every requested axis of an all-zero array.
    nonzero = np.argwhere(concrete)
    index = [slice(None)] * concrete.ndim
    for current in axes:
        if nonzero.size == 0:
            index[current] = slice(0, 0)
            continue
        start = int(nonzero[:, current].min()) if "F" in trim else None
        stop = int(nonzero[:, current].max()) + 1 if "B" in trim else None
        index[current] = slice(start, stop)
    return _finish(array[tuple(index)], traced_type=traced_type)


def _indices_from_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    function: Callable[..., object],
) -> CompositeResult:
    values = _bind_optional_positionals(
        name=function.__name__,
        args=args,
        kwargs=kwargs,
        required=1,
        optional=() if function is np.diag_indices_from else ("k",),
    )
    result = function(
        _concrete(args[0]),
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
    first = _concrete(args[0])
    second = _concrete(args[1])
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
    handlers: dict[Callable[..., Any], ArrayFunctionHandler],
) -> None:
    """Register discrete algorithms with their exact a.e. zero derivatives."""
    handlers[np.argsort] = _argsort_handler
    handlers[np.argpartition] = _argpartition_handler
    handlers[np.searchsorted] = _searchsorted_handler
    handlers[np.digitize] = _digitize_handler
    handlers[np.lexsort] = _lexsort_handler
    handlers[np.ix_] = _ix_handler
    handlers[np.union1d] = _union_handler
    handlers[np.trim_zeros] = _trim_zeros_handler
    for function in (np.argmin, np.argmax, np.nanargmin, np.nanargmax):
        handlers[function] = partial(_nanarg_handler, function=function)
    for function in (np.nonzero, np.argwhere, np.flatnonzero):
        handlers[function] = partial(_single_discrete_handler, function=function)
    for function in (np.setdiff1d, np.intersect1d, np.setxor1d):
        handlers[function] = partial(_set_operation_handler, function=function)
    for function in (np.diag_indices_from, np.tril_indices_from, np.triu_indices_from):
        handlers[function] = partial(_indices_from_handler, function=function)
    handlers[np.ravel_multi_index] = partial(_multi_index_handler, unravel=False)
    handlers[np.unravel_index] = partial(_multi_index_handler, unravel=True)
    handlers[np.isin] = partial(_membership_handler, name="isin", function=np.isin)
    in1d = getattr(np, "in1d", None)
    if callable(in1d):
        handlers[in1d] = partial(
            _membership_handler, name="in1d", function=_in1d_without_deprecation
        )
