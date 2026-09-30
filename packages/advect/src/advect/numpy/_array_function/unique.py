"""Trace NumPy unique-family functions and their structured results."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, cast

import numpy as _numpy  # noqa: ICN001 - typed module and dynamic lowering namespace

from advect.core._errors import TracingError
from advect.numpy._array_function.composite import (
    _concrete_array,
    _finish,
    _lift_composite_constant,
)
from advect.numpy._op_bindings import frontend_lowering

np: Any = _numpy

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike
    from advect.numpy._array_function.composite import CompositeResult
    from advect.numpy._array_function.emission import ArrayFunctionHandler


def _normalize_unique_axis(axis: object) -> int | None:
    if axis is None:
        return None
    if isinstance(axis, np.integer):
        return int(axis)
    if isinstance(axis, int):
        return axis
    msg = f"numpy.unique axis must be an integer or None during tracing (got {type(axis).__name__})"
    raise TracingError(msg)


def _call_numpy_unique(
    value: np.ndarray[Any, Any],
    *,
    return_index: bool,
    return_inverse: bool,
    return_counts: bool,
    axis: int | None,
    equal_nan: bool,
    sorted_values: bool,
) -> object:
    kwargs: dict[str, object] = {
        "return_index": return_index,
        "return_inverse": return_inverse,
        "return_counts": return_counts,
        "axis": axis,
        "equal_nan": equal_nan,
    }
    # ``sorted`` was added after NumPy 2.0. Its historical behavior already
    # matches the default, so only forward the keyword when a caller requests
    # the newer non-default behavior.
    if not sorted_values:
        kwargs["sorted"] = False
    return np.unique(value, **kwargs)


def _unique_result(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> object:
    positional_names = ("return_index", "return_inverse", "return_counts", "axis")
    values = dict(zip(positional_names, args[1:], strict=False)) | kwargs

    array = args[0]
    return_index = bool(values.get("return_index", False))
    return_inverse = bool(values.get("return_inverse", False))
    return_counts = bool(values.get("return_counts", False))
    axis = _normalize_unique_axis(values.get("axis"))
    equal_nan = bool(values.get("equal_nan", True))
    sorted_values = bool(values.get("sorted", True))
    concrete_result = _call_numpy_unique(
        _concrete_array(array),
        return_index=True,
        return_inverse=return_inverse,
        return_counts=return_counts,
        axis=axis,
        equal_nan=equal_nan,
        sorted_values=sorted_values,
    )
    concrete_outputs = tuple(cast("tuple[object, ...]", concrete_result))
    indices = np.asarray(concrete_outputs[1])
    source = np.ravel(array) if axis is None else array
    unique_values = np.take(source, indices, axis=axis)

    outputs: list[object] = [unique_values]
    if return_index:
        outputs.append(_lift_composite_constant(indices, array))
    cursor = 2
    if return_inverse:
        outputs.append(_lift_composite_constant(concrete_outputs[cursor], array))
        cursor += 1
    if return_counts:
        outputs.append(_lift_composite_constant(concrete_outputs[cursor], array))
    return outputs[0] if len(outputs) == 1 else tuple(outputs)


# Unique-family handlers gather the traced input at NumPy's first occurrences,
# so they are differentiable composites, not the discrete unique primitives.
@frontend_lowering("composite")
def _unique_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    return _finish(
        _unique_result(args, kwargs),
        traced_type=traced_type,
    )


def _unique_values_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    _kwargs: dict[str, Any],
) -> CompositeResult:
    return _finish(
        _unique_result(args, {"equal_nan": False}),
        traced_type=traced_type,
    )


def _named_unique_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    _kwargs: dict[str, Any],
    *,
    function: Callable[[object], object],
    unique_kwargs: dict[str, bool],
) -> CompositeResult:
    result = _unique_result(
        args,
        {"equal_nan": False, **unique_kwargs},
    )
    outputs = cast("tuple[object, ...]", result)
    result_type = type(function(np.array([0])))
    return _finish(result_type(*outputs), traced_type=traced_type)


def register_unique_handlers(
    handlers: dict[Callable[..., Any], ArrayFunctionHandler],
) -> None:
    """Register NumPy's classic and Array-API-style unique functions."""
    handlers[np.unique] = _unique_handler
    handlers[np.unique_values] = _unique_values_handler
    for function, flags in (
        (np.unique_all, ("return_index", "return_inverse", "return_counts")),
        (np.unique_counts, ("return_counts",)),
        (np.unique_inverse, ("return_inverse",)),
    ):
        handlers[function] = frontend_lowering("composite")(
            partial(
                _named_unique_handler, function=function, unique_kwargs=dict.fromkeys(flags, True)
            )
        )
