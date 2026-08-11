# ruff: noqa: A002, ANN401
"""NumPy constructors that preserve live Advect values."""

from __future__ import annotations

import operator
from typing import Any, Literal, cast

import numpy as np

from advect._array import (
    _first_sequence_tracer,
    asarray as advect_asarray,
    is_traced,
)
from advect.core._errors import TracingError
from advect.core._protocols import _snapshot_traced

type _ArrayOrder = Literal["A", "C", "F", "K"]

NOT_A_CONSTRUCTOR = object()
_CONSTRUCTORS = frozenset({np.array, np.asarray, np.asanyarray})


def _normalize_order(order: object | None, *, default: _ArrayOrder) -> _ArrayOrder:
    if order is None:
        return default
    if not isinstance(order, str):
        msg = f"order must be str, not {type(order).__name__}"
        raise TypeError(msg)
    normalized = order.upper()
    if normalized not in {"A", "C", "F", "K"}:
        msg = f"order must be one of 'A', 'C', 'F', or 'K' (got {order!r})"
        raise ValueError(msg)
    return cast("_ArrayOrder", normalized)


def _normalize_ndmin(ndmin: object) -> int:
    try:
        normalized = operator.index(cast("Any", ndmin))
    except TypeError:
        msg = f"ndmin must be an integer, not {type(ndmin).__name__}"
        raise TypeError(msg) from None
    if normalized < 0:
        msg = "ndmin must be non-negative"
        raise ValueError(msg)
    return normalized


def _validate_device(device: object | None) -> None:
    if device not in {None, "cpu"}:
        msg = f'Device not understood. Only "cpu" is allowed, but received: {device}'
        raise ValueError(msg)


def _same_dtype(value: object, dtype: object) -> bool:
    return np.dtype(cast("Any", value).dtype) == np.dtype(cast("Any", dtype))


def _known_layout(value: object) -> tuple[bool, bool] | None:
    cell = getattr(value, "_cell", None)
    layout = getattr(cell, "layout", None)
    if layout == "C":
        return True, False
    if layout == "F":
        return False, True

    try:
        _node_id, payload = _snapshot_traced(value)
    except (AttributeError, TracingError):
        return None
    flags = getattr(payload, "flags", None)
    if flags is None:
        return None
    return bool(flags.c_contiguous), bool(flags.f_contiguous)


def _order_requires_copy(value: object, order: _ArrayOrder) -> bool | None:
    if order == "K":
        return False
    layout = _known_layout(value)
    if layout is None:
        return None
    c_contiguous, f_contiguous = layout
    if order == "C":
        return not c_contiguous
    if order == "F":
        return not f_contiguous
    return not (c_contiguous or f_contiguous)


def _constant_like(value: object, anchor: object) -> Any:
    concrete = np.asarray(value)
    if concrete.dtype.kind not in {"b", "i", "u", "f", "c"}:
        msg = (
            "NumPy construction with like= supports numeric and boolean traced arrays; "
            f"got dtype {concrete.dtype}"
        )
        raise TypeError(msg)
    zeros = np.zeros_like(
        anchor,
        dtype=concrete.dtype,
        shape=concrete.shape,
    )
    if concrete.dtype.kind == "b":
        return np.logical_or(zeros, concrete)
    return zeros + concrete


def _construct_base(obj: object, *, anchor: object) -> Any:
    if _first_sequence_tracer(obj) is not None:
        return advect_asarray(obj)
    return _constant_like(obj, anchor)


def _with_ndmin(value: Any, ndmin: int) -> Any:
    ndim = int(value.ndim)
    if ndim >= ndmin:
        return value
    shape = (1,) * (ndmin - ndim) + tuple(value.shape)
    return value.reshape(shape)


def _convert_traced(
    value: Any,
    *,
    dtype: object | None,
    order: _ArrayOrder,
    copy: bool | None,
    subok: bool,
    direct: bool,
) -> Any:
    del subok  # Tracer wrappers do not preserve ndarray subclass identity.
    target_dtype = value.dtype if dtype is None else np.dtype(cast("Any", dtype))
    dtype_requires_copy = not _same_dtype(value, target_dtype)
    order_requires_copy = _order_requires_copy(value, order)

    if copy is False:
        msg = "Unable to avoid copy while creating an array as requested."
        if not direct:
            raise ValueError(msg)
        if dtype_requires_copy or order_requires_copy is True:
            raise ValueError(msg)
        if order_requires_copy is None and order != "K":
            msg = (
                "copy=False with a layout-constraining order cannot be proven during staging; "
                "omit copy= or order="
            )
            raise TracingError(msg)
        return value

    if copy is not True and not dtype_requires_copy and order_requires_copy is False:
        return value

    result = value
    astype = cast("Any", getattr(value, "astype", None))
    if dtype_requires_copy and not callable(astype):
        msg = f"Traced value {type(value).__name__} does not support NumPy dtype conversion"
        raise TypeError(msg)
    if dtype_requires_copy:
        result = astype(target_dtype, copy=True)

    if order_requires_copy is not False:
        return result.copy(order=order)
    if copy is True and result is value:
        return result.copy(order=order)
    return result


def _tracer_anchor(obj: object, like: object | None) -> object | None:
    tracer = _first_sequence_tracer(obj)
    if tracer is not None:
        return tracer
    return like if is_traced(like) else None


def array(
    object: object,
    dtype: object | None = None,
    *,
    copy: bool | None = True,
    order: str = "K",
    subok: bool = False,
    ndmin: int = 0,
    like: object | None = None,
) -> Any:
    """Create a NumPy array while preserving live values selected by ``like=``."""
    anchor = _tracer_anchor(object, like)
    if anchor is None:
        return cast("Any", np.array)(
            object,
            dtype=dtype,
            copy=copy,
            order=order,
            subok=subok,
            ndmin=ndmin,
            like=like,
        )

    normalized_order = _normalize_order(order, default="K")
    normalized_ndmin = _normalize_ndmin(ndmin)
    direct = is_traced(object)
    result = _with_ndmin(_construct_base(object, anchor=anchor), normalized_ndmin)
    return _convert_traced(
        result,
        dtype=dtype,
        order=normalized_order,
        copy=copy if direct else (False if copy is False else None),
        subok=subok,
        direct=direct,
    )


def _asarray(
    obj: object,
    dtype: object | None,
    order: str | None,
    *,
    device: object | None,
    copy: bool | None,
    like: object | None,
    subok: bool,
) -> Any:
    anchor = _tracer_anchor(obj, like)
    constructor = np.asanyarray if subok else np.asarray
    if anchor is None:
        return cast("Any", constructor)(
            obj,
            dtype=dtype,
            order=order,
            device=device,
            copy=copy,
            like=like,
        )

    _validate_device(device)
    normalized_order = _normalize_order(order, default="K")
    direct = is_traced(obj)
    result = _construct_base(obj, anchor=anchor)
    return _convert_traced(
        result,
        dtype=dtype,
        order=normalized_order,
        copy=copy if direct else (False if copy is False else None),
        subok=subok,
        direct=direct,
    )


def asarray(
    a: object,
    dtype: object | None = None,
    order: str | None = None,
    *,
    device: object | None = None,
    copy: bool | None = None,
    like: object | None = None,
) -> Any:
    """Convert to a base NumPy array without detaching live Advect values."""
    return _asarray(
        a,
        dtype,
        order,
        device=device,
        copy=copy,
        like=like,
        subok=False,
    )


def asanyarray(
    a: object,
    dtype: object | None = None,
    order: str | None = None,
    *,
    device: object | None = None,
    copy: bool | None = None,
    like: object | None = None,
) -> Any:
    """Convert to a NumPy array or subclass without detaching live Advect values."""
    return _asarray(
        a,
        dtype,
        order,
        device=device,
        copy=copy,
        like=like,
        subok=True,
    )


def handle_traced_constructor(
    self_arr: object,
    func: object,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    """Handle a NumPy constructor before the generic array-function wrapper."""
    if func not in _CONSTRUCTORS:
        return NOT_A_CONSTRUCTOR
    constructor = cast(
        "Any",
        {
            np.array: array,
            np.asarray: asarray,
            np.asanyarray: asanyarray,
        }[cast("Any", func)],
    )
    return constructor(*args, like=self_arr, **kwargs)


def construct_abstract(
    name: str,
    self_arr: object,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    """Dispatch a NumPy constructor from the backend-neutral abstract tracer."""
    constructor = cast(
        "Any",
        {
            "array": array,
            "asarray": asarray,
            "asanyarray": asanyarray,
        }.get(name),
    )
    if constructor is None:
        msg = f"Unknown NumPy constructor {name!r}"
        raise TracingError(msg)
    return constructor(*args, like=self_arr, **kwargs)


__all__ = [
    "NOT_A_CONSTRUCTOR",
    "array",
    "asanyarray",
    "asarray",
    "construct_abstract",
    "handle_traced_constructor",
]
