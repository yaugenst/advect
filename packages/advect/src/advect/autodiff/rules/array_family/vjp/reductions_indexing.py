# VJP signatures mirror NumPy op contracts
"""Native reduction adapters and explicit indexing VJP exceptions."""

from __future__ import annotations

from math import prod
from typing import Any, cast

from advect.autodiff.rules.array_family._backend_runtime import (
    _scalar_like,
    decode_array_index,
    xp,
)
from advect.autodiff.rules.array_family.jvp.common import (
    _asarray_unwrapped,
    _astype_preserving_trace,
)


def _is_traced_leaf(value: object) -> bool:
    return callable(getattr(value, "_advect_snapshot", None))


def _is_basic_index(index: object) -> bool:
    if isinstance(index, tuple):
        return all(_is_basic_index(component) for component in index)
    return isinstance(index, (int, slice)) or index is None or index is Ellipsis


def _vjp_sum(
    ans: xp.ndarray,
    *inputs: object,
    g: xp.ndarray,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    input_shape: tuple[int, ...] | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Broadcast the output cotangent across the reduced axes."""
    _ = ans
    return (
        _reduction_pullback(
            inputs,
            g,
            axis=axis,
            keepdims=keepdims,
            input_shape=input_shape,
            mean=False,
            attrs=attrs,
        ),
    )


def _vjp_mean(
    ans: xp.ndarray,
    *inputs: object,
    g: xp.ndarray,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    input_shape: tuple[int, ...] | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Broadcast and scale the output cotangent across the reduced axes."""
    _ = ans
    return (
        _reduction_pullback(
            inputs,
            g,
            axis=axis,
            keepdims=keepdims,
            input_shape=input_shape,
            mean=True,
            attrs=attrs,
        ),
    )


def _reduction_pullback(
    inputs: tuple[object, ...],
    cotangent: object,
    *,
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
    input_shape: tuple[int, ...] | None,
    mean: bool,
    attrs: dict[str, Any],
) -> xp.ndarray:
    if any(attrs.get(name) is not None for name in ("out", "where")):
        msg = "reduction derivatives do not support where/out control operands"
        raise NotImplementedError(msg)
    if not inputs:
        msg = "reduction pullback requires the source array"
        raise ValueError(msg)

    source = inputs[0]
    source_shape = tuple(input_shape) if input_shape is not None else _shape_of(source)
    axes = _reduction_axes(axis, ndim=len(source_shape))
    expanded: Any = cotangent
    if mean:
        dimensions = source_shape if axes is None else tuple(source_shape[item] for item in axes)
        divisor = prod(dimensions)
        if divisor == 0:
            msg = "mean derivative received an empty reduction axis"
            raise ValueError(msg)
        expanded = expanded / _scalar_like(divisor, expanded)
    if not keepdims and axes is not None:
        for item in axes:
            expanded = xp.expand_dims(expanded, axis=item)
    if _shape_of(expanded) != source_shape:
        expanded = xp.broadcast_to(expanded, source_shape)
    source_dtype = getattr(source, "dtype", None)
    if source_dtype is None:
        source_dtype = _asarray_unwrapped(source).dtype
    if getattr(expanded, "dtype", None) == source_dtype:
        return cast("xp.ndarray", expanded)
    return cast("xp.ndarray", _astype_preserving_trace(expanded, dtype=source_dtype))


def _reduction_axes(
    axis: int | tuple[int, ...] | None,
    *,
    ndim: int,
) -> tuple[int, ...] | None:
    if axis is None:
        return None
    raw_axes = (axis,) if isinstance(axis, int) else tuple(axis)
    normalized: list[int] = []
    for raw_axis in raw_axes:
        item = int(raw_axis)
        if item < 0:
            item += ndim
        if item < 0 or item >= ndim:
            msg = f"reduction derivative axis {raw_axis} is out of range for rank {ndim}"
            raise ValueError(msg)
        if item in normalized:
            msg = "reduction derivative axis contains duplicates"
            raise ValueError(msg)
        normalized.append(item)
    return tuple(sorted(normalized))


def _vjp_getitem(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: object,
    g: xp.ndarray,
    index: object = None,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """VJP for advect.getitem."""
    _ = ans, rest, attrs
    idx = cast("Any", decode_array_index(index))
    if _is_traced_leaf(g) or _is_traced_leaf(x):
        if not _is_basic_index(idx):
            msg = (
                "Higher-order pullbacks for advanced indexing require an explicit "
                "scatter-add primitive; only basic indices are traceable today."
            )
            raise NotImplementedError(msg)
        x_dtype = _asarray_unwrapped(x).dtype
        grad_contrib = _astype_preserving_trace(
            g,
            dtype=x_dtype,
        )
        grad = xp.zeros_like(x, dtype=x_dtype)
        if not _is_traced_leaf(grad):
            # A traced cotangent can flow through a pullback closed over concrete
            # primals. Lift the zero base into that trace before functionalizing
            # the indexed write; assigning a tracer into a raw ndarray would
            # otherwise invoke the deliberately forbidden ``__array__`` path.
            traced_zero = cast("Any", grad_contrib).sum() * xp.asarray(0, dtype=x_dtype)
            grad = grad + traced_zero
        grad[idx] = grad_contrib
        return cast("tuple[xp.ndarray]", (grad,))

    x_arr = xp.asarray(x)
    g_arr = xp.asarray(g)

    grad = xp.zeros_like(x_arr, dtype=x_arr.dtype)
    grad_contrib = xp.asarray(g_arr, dtype=x_arr.dtype)
    if _is_basic_index(idx):
        grad[idx] = grad_contrib
        return (grad,)
    try:
        xp.add.at(grad, idx, grad_contrib)
    except Exception:  # noqa: BLE001 - fall back to assignment semantics
        grad[idx] = grad[idx] + grad_contrib
    return (grad,)


def _shape_of(value: object) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        shape = _asarray_unwrapped(value).shape
    return tuple(int(dimension) for dimension in shape)


def _vjp_index_update(
    ans: xp.ndarray,
    g: xp.ndarray,
    index: object = None,
    mode: str = "set",
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray]:
    """Real-adjoint VJP using only cotangent and structural index metadata."""
    _ = ans, attrs
    idx = cast("Any", decode_array_index(index))

    if mode == "add":
        base_grad = g
    elif mode == "set":
        copy_value = getattr(g, "copy", None)
        base_grad = cast(
            "Any",
            copy_value() if callable(copy_value) else xp.asarray(g, copy=True),
        )
        base_grad[idx] = 0
    else:
        msg = f"Unsupported index_update mode {mode!r}"
        raise ValueError(msg)

    replacement_grad = cast("Any", g)[idx]
    return cast("tuple[xp.ndarray, xp.ndarray]", (base_grad, replacement_grad))
