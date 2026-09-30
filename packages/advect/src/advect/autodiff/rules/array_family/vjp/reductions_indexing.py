# VJP signatures mirror NumPy op contracts
"""Native reduction adapters and explicit indexing VJP exceptions."""

from __future__ import annotations

from math import prod
from typing import Any, cast

from advect.autodiff.rules.array_family._backend_runtime import (
    _count_like,
    _zeros_in_trace,
    decode_array_index,
    xp,
)
from advect.autodiff.rules.array_family._transpose_utils import _shape_of
from advect.autodiff.rules.array_family.jvp.common import (
    _asarray_unwrapped,
    _astype_preserving_trace,
    _normalize_axis_tuple,
)
from advect.autodiff.rules.array_family.vjp.gather import _vjp_take
from advect.core._protocols import _is_traced


def _is_basic_index(index: object) -> bool:
    if isinstance(index, tuple):
        return all(_is_basic_index(component) for component in index)
    return isinstance(index, (int, slice)) or index is None or index is Ellipsis


def _vjp_sum(
    ans: xp.ndarray,
    source: object,
    *rest: object,
    g: xp.ndarray,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Broadcast the output cotangent across the reduced axes."""
    _ = ans, rest
    return (
        _reduction_pullback(
            source,
            g,
            axis=axis,
            keepdims=keepdims,
            mean=False,
            attrs=attrs,
        ),
    )


def _vjp_mean(
    ans: xp.ndarray,
    source: object,
    *rest: object,
    g: xp.ndarray,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Broadcast and scale the output cotangent across the reduced axes."""
    _ = ans, rest
    return (
        _reduction_pullback(
            source,
            g,
            axis=axis,
            keepdims=keepdims,
            mean=True,
            attrs=attrs,
        ),
    )


def _reduction_pullback(
    source: object,
    cotangent: object,
    *,
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
    mean: bool,
    attrs: dict[str, Any],
) -> xp.ndarray:
    if any(attrs.get(name) is not None for name in ("out", "where")):
        msg = "reduction derivatives do not support where/out control operands"
        raise NotImplementedError(msg)
    source_shape = _shape_of(source)
    axes = (
        None if axis is None else tuple(sorted(_normalize_axis_tuple(axis, ndim=len(source_shape))))
    )
    expanded: Any = cotangent
    if mean:
        dimensions = source_shape if axes is None else tuple(source_shape[item] for item in axes)
        divisor = prod(dimensions)
        if divisor == 0:
            msg = "mean derivative received an empty reduction axis"
            raise ValueError(msg)
        expanded = expanded / _count_like(divisor, expanded)
    if not keepdims and axes is not None:
        for item in axes:
            expanded = xp.expand_dims(expanded, axis=item)
    if _shape_of(expanded) != source_shape:
        expanded = xp.broadcast_to(expanded, source_shape)
    return cast("xp.ndarray", expanded)


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
    if _is_traced(g) or _is_traced(x):
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
        # A traced cotangent can flow through a pullback closed over concrete
        # primals. The zero base then enters the cotangent's trace, because
        # assigning a tracer into a raw array invokes the forbidden ``__array__``.
        grad = (
            xp.zeros_like(x, dtype=x_dtype)
            if _is_traced(x)
            else _zeros_in_trace(grad_contrib, _shape_of(x), x_dtype)
        )
        grad[idx] = grad_contrib
        return (grad,)

    x_arr = xp.asarray(x)
    g_arr = xp.asarray(g)

    grad = xp.zeros_like(x_arr, dtype=x_arr.dtype)
    grad_contrib = xp.asarray(g_arr, dtype=x_arr.dtype)
    if _is_basic_index(idx):
        grad[idx] = grad_contrib
        return (grad,)
    scatter_add = getattr(xp.add, "at", None)
    if scatter_add is not None:
        scatter_add(grad, idx, grad_contrib)
        return (grad,)
    components = idx if isinstance(idx, tuple) else (idx,)
    if not any(_is_integer_array(component) for component in components):
        # Boolean masks select each position at most once.
        grad[idx] = grad_contrib
        return (grad,)
    return (_leading_integer_index_pullback(ans, x_arr, components, g=grad_contrib),)


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_integer_array(value: object) -> bool:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return False
    kind = getattr(dtype, "kind", None)
    return kind in {"i", "u"} if kind is not None else "int" in str(dtype)


def _leading_integer_index_pullback(
    ans: xp.ndarray,
    x: xp.ndarray,
    components: tuple[object, ...],
    *,
    g: xp.ndarray,
) -> xp.ndarray:
    """Scatter-add through leading integer indices without ``add.at``.

    Integer arrays and integers on the leading axes gather from those axes
    flattened into one, which is a ``take`` whose portable transpose
    accumulates repeated indices.
    """
    shape = _shape_of(x)
    if len(components) > len(shape) or not all(
        _is_integer_array(component) or _is_integer(component) for component in components
    ):
        msg = (
            "Advanced-index pullbacks on providers without add.at support "
            "boolean masks, and integer arrays and integers on the leading axes, only."
        )
        raise NotImplementedError(msg)
    leading = shape[: len(components)]
    linear: Any = None
    for component, size in zip(components, leading, strict=True):
        index: Any = component
        if _is_integer(index):
            index %= size
        else:
            index = xp.where(index < 0, index + size, index)
        linear = index if linear is None else linear * size + index
    source = xp.reshape(x, (prod(leading), *shape[len(components) :]))
    pulled, _ = _vjp_take(ans, source, linear, g=g, axis=0)
    return xp.reshape(pulled, shape)


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
    replacement_grad = cast("Any", g)[idx]

    if mode == "add":
        base_grad = g
    elif mode == "set" and not _shape_of(g):
        # An index into a rank-0 base either replaces its one element or, like
        # ``False``, selects nothing. Its cotangent may be an immutable NumPy
        # scalar, so it is not assigned into.
        base_grad = g if 0 in _shape_of(replacement_grad) else xp.zeros_like(g)
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

    return cast("tuple[xp.ndarray, xp.ndarray]", (base_grad, replacement_grad))
