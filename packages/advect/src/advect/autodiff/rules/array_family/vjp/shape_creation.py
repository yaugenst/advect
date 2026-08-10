"""Manual shape and creation VJPs retained as explicit hot-path exceptions."""

from __future__ import annotations

from typing import Any, Literal, cast

from advect.autodiff.rules.array_family._backend_runtime import _moveaxis, xp
from advect.autodiff.rules.array_family.jvp.common import _astype_preserving_trace


def _vjp_restore_shape(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    _ = ans, rest, attrs
    return (xp.reshape(g, tuple(x.shape)),)


_vjp_squeeze = _vjp_restore_shape
_vjp_expand_dims = _vjp_restore_shape


def _vjp_reshape(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    order: str | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Restore the source shape, preserving NumPy's optional order contract."""
    _ = ans, rest
    source_shape = tuple(x.shape)
    if order is not None and "_advect_array_api_version" not in attrs:
        reshape_order = cast("Literal['A', 'C', 'F']", order)
        return (xp.reshape(g, source_shape, order=reshape_order),)
    return (xp.reshape(g, source_shape),)


def _vjp_transpose(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    axes: tuple[int, ...] | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Apply the inverse axis permutation."""
    _ = ans, inputs, attrs
    rank = g.ndim
    permutation = tuple(reversed(range(rank))) if axes is None else tuple(axes)
    normalized = tuple(axis % rank for axis in permutation)
    if sorted(normalized) != list(range(rank)):
        msg = f"Invalid transpose axes {axes!r} for a rank-{rank} value"
        raise ValueError(msg)
    inverse = [0] * rank
    for output_axis, input_axis in enumerate(normalized):
        inverse[input_axis] = output_axis
    return (xp.permute_dims(g, tuple(inverse)),)


def _vjp_moveaxis(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    source: int | tuple[int, ...] = 0,
    destination: int | tuple[int, ...] = 0,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Move axes back to their source positions."""
    _ = ans, inputs, attrs
    return (_moveaxis(g, destination, source),)


def _vjp_broadcast_to(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Sum broadcast axes and restore the source shape."""
    _ = ans, rest, attrs
    source_shape = tuple(x.shape)
    grad = g
    while grad.ndim > len(source_shape):
        grad = xp.sum(grad, axis=0, dtype=grad.dtype)
    for axis, size in enumerate(source_shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = xp.sum(grad, axis=axis, dtype=grad.dtype, keepdims=True)
    reshaped = xp.reshape(grad, source_shape)
    source_dtype = getattr(x, "dtype", None)
    if source_dtype is not None and getattr(reshaped, "dtype", None) != source_dtype:
        reshaped = _astype_preserving_trace(reshaped, dtype=source_dtype)
    return (reshaped,)


def _vjp_constant_like(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[None]:
    """Return symbolic zero for a non-differentiable creation template."""
    _ = ans, inputs, g, attrs
    return (None,)


_vjp_zeros_like = _vjp_constant_like
_vjp_ones_like = _vjp_constant_like
