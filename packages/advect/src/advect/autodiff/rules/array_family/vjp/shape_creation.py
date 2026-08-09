"""Manual shape and creation VJPs retained as explicit hot-path exceptions."""

from __future__ import annotations

from typing import Any, Literal, cast

from advect.autodiff.rules.array_family._backend_runtime import _moveaxis, xp
from advect.autodiff.rules.array_family.jvp.common import _astype_preserving_trace

_SELECT_INPUTS_VJP_ATTR = "__advect_vjp_for_input_indices__"


def _input_shape(
    inputs: tuple[xp.ndarray, ...],
    input_shape: tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Resolve the source shape from lightweight saved metadata or a direct call."""
    if input_shape is not None:
        return tuple(input_shape)
    if inputs:
        return tuple(inputs[0].shape)
    msg = "Shape pullback requires input_shape metadata."
    raise RuntimeError(msg)


def _vjp_squeeze(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    input_shape: tuple[int, ...] | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Restore squeezed axes with one traceable reshape."""
    _ = ans, attrs
    return (xp.reshape(g, _input_shape(inputs, input_shape)),)


def _vjp_expand_dims(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    input_shape: tuple[int, ...] | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Remove inserted axes with one traceable reshape."""
    _ = ans, attrs
    return (xp.reshape(g, _input_shape(inputs, input_shape)),)


def _vjp_reshape(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    input_shape: tuple[int, ...] | None = None,
    order: str | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Restore the source shape, preserving NumPy's optional order contract."""
    _ = ans
    source_shape = _input_shape(inputs, input_shape)
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
    *inputs: xp.ndarray,
    g: xp.ndarray,
    input_shape: tuple[int, ...] | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Sum broadcast axes and restore the source shape."""
    _ = ans, attrs
    source_shape = _input_shape(inputs, input_shape)
    grad = g
    while grad.ndim > len(source_shape):
        grad = xp.sum(grad, axis=0, dtype=grad.dtype)
    for axis, size in enumerate(source_shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = xp.sum(grad, axis=axis, dtype=grad.dtype, keepdims=True)
    reshaped = xp.reshape(grad, source_shape)
    source_dtype = getattr(inputs[0], "dtype", None) if inputs else None
    if source_dtype is not None and getattr(reshaped, "dtype", None) != source_dtype:
        reshaped = _astype_preserving_trace(reshaped, dtype=source_dtype)
    return (reshaped,)


def _vjp_zeros_like(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[None]:
    """Return symbolic zero for the non-differentiable zeros_like template."""
    _ = ans, inputs, g, attrs
    return (None,)


def _vjp_ones_like(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[None]:
    """Return symbolic zero for the non-differentiable ones_like template."""
    _ = ans, inputs, g, attrs
    return (None,)


def _select_squeeze_input(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    active_input_indices: tuple[int, ...],
    **attrs: Any,
) -> tuple[xp.ndarray | None]:
    if 0 not in active_input_indices:
        return (None,)
    return _vjp_squeeze(ans, *inputs, g=g, **attrs)


def _select_expand_dims_input(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    active_input_indices: tuple[int, ...],
    **attrs: Any,
) -> tuple[xp.ndarray | None]:
    if 0 not in active_input_indices:
        return (None,)
    return _vjp_expand_dims(ans, *inputs, g=g, **attrs)


def _select_broadcast_to_input(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    active_input_indices: tuple[int, ...],
    **attrs: Any,
) -> tuple[xp.ndarray | None]:
    if 0 not in active_input_indices:
        return (None,)
    return _vjp_broadcast_to(ans, *inputs, g=g, **attrs)


def _select_constant_like_input(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    active_input_indices: tuple[int, ...],
    **attrs: Any,
) -> tuple[None]:
    _ = ans, inputs, g, active_input_indices, attrs
    return (None,)


setattr(_vjp_squeeze, _SELECT_INPUTS_VJP_ATTR, _select_squeeze_input)
setattr(_vjp_expand_dims, _SELECT_INPUTS_VJP_ATTR, _select_expand_dims_input)
setattr(_vjp_broadcast_to, _SELECT_INPUTS_VJP_ATTR, _select_broadcast_to_input)
setattr(_vjp_zeros_like, _SELECT_INPUTS_VJP_ATTR, _select_constant_like_input)
setattr(_vjp_ones_like, _SELECT_INPUTS_VJP_ATTR, _select_constant_like_input)
