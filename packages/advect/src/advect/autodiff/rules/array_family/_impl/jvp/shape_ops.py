"""Shape JVP rules."""

from __future__ import annotations

from typing import Any, Literal, cast

from advect.autodiff.rules.array_family._backend_runtime import _moveaxis, xp
from advect.autodiff.rules.array_family._impl.jvp.common import (
    _asarray_preserving_trace,
    _copy_if_untraced_array,
    _shape_unwrapped,
    _zeros_output_tangent,
)


def _jvp_reshape(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    order: str = "C",
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.reshape."""
    _ = x, rest
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    reshape_order = cast("Literal['A', 'C', 'F']", order)
    if "_advect_array_api_version" in attrs:
        return xp.reshape(tangent, _shape_unwrapped(ans))
    return xp.reshape(tangent, _shape_unwrapped(ans), order=reshape_order)


def _jvp_ravel(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    order: str = "C",
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.ravel."""
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    ravel_order = cast("Literal['A', 'C', 'F', 'K']", order)
    return xp.ravel(_asarray_preserving_trace(tangent), order=ravel_order)


def _jvp_squeeze(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axis: int | tuple[int, ...] | None = None,
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.squeeze."""
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return xp.squeeze(_asarray_preserving_trace(tangent), axis=axis)


def _jvp_expand_dims(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axis: int | tuple[int, ...] = 0,
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.expand_dims."""
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return xp.expand_dims(_asarray_preserving_trace(tangent), axis=axis)


def _jvp_transpose(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axes: tuple[int, ...] | None = None,
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.transpose."""
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return xp.transpose(_asarray_preserving_trace(tangent), axes=axes)


def _jvp_swapaxes(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axis1: int = 0,
    axis2: int = 1,
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.swapaxes."""
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return xp.swapaxes(_asarray_preserving_trace(tangent), axis1, axis2)


def _jvp_moveaxis(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    source: int | tuple[int, ...] = 0,
    destination: int | tuple[int, ...] = 0,
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.moveaxis."""
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return _moveaxis(_asarray_preserving_trace(tangent), source, destination)


def _jvp_broadcast_to(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    shape: tuple[int, ...] | None = None,
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.broadcast_to."""
    _ = x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    out_shape = _shape_unwrapped(ans) if shape is None else tuple(shape)
    return cast(
        "xp.ndarray[Any, Any]",
        _copy_if_untraced_array(
            xp.broadcast_to(_asarray_preserving_trace(tangent), out_shape),
        ),
    )


def _jvp_flip(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axis: int | tuple[int, ...] | None = None,
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.flip."""
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return xp.flip(_asarray_preserving_trace(tangent), axis=axis)


def _jvp_fliplr(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.fliplr."""
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return xp.fliplr(_asarray_preserving_trace(tangent))


def _jvp_flipud(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.flipud."""
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return xp.flipud(_asarray_preserving_trace(tangent))


def _jvp_roll(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    shift: int | tuple[int, ...] = 0,
    axis: int | tuple[int, ...] | None = None,
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.roll."""
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return xp.roll(_asarray_preserving_trace(tangent), shift=shift, axis=axis)


def _jvp_rot90(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    k: int = 1,
    axes: tuple[int, int] = (0, 1),
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.rot90."""
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return xp.rot90(_asarray_preserving_trace(tangent), k=int(k), axes=axes)


def _jvp_rollaxis(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axis: int,
    start: int = 0,
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.rollaxis."""
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return xp.rollaxis(_asarray_preserving_trace(tangent), axis=axis, start=start)


def _jvp_triu(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    k: int = 0,
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.triu."""
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return xp.triu(_asarray_preserving_trace(tangent), k=int(k))


def _jvp_tril(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    k: int = 0,
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.tril."""
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return xp.tril(_asarray_preserving_trace(tangent), k=int(k))


def _jvp_atleast_1d(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.atleast_1d."""
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return cast("xp.ndarray[Any, Any]", xp.atleast_1d(tangent))


def _jvp_atleast_2d(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.atleast_2d."""
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return cast("xp.ndarray[Any, Any]", xp.atleast_2d(tangent))


def _jvp_atleast_3d(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.atleast_3d."""
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return cast("xp.ndarray[Any, Any]", xp.atleast_3d(tangent))
