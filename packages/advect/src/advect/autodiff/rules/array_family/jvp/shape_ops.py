"""Shape JVP rules; most apply the linear shape operation to the tangent."""

from __future__ import annotations

from typing import Any, cast

from advect.autodiff.rules.array_family._backend_runtime import _moveaxis, xp
from advect.autodiff.rules.array_family._transpose_utils import (
    _layout_order,
    _ravel_axes,
    _reshape_in_order,
)
from advect.autodiff.rules.array_family.jvp.common import (
    _asarray_preserving_trace,
    _copy_if_untraced_array,
    _shape_unwrapped,
    linear_jvp,
)


def _jvp_reshape(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray, ...],
    order: str = "C",
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.reshape."""
    _ = rest, attrs
    return _reshape_in_order(tangents[0], _shape_unwrapped(ans), _layout_order(x, order))


def _jvp_ravel(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray, ...],
    order: str = "C",
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.ravel, reading the tangent in the primal's order."""
    _ = ans, rest, attrs
    axes = _ravel_axes(x, order)
    source = _asarray_preserving_trace(tangents[0])
    if axes != tuple(sorted(axes)):
        source = xp.permute_dims(source, axes)
    return xp.reshape(source, (-1,))


def _jvp_broadcast_to(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray, ...],
    shape: tuple[int, ...] | None = None,
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.broadcast_to."""
    _ = x, rest, attrs
    out_shape = _shape_unwrapped(ans) if shape is None else tuple(shape)
    return cast(
        "xp.ndarray[Any, Any]",
        _copy_if_untraced_array(
            xp.broadcast_to(_asarray_preserving_trace(tangents[0]), out_shape),
        ),
    )


type _Axis = int | tuple[int, ...]


@linear_jvp
def _jvp_squeeze(t: Any, axis: _Axis | None = None, **_: Any) -> Any:
    return xp.squeeze(t, axis=axis)


@linear_jvp
def _jvp_expand_dims(t: Any, axis: _Axis = 0, **_: Any) -> Any:
    return xp.expand_dims(t, axis=axis)


@linear_jvp
def _jvp_transpose(t: Any, axes: tuple[int, ...] | None = None, **_: Any) -> Any:
    return xp.transpose(t, axes=axes)


@linear_jvp
def _jvp_swapaxes(t: Any, axis1: int = 0, axis2: int = 1, **_: Any) -> Any:
    return xp.swapaxes(t, axis1, axis2)


@linear_jvp
def _jvp_moveaxis(t: Any, source: _Axis = 0, destination: _Axis = 0, **_: Any) -> Any:
    return _moveaxis(t, source, destination)


@linear_jvp
def _jvp_flip(t: Any, axis: _Axis | None = None, **_: Any) -> Any:
    return xp.flip(t, axis=axis)


@linear_jvp
def _jvp_fliplr(t: Any, **_: Any) -> Any:
    return xp.fliplr(t)


@linear_jvp
def _jvp_flipud(t: Any, **_: Any) -> Any:
    return xp.flipud(t)


@linear_jvp
def _jvp_roll(t: Any, shift: _Axis = 0, axis: _Axis | None = None, **_: Any) -> Any:
    return xp.roll(t, shift=shift, axis=axis)


@linear_jvp
def _jvp_rot90(t: Any, k: int = 1, axes: tuple[int, int] = (0, 1), **_: Any) -> Any:
    return xp.rot90(t, k=k, axes=axes)


@linear_jvp
def _jvp_rollaxis(t: Any, axis: int, start: int = 0, **_: Any) -> Any:
    return xp.rollaxis(t, axis=axis, start=start)


@linear_jvp
def _jvp_triu(t: Any, k: int = 0, **_: Any) -> Any:
    return xp.triu(t, k=k)


@linear_jvp
def _jvp_tril(t: Any, k: int = 0, **_: Any) -> Any:
    return xp.tril(t, k=k)


@linear_jvp
def _jvp_atleast_1d(t: Any, **_: Any) -> Any:
    return xp.atleast_1d(t)


@linear_jvp
def _jvp_atleast_2d(t: Any, **_: Any) -> Any:
    return xp.atleast_2d(t)


@linear_jvp
def _jvp_atleast_3d(t: Any, **_: Any) -> Any:
    return xp.atleast_3d(t)
