"""Traceable real adjoints for NumPy FFT primitives."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from advect.autodiff.rules.array_family._backend_runtime import _zero_pad_axis, xp
from advect.autodiff.rules.array_family._transpose_utils import (
    _adjoint_fft_norm as _adjoint_norm,
    _axis_slice,
    _normalize_axis,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.autodiff.rules.array_family._transpose_utils import FFTNorm


def _resize_axis_adjoint(
    value: xp.ndarray,
    *,
    target_length: int,
    axis: int,
) -> xp.ndarray:
    """Transpose NumPy's crop-or-zero-pad behavior for one transform axis."""
    normalized_axis = _normalize_axis(axis, ndim=value.ndim, op_name="FFT")
    current_length = int(value.shape[normalized_axis])
    if target_length == current_length:
        return value
    if target_length < current_length:
        return value[
            _axis_slice(
                ndim=value.ndim,
                axis=normalized_axis,
                stop=target_length,
            )
        ]
    return _zero_pad_axis(
        value,
        axis=normalized_axis,
        before=0,
        after=target_length - current_length,
    )


def _resize_axes_adjoint(
    value: xp.ndarray,
    *,
    target_shape: tuple[int, ...],
    axes: tuple[int, ...],
) -> xp.ndarray:
    result = value
    for axis, target_length in zip(axes, target_shape, strict=True):
        result = _resize_axis_adjoint(
            result,
            target_length=target_length,
            axis=axis,
        )
    return result


def _transform_axes(
    *,
    ndim: int,
    shape: tuple[int, ...] | None,
    axes: tuple[int, ...] | None,
) -> tuple[int, ...]:
    if axes is None:
        if shape is None:
            return tuple(range(ndim))
        return tuple(range(ndim - len(shape), ndim))
    return tuple(_normalize_axis(axis, ndim=ndim, op_name="FFT") for axis in axes)


def _complex_fftn_vjp(adjoint: str) -> Callable[..., tuple[xp.ndarray]]:
    """Transpose a complex N-D FFT through its adjoint transform ``adjoint``."""

    def vjp(
        ans: xp.ndarray,
        x: xp.ndarray,
        *rest: xp.ndarray,
        g: xp.ndarray,
        s: tuple[int, ...] | None = None,
        axes: tuple[int, ...] | None = None,
        norm: FFTNorm | None = None,
        **attrs: Any,
    ) -> tuple[xp.ndarray]:
        _ = ans, rest, attrs
        transformed = getattr(xp.fft, adjoint)(g, s=s, axes=axes, norm=_adjoint_norm(norm))
        normalized_axes = _transform_axes(ndim=x.ndim, shape=s, axes=axes)
        target_shape = tuple(int(x.shape[axis]) for axis in normalized_axes)
        return (
            _resize_axes_adjoint(
                transformed,
                target_shape=target_shape,
                axes=normalized_axes,
            ),
        )

    return vjp


_vjp_fftn = _complex_fftn_vjp("ifftn")
_vjp_ifftn = _complex_fftn_vjp("fftn")


def _vjp_rfftn(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    s: tuple[int, ...] | None = None,
    axes: tuple[int, ...] | None = None,
    norm: FFTNorm | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Embed the final half-spectrum axis and apply the full N-D FFT adjoint."""
    _ = ans, rest, attrs
    normalized_axes = _transform_axes(ndim=x.ndim, shape=s, axes=axes)
    if not normalized_axes:
        msg = "rfftn transpose requires at least one transform axis"
        raise ValueError(msg)
    transform_shape = (
        tuple(int(x.shape[axis]) for axis in normalized_axes) if s is None else tuple(s)
    )
    real_axis = normalized_axes[-1]
    if transform_shape[-1] < int(g.shape[real_axis]):
        msg = "rfftn cotangent is longer than its full transform axis"
        raise ValueError(msg)
    spectrum = _resize_axis_adjoint(g, target_length=transform_shape[-1], axis=real_axis)
    transformed = xp.real(
        xp.fft.ifftn(
            spectrum,
            s=transform_shape,
            axes=normalized_axes,
            norm=_adjoint_norm(norm),
        )
    )
    target_shape = tuple(int(x.shape[axis]) for axis in normalized_axes)
    return (
        _resize_axes_adjoint(
            transformed,
            target_shape=target_shape,
            axes=normalized_axes,
        ),
    )


def _vjp_irfftn(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    s: tuple[int, ...] | None = None,
    axes: tuple[int, ...] | None = None,
    norm: FFTNorm | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Apply the weighted half-spectrum adjoint of an N-D inverse real FFT."""
    _ = ans, rest, attrs
    normalized_axes = _transform_axes(ndim=x.ndim, shape=s, axes=axes)
    if not normalized_axes:
        msg = "irfftn transpose requires at least one transform axis"
        raise ValueError(msg)
    transform_shape = (
        tuple(int(g.shape[axis]) for axis in normalized_axes) if s is None else tuple(s)
    )
    spectrum = xp.fft.rfftn(
        g,
        s=transform_shape,
        axes=normalized_axes,
        norm=_adjoint_norm(norm),
    )
    real_axis = normalized_axes[-1]
    half_length = int(spectrum.shape[real_axis])
    weights = xp.ones((half_length,), dtype=xp.real(spectrum).dtype)
    if half_length > 1:
        endpoint = half_length - 1 if transform_shape[-1] % 2 == 0 else half_length
        if endpoint > 1:
            weights[1:endpoint] = 2
    weight_shape = [1] * spectrum.ndim
    weight_shape[real_axis] = half_length
    weighted = spectrum * xp.reshape(weights, tuple(weight_shape))
    target_shape = tuple(int(x.shape[axis]) for axis in normalized_axes)
    return (
        _resize_axes_adjoint(
            weighted,
            target_shape=target_shape,
            axes=normalized_axes,
        ),
    )


def _one_axis(nd_vjp: Callable[..., tuple[xp.ndarray]]) -> Callable[..., tuple[xp.ndarray]]:
    """Transpose a one-axis transform as its N-D form over ``axes=(axis,)``."""

    def vjp(
        ans: xp.ndarray,
        x: xp.ndarray,
        *rest: xp.ndarray,
        g: xp.ndarray,
        n: int | None = None,
        axis: int = -1,
        norm: FFTNorm | None = None,
        **attrs: Any,
    ) -> tuple[xp.ndarray]:
        s = None if n is None else (n,)
        return nd_vjp(ans, x, *rest, g=g, s=s, axes=(axis,), norm=norm, **attrs)

    return vjp


_vjp_fft = _one_axis(_vjp_fftn)
_vjp_ifft = _one_axis(_vjp_ifftn)
_vjp_rfft = _one_axis(_vjp_rfftn)
_vjp_irfft = _one_axis(_vjp_irfftn)
_vjp_fft2 = partial(_vjp_fftn, axes=(-2, -1))
_vjp_ifft2 = partial(_vjp_ifftn, axes=(-2, -1))
_vjp_rfft2 = partial(_vjp_rfftn, axes=(-2, -1))
_vjp_irfft2 = partial(_vjp_irfftn, axes=(-2, -1))


def _vjp_fftshift(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    axes: int | tuple[int, ...] | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    _ = ans, inputs, attrs
    return (xp.fft.ifftshift(g, axes=axes),)


def _vjp_ifftshift(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    axes: int | tuple[int, ...] | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    _ = ans, inputs, attrs
    return (xp.fft.fftshift(g, axes=axes),)
