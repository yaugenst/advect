"""Traceable real adjoints for NumPy FFT primitives."""

from __future__ import annotations

from typing import Any, Literal, cast

from advect.autodiff.rules.array_family._backend_runtime import _array_constructor_like, xp

type FFTNorm = Literal["backward", "ortho", "forward"]


def _adjoint_norm(norm: FFTNorm | None) -> FFTNorm:
    if norm in {None, "backward"}:
        return "forward"
    if norm == "forward":
        return "backward"
    return "ortho"


def _normalize_axis(axis: int, *, ndim: int) -> int:
    normalized = int(axis)
    if normalized < 0:
        normalized += ndim
    if normalized < 0 or normalized >= ndim:
        msg = f"FFT axis {axis} is out of bounds for rank {ndim}"
        raise ValueError(msg)
    return normalized


def _axis_slice(
    *,
    ndim: int,
    axis: int,
    start: int | None = None,
    stop: int | None = None,
) -> tuple[slice, ...]:
    result = [slice(None)] * ndim
    result[axis] = slice(start, stop)
    return tuple(result)


def _resize_axis_adjoint(
    value: xp.ndarray,
    *,
    target_length: int,
    axis: int,
) -> xp.ndarray:
    """Transpose NumPy's crop-or-zero-pad behavior for one transform axis."""
    normalized_axis = _normalize_axis(axis, ndim=value.ndim)
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
    pad_width = [(0, 0)] * value.ndim
    pad_width[normalized_axis] = (0, target_length - current_length)
    return cast(
        "xp.ndarray",
        xp.pad(value, tuple(pad_width), mode="constant", constant_values=0),
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
    return tuple(_normalize_axis(axis, ndim=ndim) for axis in axes)


def _vjp_fft(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    n: int | None = None,
    axis: int = -1,
    norm: FFTNorm | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    _ = ans, rest, attrs
    transformed = xp.fft.ifft(g, n=n, axis=axis, norm=_adjoint_norm(norm))
    return (
        _resize_axis_adjoint(
            transformed,
            target_length=int(x.shape[_normalize_axis(axis, ndim=x.ndim)]),
            axis=axis,
        ),
    )


def _vjp_ifft(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    n: int | None = None,
    axis: int = -1,
    norm: FFTNorm | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    _ = ans, rest, attrs
    transformed = xp.fft.fft(g, n=n, axis=axis, norm=_adjoint_norm(norm))
    return (
        _resize_axis_adjoint(
            transformed,
            target_length=int(x.shape[_normalize_axis(axis, ndim=x.ndim)]),
            axis=axis,
        ),
    )


def _vjp_fftn(
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
    transformed = xp.fft.ifftn(g, s=s, axes=axes, norm=_adjoint_norm(norm))
    normalized_axes = _transform_axes(ndim=x.ndim, shape=s, axes=axes)
    target_shape = tuple(int(x.shape[axis]) for axis in normalized_axes)
    return (
        _resize_axes_adjoint(
            transformed,
            target_shape=target_shape,
            axes=normalized_axes,
        ),
    )


def _vjp_ifftn(
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
    transformed = xp.fft.fftn(g, s=s, axes=axes, norm=_adjoint_norm(norm))
    normalized_axes = _transform_axes(ndim=x.ndim, shape=s, axes=axes)
    target_shape = tuple(int(x.shape[axis]) for axis in normalized_axes)
    return (
        _resize_axes_adjoint(
            transformed,
            target_shape=target_shape,
            axes=normalized_axes,
        ),
    )


def _vjp_fft2(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    s: tuple[int, ...] | None = None,
    axes: tuple[int, ...] | None = (-2, -1),
    norm: FFTNorm | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    return _vjp_fftn(
        ans,
        x,
        *rest,
        g=g,
        s=s,
        axes=axes,
        norm=norm,
        **attrs,
    )


def _vjp_ifft2(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    s: tuple[int, ...] | None = None,
    axes: tuple[int, ...] | None = (-2, -1),
    norm: FFTNorm | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    return _vjp_ifftn(
        ans,
        x,
        *rest,
        g=g,
        s=s,
        axes=axes,
        norm=norm,
        **attrs,
    )


def _vjp_rfft(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    n: int | None = None,
    axis: int = -1,
    norm: FFTNorm | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Embed the half spectrum, then apply the full complex FFT adjoint."""
    _ = ans, rest, attrs
    normalized_axis = _normalize_axis(axis, ndim=x.ndim)
    transform_length = int(x.shape[normalized_axis]) if n is None else int(n)
    half_length = int(g.shape[normalized_axis])
    missing_length = transform_length - half_length
    if missing_length < 0:
        msg = "rfft cotangent is longer than its full transform length"
        raise ValueError(msg)
    if missing_length:
        zeros_shape = list(g.shape)
        zeros_shape[normalized_axis] = missing_length
        zeros = _array_constructor_like(
            g,
            "zeros",
            tuple(zeros_shape),
            dtype=g.dtype,
        )
        spectrum = xp.concatenate((g, zeros), axis=normalized_axis)
    else:
        spectrum = g
    transformed = xp.real(
        xp.fft.ifft(
            spectrum,
            n=transform_length,
            axis=normalized_axis,
            norm=_adjoint_norm(norm),
        )
    )
    return (
        _resize_axis_adjoint(
            transformed,
            target_length=int(x.shape[normalized_axis]),
            axis=normalized_axis,
        ),
    )


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
        tuple(int(x.shape[axis]) for axis in normalized_axes)
        if s is None
        else tuple(int(length) for length in s)
    )
    real_axis = normalized_axes[-1]
    missing_length = transform_shape[-1] - int(g.shape[real_axis])
    if missing_length < 0:
        msg = "rfftn cotangent is longer than its full transform axis"
        raise ValueError(msg)
    if missing_length:
        zeros_shape = list(g.shape)
        zeros_shape[real_axis] = missing_length
        zeros = _array_constructor_like(
            g,
            "zeros",
            tuple(zeros_shape),
            dtype=g.dtype,
        )
        spectrum = xp.concatenate((g, zeros), axis=real_axis)
    else:
        spectrum = g
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


def _vjp_rfft2(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    s: tuple[int, ...] | None = None,
    axes: tuple[int, ...] | None = (-2, -1),
    norm: FFTNorm | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    return _vjp_rfftn(
        ans,
        x,
        *rest,
        g=g,
        s=s,
        axes=axes,
        norm=norm,
        **attrs,
    )


def _vjp_irfft(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    n: int | None = None,
    axis: int = -1,
    norm: FFTNorm | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Apply the weighted half-spectrum adjoint of a real inverse FFT."""
    _ = ans, rest, attrs
    normalized_axis = _normalize_axis(axis, ndim=g.ndim)
    transform_length = int(g.shape[normalized_axis]) if n is None else int(n)
    spectrum = xp.fft.rfft(
        g,
        n=transform_length,
        axis=normalized_axis,
        norm=_adjoint_norm(norm),
    )
    half_length = int(spectrum.shape[normalized_axis])
    weights = xp.ones((half_length,), dtype=xp.real(spectrum).dtype)
    if half_length > 1:
        endpoint = half_length - 1 if transform_length % 2 == 0 else half_length
        if endpoint > 1:
            weights[1:endpoint] = 2
    weight_shape = [1] * spectrum.ndim
    weight_shape[normalized_axis] = half_length
    weighted = spectrum * xp.reshape(weights, tuple(weight_shape))
    return (
        _resize_axis_adjoint(
            weighted,
            target_length=int(x.shape[_normalize_axis(axis, ndim=x.ndim)]),
            axis=axis,
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
        tuple(int(g.shape[axis]) for axis in normalized_axes)
        if s is None
        else tuple(int(length) for length in s)
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


def _vjp_irfft2(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    s: tuple[int, ...] | None = None,
    axes: tuple[int, ...] | None = (-2, -1),
    norm: FFTNorm | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    return _vjp_irfftn(
        ans,
        x,
        *rest,
        g=g,
        s=s,
        axes=axes,
        norm=norm,
        **attrs,
    )


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


__all__ = [
    "_vjp_fft",
    "_vjp_fft2",
    "_vjp_fftn",
    "_vjp_fftshift",
    "_vjp_ifft",
    "_vjp_ifft2",
    "_vjp_ifftn",
    "_vjp_ifftshift",
    "_vjp_irfft",
    "_vjp_irfft2",
    "_vjp_irfftn",
    "_vjp_rfft",
    "_vjp_rfft2",
    "_vjp_rfftn",
]
