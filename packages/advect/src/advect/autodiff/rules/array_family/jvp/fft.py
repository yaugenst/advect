"""Fft JVP rules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from advect.autodiff.rules.array_family._backend_runtime import xp
from advect.autodiff.rules.array_family._transpose_utils import (
    _adjoint_fft_norm as _adjoint_norm,
)
from advect.autodiff.rules.array_family.jvp.common import _zeros_output_tangent

if TYPE_CHECKING:
    from advect.autodiff.rules.array_family._transpose_utils import FFTNorm


def _jvp_fft(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    n: int | None = None,
    axis: int = -1,
    norm: FFTNorm | None = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return cast("xp.ndarray[Any, Any]", xp.fft.fft(tangent, n=n, axis=axis, norm=norm))


def _jvp_ifft(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    n: int | None = None,
    axis: int = -1,
    norm: FFTNorm | None = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return cast("xp.ndarray[Any, Any]", xp.fft.ifft(tangent, n=n, axis=axis, norm=norm))


def _jvp_fft2(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    s: tuple[int, ...] | None = None,
    axes: tuple[int, ...] | None = (-2, -1),
    norm: FFTNorm | None = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return cast("xp.ndarray[Any, Any]", xp.fft.fft2(tangent, s=s, axes=axes, norm=norm))


def _jvp_ifft2(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    s: tuple[int, ...] | None = None,
    axes: tuple[int, ...] | None = (-2, -1),
    norm: FFTNorm | None = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return cast("xp.ndarray[Any, Any]", xp.fft.ifft2(tangent, s=s, axes=axes, norm=norm))


def _jvp_fftn(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    s: tuple[int, ...] | None = None,
    axes: tuple[int, ...] | None = None,
    norm: FFTNorm | None = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return cast("xp.ndarray[Any, Any]", xp.fft.fftn(tangent, s=s, axes=axes, norm=norm))


def _jvp_ifftn(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    s: tuple[int, ...] | None = None,
    axes: tuple[int, ...] | None = None,
    norm: FFTNorm | None = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return cast("xp.ndarray[Any, Any]", xp.fft.ifftn(tangent, s=s, axes=axes, norm=norm))


def _jvp_rfft(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    n: int | None = None,
    axis: int = -1,
    norm: FFTNorm | None = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return cast("xp.ndarray[Any, Any]", xp.fft.rfft(tangent, n=n, axis=axis, norm=norm))


def _jvp_rfft2(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    s: tuple[int, ...] | None = None,
    axes: tuple[int, ...] | None = (-2, -1),
    norm: FFTNorm | None = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return cast("xp.ndarray[Any, Any]", xp.fft.rfft2(tangent, s=s, axes=axes, norm=norm))


def _jvp_rfftn(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    s: tuple[int, ...] | None = None,
    axes: tuple[int, ...] | None = None,
    norm: FFTNorm | None = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return cast("xp.ndarray[Any, Any]", xp.fft.rfftn(tangent, s=s, axes=axes, norm=norm))


def _jvp_irfft(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    n: int | None = None,
    axis: int = -1,
    norm: FFTNorm | None = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return cast("xp.ndarray[Any, Any]", xp.fft.irfft(tangent, n=n, axis=axis, norm=norm))


def _jvp_irfft2(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    s: tuple[int, ...] | None = None,
    axes: tuple[int, ...] | None = (-2, -1),
    norm: FFTNorm | None = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return cast("xp.ndarray[Any, Any]", xp.fft.irfft2(tangent, s=s, axes=axes, norm=norm))


def _jvp_irfftn(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    s: tuple[int, ...] | None = None,
    axes: tuple[int, ...] | None = None,
    norm: FFTNorm | None = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    if s is not None and axes is None:
        axes = tuple(range(tangent.ndim - len(s), tangent.ndim))
    return cast("xp.ndarray[Any, Any]", xp.fft.irfftn(tangent, s=s, axes=axes, norm=norm))


def _jvp_hfft(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    n: int | None = None,
    axis: int = -1,
    norm: FFTNorm | None = None,
    **attrs: Any,
) -> xp.ndarray:
    """Differentiate hfft through its conjugated inverse-real FFT identity."""
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return cast(
        "xp.ndarray[Any, Any]",
        xp.fft.irfft(
            xp.conj(tangent),
            n=n,
            axis=axis,
            norm=_adjoint_norm(norm),
        ),
    )


def _jvp_ihfft(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    n: int | None = None,
    axis: int = -1,
    norm: FFTNorm | None = None,
    **attrs: Any,
) -> xp.ndarray:
    """Differentiate ihfft through its conjugated real FFT identity."""
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return cast(
        "xp.ndarray[Any, Any]",
        xp.conj(
            xp.fft.rfft(
                tangent,
                n=n,
                axis=axis,
                norm=_adjoint_norm(norm),
            )
        ),
    )


def _jvp_fftshift(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axes: int | tuple[int, ...] | None = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return cast("xp.ndarray[Any, Any]", xp.fft.fftshift(tangent, axes=axes))


def _jvp_ifftshift(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axes: int | tuple[int, ...] | None = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return cast("xp.ndarray[Any, Any]", xp.fft.ifftshift(tangent, axes=axes))
