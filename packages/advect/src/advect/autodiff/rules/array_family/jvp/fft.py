"""Fft JVP rules: each transform is linear, so it applies itself to the tangent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from advect.autodiff.rules.array_family._backend_runtime import xp
from advect.autodiff.rules.array_family._transpose_utils import (
    _adjoint_fft_norm as _adjoint_norm,
)
from advect.autodiff.rules.array_family.jvp.common import linear_jvp

if TYPE_CHECKING:
    from advect.autodiff.rules.array_family._transpose_utils import FFTNorm
    from advect.autodiff.rules.array_family.jvp.common import _JVPFn

type _Ints = tuple[int, ...] | None


def _along_axis(name: str) -> _JVPFn:
    @linear_jvp
    def jvp(
        t: Any, n: int | None = None, axis: int = -1, norm: FFTNorm | None = None, **_: Any
    ) -> Any:
        return getattr(xp.fft, name)(t, n=n, axis=axis, norm=norm)

    return jvp


def _over_axes(name: str, default_axes: _Ints) -> _JVPFn:
    @linear_jvp
    def jvp(
        t: Any, s: _Ints = None, axes: _Ints = default_axes, norm: FFTNorm | None = None, **_: Any
    ) -> Any:
        return getattr(xp.fft, name)(t, s=s, axes=axes, norm=norm)

    return jvp


_jvp_fft = _along_axis("fft")
_jvp_ifft = _along_axis("ifft")
_jvp_rfft = _along_axis("rfft")
_jvp_irfft = _along_axis("irfft")
# The Array API has no two-dimensional transforms; each is its n-dimensional
# transform over the last two axes by default.
_jvp_fft2 = _over_axes("fftn", (-2, -1))
_jvp_ifft2 = _over_axes("ifftn", (-2, -1))
_jvp_rfft2 = _over_axes("rfftn", (-2, -1))
_jvp_irfft2 = _over_axes("irfftn", (-2, -1))
_jvp_fftn = _over_axes("fftn", None)
_jvp_ifftn = _over_axes("ifftn", None)
_jvp_rfftn = _over_axes("rfftn", None)


@linear_jvp
def _jvp_irfftn(
    t: Any, s: _Ints = None, axes: _Ints = None, norm: FFTNorm | None = None, **_: Any
) -> Any:
    if s is not None and axes is None:
        axes = tuple(range(t.ndim - len(s), t.ndim))
    return xp.fft.irfftn(t, s=s, axes=axes, norm=norm)


@linear_jvp
def _jvp_hfft(
    t: Any, n: int | None = None, axis: int = -1, norm: FFTNorm | None = None, **_: Any
) -> Any:
    """Differentiate hfft through its conjugated inverse-real FFT identity."""
    return xp.fft.irfft(xp.conj(t), n=n, axis=axis, norm=_adjoint_norm(norm))


@linear_jvp
def _jvp_ihfft(
    t: Any, n: int | None = None, axis: int = -1, norm: FFTNorm | None = None, **_: Any
) -> Any:
    """Differentiate ihfft through its conjugated real FFT identity."""
    return xp.conj(xp.fft.rfft(t, n=n, axis=axis, norm=_adjoint_norm(norm)))


@linear_jvp
def _jvp_fftshift(t: Any, axes: int | tuple[int, ...] | None = None, **_: Any) -> Any:
    return xp.fft.fftshift(t, axes=axes)


@linear_jvp
def _jvp_ifftshift(t: Any, axes: int | tuple[int, ...] | None = None, **_: Any) -> Any:
    return xp.fft.ifftshift(t, axes=axes)
