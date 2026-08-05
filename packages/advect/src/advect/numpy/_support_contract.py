"""Compact public support declarations for the concrete NumPy frontend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type NumpyCallableKind = Literal["array_method", "function", "ufunc_call", "ufunc_method"]
type NumpyMode = Literal["dynamic", "staged", "serialized"]

_ALL_MODES: tuple[NumpyMode, ...] = ("dynamic", "staged", "serialized")
_DYNAMIC_ONLY: tuple[NumpyMode, ...] = ("dynamic",)


@dataclass(frozen=True, slots=True)
class NumpySupportDeclaration:
    """One public callable spelling and its conservative runtime contract."""

    kind: NumpyCallableKind
    callable: str
    modes: tuple[NumpyMode, ...]
    has_derivatives: bool


def _names(value: str) -> tuple[str, ...]:
    return tuple(value.split())


def _group(
    kind: NumpyCallableKind,
    modes: tuple[NumpyMode, ...],
    names: str,
    *,
    has_derivatives: bool,
) -> tuple[NumpySupportDeclaration, ...]:
    return tuple(
        NumpySupportDeclaration(kind, f"numpy.{name}", modes, has_derivatives)
        for name in _names(names)
    )


_DECLARATIONS = (
    *_group(
        "array_method",
        _ALL_MODES,
        "ndarray.astype ndarray.copy ndarray.item ndarray.reshape ndarray.sum",
        has_derivatives=True,
    ),
    *_group("array_method", _DYNAMIC_ONLY, "ndarray.transpose", has_derivatives=True),
    *_group(
        "function",
        _ALL_MODES,
        """
        all any arange argmax argmin argsort count_nonzero empty empty_like eye
        ones searchsorted size zeros
        """,
        has_derivatives=False,
    ),
    *_group(
        "function",
        _ALL_MODES,
        """
        angle array asanyarray asarray astype average broadcast_to clip compress
        concatenate convolve copy correlate cross cumprod cumsum cumulative_prod
        cumulative_sum diagonal diff dot expand_dims fft.fft fft.fft2 fft.fftn
        fft.fftshift fft.hfft fft.ifft fft.ifft2 fft.ifftn fft.ifftshift fft.ihfft
        fft.irfft fft.irfft2 fft.irfftn fft.rfft fft.rfft2 fft.rfftn flip full
        full_like gradient imag linalg.cholesky linalg.cross linalg.det
        linalg.diagonal linalg.eigh linalg.eigvalsh linalg.inv linalg.matmul
        linalg.matrix_norm linalg.matrix_power linalg.matrix_transpose linalg.norm
        linalg.outer linalg.pinv linalg.qr linalg.slogdet linalg.solve linalg.svd
        linalg.svdvals linalg.tensordot linalg.trace linalg.vecdot
        linalg.vector_norm matrix_transpose max mean min moveaxis nanmax nanmean
        nanmin nanprod nanstd nansum nanvar ones_like outer prod real repeat reshape
        roll sort squeeze stack std sum take take_along_axis tensordot tile
        trace transpose tril triu var where zeros_like
        """,
        has_derivatives=True,
    ),
    *_group(
        "function",
        _DYNAMIC_ONLY,
        """
        allclose argpartition argwhere array_equal array_equiv can_cast common_type
        diag_indices_from digitize flatnonzero identity in1d isclose iscomplex
        iscomplexobj isin isneginf isposinf isreal isrealobj ix_ lexsort
        linalg.matrix_rank nanargmax nanargmin ndim nonzero ravel_multi_index
        result_type shape tri tril_indices_from triu_indices_from unique
        unique_counts unique_inverse unravel_index
        """,
        has_derivatives=False,
    ),
    *_group(
        "function",
        _DYNAMIC_ONLY,
        """
        amax amin append apply_along_axis apply_over_axes around array_split
        atleast_1d atleast_2d atleast_3d bincount block broadcast_arrays choose
        column_stack copyto corrcoef cov delete diag diagflat dsplit dstack ediff1d
        einsum extract fill_diagonal fix fliplr flipud geomspace histogram
        histogram2d histogram_bin_edges histogramdd hsplit hstack i0 inner insert
        interp intersect1d kron lib.scimath.arccos lib.scimath.arcsin
        lib.scimath.arctanh lib.scimath.log lib.scimath.log10 lib.scimath.log2
        lib.scimath.logn lib.scimath.power lib.scimath.sqrt
        lib.stride_tricks.sliding_window_view linalg.cond linalg.eig
        linalg.eigvals linalg.lstsq linalg.multi_dot linalg.tensorinv
        linalg.tensorsolve linspace logspace median meshgrid nan_to_num nancumprod
        nancumsum nanmedian nanpercentile nanquantile pad partition percentile
        piecewise place poly polyadd polyder polydiv polyfit polyint polymul polysub
        polyval ptp put put_along_axis putmask quantile ravel real_if_close resize
        rollaxis roots rot90 round row_stack select setdiff1d setxor1d sinc
        sort_complex split swapaxes trapezoid trapz trim_zeros union1d unique_all
        unique_values unstack unwrap vander vdot vsplit vstack
        """,
        has_derivatives=True,
    ),
    *_group(
        "ufunc_call",
        _ALL_MODES,
        """
        bitwise_and bitwise_or bitwise_xor equal greater greater_equal invert
        isfinite isinf isnan less less_equal logical_and logical_not logical_or
        logical_xor not_equal signbit
        """,
        has_derivatives=False,
    ),
    *_group(
        "ufunc_call",
        _ALL_MODES,
        """
        absolute add arccos arccosh arcsin arcsinh arctan arctan2 arctanh ceil
        conjugate copysign cos cosh divide exp expm1 floor floor_divide heaviside
        hypot ldexp log log10 log1p log2 logaddexp matmul matvec maximum minimum
        multiply negative nextafter positive power reciprocal remainder rint sign
        sin sinh spacing sqrt square subtract tan tanh trunc vecdot vecmat
        """,
        has_derivatives=True,
    ),
    *_group(
        "ufunc_call",
        _DYNAMIC_ONLY,
        """
        cbrt deg2rad degrees divmod exp2 fabs float_power fmax fmin fmod frexp
        logaddexp2 modf rad2deg radians
        """,
        has_derivatives=True,
    ),
    *_group(
        "ufunc_method",
        _ALL_MODES,
        """
        bitwise_and.outer bitwise_or.outer bitwise_xor.outer equal.outer
        greater.outer greater_equal.outer less.outer less_equal.outer
        logical_and.outer logical_or.outer logical_xor.outer not_equal.outer
        """,
        has_derivatives=False,
    ),
    *_group(
        "ufunc_method",
        _ALL_MODES,
        """
        add.accumulate add.outer add.reduce arctan2.outer copysign.outer
        divide.outer floor_divide.outer heaviside.outer hypot.outer ldexp.outer
        logaddexp.outer maximum.outer minimum.outer multiply.accumulate
        multiply.outer multiply.reduce nextafter.outer power.outer remainder.outer
        subtract.outer
        """,
        has_derivatives=True,
    ),
    *_group(
        "ufunc_method",
        _DYNAMIC_ONLY,
        "float_power.outer fmax.outer fmin.outer fmod.outer logaddexp2.outer",
        has_derivatives=True,
    ),
)


def numpy_support_declarations() -> tuple[NumpySupportDeclaration, ...]:
    """Return the declared public NumPy support contract."""
    return _DECLARATIONS


__all__ = ["NumpySupportDeclaration", "numpy_support_declarations"]
