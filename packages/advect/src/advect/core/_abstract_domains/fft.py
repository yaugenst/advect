"""Abstract registrations and evaluators for Fourier transforms."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from advect.core._abstract_helpers import (
    complex_dtype,
    dtype_name,
    fft_shape,
    fftn_shape,
    real_dtype,
)
from advect.core._abstract_model import ArraySpec, rule

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from advect.core._abstract_model import AbstractRule, ResultEvaluator


RULES: dict[str, AbstractRule] = {
    "array_ext.fft.fft": rule(
        "fft",
        1,
        positional=("n", "axis", "norm"),
        allowed=("n", "axis", "norm"),
    ),
    "array_ext.fft.fft2": rule(
        "fftn",
        1,
        positional=("s", "axes", "norm"),
        allowed=("s", "axes", "norm"),
    ),
    "array_ext.fft.fftfreq": rule(
        "fftfreq",
        0,
        positional=("n",),
        allowed=("d", "dtype", "n"),
        required=("dtype", "n"),
    ),
    "array_ext.fft.fftn": rule(
        "fftn",
        1,
        positional=("s", "axes", "norm"),
        allowed=("s", "axes", "norm"),
    ),
    "array_ext.fft.fftshift": rule(
        "same",
        1,
        positional=("axes",),
        allowed=("axes",),
    ),
    "array_ext.fft.hfft": rule(
        "irfft",
        1,
        positional=("n", "axis", "norm"),
        allowed=("n", "axis", "norm"),
    ),
    "array_ext.fft.ifft": rule(
        "fft",
        1,
        positional=("n", "axis", "norm"),
        allowed=("n", "axis", "norm"),
    ),
    "array_ext.fft.ifft2": rule(
        "fftn",
        1,
        positional=("s", "axes", "norm"),
        allowed=("s", "axes", "norm"),
    ),
    "array_ext.fft.ifftn": rule(
        "fftn",
        1,
        positional=("s", "axes", "norm"),
        allowed=("s", "axes", "norm"),
    ),
    "array_ext.fft.ifftshift": rule(
        "same",
        1,
        positional=("axes",),
        allowed=("axes",),
    ),
    "array_ext.fft.ihfft": rule(
        "rfft",
        1,
        positional=("n", "axis", "norm"),
        allowed=("n", "axis", "norm"),
    ),
    "array_ext.fft.irfft": rule(
        "irfft",
        1,
        positional=("n", "axis", "norm"),
        allowed=("n", "axis", "norm"),
    ),
    "array_ext.fft.irfft2": rule(
        "irfftn",
        1,
        positional=("s", "axes", "norm"),
        allowed=("s", "axes", "norm"),
    ),
    "array_ext.fft.irfftn": rule(
        "irfftn",
        1,
        positional=("s", "axes", "norm"),
        allowed=("s", "axes", "norm"),
    ),
    "array_ext.fft.rfft": rule(
        "rfft",
        1,
        positional=("n", "axis", "norm"),
        allowed=("n", "axis", "norm"),
    ),
    "array_ext.fft.rfft2": rule(
        "rfftn",
        1,
        positional=("s", "axes", "norm"),
        allowed=("s", "axes", "norm"),
    ),
    "array_ext.fft.rfftfreq": rule(
        "rfftfreq",
        0,
        positional=("n",),
        allowed=("d", "dtype", "n"),
        required=("dtype", "n"),
    ),
    "array_ext.fft.rfftn": rule(
        "rfftn",
        1,
        positional=("s", "axes", "norm"),
        allowed=("s", "axes", "norm"),
    ),
}


def _frequency_grid(
    _specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
    *,
    real: bool,
) -> tuple[ArraySpec, ...]:
    n = attrs["n"]
    name = "rfftfreq" if real else "fftfreq"
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"{name} n must be a positive integer")
    d = attrs.get("d", 1.0)
    if isinstance(d, bool) or not isinstance(d, (int, float)) or d == 0:
        raise ValueError(f"{name} d must be a nonzero real scalar")
    size = n // 2 + 1 if real else n
    return (ArraySpec((size,), dtype_name(attrs["dtype"])),)


def _fft_family(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
    *,
    kind: str,
) -> tuple[ArraySpec, ...]:
    first = specs[0]
    shape = fft_shape(
        first.shape,
        n=attrs.get("n"),
        axis=attrs.get("axis", -1),
        real_output=kind == "rfft",
        inverse_real=kind == "irfft",
    )
    result_dtype = real_dtype(first.dtype) if kind == "irfft" else complex_dtype(first.dtype)
    return (ArraySpec(shape, result_dtype),)


def _fftn_family(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
    *,
    kind: str,
) -> tuple[ArraySpec, ...]:
    first = specs[0]
    shape = fftn_shape(
        first.shape,
        sizes=attrs.get("s"),
        axes=attrs.get("axes"),
        real_output=kind == "rfftn",
        inverse_real=kind == "irfftn",
    )
    result_dtype = real_dtype(first.dtype) if kind == "irfftn" else complex_dtype(first.dtype)
    return (ArraySpec(shape, result_dtype),)


EVALUATORS: dict[str, ResultEvaluator] = {
    "fftfreq": partial(_frequency_grid, real=False),
    "rfftfreq": partial(_frequency_grid, real=True),
    **{kind: partial(_fft_family, kind=kind) for kind in ("fft", "rfft", "irfft")},
    **{kind: partial(_fftn_family, kind=kind) for kind in ("fftn", "rfftn", "irfftn")},
}
