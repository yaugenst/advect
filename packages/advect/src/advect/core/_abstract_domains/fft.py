"""Abstract registrations and evaluators for Fourier transforms."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from advect.core._abstract_helpers import (
    fft_dtype,
    fft_shape,
    fftn_shape,
)
from advect.core._abstract_model import ArraySpec, rule

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from advect.core._abstract_model import AbstractRule, ResultEvaluator


# Each call schema lists its evaluator kinds and the transforms that share them.
_TRANSFORMS = {
    ("n", "axis", "norm"): {"fft": "fft ifft", "rfft": "rfft ihfft", "irfft": "irfft hfft"},
    ("s", "axes", "norm"): {
        "fft2": "fft2 ifft2",
        "fftn": "fftn ifftn",
        "rfft2": "rfft2",
        "rfftn": "rfftn",
        "irfft2": "irfft2",
        "irfftn": "irfftn",
    },
}
RULES: dict[str, AbstractRule] = {
    **{
        f"array_ext.fft.{name}": rule(kind, 1, positional=schema, allowed=schema)
        for schema, kinds in _TRANSFORMS.items()
        for kind, names in kinds.items()
        for name in names.split()
    },
    **{
        f"array_ext.fft.{kind}": rule(
            kind, 0, positional=("n",), allowed=("d", "dtype", "n"), required=("dtype", "n")
        )
        for kind in ("fftfreq", "rfftfreq")
    },
    **{
        f"array_ext.fft.{name}": rule("same", 1, positional=("axes",), allowed=("axes",))
        for name in ("fftshift", "ifftshift")
    },
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
    return (ArraySpec((size,), attrs["dtype"]),)


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
    return (ArraySpec(shape, fft_dtype(first.dtype, real_output=kind == "irfft")),)


def _fftn_family(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
    *,
    kind: str,
    default_axes: tuple[int, ...] | None = None,
) -> tuple[ArraySpec, ...]:
    first = specs[0]
    axes = attrs.get("axes", default_axes)
    shape = fftn_shape(
        first.shape,
        sizes=attrs.get("s"),
        axes=axes,
        real_output=kind == "rfftn",
        inverse_real=kind == "irfftn",
    )
    dtype = first.dtype
    if kind == "irfftn" and len(first.shape if axes is None else axes) > 1:
        # NumPy transforms every axis but the last one as complex first.
        dtype = fft_dtype(dtype, real_output=False)
    return (ArraySpec(shape, fft_dtype(dtype, real_output=kind == "irfftn")),)


EVALUATORS: dict[str, ResultEvaluator] = {
    "fftfreq": partial(_frequency_grid, real=False),
    "rfftfreq": partial(_frequency_grid, real=True),
    **{kind: partial(_fft_family, kind=kind) for kind in ("fft", "rfft", "irfft")},
    **{kind: partial(_fftn_family, kind=kind) for kind in ("fftn", "rfftn", "irfftn")},
    # NumPy's two-dimensional transforms default to the last two axes.
    **{
        f"{kind[:-1]}2": partial(_fftn_family, kind=kind, default_axes=(-2, -1))
        for kind in ("fftn", "rfftn", "irfftn")
    },
}
