# ruff: noqa: A002, PLR0913
# SciPy-compatible names/signatures intentionally trigger these rules.
"""Traceable counterparts to frequently used ``scipy.ndimage`` operations."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from scipy import ndimage as _scipy_ndimage

from advect.core._context import is_tracing
from advect.scipy._ndimage.common import (
    _call_primitive,
    _finish_output,
    _ndim_of,
    _normalize_axes,
    _normalize_axis,
    _normalize_modes,
    _normalize_origins,
    _normalize_output,
    _normalize_sequence,
    _operand_dtype,
    _output_dtype,
    _require_numpy_values,
    _static_scalar,
    _traceable_astype,
    _validate_ufunc_output_cast,
)
from advect.scipy._ndimage.filters import (
    _convolve1d_primitive,
    _convolve_primitive,
    _correlate1d_primitive,
    _correlate_primitive,
    _correlation_call,
    _gaussian_filter1d_primitive,
    _gaussian_filter_primitive,
    _uniform_filter1d_primitive,
    _uniform_filter_primitive,
)
from advect.scipy._ndimage.morphology import (
    _grey_dilation_primitive,
    _grey_erosion_primitive,
    _maximum_filter1d_primitive,
    _maximum_filter_primitive,
    _median_filter_primitive,
    _minimum_filter1d_primitive,
    _minimum_filter_primitive,
    _percentile_filter_primitive,
    _rank_filter_primitive,
    _selection_call,
)

if TYPE_CHECKING:
    from typing import Protocol

    class _LoweringMetadata(Protocol):
        __advect_lowering__: str


def gaussian_filter(
    input: object,
    sigma: object,
    order: object = 0,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    truncate: object = 4.0,
    *,
    radius: object = None,
    axes: object = None,
) -> object:
    """Apply a multidimensional Gaussian filter with exact boundary adjoints."""
    if not is_tracing():
        _require_numpy_values("gaussian_filter", input, output)
        return _scipy_ndimage.gaussian_filter(
            input,
            sigma,
            order=order,
            output=output,
            mode=mode,
            cval=cval,
            truncate=truncate,
            radius=radius,
            axes=axes,
        )
    normalized_axes = _normalize_axes(axes, _ndim_of(input))
    return _call_primitive(
        _gaussian_filter_primitive,
        name="gaussian_filter",
        input=input,
        output=output,
        operands={"cval": cval},
        static={
            "axes": normalized_axes,
            "sigmas": _normalize_sequence(sigma, len(normalized_axes)),
            "orders": _normalize_sequence(order, len(normalized_axes)),
            "modes": _normalize_modes(mode, len(normalized_axes)),
            "truncate": float(cast("Any", _static_scalar(truncate))),
            "radii": _normalize_sequence(radius, len(normalized_axes)),
        },
    )


def gaussian_filter1d(
    input: object,
    sigma: object,
    axis: object = -1,
    order: object = 0,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    truncate: object = 4.0,
    *,
    radius: object = None,
) -> object:
    """Apply a one-dimensional Gaussian filter along ``axis``."""
    if not is_tracing():
        _require_numpy_values("gaussian_filter1d", input, output)
        return _scipy_ndimage.gaussian_filter1d(
            input,
            sigma,
            axis=axis,
            order=order,
            output=output,
            mode=mode,
            cval=cval,
            truncate=truncate,
            radius=radius,
        )
    normalized_axis = _normalize_axis(axis, _ndim_of(input))
    return _call_primitive(
        _gaussian_filter1d_primitive,
        name="gaussian_filter1d",
        input=input,
        output=output,
        operands={"cval": cval},
        static={
            "axes": (normalized_axis,),
            "sigmas": (_static_scalar(sigma),),
            "orders": (_static_scalar(order),),
            "modes": (str(_static_scalar(mode)),),
            "truncate": float(cast("Any", _static_scalar(truncate))),
            "radii": (_static_scalar(radius),),
        },
    )


def uniform_filter(
    input: object,
    size: object = 3,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object:
    """Apply a multidimensional uniform filter."""
    if not is_tracing():
        _require_numpy_values("uniform_filter", input, output)
        return _scipy_ndimage.uniform_filter(
            input,
            size=size,
            output=output,
            mode=mode,
            cval=cval,
            origin=origin,
            axes=axes,
        )
    normalized_axes = _normalize_axes(axes, _ndim_of(input))
    return _call_primitive(
        _uniform_filter_primitive,
        name="uniform_filter",
        input=input,
        output=output,
        operands={"cval": cval},
        static={
            "axes": normalized_axes,
            "sizes": tuple(
                operator.index(cast("Any", item))
                for item in _normalize_sequence(size, len(normalized_axes))
            ),
            "origins": _normalize_origins(origin, len(normalized_axes)),
            "modes": _normalize_modes(mode, len(normalized_axes)),
        },
    )


def uniform_filter1d(
    input: object,
    size: object,
    axis: object = -1,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
) -> object:
    """Apply a one-dimensional uniform filter along ``axis``."""
    if not is_tracing():
        _require_numpy_values("uniform_filter1d", input, output)
        return _scipy_ndimage.uniform_filter1d(
            input,
            size,
            axis=axis,
            output=output,
            mode=mode,
            cval=cval,
            origin=origin,
        )
    normalized_axis = _normalize_axis(axis, _ndim_of(input))
    return _call_primitive(
        _uniform_filter1d_primitive,
        name="uniform_filter1d",
        input=input,
        output=output,
        operands={"cval": cval},
        static={
            "axes": (normalized_axis,),
            "sizes": (operator.index(cast("Any", _static_scalar(size))),),
            "origins": (operator.index(cast("Any", _static_scalar(origin))),),
            "modes": (str(_static_scalar(mode)),),
        },
    )


def convolve(
    input: object,
    weights: object,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object:
    """Multidimensional convolution with differentiable input and weights."""
    if not is_tracing():
        _require_numpy_values("convolve", input, weights, output)
        return _scipy_ndimage.convolve(
            input,
            weights,
            output=output,
            mode=mode,
            cval=cval,
            origin=origin,
            axes=axes,
        )
    return _correlation_call(
        "convolve",
        _convolve_primitive,
        input,
        weights,
        output,
        mode,
        cval,
        origin,
        axes,
        one_dimensional=False,
    )


def correlate(
    input: object,
    weights: object,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object:
    """Multidimensional correlation with differentiable input and weights."""
    if not is_tracing():
        _require_numpy_values("correlate", input, weights, output)
        return _scipy_ndimage.correlate(
            input,
            weights,
            output=output,
            mode=mode,
            cval=cval,
            origin=origin,
            axes=axes,
        )
    return _correlation_call(
        "correlate",
        _correlate_primitive,
        input,
        weights,
        output,
        mode,
        cval,
        origin,
        axes,
        one_dimensional=False,
    )


def convolve1d(
    input: object,
    weights: object,
    axis: object = -1,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
) -> object:
    """One-dimensional convolution with differentiable input and weights."""
    if not is_tracing():
        _require_numpy_values("convolve1d", input, weights, output)
        return _scipy_ndimage.convolve1d(
            input,
            weights,
            axis=axis,
            output=output,
            mode=mode,
            cval=cval,
            origin=origin,
        )
    return _correlation_call(
        "convolve1d",
        _convolve1d_primitive,
        input,
        weights,
        output,
        mode,
        cval,
        origin,
        axis,
        one_dimensional=True,
    )


def correlate1d(
    input: object,
    weights: object,
    axis: object = -1,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
) -> object:
    """One-dimensional correlation with differentiable input and weights."""
    if not is_tracing():
        _require_numpy_values("correlate1d", input, weights, output)
        return _scipy_ndimage.correlate1d(
            input,
            weights,
            axis=axis,
            output=output,
            mode=mode,
            cval=cval,
            origin=origin,
        )
    return _correlation_call(
        "correlate1d",
        _correlate1d_primitive,
        input,
        weights,
        output,
        mode,
        cval,
        origin,
        axis,
        one_dimensional=True,
    )


def laplace(
    input: object,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    *,
    axes: object = None,
) -> object:
    """Apply the discrete multidimensional Laplace operator."""
    if not is_tracing():
        _require_numpy_values("laplace", input, output)
        return _scipy_ndimage.laplace(
            input,
            output=output,
            mode=mode,
            cval=cval,
            axes=axes,
        )
    normalized_axes = _normalize_axes(axes, _ndim_of(input))
    if not normalized_axes:
        return _finish_output(
            input,
            input=input,
            output=output,
            operation="scipy.ndimage.laplace output=",
        )
    modes = _normalize_modes(mode, len(normalized_axes))
    dtype = _output_dtype(input, output)
    terms: list[Any] = [
        correlate1d(
            input,
            (1.0, -2.0, 1.0),
            axis=axis,
            output=dtype,
            mode=axis_mode,
            cval=cval,
        )
        for axis, axis_mode in zip(normalized_axes, modes, strict=True)
    ]
    result = terms[0]
    for term in terms[1:]:
        result = result + term
    return _finish_output(
        result,
        input=input,
        output=output,
        operation="scipy.ndimage.laplace output=",
    )


def gaussian_laplace(
    input: object,
    sigma: object,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    *,
    axes: object = None,
    **kwargs: object,
) -> object:
    """Apply a Laplacian of Gaussian filter."""
    if not is_tracing():
        _require_numpy_values("gaussian_laplace", input, output)
        return _scipy_ndimage.gaussian_laplace(
            input,
            sigma,
            output=output,
            mode=mode,
            cval=cval,
            axes=axes,
            **kwargs,
        )
    ndim = _ndim_of(input)
    normalized_axes = _normalize_axes(axes, ndim)
    if not normalized_axes:
        return _finish_output(
            input,
            input=input,
            output=output,
            operation="scipy.ndimage.gaussian_laplace output=",
        )
    selected_sigmas = _normalize_sequence(sigma, len(normalized_axes))
    if len(normalized_axes) < ndim:
        full_sigmas: list[object] = [0] * ndim
        for axis, axis_sigma in zip(normalized_axes, selected_sigmas, strict=True):
            full_sigmas[axis] = axis_sigma
    else:
        # SciPy preserves sequence order when every physical axis is selected,
        # even if ``axes`` itself is unsorted.
        full_sigmas = list(selected_sigmas)
    modes = _normalize_modes(mode, len(normalized_axes))
    dtype = _output_dtype(input, output)
    terms: list[Any] = []
    for axis, axis_mode in zip(normalized_axes, modes, strict=True):
        orders = [0] * ndim
        orders[axis] = 2
        terms.append(
            gaussian_filter(
                input,
                tuple(full_sigmas),
                order=tuple(orders),
                output=dtype,
                mode=axis_mode,
                cval=cval,
                **kwargs,
            )
        )
    result = terms[0]
    for term in terms[1:]:
        result = result + term
    return _finish_output(
        result,
        input=input,
        output=output,
        operation="scipy.ndimage.gaussian_laplace output=",
    )


def _edge_filter(
    name: str,
    input: object,
    axis: object,
    output: object,
    mode: object,
    cval: object,
    *,
    smoothing: tuple[float, float, float],
) -> object:
    ndim = _ndim_of(input)
    normalized_axis = _normalize_axis(axis, ndim)
    modes = _normalize_modes(mode, ndim)
    dtype = _output_dtype(input, output)
    result = correlate1d(
        input,
        (-1.0, 0.0, 1.0),
        axis=normalized_axis,
        output=dtype,
        mode=modes[normalized_axis],
        cval=cval,
    )
    for other_axis in range(ndim):
        if other_axis == normalized_axis:
            continue
        result = correlate1d(
            result,
            smoothing,
            axis=other_axis,
            output=dtype,
            mode=modes[other_axis],
            cval=cval,
        )
    return _finish_output(
        result,
        input=input,
        output=output,
        operation=f"scipy.ndimage.{name} output=",
    )


def sobel(
    input: object,
    axis: object = -1,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
) -> object:
    """Calculate an axis-specific Sobel filter."""
    if not is_tracing():
        _require_numpy_values("sobel", input, output)
        return _scipy_ndimage.sobel(
            input,
            axis=axis,
            output=output,
            mode=mode,
            cval=cval,
        )
    return _edge_filter(
        "sobel",
        input,
        axis,
        output,
        mode,
        cval,
        smoothing=(1.0, 2.0, 1.0),
    )


def prewitt(
    input: object,
    axis: object = -1,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
) -> object:
    """Calculate an axis-specific Prewitt filter."""
    if not is_tracing():
        _require_numpy_values("prewitt", input, output)
        return _scipy_ndimage.prewitt(
            input,
            axis=axis,
            output=output,
            mode=mode,
            cval=cval,
        )
    return _edge_filter(
        "prewitt",
        input,
        axis,
        output,
        mode,
        cval,
        smoothing=(1.0, 1.0, 1.0),
    )


for _composite in (laplace, gaussian_laplace, sobel, prewitt):
    cast("_LoweringMetadata", _composite).__advect_lowering__ = "composite"


def maximum_filter(
    input: object,
    size: object = None,
    footprint: object = None,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object:
    """Calculate a multidimensional maximum filter with symmetric tie gradients."""
    if not is_tracing():
        _require_numpy_values("maximum_filter", input, footprint, output)
        return _scipy_ndimage.maximum_filter(
            input,
            size=size,
            footprint=footprint,
            output=output,
            mode=mode,
            cval=cval,
            origin=origin,
            axes=axes,
        )
    return _selection_call(
        "maximum_filter",
        _maximum_filter_primitive,
        input,
        size=size,
        footprint=footprint,
        structure=None,
        output=output,
        mode=mode,
        cval=cval,
        origin=origin,
        axes=axes,
    )


def minimum_filter(
    input: object,
    size: object = None,
    footprint: object = None,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object:
    """Calculate a multidimensional minimum filter with symmetric tie gradients."""
    if not is_tracing():
        _require_numpy_values("minimum_filter", input, footprint, output)
        return _scipy_ndimage.minimum_filter(
            input,
            size=size,
            footprint=footprint,
            output=output,
            mode=mode,
            cval=cval,
            origin=origin,
            axes=axes,
        )
    return _selection_call(
        "minimum_filter",
        _minimum_filter_primitive,
        input,
        size=size,
        footprint=footprint,
        structure=None,
        output=output,
        mode=mode,
        cval=cval,
        origin=origin,
        axes=axes,
    )


def maximum_filter1d(
    input: object,
    size: object,
    axis: object = -1,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
) -> object:
    """Calculate a one-dimensional maximum filter."""
    if not is_tracing():
        _require_numpy_values("maximum_filter1d", input, output)
        return _scipy_ndimage.maximum_filter1d(
            input,
            size,
            axis=axis,
            output=output,
            mode=mode,
            cval=cval,
            origin=origin,
        )
    return _selection_call(
        "maximum_filter1d",
        _maximum_filter1d_primitive,
        input,
        size=size,
        footprint=None,
        structure=None,
        output=output,
        mode=mode,
        cval=cval,
        origin=origin,
        axes=axis,
        one_dimensional=True,
    )


def minimum_filter1d(
    input: object,
    size: object,
    axis: object = -1,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
) -> object:
    """Calculate a one-dimensional minimum filter."""
    if not is_tracing():
        _require_numpy_values("minimum_filter1d", input, output)
        return _scipy_ndimage.minimum_filter1d(
            input,
            size,
            axis=axis,
            output=output,
            mode=mode,
            cval=cval,
            origin=origin,
        )
    return _selection_call(
        "minimum_filter1d",
        _minimum_filter1d_primitive,
        input,
        size=size,
        footprint=None,
        structure=None,
        output=output,
        mode=mode,
        cval=cval,
        origin=origin,
        axes=axis,
        one_dimensional=True,
    )


def grey_dilation(
    input: object,
    size: object = None,
    footprint: object = None,
    structure: object = None,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object:
    """Calculate a greyscale dilation with symmetric tie gradients."""
    if not is_tracing():
        _require_numpy_values("grey_dilation", input, footprint, structure, output)
        return _scipy_ndimage.grey_dilation(
            input,
            size=size,
            footprint=footprint,
            structure=structure,
            output=output,
            mode=mode,
            cval=cval,
            origin=origin,
            axes=axes,
        )
    if size is None and footprint is None and structure is None:
        raise ValueError("size, footprint, or structure must be specified")
    return _selection_call(
        "grey_dilation",
        _grey_dilation_primitive,
        input,
        size=size,
        footprint=footprint,
        structure=structure,
        output=output,
        mode=mode,
        cval=cval,
        origin=origin,
        axes=axes,
    )


def grey_erosion(
    input: object,
    size: object = None,
    footprint: object = None,
    structure: object = None,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object:
    """Calculate a greyscale erosion with symmetric tie gradients."""
    if not is_tracing():
        _require_numpy_values("grey_erosion", input, footprint, structure, output)
        return _scipy_ndimage.grey_erosion(
            input,
            size=size,
            footprint=footprint,
            structure=structure,
            output=output,
            mode=mode,
            cval=cval,
            origin=origin,
            axes=axes,
        )
    if size is None and footprint is None and structure is None:
        raise ValueError("size, footprint, or structure must be specified")
    return _selection_call(
        "grey_erosion",
        _grey_erosion_primitive,
        input,
        size=size,
        footprint=footprint,
        structure=structure,
        output=output,
        mode=mode,
        cval=cval,
        origin=origin,
        axes=axes,
    )


def median_filter(
    input: object,
    size: object = None,
    footprint: object = None,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object:
    """Calculate a multidimensional median filter."""
    if not is_tracing():
        _require_numpy_values("median_filter", input, footprint, output)
        return _scipy_ndimage.median_filter(
            input,
            size=size,
            footprint=footprint,
            output=output,
            mode=mode,
            cval=cval,
            origin=origin,
            axes=axes,
        )
    return _selection_call(
        "median_filter",
        _median_filter_primitive,
        input,
        size=size,
        footprint=footprint,
        structure=None,
        output=output,
        mode=mode,
        cval=cval,
        origin=origin,
        axes=axes,
    )


def rank_filter(
    input: object,
    rank: object,
    size: object = None,
    footprint: object = None,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object:
    """Calculate a multidimensional rank filter."""
    if not is_tracing():
        _require_numpy_values("rank_filter", input, footprint, output)
        return _scipy_ndimage.rank_filter(
            input,
            rank,
            size=size,
            footprint=footprint,
            output=output,
            mode=mode,
            cval=cval,
            origin=origin,
            axes=axes,
        )
    return _selection_call(
        "rank_filter",
        _rank_filter_primitive,
        input,
        size=size,
        footprint=footprint,
        structure=None,
        output=output,
        mode=mode,
        cval=cval,
        origin=origin,
        axes=axes,
        rank_value=rank,
    )


def percentile_filter(
    input: object,
    percentile: object,
    size: object = None,
    footprint: object = None,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object:
    """Calculate a multidimensional percentile filter."""
    if not is_tracing():
        _require_numpy_values("percentile_filter", input, footprint, output)
        return _scipy_ndimage.percentile_filter(
            input,
            percentile,
            size=size,
            footprint=footprint,
            output=output,
            mode=mode,
            cval=cval,
            origin=origin,
            axes=axes,
        )
    return _selection_call(
        "percentile_filter",
        _percentile_filter_primitive,
        input,
        size=size,
        footprint=footprint,
        structure=None,
        output=output,
        mode=mode,
        cval=cval,
        origin=origin,
        axes=axes,
        rank_value=percentile,
    )


def _grey_open_or_close(
    name: Literal["grey_opening", "grey_closing"],
    input: object,
    *,
    size: object,
    footprint: object,
    structure: object,
    output: object,
    mode: object,
    cval: object,
    origin: object,
    axes: object,
) -> object:
    options = {
        "size": size,
        "footprint": footprint,
        "structure": structure,
        "mode": mode,
        "cval": cval,
        "origin": origin,
        "axes": axes,
    }
    if not is_tracing():
        _require_numpy_values(name, input, footprint, structure, output)
        return getattr(_scipy_ndimage, name)(input, output=output, **options)
    first, second = (
        (grey_erosion, grey_dilation) if name == "grey_opening" else (grey_dilation, grey_erosion)
    )
    intermediate = first(input, output=_operand_dtype(input), **options)
    return second(intermediate, output=output, **options)


def grey_opening(
    input: object,
    size: object = None,
    footprint: object = None,
    structure: object = None,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object:
    """Apply greyscale erosion followed by greyscale dilation."""
    return _grey_open_or_close(
        "grey_opening",
        input,
        size=size,
        footprint=footprint,
        structure=structure,
        output=output,
        mode=mode,
        cval=cval,
        origin=origin,
        axes=axes,
    )


def grey_closing(
    input: object,
    size: object = None,
    footprint: object = None,
    structure: object = None,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object:
    """Apply greyscale dilation followed by greyscale erosion."""
    return _grey_open_or_close(
        "grey_closing",
        input,
        size=size,
        footprint=footprint,
        structure=structure,
        output=output,
        mode=mode,
        cval=cval,
        origin=origin,
        axes=axes,
    )


def _morphology_pair(
    input: object,
    *,
    size: object,
    footprint: object,
    structure: object,
    mode: object,
    cval: object,
    origin: object,
    axes: object,
    output: object,
) -> tuple[Any, Any, object | None, np.dtype[Any]]:
    destination = _normalize_output(input, output).destination
    dtype = _operand_dtype(input if destination is None else destination)
    dilated = grey_dilation(
        input,
        size=size,
        footprint=footprint,
        structure=structure,
        output=_operand_dtype(input),
        mode=mode,
        cval=cval,
        origin=origin,
        axes=axes,
    )
    eroded = grey_erosion(
        input,
        size=size,
        footprint=footprint,
        structure=structure,
        output=dtype,
        mode=mode,
        cval=cval,
        origin=origin,
        axes=axes,
    )
    return dilated, eroded, destination, dtype


def morphological_gradient(
    input: object,
    size: object = None,
    footprint: object = None,
    structure: object = None,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object:
    """Calculate the difference between greyscale dilation and erosion."""
    if not is_tracing():
        _require_numpy_values("morphological_gradient", input, footprint, structure, output)
        return _scipy_ndimage.morphological_gradient(
            input,
            size=size,
            footprint=footprint,
            structure=structure,
            output=output,
            mode=mode,
            cval=cval,
            origin=origin,
            axes=axes,
        )
    dilated, eroded, destination, dtype = _morphology_pair(
        input,
        size=size,
        footprint=footprint,
        structure=structure,
        mode=mode,
        cval=cval,
        origin=origin,
        axes=axes,
        output=output,
    )
    if destination is not None:
        _validate_ufunc_output_cast(np.subtract, dilated, eroded, dtype)
    result = dilated - eroded
    return _finish_output(
        result,
        input=input,
        output=destination,
        operation="scipy.ndimage.morphological_gradient output=",
    )


def morphological_laplace(
    input: object,
    size: object = None,
    footprint: object = None,
    structure: object = None,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object:
    """Calculate the morphological Laplace operator."""
    if not is_tracing():
        _require_numpy_values("morphological_laplace", input, footprint, structure, output)
        return _scipy_ndimage.morphological_laplace(
            input,
            size=size,
            footprint=footprint,
            structure=structure,
            output=output,
            mode=mode,
            cval=cval,
            origin=origin,
            axes=axes,
        )
    dilated, eroded, destination, dtype = _morphology_pair(
        input,
        size=size,
        footprint=footprint,
        structure=structure,
        mode=mode,
        cval=cval,
        origin=origin,
        axes=axes,
        output=output,
    )
    if destination is not None:
        _validate_ufunc_output_cast(np.add, dilated, eroded, dtype)
        _validate_ufunc_output_cast(np.subtract, eroded, input, dtype)
    result = _traceable_astype(dilated + eroded, dtype)
    result = _traceable_astype(result - input, dtype)
    result = _traceable_astype(result - input, dtype)
    return _finish_output(
        result,
        input=input,
        output=destination,
        operation="scipy.ndimage.morphological_laplace output=",
    )


def _tophat(
    name: Literal["white_tophat", "black_tophat"],
    input: object,
    *,
    size: object,
    footprint: object,
    structure: object,
    output: object,
    mode: object,
    cval: object,
    origin: object,
    axes: object,
) -> object:
    options = {
        "size": size,
        "footprint": footprint,
        "structure": structure,
        "mode": mode,
        "cval": cval,
        "origin": origin,
        "axes": axes,
    }
    if not is_tracing():
        _require_numpy_values(name, input, footprint, structure, output)
        return getattr(_scipy_ndimage, name)(input, output=output, **options)
    morphology = grey_opening if name == "white_tophat" else grey_closing
    filtered = cast(
        "Any",
        morphology(input, output=_output_dtype(input, output), **options),
    )
    input_value = cast("Any", input)
    left, right = (input_value, filtered) if name == "white_tophat" else (filtered, input_value)
    if _operand_dtype(left) == np.dtype(bool) and _operand_dtype(right) == np.dtype(bool):
        result = np.bitwise_xor(left, right)
    else:
        _validate_ufunc_output_cast(np.subtract, left, right, _operand_dtype(filtered))
        result = left - right
    return _finish_output(
        result,
        input=input,
        output=output,
        operation=f"scipy.ndimage.{name} output=",
    )


def white_tophat(
    input: object,
    size: object = None,
    footprint: object = None,
    structure: object = None,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object:
    """Calculate the difference between the input and its greyscale opening."""
    return _tophat(
        "white_tophat",
        input=input,
        size=size,
        footprint=footprint,
        structure=structure,
        output=output,
        mode=mode,
        cval=cval,
        origin=origin,
        axes=axes,
    )


def black_tophat(
    input: object,
    size: object = None,
    footprint: object = None,
    structure: object = None,
    output: object = None,
    mode: object = "reflect",
    cval: object = 0.0,
    origin: object = 0,
    *,
    axes: object = None,
) -> object:
    """Calculate the difference between greyscale closing and the input."""
    return _tophat(
        "black_tophat",
        input=input,
        size=size,
        footprint=footprint,
        structure=structure,
        output=output,
        mode=mode,
        cval=cval,
        origin=origin,
        axes=axes,
    )


for _composite in (
    grey_opening,
    grey_closing,
    morphological_gradient,
    morphological_laplace,
    white_tophat,
    black_tophat,
):
    cast("_LoweringMetadata", _composite).__advect_lowering__ = "composite"


__all__ = [
    "black_tophat",
    "convolve",
    "convolve1d",
    "correlate",
    "correlate1d",
    "gaussian_filter",
    "gaussian_filter1d",
    "gaussian_laplace",
    "grey_closing",
    "grey_dilation",
    "grey_erosion",
    "grey_opening",
    "laplace",
    "maximum_filter",
    "maximum_filter1d",
    "median_filter",
    "minimum_filter",
    "minimum_filter1d",
    "morphological_gradient",
    "morphological_laplace",
    "percentile_filter",
    "prewitt",
    "rank_filter",
    "sobel",
    "uniform_filter",
    "uniform_filter1d",
    "white_tophat",
]
