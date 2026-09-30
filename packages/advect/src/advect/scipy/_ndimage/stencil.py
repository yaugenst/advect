# ruff: noqa: A002, ANN401
# SciPy-compatible names/signatures and primitive rule schemas intentionally trigger these rules.
"""Evaluate linear stencils and implement their exact boundary transpose.

Padding, index folding, weight transposition, and the private staged stencil
transpose primitive live here.  Linear filter installers consume this module;
public SciPy signatures do not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from scipy import ndimage as _scipy_ndimage

from advect.core import ArraySpec, primitive
from advect.scipy._frontend import _is_traced_value
from advect.scipy._ndimage.common import (
    _mode_name,
    _operand_dtype,
    _require_numpy_values,
    _zero_tangent,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from advect.core import AbstractValue


type _PadWidth = tuple[tuple[int, int], ...]


def _numpy_pad_mode(
    mode: str,
) -> Literal["constant", "edge", "reflect", "symmetric", "wrap"]:
    return cast(
        'Literal["constant", "edge", "reflect", "symmetric", "wrap"]',
        {
            "constant": "constant",
            "mirror": "reflect",
            "nearest": "edge",
            "reflect": "symmetric",
            "wrap": "wrap",
        }[_mode_name(mode)],
    )


def _pad_width(
    ndim: int,
    axes: tuple[int, ...],
    extents: tuple[tuple[int, int], ...],
) -> _PadWidth:
    """Return the padding that lets reads at ``extents`` offsets stay in bounds."""
    pad = {
        axis: (max(0, -low), max(0, high)) for axis, (low, high) in zip(axes, extents, strict=True)
    }
    return tuple(pad.get(axis, (0, 0)) for axis in range(ndim))


def _pad_numpy(
    value: np.ndarray,
    pad_width: _PadWidth,
    *,
    axes: tuple[int, ...],
    modes: tuple[str, ...],
    cval: Any,
) -> np.ndarray:
    """Pad ``axes`` by their boundary modes; :func:`_fold_numpy` is the adjoint.

    Constant padding stores ``cval`` at ``value``'s dtype.
    """
    for axis, mode in zip(axes, modes, strict=True):
        widths = [(0, 0)] * value.ndim
        widths[axis] = pad_width[axis]
        if _mode_name(mode) == "constant":
            value = np.pad(value, widths, mode="constant", constant_values=cval)
        else:
            value = np.pad(value, widths, mode=_numpy_pad_mode(mode))
    return value


def _normalize_indices(
    raw: np.ndarray,
    *,
    length: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    normalized_mode = _mode_name(mode)
    if normalized_mode == "nearest":
        indices = np.clip(raw, 0, length - 1)
        valid = None
    elif normalized_mode == "wrap":
        indices = np.remainder(raw, length)
        valid = None
    elif normalized_mode == "reflect":
        folded = np.remainder(raw, 2 * length)
        indices = np.where(folded < length, folded, 2 * length - folded - 1)
        valid = None
    elif normalized_mode == "mirror":
        if length == 1:
            indices = np.zeros_like(raw)
        else:
            period = 2 * (length - 1)
            folded = np.remainder(raw, period)
            indices = np.where(folded < length, folded, period - folded)
        valid = None
    elif normalized_mode == "constant":
        valid = (raw >= 0) & (raw < length)
        indices = np.clip(raw, 0, length - 1)
    else:
        msg = f"boundary mode not supported: {mode!r}"
        raise RuntimeError(msg)
    return indices, valid


def _shift_axis(value: Any, *, axis: int, offset: int, mode: str, cval: Any) -> Any:
    length = value.shape[axis]
    if length == 0:
        return value
    indices, valid = _normalize_indices(np.arange(length) + offset, length=length, mode=mode)
    gathered = np.take(value, indices, axis=axis)
    if valid is None:
        return gathered
    mask_shape = [1] * value.ndim
    mask_shape[axis] = length
    mask = np.reshape(valid, tuple(mask_shape))
    return np.where(mask, gathered, np.zeros_like(gathered) + cval)


def _shift(
    value: Any,
    *,
    axes: tuple[int, ...],
    offsets: tuple[int, ...],
    modes: tuple[str, ...],
    cval: Any,
) -> Any:
    shifted = value
    for axis, offset, mode in zip(axes, offsets, modes, strict=True):
        shifted = _shift_axis(shifted, axis=axis, offset=offset, mode=mode, cval=cval)
    return shifted


def _correlate_stencil(
    input: Any,
    weights: Any,
    *,
    axes: tuple[int, ...],
    origins: tuple[int, ...],
    modes: tuple[str, ...],
    cval: Any,
    convolution: bool,
) -> Any:
    result: Any | None = None
    for _index, offsets, coefficient in _stencil_entries(
        weights,
        origins=origins,
        convolution=convolution,
    ):
        term = coefficient * _shift(
            input,
            axes=axes,
            offsets=offsets,
            modes=modes,
            cval=cval,
        )
        result = term if result is None else result + term
    if result is None:
        return np.zeros_like(input)
    return result


def _stencil_entries(
    weights: Any,
    *,
    origins: tuple[int, ...],
    convolution: bool,
) -> Iterator[tuple[tuple[int, ...], tuple[int, ...], Any]]:
    weight_shape = tuple(int(size) for size in weights.shape)
    centers = tuple(size // 2 for size in weight_shape)
    complex_weights = np.issubdtype(_operand_dtype(weights), np.complexfloating)
    # Convolution reads the kernel reflected through its center.
    sign = -1 if convolution else 1
    for index in np.ndindex(weight_shape):
        offsets = tuple(
            sign * (item - center - origin)
            for item, center, origin in zip(index, centers, origins, strict=True)
        )
        coefficient = weights[index]
        if complex_weights and not convolution:
            coefficient = np.conj(coefficient)
        yield index, offsets, coefficient


def _fold_axis_numpy(
    cotangent: np.ndarray,
    *,
    axis: int,
    length: int,
    before: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    moved = np.moveaxis(cotangent, axis, 0)
    after = moved.shape[0] - before - length
    result = np.array(moved[before : before + length], copy=True)
    boundary_cotangent = np.zeros((), dtype=cotangent.dtype)
    lower = moved[:before]
    upper = moved[before + length :]
    if _mode_name(mode) == "constant":
        boundary_cotangent = np.sum(lower) + np.sum(upper)
    else:
        if before:
            lower_indices, _valid = _normalize_indices(
                np.arange(-before, 0),
                length=length,
                mode=mode,
            )
            np.add.at(result, lower_indices, lower)
        if after:
            upper_indices, _valid = _normalize_indices(
                np.arange(length, length + after),
                length=length,
                mode=mode,
            )
            np.add.at(result, upper_indices, upper)
    return np.moveaxis(result, 0, axis), np.asarray(
        boundary_cotangent,
        dtype=cotangent.dtype,
    )


def _padded_transpose_numpy(
    cotangent: np.ndarray,
    stencil: Callable[[np.ndarray], np.ndarray],
    *,
    axes: tuple[int, ...],
    extents: tuple[tuple[int, int], ...],
    modes: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Transpose a boundary-mode stencil whose reads span ``extents`` per axis.

    ``stencil`` applies the zero-padded adjoint to the embedded cotangent. The
    padding is then folded back onto the samples each boundary mode read, and
    constant-mode padding becomes the boundary-value cotangent.
    """
    if 0 in cotangent.shape:
        return np.zeros_like(cotangent), np.zeros((), dtype=cotangent.dtype)
    pad_width = _pad_width(cotangent.ndim, axes, extents)
    return _fold_numpy(
        stencil(np.pad(cotangent, pad_width)),
        pad_width,
        axes=axes,
        modes=modes,
        shape=cotangent.shape,
    )


def _fold_numpy(
    padded: np.ndarray,
    pad_width: _PadWidth,
    *,
    axes: tuple[int, ...],
    modes: tuple[str, ...],
    shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Fold padding back onto the samples it read, and constant padding onto ``cval``."""
    boundary_cotangent = np.zeros((), dtype=padded.dtype)
    for axis, mode in reversed(tuple(zip(axes, modes, strict=True))):
        padded, contribution = _fold_axis_numpy(
            padded,
            axis=axis,
            length=shape[axis],
            before=pad_width[axis][0],
            mode=mode,
        )
        boundary_cotangent = boundary_cotangent + contribution
    return padded, boundary_cotangent


def _stencil_input_transpose_numpy(
    cotangent: np.ndarray,
    weights: np.ndarray,
    *,
    axes: tuple[int, ...],
    origins: tuple[int, ...],
    modes: tuple[str, ...],
    convolution: bool,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = [
        entry_offsets
        for _index, entry_offsets, _coefficient in _stencil_entries(
            weights,
            origins=origins,
            convolution=convolution,
        )
    ]
    function = _scipy_ndimage.correlate if convolution else _scipy_ndimage.convolve
    return _padded_transpose_numpy(
        cotangent,
        lambda embedded: function(
            embedded,
            weights,
            output=cotangent.dtype,
            mode="constant",
            cval=0,
            origin=origins,
            axes=axes,
        ),
        axes=axes,
        extents=tuple((min(items), max(items)) for items in zip(*offsets, strict=True)),
        modes=modes,
    )


def _stencil_weight_transpose(
    cotangent: Any,
    input: Any,
    weights: Any,
    cval: Any,
    *,
    axes: tuple[int, ...],
    origins: tuple[int, ...],
    modes: tuple[str, ...],
    convolution: bool,
) -> Any:
    contributions = []
    for _index, offsets, _coefficient in _stencil_entries(
        weights,
        origins=origins,
        convolution=convolution,
    ):
        candidate = _shift(
            input,
            axes=axes,
            offsets=offsets,
            modes=modes,
            cval=cval,
        )
        contribution = np.sum(np.conj(candidate) * cotangent)
        contributions.append(np.conj(contribution) if not convolution else contribution)
    if not contributions:
        return np.zeros_like(weights)
    return np.reshape(np.stack(contributions), weights.shape)


@primitive(
    name="scipy.ndimage._stencil_input_transpose",
    static_argnames=("axes", "origins", "modes", "convolution"),
)
def _stencil_input_transpose_primitive(
    cotangent: Any,
    weights: Any,
    *,
    axes: tuple[int, ...],
    origins: tuple[int, ...],
    modes: tuple[str, ...],
    convolution: bool,
) -> tuple[Any, Any]:
    _require_numpy_values("_stencil_input_transpose", cotangent, weights)
    return _stencil_input_transpose_numpy(
        cotangent,
        weights,
        axes=axes,
        origins=origins,
        modes=modes,
        convolution=convolution,
    )


@_stencil_input_transpose_primitive.def_abstract
def _stencil_input_transpose_abstract(
    cotangent: AbstractValue,
    weights: AbstractValue,
    **static: Any,
) -> tuple[ArraySpec, ArraySpec]:
    del weights, static
    return (
        ArraySpec(
            cotangent.spec.shape,
            cotangent.spec.dtype,
            device=cotangent.spec.device,
        ),
        ArraySpec((), cotangent.spec.dtype, device=cotangent.spec.device),
    )


@_stencil_input_transpose_primitive.def_jvp
def _stencil_input_transpose_jvp(
    output: Any,
    primals: tuple[Any, ...],
    tangents: tuple[Any | None, ...],
    **static: Any,
) -> tuple[Any, Any]:
    cotangent, weights = primals
    cotangent_tangent, weights_tangent = tangents
    result: tuple[Any, Any] | None = None
    if cotangent_tangent is not None:
        result = _stencil_input_transpose_primitive(cotangent_tangent, weights, **static)
    if weights_tangent is not None:
        weight_term = _stencil_input_transpose_primitive(cotangent, weights_tangent, **static)
        result = (
            weight_term
            if result is None
            else (result[0] + weight_term[0], result[1] + weight_term[1])
        )
    return (np.zeros_like(output[0]), np.zeros_like(output[1])) if result is None else result


@_stencil_input_transpose_primitive.def_transpose
def _stencil_input_transpose_transpose(
    output_cotangent: tuple[Any, Any],
    primals: tuple[Any, ...],
    output: Any,
    **static: Any,
) -> tuple[Any, Any]:
    del output
    cotangent, weights = primals
    # An output leaf that nothing read arrives as ``None``, a symbolic zero.
    input_cotangent = _zero_tangent(cotangent, output_cotangent[0])
    boundary_cotangent = 0 if output_cotangent[1] is None else output_cotangent[1]
    return (
        _correlate_stencil(input_cotangent, weights, cval=boundary_cotangent, **static),
        _stencil_weight_transpose(
            cotangent, input_cotangent, weights, boundary_cotangent, **static
        ),
    )


def _stencil_input_transpose(
    cotangent: Any,
    weights: Any,
    *,
    axes: tuple[int, ...],
    origins: tuple[int, ...],
    modes: tuple[str, ...],
    convolution: bool,
) -> tuple[Any, Any]:
    if not _is_traced_value(cotangent) and not _is_traced_value(weights):
        return _stencil_input_transpose_numpy(
            cotangent,
            weights,
            axes=axes,
            origins=origins,
            modes=modes,
            convolution=convolution,
        )
    return _stencil_input_transpose_primitive(
        cotangent,
        weights,
        axes=axes,
        origins=origins,
        modes=modes,
        convolution=convolution,
    )
