# ruff: noqa: A001, A002, ANN401, EM101, PLR0913, PLR2004, RUF059, S101, TRY003
# SciPy-compatible names/signatures and primitive rule schemas intentionally trigger these rules.
"""Primitive and derivative engines for :mod:`advect.scipy.ndimage`."""

from __future__ import annotations

import math
import operator
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from scipy import ndimage as _scipy_ndimage

from advect.core import ArraySpec, primitive
from advect.scipy._frontend import (
    _array_operand,
    _is_traced_value,
    _replace_out as _replace_traced_out,
    _require_numpy_values as _require_scipy_numpy_values,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from numpy.typing import DTypeLike

    from advect.core import AbstractValue
    from advect.core._primitive import Primitive


_MODE_ALIASES = {
    "grid-constant": "constant",
    "grid-mirror": "reflect",
    "grid-wrap": "wrap",
}

type _NeighborhoodEntry = tuple[int, tuple[int, ...], tuple[int, ...]]
type _PadWidth = tuple[tuple[int, int], ...]
type _NeighborhoodSlices = tuple[tuple[slice, ...], ...]
type _HomogeneousNeighborhoodData = tuple[np.ndarray, _PadWidth, str, _NeighborhoodSlices]
type _SelectionPullback = tuple[np.ndarray, np.ndarray, np.ndarray]


def _require_numpy_values(name: str, *values: object) -> None:
    _require_scipy_numpy_values("ndimage", name, *values)


def _numpy_dtype(dtype: object) -> np.dtype[Any]:
    try:
        return np.dtype(cast("DTypeLike", dtype))
    except (TypeError, ValueError) as error:
        msg = f"advect.scipy.ndimage requires a NumPy dtype; got {dtype!r}"
        raise TypeError(msg) from error


def _operand_dtype(value: Any) -> np.dtype[Any]:
    dtype = getattr(value, "dtype", None)
    return np.asarray(value).dtype if dtype is None else _numpy_dtype(dtype)


def _traceable_astype(value: Any, dtype: object) -> Any:
    normalized = _numpy_dtype(dtype)
    if _operand_dtype(value) == normalized:
        return value
    astype = getattr(value, "astype", None)
    if callable(astype):
        return astype(normalized)
    return np.asarray(value, dtype=normalized)


def _replace_out(destination: object, replacement: object, *, operation: str) -> object:
    return _replace_traced_out(
        destination,
        replacement,
        argument="output",
        operation=operation,
    )


def _normalize_axes(axes: object, ndim: int) -> tuple[int, ...]:
    if axes is None:
        return tuple(range(ndim))
    if np.isscalar(axes):
        raw = (operator.index(cast("Any", axes)),)
    elif isinstance(axes, Iterable):
        try:
            raw = tuple(operator.index(cast("Any", axis)) for axis in axes)
        except TypeError as error:
            msg = "axes must be an integer, iterable of integers, or None"
            raise ValueError(msg) from error
    else:
        msg = "axes must be an integer, iterable of integers, or None"
        raise ValueError(msg)
    normalized: list[int] = []
    for axis in raw:
        if axis < -ndim or axis >= ndim:
            msg = f"specified axis: {axis} is out of range"
            raise ValueError(msg)
        normalized.append(axis % ndim if axis < 0 else axis)
    if len(set(normalized)) != len(normalized):
        msg = "axes must be unique"
        raise ValueError(msg)
    return tuple(normalized)


def _normalize_axis(axis: object, ndim: int) -> int:
    normalized = operator.index(cast("Any", axis))
    if normalized < -ndim or normalized >= ndim:
        raise np.exceptions.AxisError(normalized, ndim=ndim)
    return normalized % ndim


def _static_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if _is_traced_value(value):
        msg = "ndimage configuration arguments must be concrete while tracing"
        raise TypeError(msg)
    return value


def _normalize_sequence(value: object, rank: int) -> tuple[object, ...]:
    if not isinstance(value, str) and isinstance(value, Iterable):
        normalized = tuple(_static_scalar(item) for item in value)
        if len(normalized) != rank:
            msg = "sequence argument must have length equal to input rank"
            raise RuntimeError(msg)
        return normalized
    scalar = _static_scalar(value)
    return (scalar,) * rank


def _normalize_modes(mode: object, rank: int) -> tuple[str, ...]:
    values = _normalize_sequence(mode, rank)
    return tuple(str(value) for value in values)


def _normalize_origins(origin: object, rank: int) -> tuple[int, ...]:
    return tuple(operator.index(cast("Any", item)) for item in _normalize_sequence(origin, rank))


def _shape_of(value: object) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return np.asarray(value).shape
    return tuple(int(size) for size in shape)


def _ndim_of(value: object) -> int:
    return len(_shape_of(value))


@dataclass(frozen=True, slots=True)
class _OutputChoice:
    destination: object | None
    dtype: str | None


def _normalize_output(input: object, output: object) -> _OutputChoice:
    if output is None:
        return _OutputChoice(None, None)
    if _is_traced_value(output) or isinstance(output, np.ndarray):
        _require_numpy_values("output", output)
        if _shape_of(output) != _shape_of(input):
            raise RuntimeError("output shape not correct")
        return _OutputChoice(output, _operand_dtype(output).str)
    return _OutputChoice(None, _numpy_dtype(output).str)


def _output_dtype(input: object, output: object) -> np.dtype[Any]:
    choice = _normalize_output(input, output)
    return _operand_dtype(input) if choice.dtype is None else _numpy_dtype(choice.dtype)


def _runtime_output(output_dtype: str | None) -> object:
    return None if output_dtype is None else _numpy_dtype(output_dtype)


def _sample_array(value: AbstractValue, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    sample_shape = value.spec.shape if shape is None else shape
    return np.ones(sample_shape, dtype=_numpy_dtype(value.spec.dtype))


def _result_spec(
    input: AbstractValue,
    result: object,
) -> ArraySpec:
    return ArraySpec(
        input.spec.shape,
        np.asarray(result).dtype.name,
        device=input.spec.device,
    )


def _finish_output(
    result: object,
    *,
    input: object,
    output: object,
    operation: str,
) -> object:
    choice = _normalize_output(input, output)
    dtype = _operand_dtype(input) if choice.dtype is None else _numpy_dtype(choice.dtype)
    replacement = _traceable_astype(result, dtype)
    if choice.destination is None:
        return replacement
    return _replace_out(choice.destination, replacement, operation=operation)


def _call_primitive(
    primitive_function: Primitive[..., Any],
    *,
    name: str,
    input: object,
    output: object,
    operands: dict[str, object],
    static: dict[str, object],
) -> object:
    choice = _normalize_output(input, output)
    replacement = primitive_function(
        input=_array_operand(input),
        output_dtype=choice.dtype,
        **{key: _array_operand(value) for key, value in operands.items()},
        **static,
    )
    if choice.destination is None:
        return replacement
    return _replace_out(
        choice.destination,
        replacement,
        operation=f"scipy.ndimage.{name} output=",
    )


def _mode_name(mode: str) -> str:
    return _MODE_ALIASES.get(mode, mode)


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


def _pad_numpy(
    input: np.ndarray,
    pad_width: _PadWidth,
    *,
    mode: str,
    cval: np.ndarray,
) -> np.ndarray:
    if _mode_name(mode) == "constant":
        return np.pad(input, pad_width, mode="constant", constant_values=cval)
    return np.pad(input, pad_width, mode=_numpy_pad_mode(mode))


def _axis_indices(
    length: int,
    *,
    offset: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    raw = np.arange(length) + offset
    return _normalize_indices(raw, length=length, mode=mode)


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
    indices, valid = _axis_indices(length, offset=offset, mode=mode)
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
    for index in np.ndindex(weight_shape):
        if convolution:
            offsets = tuple(
                center - item + origin
                for item, center, origin in zip(index, centers, origins, strict=True)
            )
        else:
            offsets = tuple(
                item - center - origin
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


def _stencil_input_transpose_numpy(
    cotangent: np.ndarray,
    weights: np.ndarray,
    *,
    axes: tuple[int, ...],
    origins: tuple[int, ...],
    modes: tuple[str, ...],
    convolution: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if 0 in cotangent.shape:
        return np.zeros_like(cotangent), np.zeros((), dtype=cotangent.dtype)
    offsets = [
        entry_offsets
        for _index, entry_offsets, _coefficient in _stencil_entries(
            weights,
            origins=origins,
            convolution=convolution,
        )
    ]
    before = tuple(max(0, -min(items)) for items in zip(*offsets, strict=True))
    after = tuple(max(0, *items) for items in zip(*offsets, strict=True))
    pad_by_axis = dict(zip(axes, zip(before, after, strict=True), strict=True))
    pad_width = tuple(pad_by_axis.get(axis, (0, 0)) for axis in range(cotangent.ndim))
    padded_shape = tuple(
        size + lower + upper
        for size, (lower, upper) in zip(cotangent.shape, pad_width, strict=True)
    )
    embedded = np.zeros(padded_shape, dtype=cotangent.dtype)
    center = tuple(
        slice(lower, lower + size)
        for size, (lower, _upper) in zip(cotangent.shape, pad_width, strict=True)
    )
    embedded[center] = cotangent
    function = _scipy_ndimage.correlate if convolution else _scipy_ndimage.convolve
    result = function(
        embedded,
        weights,
        output=cotangent.dtype,
        mode="constant",
        cval=0,
        origin=origins,
        axes=axes,
    )
    boundary_cotangent = np.zeros((), dtype=cotangent.dtype)
    for axis, lower, mode in reversed(tuple(zip(axes, before, modes, strict=True))):
        result, contribution = _fold_axis_numpy(
            result,
            axis=axis,
            length=cotangent.shape[axis],
            before=lower,
            mode=mode,
        )
        boundary_cotangent = boundary_cotangent + contribution
    return result, boundary_cotangent


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


def _project_cotangent(value: Any, primal: Any, output: Any) -> Any:
    primal_dtype = _operand_dtype(primal)
    output_dtype = _operand_dtype(output)
    if not np.issubdtype(primal_dtype, np.inexact) or not np.issubdtype(
        output_dtype,
        np.inexact,
    ):
        return np.zeros_like(primal)
    if not np.issubdtype(primal_dtype, np.complexfloating) and np.issubdtype(
        _operand_dtype(value),
        np.complexfloating,
    ):
        value = np.real(value)
    return _traceable_astype(value, primal_dtype)


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
    *,
    axes: tuple[int, ...],
    origins: tuple[int, ...],
    modes: tuple[str, ...],
    convolution: bool,
) -> tuple[ArraySpec, ArraySpec]:
    del weights, axes, origins, modes, convolution
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
    *,
    axes: tuple[int, ...],
    origins: tuple[int, ...],
    modes: tuple[str, ...],
    convolution: bool,
) -> tuple[Any, Any]:
    cotangent, weights = primals
    cotangent_tangent, weights_tangent = tangents
    result: tuple[Any, Any] | None = None
    if cotangent_tangent is not None:
        result = _stencil_input_transpose_primitive(
            cotangent_tangent,
            weights,
            axes=axes,
            origins=origins,
            modes=modes,
            convolution=convolution,
        )
    if weights_tangent is not None:
        weight_term = _stencil_input_transpose_primitive(
            cotangent,
            weights_tangent,
            axes=axes,
            origins=origins,
            modes=modes,
            convolution=convolution,
        )
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
    *,
    axes: tuple[int, ...],
    origins: tuple[int, ...],
    modes: tuple[str, ...],
    convolution: bool,
) -> tuple[Any, Any]:
    del output
    cotangent, weights = primals
    input_cotangent, boundary_cotangent = output_cotangent
    return (
        _correlate_stencil(
            input_cotangent,
            weights,
            axes=axes,
            origins=origins,
            modes=modes,
            cval=boundary_cotangent,
            convolution=convolution,
        ),
        _stencil_weight_transpose(
            cotangent,
            input_cotangent,
            weights,
            boundary_cotangent,
            axes=axes,
            origins=origins,
            modes=modes,
            convolution=convolution,
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


def _gaussian_kernel(sigma: float, order: int, radius: int) -> np.ndarray:
    if order < 0:
        raise ValueError("order must be non-negative")
    exponents = np.arange(order + 1)
    sigma_squared = sigma * sigma
    positions = np.arange(-radius, radius + 1)
    gaussian = np.exp(-0.5 / sigma_squared * positions**2)
    gaussian = gaussian / gaussian.sum()
    if order == 0:
        return gaussian
    coefficients = np.zeros(order + 1)
    coefficients[0] = 1
    derivative = np.diag(exponents[1:], 1)
    product = np.diag(np.ones(order) / -sigma_squared, -1)
    for _ in range(order):
        coefficients = (derivative + product).dot(coefficients)
    polynomial = (positions[:, None] ** exponents).dot(coefficients)
    return polynomial * gaussian


def _zero_tangent(primal: Any, tangent: Any | None) -> Any:
    return np.zeros_like(primal) if tangent is None else tangent


def _cast_tangent(tangent: Any, output: Any) -> Any:
    dtype = _operand_dtype(output)
    if not np.issubdtype(dtype, np.inexact):
        return np.zeros_like(output)
    return _traceable_astype(tangent, dtype)


def _validate_ufunc_output_cast(
    ufunc: np.ufunc,
    left: object,
    right: object,
    output_dtype: object,
) -> None:
    """Apply NumPy's own in-place casting check used by SciPy composites."""
    ufunc(
        np.zeros((), dtype=_operand_dtype(left)),
        np.zeros((), dtype=_operand_dtype(right)),
        out=np.zeros((), dtype=_numpy_dtype(output_dtype)),
    )


@dataclass(frozen=True, slots=True)
class _Neighborhood:
    axes: tuple[int, ...]
    shape: tuple[int, ...]
    footprint: tuple[bool, ...]
    origins: tuple[int, ...]
    modes: tuple[str, ...]


def _static_footprint(value: object) -> np.ndarray:
    if _is_traced_value(value):
        msg = "footprint is non-differentiable configuration and must be concrete"
        raise TypeError(msg)
    return np.asarray(value, dtype=bool)


def _neighborhood(
    input: object,
    *,
    size: object,
    footprint: object,
    structure: object,
    origin: object,
    mode: object,
    axes: object,
    rank_filter: bool,
) -> _Neighborhood:
    input_ndim = _ndim_of(input)
    input_axes = _normalize_axes(axes, input_ndim)
    origins_sequence = _normalize_origins(origin, len(input_axes))
    modes_sequence = _normalize_modes(mode, len(input_axes))
    origins_by_axis = dict(zip(input_axes, origins_sequence, strict=True))
    modes_by_axis = dict(zip(input_axes, modes_sequence, strict=True))

    explicit_footprint: np.ndarray | None = None
    if footprint is not None:
        explicit_footprint = _static_footprint(footprint)

    has_structure = structure is not None
    if explicit_footprint is not None:
        shape = explicit_footprint.shape
        flat = tuple(bool(item) for item in explicit_footprint.flat)
        separable = not rank_filter and not has_structure and bool(explicit_footprint.all())
    elif has_structure:
        shape = _shape_of(structure)
        flat = (True,) * math.prod(shape)
        separable = False
    else:
        if size is None:
            msg = "no footprint or filter size provided"
            raise RuntimeError(msg)
        sizes = tuple(
            operator.index(cast("Any", item)) for item in _normalize_sequence(size, len(input_axes))
        )
        if rank_filter:
            shape = sizes
            flat = (True,) * math.prod(shape)
            separable = False
        else:
            active = tuple(
                (axis, filter_size)
                for axis, filter_size in zip(input_axes, sizes, strict=True)
                if filter_size > 1
            )
            if not active:
                return _Neighborhood((), (), (True,), (), ())
            input_axes = tuple(axis for axis, _ in active)
            shape = tuple(filter_size for _, filter_size in active)
            flat = (True,) * math.prod(shape)
            separable = True

    kernel_axes = input_axes if separable else tuple(sorted(input_axes))
    if len(shape) != len(kernel_axes):
        msg = f"footprint.ndim ({len(shape)}) must match len(axes) ({len(input_axes)})"
        raise RuntimeError(msg)
    if separable or len(input_axes) < input_ndim:
        kernel_origins = tuple(origins_by_axis[axis] for axis in kernel_axes)
        kernel_modes = tuple(modes_by_axis[axis] for axis in kernel_axes)
    else:
        # SciPy leaves full-rank origin and mode sequences in physical kernel
        # order, even when ``axes`` is an unsorted permutation.
        kernel_origins = origins_sequence
        kernel_modes = modes_sequence
    return _Neighborhood(
        axes=kernel_axes,
        shape=shape,
        footprint=flat,
        origins=kernel_origins,
        modes=kernel_modes,
    )


def _neighborhood_entries(
    neighborhood: _Neighborhood,
    *,
    dilation: bool,
) -> Iterator[_NeighborhoodEntry]:
    if not neighborhood.axes:
        yield 0, (), ()
        return
    centers = tuple(size // 2 for size in neighborhood.shape)
    for flat_index, index in enumerate(np.ndindex(neighborhood.shape)):
        if not neighborhood.footprint[flat_index]:
            continue
        if dilation:
            offsets = tuple(
                center - item + origin
                for item, center, origin in zip(
                    index,
                    centers,
                    neighborhood.origins,
                    strict=True,
                )
            )
        else:
            offsets = tuple(
                item - center - origin
                for item, center, origin in zip(
                    index,
                    centers,
                    neighborhood.origins,
                    strict=True,
                )
            )
        yield flat_index, index, offsets


def _neighborhood_candidates(
    input: Any,
    input_tangent: Any,
    structure: Any,
    structure_tangent: Any | None,
    cval: Any,
    cval_tangent: Any,
    *,
    neighborhood: _Neighborhood,
    dilation: bool,
    has_structure: bool,
) -> Iterator[tuple[Any, Any]]:
    structure_delta = None if structure_tangent is None else structure_tangent
    for _flat_index, index, offsets in _neighborhood_entries(
        neighborhood,
        dilation=dilation,
    ):
        candidate = _shift(
            input,
            axes=neighborhood.axes,
            offsets=offsets,
            modes=neighborhood.modes,
            cval=cval,
        )
        candidate_tangent = _shift(
            input_tangent,
            axes=neighborhood.axes,
            offsets=offsets,
            modes=neighborhood.modes,
            cval=cval_tangent,
        )
        if has_structure:
            delta = structure[index]
            tangent_delta = 0 if structure_delta is None else structure_delta[index]
            if dilation:
                candidate = candidate + delta
                candidate_tangent = candidate_tangent + tangent_delta
            else:
                candidate = candidate - delta
                candidate_tangent = candidate_tangent - tangent_delta
        yield candidate, candidate_tangent


def _selection_jvp(
    output: Any,
    input: Any,
    input_tangent: Any | None,
    structure: Any,
    structure_tangent: Any | None,
    cval: Any,
    cval_tangent: Any | None,
    *,
    neighborhood: _Neighborhood,
    selection: str,
    dilation: bool,
    rank: int | None,
    has_structure: bool,
) -> Any:
    if (
        has_structure
        and rank is None
        and not any(
            _is_traced_value(value)
            for value in (
                output,
                input,
                input_tangent,
                structure,
                structure_tangent,
                cval,
                cval_tangent,
            )
        )
    ):
        result = _homogeneous_selection_jvp_numpy(
            output,
            input,
            input_tangent,
            structure,
            structure_tangent,
            cval,
            cval_tangent,
            neighborhood=neighborhood,
            dilation=dilation,
        )
        if result is not None:
            return result
    tangent_input = _zero_tangent(input, input_tangent)
    tangent_cval = 0 if cval_tangent is None else cval_tangent
    if (
        not has_structure
        and rank is None
        and len(neighborhood.axes) == 1
        and sum(neighborhood.footprint) <= 9
        and not any(
            _is_traced_value(value) for value in (output, input, tangent_input, cval, tangent_cval)
        )
    ):
        entries = tuple(_neighborhood_entries(neighborhood, dilation=dilation))
        chosen = _small_axis_selection_sources_numpy(
            input,
            cval,
            output,
            entries=entries,
            neighborhood=neighborhood,
        )
        if chosen is not None:
            axis = neighborhood.axes[0]
            gathered = np.take_along_axis(
                tangent_input,
                np.clip(chosen, 0, input.shape[axis] - 1),
                axis=axis,
            )
            return _cast_tangent(np.where(chosen >= 0, gathered, tangent_cval), output)
    if not has_structure and not any(
        _is_traced_value(value) for value in (output, input, tangent_input, cval, tangent_cval)
    ):
        plateau_result = _plateau_selection_jvp_numpy(
            output,
            input,
            tangent_input,
            cval,
            tangent_cval,
            neighborhood=neighborhood,
            dilation=dilation,
        )
        if plateau_result is not None:
            return plateau_result
        chosen = _unique_value_selection_sources_numpy(input, cval, output)
        if chosen is not None:
            selected = tangent_input.ravel()[np.clip(chosen, 0, input.size - 1)]
            return _cast_tangent(np.where(chosen >= 0, selected, tangent_cval), output)
        small_result = _small_flat_selection_jvp_numpy(
            output,
            input,
            tangent_input,
            cval,
            tangent_cval,
            entries=tuple(_neighborhood_entries(neighborhood, dilation=dilation)),
            neighborhood=neighborhood,
        )
        if small_result is not None:
            return small_result
    candidates = list(
        _neighborhood_candidates(
            input,
            tangent_input,
            structure,
            structure_tangent,
            cval,
            tangent_cval,
            neighborhood=neighborhood,
            dilation=dilation,
            has_structure=has_structure,
        )
    )
    if not candidates:
        return np.zeros_like(output)
    values = [candidate for candidate, _ in candidates]
    if rank is None:
        selected = values[0]
        reducer = np.maximum if selection == "maximum" else np.minimum
        for candidate in values[1:]:
            selected = reducer(selected, candidate)
    else:
        selected = np.sort(np.stack(values, axis=0), axis=0)[rank]
    tangent_sum = np.zeros_like(output)
    winner_count = np.zeros_like(output)
    for candidate, candidate_tangent in candidates:
        winner = candidate == selected
        tangent_sum = tangent_sum + np.where(winner, candidate_tangent, np.zeros_like(output))
        winner_count = winner_count + np.where(winner, np.ones_like(output), np.zeros_like(output))
    return _cast_tangent(tangent_sum / winner_count, output)


def _neighborhood_destinations_numpy(
    input_shape: tuple[int, ...],
    entries: tuple[_NeighborhoodEntry, ...],
    neighborhood: _Neighborhood,
) -> np.ndarray:
    source_indices = np.arange(math.prod(input_shape), dtype=np.intp).reshape(input_shape)
    destinations = np.broadcast_to(source_indices, (len(entries), *input_shape)).copy()
    valid_destinations: np.ndarray | None = None
    for offset_axis, (axis, mode) in enumerate(
        zip(neighborhood.axes, neighborhood.modes, strict=True)
    ):
        length = input_shape[axis]
        positions = np.arange(length)
        offsets = np.asarray([entry[2][offset_axis] for entry in entries])
        indices, valid = _normalize_indices(
            positions[np.newaxis, :] + offsets[:, np.newaxis],
            length=length,
            mode=mode,
        )
        broadcast_shape = [1] * (len(input_shape) + 1)
        broadcast_shape[0] = len(entries)
        broadcast_shape[axis + 1] = length
        adjustment = (indices - positions) * math.prod(input_shape[axis + 1 :])
        destinations += adjustment.reshape(broadcast_shape)
        if valid is not None:
            broadcast_valid = valid.reshape(broadcast_shape)
            valid_destinations = (
                broadcast_valid
                if valid_destinations is None
                else valid_destinations & broadcast_valid
            )
    if valid_destinations is not None:
        destinations = np.where(valid_destinations, destinations, -1)
    return destinations


def _neighborhood_destination_numpy(
    source_indices: np.ndarray,
    neighborhood: _Neighborhood,
    offsets: tuple[int, ...],
) -> np.ndarray:
    destination = np.array(source_indices, copy=True)
    valid_destination: np.ndarray | None = None
    for axis, offset, mode in zip(
        neighborhood.axes,
        offsets,
        neighborhood.modes,
        strict=True,
    ):
        length = source_indices.shape[axis]
        positions = np.arange(length)
        indices, valid = _normalize_indices(
            positions + offset,
            length=length,
            mode=mode,
        )
        broadcast_shape = [1] * source_indices.ndim
        broadcast_shape[axis] = length
        adjustment = (indices - positions) * math.prod(source_indices.shape[axis + 1 :])
        destination += adjustment.reshape(broadcast_shape)
        if valid is not None:
            broadcast_valid = valid.reshape(broadcast_shape)
            valid_destination = (
                broadcast_valid
                if valid_destination is None
                else valid_destination & broadcast_valid
            )
    if valid_destination is not None:
        destination = np.where(valid_destination, destination, -1)
    return destination


def _scatter_numpy(
    destinations: np.ndarray,
    weights: np.ndarray,
    *,
    size: int,
    dtype: np.dtype[Any],
) -> np.ndarray:
    if np.iscomplexobj(weights):
        scattered = np.bincount(
            destinations,
            weights=np.real(weights),
            minlength=size,
        ) + 1j * np.bincount(
            destinations,
            weights=np.imag(weights),
            minlength=size,
        )
    else:
        scattered = np.bincount(destinations, weights=weights, minlength=size)
    return np.asarray(scattered, dtype=dtype)


def _unique_value_selection_sources_numpy(
    input: np.ndarray,
    cval: np.ndarray,
    output: np.ndarray,
) -> np.ndarray | None:
    if input.dtype != output.dtype or np.any(np.isnan(input)) or np.any(np.isnan(output)):
        return None
    flat_input = input.ravel()
    sample = flat_input[:: max(1, flat_input.size // 128)]
    if np.unique(sample).size != sample.size:
        return None
    unique_values, inverse = np.unique(input, return_inverse=True)
    if unique_values.size != input.size:
        return None
    boundary_value = np.asarray(cval, dtype=output.dtype)
    if np.any(unique_values == boundary_value):
        return None
    flat_output = output.ravel()
    ranks = np.searchsorted(unique_values, flat_output)
    bounded_ranks = np.minimum(ranks, unique_values.size - 1)
    from_input = (ranks < unique_values.size) & (unique_values[bounded_ranks] == flat_output)
    if np.any(~from_input & (flat_output != boundary_value)):
        return None
    source_by_rank = np.empty(input.size, dtype=np.intp)
    source_by_rank[inverse.ravel()] = np.arange(input.size)
    chosen = np.full(input.size, -1, dtype=np.intp)
    chosen[from_input] = source_by_rank[ranks[from_input]]
    return chosen.reshape(output.shape)


def _unique_value_selection_transpose_numpy(
    cotangent: np.ndarray,
    input: np.ndarray,
    structure: np.ndarray,
    cval: np.ndarray,
    output: np.ndarray,
) -> _SelectionPullback | None:
    chosen = _unique_value_selection_sources_numpy(input, cval, output)
    if chosen is None:
        return None
    from_input = chosen.ravel() >= 0
    flat_cotangent = cotangent.ravel()
    scattered = _scatter_numpy(
        chosen.ravel()[from_input],
        flat_cotangent[from_input],
        size=input.size,
        dtype=cotangent.dtype,
    )
    return (
        _project_cotangent(scattered.reshape(input.shape), input, output),
        np.zeros_like(structure),
        _project_cotangent(np.sum(flat_cotangent[~from_input]), cval, output),
    )


def _small_axis_selection_sources_numpy(
    input: np.ndarray,
    cval: np.ndarray,
    output: np.ndarray,
    *,
    entries: tuple[_NeighborhoodEntry, ...],
    neighborhood: _Neighborhood,
) -> np.ndarray | None:
    (axis,) = neighborhood.axes
    (mode,) = neighborhood.modes
    length = input.shape[axis]
    offsets = tuple(entry[2][0] for entry in entries)
    before = max(0, -min(offsets))
    after = max(0, *offsets)
    pad_width = [(0, 0)] * input.ndim
    pad_width[axis] = (before, after)
    padded = _pad_numpy(
        input,
        tuple(pad_width),
        mode=mode,
        cval=cval,
    )
    broadcast_shape = [1] * input.ndim
    broadcast_shape[axis] = length
    winners = []
    sources = []
    for offset in offsets:
        indices, valid = _axis_indices(length, offset=offset, mode=mode)
        source = indices
        if valid is not None:
            source = np.where(valid, indices, -1)
        slices = [slice(None)] * input.ndim
        start = before + offset
        slices[axis] = slice(start, start + length)
        candidate = padded[tuple(slices)]
        if candidate.dtype != output.dtype:
            return None
        winners.append(candidate == output)
        sources.append(source.reshape(broadcast_shape))
    chosen = np.broadcast_to(sources[-1], input.shape)
    for winner, source in reversed(tuple(zip(winners[:-1], sources[:-1], strict=True))):
        chosen = np.where(winner, source, chosen)
    matched = np.array(winners[0], copy=True)
    conflict = winners[0] & (sources[0] != chosen)
    for winner, source in zip(winners[1:], sources[1:], strict=True):
        matched |= winner
        conflict |= winner & (source != chosen)
    if np.any(conflict) or np.any(~matched):
        return None
    return chosen


def _small_axis_selection_transpose_numpy(
    cotangent: np.ndarray,
    input: np.ndarray,
    structure: np.ndarray,
    cval: np.ndarray,
    output: np.ndarray,
    *,
    entries: tuple[_NeighborhoodEntry, ...],
    neighborhood: _Neighborhood,
) -> _SelectionPullback | None:
    chosen = _small_axis_selection_sources_numpy(
        input,
        cval,
        output,
        entries=entries,
        neighborhood=neighborhood,
    )
    if chosen is None:
        return None
    (axis,) = neighborhood.axes
    length = input.shape[axis]
    moved_chosen = np.moveaxis(chosen, axis, -1).reshape(-1, length)
    moved_cotangent = np.moveaxis(cotangent, axis, -1).reshape(-1, length)
    valid = moved_chosen >= 0
    destinations = np.arange(moved_chosen.shape[0], dtype=np.intp)[:, np.newaxis] * length
    destinations = destinations + moved_chosen
    scattered = _scatter_numpy(
        destinations[valid],
        moved_cotangent[valid],
        size=input.size,
        dtype=cotangent.dtype,
    )
    moved_shape = np.moveaxis(input, axis, -1).shape
    input_cotangent = np.moveaxis(scattered.reshape(moved_shape), -1, axis)
    return (
        _project_cotangent(input_cotangent, input, output),
        np.zeros_like(structure),
        _project_cotangent(np.sum(moved_cotangent[~valid]), cval, output),
    )


def _homogeneous_neighborhood_data_numpy(
    input: np.ndarray,
    cval: np.ndarray,
    *,
    entries: tuple[_NeighborhoodEntry, ...],
    neighborhood: _Neighborhood,
) -> _HomogeneousNeighborhoodData | None:
    normalized_modes = {_mode_name(mode) for mode in neighborhood.modes}
    if len(normalized_modes) != 1:
        return None
    offsets = tuple(entry[2] for entry in entries)
    before = tuple(max(0, -min(items)) for items in zip(*offsets, strict=True))
    after = tuple(max(0, *items) for items in zip(*offsets, strict=True))
    padding = dict(zip(neighborhood.axes, zip(before, after, strict=True), strict=True))
    pad_width = tuple(padding.get(axis, (0, 0)) for axis in range(input.ndim))
    mode = neighborhood.modes[0]
    padded = _pad_numpy(input, pad_width, mode=mode, cval=cval)
    slices = []
    for _flat_index, _index, entry_offsets in entries:
        entry_slices = [slice(None)] * input.ndim
        for axis, lower, offset in zip(
            neighborhood.axes,
            before,
            entry_offsets,
            strict=True,
        ):
            start = lower + offset
            entry_slices[axis] = slice(start, start + input.shape[axis])
        slices.append(tuple(entry_slices))
    return padded, pad_width, mode, tuple(slices)


def _small_flat_selection_data_numpy(
    input: np.ndarray,
    cval: np.ndarray,
    output: np.ndarray,
    *,
    entries: tuple[_NeighborhoodEntry, ...],
    neighborhood: _Neighborhood,
) -> _HomogeneousNeighborhoodData | None:
    if len(entries) > 32 or input.dtype != output.dtype:
        return None
    data = _homogeneous_neighborhood_data_numpy(
        input,
        cval,
        entries=entries,
        neighborhood=neighborhood,
    )
    if data is None:
        return None
    padded, pad_width, mode, slices = data
    winners = np.empty((len(entries), *input.shape), dtype=bool)
    for position, entry_slices in enumerate(slices):
        np.equal(padded[entry_slices], output, out=winners[position])
    return winners, pad_width, mode, slices


def _small_flat_selection_jvp_numpy(
    output: np.ndarray,
    input: np.ndarray,
    input_tangent: np.ndarray,
    cval: np.ndarray,
    cval_tangent: Any,
    *,
    entries: tuple[_NeighborhoodEntry, ...],
    neighborhood: _Neighborhood,
) -> np.ndarray | None:
    data = _small_flat_selection_data_numpy(
        input,
        cval,
        output,
        entries=entries,
        neighborhood=neighborhood,
    )
    if data is None:
        return None
    winners, pad_width, mode, slices = data
    padded_tangent = _pad_numpy(input_tangent, pad_width, mode=mode, cval=cval_tangent)
    tangent_sum = np.zeros_like(input_tangent)
    for winner, entry_slices in zip(winners, slices, strict=True):
        np.add(tangent_sum, padded_tangent[entry_slices], out=tangent_sum, where=winner)
    return _cast_tangent(tangent_sum / np.sum(winners, axis=0), output)


def _small_flat_selection_transpose_numpy(
    cotangent: np.ndarray,
    input: np.ndarray,
    structure: np.ndarray,
    cval: np.ndarray,
    output: np.ndarray,
    *,
    entries: tuple[_NeighborhoodEntry, ...],
    neighborhood: _Neighborhood,
    active_inputs: set[int],
) -> _SelectionPullback | None:
    data = _small_flat_selection_data_numpy(
        input,
        cval,
        output,
        entries=entries,
        neighborhood=neighborhood,
    )
    if data is None:
        return None
    winners, _pad_width, _mode, _slices = data
    active = cotangent / np.sum(winners, axis=0)
    destinations = _neighborhood_destinations_numpy(input.shape, entries, neighborhood)
    valid = destinations >= 0
    live = valid & winners
    weights = np.broadcast_to(active, winners.shape)
    input_cotangent = (
        _scatter_numpy(
            destinations[live],
            weights[live],
            size=input.size,
            dtype=cotangent.dtype,
        ).reshape(input.shape)
        if 0 in active_inputs
        else np.zeros_like(input)
    )
    boundary_cotangent = (
        np.sum(weights[~valid & winners], dtype=cotangent.dtype)
        if 2 in active_inputs
        else np.zeros_like(cval)
    )
    return (
        _project_cotangent(input_cotangent, input, output),
        np.zeros_like(structure),
        _project_cotangent(boundary_cotangent, cval, output),
    )


def _homogeneous_selection_entry_numpy(
    input: np.ndarray,
    structure: np.ndarray,
    cval: np.ndarray,
    output: np.ndarray,
    *,
    entries: tuple[_NeighborhoodEntry, ...],
    neighborhood: _Neighborhood,
    dilation: bool,
) -> np.ndarray | None:
    if not neighborhood.axes:
        return None
    data = _homogeneous_neighborhood_data_numpy(
        input,
        cval,
        entries=entries,
        neighborhood=neighborhood,
    )
    if data is None:
        return None
    padded, _pad_width, _mode, slices = data
    index_dtype = np.min_scalar_type(len(entries))
    chosen = np.zeros(input.shape, dtype=index_dtype)
    winner_count = np.zeros(input.shape, dtype=index_dtype)
    if np.result_type(padded.dtype, structure.dtype) != output.dtype:
        return None
    candidate = np.empty_like(output)
    winner = np.empty(input.shape, dtype=bool)
    operation = np.add if dilation else np.subtract
    for position, ((_flat_index, index, _entry_offsets), entry_slices) in enumerate(
        zip(entries, slices, strict=True)
    ):
        operation(padded[entry_slices], structure[index], out=candidate)
        np.equal(candidate, output, out=winner)
        np.putmask(chosen, winner, position)
        winner_count += winner
    if np.any(winner_count != 1):
        return None
    return chosen


def _chosen_entry_destinations_numpy(
    chosen: np.ndarray,
    entries: tuple[_NeighborhoodEntry, ...],
    neighborhood: _Neighborhood,
) -> np.ndarray:
    if neighborhood.axes == tuple(range(chosen.ndim)) and all(
        _mode_name(mode) == "constant" for mode in neighborhood.modes
    ):
        offsets = np.asarray([entry[2] for entry in entries], dtype=np.intp)
        destination = np.zeros(chosen.shape, dtype=np.intp)
        valid = np.ones(chosen.shape, dtype=bool)
        for offset_axis, axis in enumerate(neighborhood.axes):
            position_shape = [1] * chosen.ndim
            position_shape[axis] = chosen.shape[axis]
            source = np.arange(chosen.shape[axis]).reshape(position_shape)
            source = source + offsets[:, offset_axis][chosen]
            valid &= (source >= 0) & (source < chosen.shape[axis])
            destination += source * math.prod(chosen.shape[axis + 1 :])
        np.putmask(destination, ~valid, -1)
        return destination
    destination = np.arange(chosen.size, dtype=np.intp).reshape(chosen.shape)
    valid_destination: np.ndarray | None = None
    for offset_axis, (axis, mode) in enumerate(
        zip(neighborhood.axes, neighborhood.modes, strict=True)
    ):
        length = chosen.shape[axis]
        positions = np.arange(length)
        offsets = np.asarray([entry[2][offset_axis] for entry in entries])
        indices, valid = _normalize_indices(
            positions[np.newaxis, :] + offsets[:, np.newaxis],
            length=length,
            mode=mode,
        )
        if valid is not None:
            indices = np.where(valid, indices, -1)
        broadcast_shape = [len(entries), *((1,) * chosen.ndim)]
        broadcast_shape[axis + 1] = length
        chosen_index = chosen[np.newaxis, ...]
        source = np.take_along_axis(
            indices.reshape(broadcast_shape),
            chosen_index,
            axis=0,
        )[0]
        position_shape = [1] * chosen.ndim
        position_shape[axis] = length
        destination += (source - positions.reshape(position_shape)) * math.prod(
            chosen.shape[axis + 1 :]
        )
        if valid is not None:
            source_valid = source >= 0
            valid_destination = (
                source_valid if valid_destination is None else valid_destination & source_valid
            )
    if valid_destination is not None:
        destination = np.where(valid_destination, destination, -1)
    return destination


def _homogeneous_selection_jvp_numpy(
    output: np.ndarray,
    input: np.ndarray,
    input_tangent: np.ndarray | None,
    structure: np.ndarray,
    structure_tangent: np.ndarray | None,
    cval: np.ndarray,
    cval_tangent: np.ndarray | None,
    *,
    neighborhood: _Neighborhood,
    dilation: bool,
) -> np.ndarray | None:
    entries = tuple(_neighborhood_entries(neighborhood, dilation=dilation))
    chosen_entry = _homogeneous_selection_entry_numpy(
        input,
        structure,
        cval,
        output,
        entries=entries,
        neighborhood=neighborhood,
        dilation=dilation,
    )
    if chosen_entry is None:
        return None
    if input_tangent is None and structure_tangent is None and cval_tangent is None:
        return np.zeros_like(output)
    result: Any = 0
    if input_tangent is not None or cval_tangent is not None:
        chosen_source = _chosen_entry_destinations_numpy(
            chosen_entry,
            entries,
            neighborhood,
        )
        valid_source = chosen_source >= 0
        tangent_input = _zero_tangent(input, input_tangent)
        selected_input = tangent_input.ravel()[np.clip(chosen_source, 0, input.size - 1)]
        result = np.where(
            valid_source,
            selected_input,
            0 if cval_tangent is None else cval_tangent,
        )
    if structure_tangent is not None:
        entry_indices = np.asarray([entry[0] for entry in entries], dtype=np.intp)
        selected_structure = structure_tangent.ravel()[entry_indices[chosen_entry]]
        result = result + selected_structure if dilation else result - selected_structure
    return _cast_tangent(result, output)


def _unique_candidate_selection_transpose_numpy(
    cotangent: np.ndarray,
    input: np.ndarray,
    structure: np.ndarray,
    cval: np.ndarray,
    output: np.ndarray,
    *,
    neighborhood: _Neighborhood,
    dilation: bool,
    active_inputs: set[int],
) -> _SelectionPullback | None:
    entries = tuple(_neighborhood_entries(neighborhood, dilation=dilation))
    chosen_entry = _homogeneous_selection_entry_numpy(
        input,
        structure,
        cval,
        output,
        entries=entries,
        neighborhood=neighborhood,
        dilation=dilation,
    )
    if chosen_entry is not None:
        flat_cotangent = cotangent.ravel()
        needs_source = bool(active_inputs & {0, 2})
        chosen_source = (
            _chosen_entry_destinations_numpy(
                chosen_entry,
                entries,
                neighborhood,
            )
            if needs_source
            else None
        )
        valid_source = None if chosen_source is None else chosen_source.ravel() >= 0
        input_cotangent = (
            _scatter_numpy(
                cast("np.ndarray", chosen_source).ravel()[cast("np.ndarray", valid_source)],
                flat_cotangent[cast("np.ndarray", valid_source)],
                size=input.size,
                dtype=cotangent.dtype,
            ).reshape(input.shape)
            if 0 in active_inputs
            else np.zeros_like(input)
        )
        entry_indices = np.asarray([entry[0] for entry in entries], dtype=np.intp)
        structure_cotangent = (
            _scatter_numpy(
                entry_indices[chosen_entry.ravel()],
                (1 if dilation else -1) * flat_cotangent,
                size=structure.size,
                dtype=cotangent.dtype,
            ).reshape(structure.shape)
            if 1 in active_inputs
            else np.zeros_like(structure)
        )
        boundary_cotangent = (
            np.sum(flat_cotangent[~cast("np.ndarray", valid_source)])
            if 2 in active_inputs
            else np.zeros_like(cval)
        )
        return (
            _project_cotangent(input_cotangent, input, output),
            _project_cotangent(structure_cotangent, structure, output),
            _project_cotangent(boundary_cotangent, cval, output),
        )
    source_indices = np.arange(input.size, dtype=np.intp).reshape(input.shape)
    flat_input = input.ravel()
    chosen_source = np.full(input.shape, -2, dtype=np.intp)
    track_entry = 1 in active_inputs
    chosen_entry = np.full(input.shape, -1, dtype=np.intp) if track_entry else None
    has_constant_mode = any(_mode_name(mode) == "constant" for mode in neighborhood.modes)
    conflict = np.zeros(input.shape, dtype=bool)
    for flat_index, index, offsets in entries:
        source = _neighborhood_destination_numpy(
            source_indices,
            neighborhood,
            offsets,
        )
        candidate = (
            np.where(
                source >= 0,
                flat_input[np.clip(source, 0, input.size - 1)],
                cval,
            )
            if has_constant_mode
            else flat_input[source]
        )
        candidate = candidate + structure[index] if dilation else candidate - structure[index]
        if candidate.dtype != output.dtype:
            return None
        winner = candidate == output
        unset = chosen_source == -2
        conflict |= winner & ~unset & (chosen_source != source)
        if chosen_entry is not None:
            conflict |= winner & ~unset & (chosen_entry != flat_index)
        chosen_source = np.where(winner & unset, source, chosen_source)
        if chosen_entry is not None:
            chosen_entry = np.where(winner & unset, flat_index, chosen_entry)
    if np.any(conflict) or np.any(chosen_source == -2):
        return None
    flat_cotangent = cotangent.ravel()
    valid_source = chosen_source.ravel() >= 0
    input_cotangent = (
        _scatter_numpy(
            chosen_source.ravel()[valid_source],
            flat_cotangent[valid_source],
            size=input.size,
            dtype=cotangent.dtype,
        ).reshape(input.shape)
        if 0 in active_inputs
        else np.zeros_like(input)
    )
    structure_cotangent = (
        _scatter_numpy(
            cast("np.ndarray", chosen_entry).ravel(),
            (1 if dilation else -1) * flat_cotangent,
            size=structure.size,
            dtype=cotangent.dtype,
        ).reshape(structure.shape)
        if 1 in active_inputs
        else np.zeros_like(structure)
    )
    boundary_cotangent = (
        np.sum(flat_cotangent[~valid_source]) if 2 in active_inputs else np.zeros_like(cval)
    )
    return (
        _project_cotangent(input_cotangent, input, output),
        _project_cotangent(structure_cotangent, structure, output),
        _project_cotangent(boundary_cotangent, cval, output),
    )


def _limited_unique_values_numpy(
    value: np.ndarray,
    *,
    limit: int = 8,
) -> np.ndarray | None:
    flat = value.ravel()
    if flat.size == 0:
        return np.empty(0, dtype=value.dtype)
    stride = max(1, flat.size // 64)
    if np.unique(flat[::stride]).size > limit:
        return None
    covered = np.zeros(flat.shape, dtype=bool)
    selected_values: list[Any] = []
    for _ in range(limit):
        if np.all(covered):
            return np.asarray(selected_values, dtype=value.dtype)
        index = int(np.argmax(~covered))
        selected = flat[index]
        matches = flat == selected
        if not matches[index]:
            return None
        covered |= matches
        selected_values.append(selected)
    return np.asarray(selected_values, dtype=value.dtype) if np.all(covered) else None


def _plateau_selection_data_numpy(
    input: np.ndarray,
    cval: np.ndarray,
    output: np.ndarray,
    *,
    neighborhood: _Neighborhood,
    dilation: bool,
) -> (
    tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        tuple[int, ...],
        tuple[int, ...],
        np.ndarray | None,
    ]
    | None
):
    if not all(neighborhood.footprint) or input.dtype != output.dtype:
        return None
    selected_values = _limited_unique_values_numpy(output)
    if selected_values is None or selected_values.size == 0:
        return None
    value_shape = (selected_values.size, *((1,) * input.ndim))
    winners = input[np.newaxis, ...] == selected_values.reshape(value_shape)
    boundary_value = np.asarray(cval, dtype=output.dtype)
    boundary_winners = selected_values == boundary_value
    filter_origins = tuple(-origin if dilation else origin for origin in neighborhood.origins)
    filter_axes = tuple(axis + 1 for axis in neighborhood.axes)
    boundary_fraction: np.ndarray | None = None
    if any(_mode_name(mode) == "constant" for mode in neighborhood.modes):
        valid_fraction = _scipy_ndimage.uniform_filter(
            np.ones_like(input, dtype=float),
            size=neighborhood.shape,
            mode=neighborhood.modes,
            cval=0,
            origin=filter_origins,
            axes=neighborhood.axes,
        )
        boundary_fraction = 1 - valid_fraction
    return (
        selected_values,
        winners,
        boundary_winners,
        filter_origins,
        filter_axes,
        boundary_fraction,
    )


def _plateau_selection_jvp_numpy(
    output: np.ndarray,
    input: np.ndarray,
    input_tangent: np.ndarray,
    cval: np.ndarray,
    cval_tangent: Any,
    *,
    neighborhood: _Neighborhood,
    dilation: bool,
) -> np.ndarray | None:
    data = _plateau_selection_data_numpy(
        input,
        cval,
        output,
        neighborhood=neighborhood,
        dilation=dilation,
    )
    if data is None:
        return None
    selected_values, winners, boundary_winners, filter_origins, filter_axes, boundary = data
    winner_values = winners.astype(input_tangent.dtype)
    complementary_pair = (
        selected_values.size == 2 and boundary is None and np.all(winners[0] | winners[1])
    )
    if complementary_pair:
        payload = np.stack((winner_values[0], winner_values[0] * input_tangent, input_tangent))
    else:
        payload = np.concatenate(
            (winner_values, winner_values * input_tangent[np.newaxis, ...]),
            axis=0,
        )
    filtered = _scipy_ndimage.uniform_filter(
        payload,
        size=neighborhood.shape,
        mode=neighborhood.modes,
        cval=0,
        origin=filter_origins,
        axes=filter_axes,
    )
    if complementary_pair:
        first_count, first_sum, total_sum = filtered
        winner_count = np.stack((first_count, 1 - first_count))
        tangent_sum = np.stack((first_sum, total_sum - first_sum))
    else:
        winner_count, tangent_sum = np.split(filtered, 2, axis=0)
    if boundary is not None:
        boundary_terms = boundary_winners.reshape((selected_values.size, *((1,) * input.ndim)))
        winner_count = winner_count + boundary_terms * boundary
        tangent_sum = tangent_sum + boundary_terms * boundary * cval_tangent
    output_winners = output[np.newaxis, ...] == selected_values.reshape(
        (selected_values.size, *((1,) * input.ndim))
    )
    result = np.sum(
        np.where(
            output_winners,
            tangent_sum / np.where(winner_count == 0, 1, winner_count),
            0,
        ),
        axis=0,
    )
    return _cast_tangent(result, output)


def _plateau_selection_transpose_numpy(
    cotangent: np.ndarray,
    input: np.ndarray,
    structure: np.ndarray,
    cval: np.ndarray,
    output: np.ndarray,
    *,
    neighborhood: _Neighborhood,
    dilation: bool,
    active_inputs: set[int],
) -> _SelectionPullback | None:
    if 1 in active_inputs:
        return None
    data = _plateau_selection_data_numpy(
        input,
        cval,
        output,
        neighborhood=neighborhood,
        dilation=dilation,
    )
    if data is None:
        return None
    selected_values, winners, boundary_winners, filter_origins, filter_axes, boundary = data
    filter_size = math.prod(neighborhood.shape)
    complementary_pair = (
        selected_values.size == 2 and boundary is None and np.all(winners[0] | winners[1])
    )
    winner_count = _scipy_ndimage.uniform_filter(
        (winners[:1] if complementary_pair else winners).astype(cotangent.dtype),
        size=neighborhood.shape,
        mode=neighborhood.modes,
        cval=0,
        origin=filter_origins,
        axes=filter_axes,
    )
    if complementary_pair:
        winner_count = np.concatenate((winner_count, 1 - winner_count), axis=0)
    if boundary is not None:
        winner_count = (
            winner_count
            + boundary_winners.reshape((selected_values.size, *((1,) * input.ndim))) * boundary
        )
    winner_count = winner_count * filter_size
    output_winners = output[np.newaxis, ...] == selected_values.reshape(
        (selected_values.size, *((1,) * input.ndim))
    )
    active = np.where(
        output_winners,
        cotangent[np.newaxis, ...] / np.where(winner_count == 0, 1, winner_count),
        0,
    )
    boundary_cotangent = (
        np.sum(
            active
            * boundary_winners.reshape((selected_values.size, *((1,) * input.ndim)))
            * boundary
            * filter_size
        )
        if boundary is not None and 2 in active_inputs
        else np.zeros_like(cval)
    )
    symmetric_reflect = all(
        _mode_name(mode) == "reflect" for mode in neighborhood.modes
    ) and not any(neighborhood.origins)
    if symmetric_reflect:
        routed = (
            _scipy_ndimage.uniform_filter(
                active,
                size=neighborhood.shape,
                mode="reflect",
                axes=filter_axes,
            )
            * filter_size
        )
    else:
        routed = active
        configurations = tuple(
            zip(
                neighborhood.axes,
                neighborhood.shape,
                neighborhood.origins,
                neighborhood.modes,
                strict=True,
            )
        )
        for axis, size, origin, mode in reversed(configurations):
            routed, _boundary = _stencil_input_transpose_numpy(
                routed,
                np.ones(size, dtype=cotangent.dtype),
                axes=(axis + 1,),
                origins=(origin,),
                modes=(mode,),
                convolution=dilation,
            )
    input_cotangent = (
        np.sum(np.where(winners, routed, 0), axis=0) if 0 in active_inputs else np.zeros_like(input)
    )
    return (
        _project_cotangent(input_cotangent, input, output),
        np.zeros_like(structure),
        _project_cotangent(boundary_cotangent, cval, output),
    )


def _fast_selection_transpose_numpy(
    cotangent: np.ndarray,
    input: np.ndarray,
    structure: np.ndarray,
    cval: np.ndarray,
    output: np.ndarray,
    *,
    neighborhood: _Neighborhood,
    dilation: bool,
    has_structure: bool,
    active_inputs: set[int],
) -> _SelectionPullback | None:
    if has_structure:
        return _unique_candidate_selection_transpose_numpy(
            cotangent,
            input,
            structure,
            cval,
            output,
            neighborhood=neighborhood,
            dilation=dilation,
            active_inputs=active_inputs,
        )
    if 1 in active_inputs:
        return None
    if len(neighborhood.axes) == 1 and sum(neighborhood.footprint) <= 9:
        axis_result = _small_axis_selection_transpose_numpy(
            cotangent,
            input,
            structure,
            cval,
            output,
            entries=tuple(_neighborhood_entries(neighborhood, dilation=dilation)),
            neighborhood=neighborhood,
        )
        if axis_result is not None:
            return axis_result
    plateau_result = _plateau_selection_transpose_numpy(
        cotangent,
        input,
        structure,
        cval,
        output,
        neighborhood=neighborhood,
        dilation=dilation,
        active_inputs=active_inputs,
    )
    if plateau_result is not None:
        return plateau_result
    unique_result = _unique_value_selection_transpose_numpy(
        cotangent,
        input,
        structure,
        cval,
        output,
    )
    if unique_result is not None:
        return unique_result
    return _small_flat_selection_transpose_numpy(
        cotangent,
        input,
        structure,
        cval,
        output,
        entries=tuple(_neighborhood_entries(neighborhood, dilation=dilation)),
        neighborhood=neighborhood,
        active_inputs=active_inputs,
    )


def _selection_transpose_numpy(
    cotangent: np.ndarray,
    input: np.ndarray,
    structure: np.ndarray,
    cval: np.ndarray,
    output: np.ndarray,
    *,
    neighborhood: _Neighborhood,
    selection: str,
    dilation: bool,
    rank: int | None,
    has_structure: bool,
    active_input_indices: tuple[int, ...] | None = None,
) -> _SelectionPullback:
    if input.size == 0:
        return np.zeros_like(input), np.zeros_like(structure), np.zeros_like(cval)
    active_inputs = {0, 1, 2} if active_input_indices is None else set(active_input_indices)
    fast_result = _fast_selection_transpose_numpy(
        cotangent,
        input,
        structure,
        cval,
        output,
        neighborhood=neighborhood,
        dilation=dilation,
        has_structure=has_structure,
        active_inputs=active_inputs,
    )
    if fast_result is not None:
        return fast_result
    entries = tuple(_neighborhood_entries(neighborhood, dilation=dilation))
    if not entries:
        return np.zeros_like(input), np.zeros_like(structure), np.zeros_like(cval)
    destination_stack = _neighborhood_destinations_numpy(input.shape, entries, neighborhood)
    valid_destinations = destination_stack >= 0
    flat_input = input.ravel()
    values = np.where(
        valid_destinations,
        flat_input[np.clip(destination_stack, 0, input.size - 1)],
        cval,
    )
    if has_structure:
        deltas = np.asarray([structure[index] for _flat, index, _offsets in entries])
        deltas = deltas.reshape((len(entries), *((1,) * input.ndim)))
        values = values + deltas if dilation else values - deltas
    if values.dtype == output.dtype:
        selected = output
    elif rank is None:
        selected = values[0]
        reducer = np.maximum if selection == "maximum" else np.minimum
        for candidate in values[1:]:
            selected = reducer(selected, candidate)
    else:
        selected = np.sort(values, axis=0)[rank]
    winners = values == selected
    winner_count = np.sum(winners, axis=0)
    active = np.where(
        winners,
        cotangent[np.newaxis, ...] / winner_count,
        np.zeros_like(values),
    )
    flat_destinations = destination_stack.ravel()
    weights = active.ravel()
    flat_winners = winners.ravel()
    valid = flat_destinations >= 0
    live = valid & flat_winners
    if 0 not in active_inputs:
        input_cotangent = np.zeros_like(input)
    else:
        input_cotangent = _scatter_numpy(
            flat_destinations[live],
            weights[live],
            size=input.size,
            dtype=cotangent.dtype,
        ).reshape(input.shape)
    boundary_cotangent = (
        np.sum(weights[~valid & flat_winners], dtype=cotangent.dtype)
        if 2 in active_inputs
        else np.zeros_like(cval)
    )
    if has_structure and 1 in active_inputs:
        active_sums = np.sum(active, axis=tuple(range(1, active.ndim)))
        structure_terms = np.zeros(math.prod(neighborhood.shape), dtype=cotangent.dtype)
        for (flat_index, _index, _offsets), contribution in zip(
            entries,
            active_sums,
            strict=True,
        ):
            structure_terms[flat_index] = (1 if dilation else -1) * contribution
        structure_cotangent = structure_terms.reshape(structure.shape)
    else:
        structure_cotangent = np.zeros_like(structure)
    return (
        _project_cotangent(input_cotangent, input, output),
        _project_cotangent(structure_cotangent, structure, output),
        _project_cotangent(boundary_cotangent, cval, output),
    )


@primitive(
    name="scipy.ndimage._selection_transpose",
    static_argnames=(
        "axes",
        "shape",
        "footprint",
        "origins",
        "modes",
        "selection",
        "dilation",
        "rank",
        "has_structure",
        "selected_input_indices",
    ),
)
def _selection_transpose_primitive(
    cotangent: Any,
    input: Any,
    structure: Any,
    cval: Any,
    output: Any,
    *,
    axes: tuple[int, ...],
    shape: tuple[int, ...],
    footprint: tuple[bool, ...],
    origins: tuple[int, ...],
    modes: tuple[str, ...],
    selection: str,
    dilation: bool,
    rank: int | None,
    has_structure: bool,
    selected_input_indices: tuple[int, ...] | None = None,
) -> tuple[Any, Any, Any]:
    _require_numpy_values("_selection_transpose", cotangent, input, structure, cval, output)
    return _selection_transpose_numpy(
        cotangent,
        input,
        structure,
        cval,
        output,
        neighborhood=_Neighborhood(axes, shape, footprint, origins, modes),
        selection=selection,
        dilation=dilation,
        rank=rank,
        has_structure=has_structure,
        active_input_indices=selected_input_indices,
    )


@_selection_transpose_primitive.def_abstract
def _selection_transpose_abstract(
    cotangent: AbstractValue,
    input: AbstractValue,
    structure: AbstractValue,
    cval: AbstractValue,
    output: AbstractValue,
    *,
    axes: tuple[int, ...],
    shape: tuple[int, ...],
    footprint: tuple[bool, ...],
    origins: tuple[int, ...],
    modes: tuple[str, ...],
    selection: str,
    dilation: bool,
    rank: int | None,
    has_structure: bool,
    selected_input_indices: tuple[int, ...] | None,
) -> tuple[ArraySpec, ArraySpec, ArraySpec]:
    del cotangent, output, axes, shape, footprint, origins, modes
    del selection, dilation, rank, has_structure, selected_input_indices
    return input.spec, structure.spec, cval.spec


@_selection_transpose_primitive.def_jvp
def _selection_transpose_jvp(
    output: Any,
    primals: tuple[Any, ...],
    tangents: tuple[Any | None, ...],
    *,
    axes: tuple[int, ...],
    shape: tuple[int, ...],
    footprint: tuple[bool, ...],
    origins: tuple[int, ...],
    modes: tuple[str, ...],
    selection: str,
    dilation: bool,
    rank: int | None,
    has_structure: bool,
    selected_input_indices: tuple[int, ...] | None,
) -> tuple[Any, Any, Any]:
    _cotangent, input, structure, cval, primal_output = primals
    cotangent_tangent = tangents[0]
    if cotangent_tangent is None:
        return tuple(np.zeros_like(value) for value in output)
    return _selection_transpose_primitive(
        cotangent_tangent,
        input,
        structure,
        cval,
        primal_output,
        axes=axes,
        shape=shape,
        footprint=footprint,
        origins=origins,
        modes=modes,
        selection=selection,
        dilation=dilation,
        rank=rank,
        has_structure=has_structure,
        selected_input_indices=selected_input_indices,
    )


@_selection_transpose_primitive.def_transpose
def _selection_transpose_transpose(
    output_cotangent: tuple[Any, Any, Any],
    primals: tuple[Any, ...],
    output: Any,
    *,
    axes: tuple[int, ...],
    shape: tuple[int, ...],
    footprint: tuple[bool, ...],
    origins: tuple[int, ...],
    modes: tuple[str, ...],
    selection: str,
    dilation: bool,
    rank: int | None,
    has_structure: bool,
    selected_input_indices: tuple[int, ...] | None,
) -> tuple[Any, Any, Any, Any, Any]:
    del output
    _cotangent, input, structure, cval, primal_output = primals
    input_cotangent, structure_cotangent, cval_cotangent = output_cotangent
    selected = {0, 1, 2} if selected_input_indices is None else set(selected_input_indices)
    return (
        _selection_jvp(
            primal_output,
            input,
            input_cotangent if 0 in selected else None,
            structure,
            structure_cotangent if 1 in selected else None,
            cval,
            cval_cotangent if 2 in selected else None,
            neighborhood=_Neighborhood(axes, shape, footprint, origins, modes),
            selection=selection,
            dilation=dilation,
            rank=rank,
            has_structure=has_structure,
        ),
        np.zeros_like(input),
        np.zeros_like(structure),
        np.zeros_like(cval),
        np.zeros_like(primal_output),
    )


def _selection_transpose(
    cotangent: Any,
    input: Any,
    structure: Any,
    cval: Any,
    output: Any,
    *,
    neighborhood: _Neighborhood,
    selection: str,
    dilation: bool,
    rank: int | None,
    has_structure: bool,
    active_input_indices: tuple[int, ...] | None = None,
) -> tuple[Any, Any, Any]:
    if not np.issubdtype(_operand_dtype(output), np.inexact):
        return np.zeros_like(input), np.zeros_like(structure), np.zeros_like(cval)
    if not any(_is_traced_value(value) for value in (cotangent, input, structure, cval, output)):
        return _selection_transpose_numpy(
            cotangent,
            input,
            structure,
            cval,
            output,
            neighborhood=neighborhood,
            selection=selection,
            dilation=dilation,
            rank=rank,
            has_structure=has_structure,
            active_input_indices=active_input_indices,
        )
    return _selection_transpose_primitive(
        cotangent,
        input,
        structure,
        cval,
        output,
        axes=neighborhood.axes,
        shape=neighborhood.shape,
        footprint=neighborhood.footprint,
        origins=neighborhood.origins,
        modes=neighborhood.modes,
        selection=selection,
        dilation=dilation,
        rank=rank,
        has_structure=has_structure,
        selected_input_indices=active_input_indices,
    )


def _run_gaussian(
    input: Any,
    cval: Any,
    output: object,
    *,
    one_dimensional: bool,
    axes: tuple[int, ...],
    sigmas: tuple[object, ...],
    orders: tuple[object, ...],
    modes: tuple[str, ...],
    truncate: float,
    radii: tuple[object, ...],
) -> Any:
    if one_dimensional:
        return _scipy_ndimage.gaussian_filter1d(
            input,
            sigmas[0],
            axis=axes[0],
            order=orders[0],
            output=output,
            mode=modes[0],
            cval=cval,
            truncate=truncate,
            radius=radii[0],
        )
    return _scipy_ndimage.gaussian_filter(
        input,
        sigmas,
        order=orders,
        output=output,
        mode=modes,
        cval=cval,
        truncate=truncate,
        radius=radii,
        axes=axes,
    )


def _install_gaussian(name: str, *, one_dimensional: bool) -> Primitive[..., Any]:
    @primitive(
        name=f"scipy.ndimage.{name}",
        static_argnames=(
            "axes",
            "sigmas",
            "orders",
            "modes",
            "truncate",
            "radii",
            "output_dtype",
        ),
    )
    def concrete(
        input: Any,
        cval: Any,
        *,
        axes: tuple[int, ...],
        sigmas: tuple[object, ...],
        orders: tuple[object, ...],
        modes: tuple[str, ...],
        truncate: float,
        radii: tuple[object, ...],
        output_dtype: str | None,
    ) -> Any:
        _require_numpy_values(name, input, cval)
        return _run_gaussian(
            input,
            cval,
            _runtime_output(output_dtype),
            one_dimensional=one_dimensional,
            axes=axes,
            sigmas=sigmas,
            orders=orders,
            modes=modes,
            truncate=truncate,
            radii=radii,
        )

    @concrete.def_abstract
    def abstract(
        input: AbstractValue,
        cval: AbstractValue,
        *,
        axes: tuple[int, ...],
        sigmas: tuple[object, ...],
        orders: tuple[object, ...],
        modes: tuple[str, ...],
        truncate: float,
        radii: tuple[object, ...],
        output_dtype: str | None,
    ) -> ArraySpec:
        sample_shape = (1,) * len(input.spec.shape)
        result = _run_gaussian(
            _sample_array(input, shape=sample_shape),
            _sample_array(cval, shape=()),
            _runtime_output(output_dtype),
            one_dimensional=one_dimensional,
            axes=axes,
            sigmas=sigmas,
            orders=orders,
            modes=modes,
            truncate=truncate,
            radii=radii,
        )
        return _result_spec(input, result)

    @concrete.def_jvp
    def jvp_rule(
        output: Any,
        primals: tuple[Any, ...],
        tangents: tuple[Any | None, ...],
        *,
        axes: tuple[int, ...],
        sigmas: tuple[object, ...],
        orders: tuple[object, ...],
        modes: tuple[str, ...],
        truncate: float,
        radii: tuple[object, ...],
        output_dtype: str | None,
    ) -> Any:
        del output_dtype
        input, cval = primals
        input_tangent, cval_tangent = tangents
        if input_tangent is None and cval_tangent is None:
            return np.zeros_like(output)
        result = concrete(
            input=_zero_tangent(input, input_tangent),
            cval=0 if cval_tangent is None else cval_tangent,
            axes=axes,
            sigmas=sigmas,
            orders=orders,
            modes=modes,
            truncate=truncate,
            radii=radii,
            output_dtype=_operand_dtype(output).str,
        )
        return _cast_tangent(result, output)

    @concrete.def_transpose
    def transpose_rule(
        cotangent: Any,
        primals: tuple[Any, ...],
        output: Any,
        *,
        axes: tuple[int, ...],
        sigmas: tuple[object, ...],
        orders: tuple[object, ...],
        modes: tuple[str, ...],
        truncate: float,
        radii: tuple[object, ...],
        output_dtype: str | None,
        active_input_indices: tuple[int, ...] | None = None,
    ) -> tuple[Any | None, Any | None]:
        del output_dtype
        input, cval = primals
        active = {0, 1} if active_input_indices is None else set(active_input_indices)
        if not active:
            return None, None
        result = cotangent
        boundary_cotangent = np.zeros((), dtype=_operand_dtype(cotangent))
        configurations = tuple(zip(axes, sigmas, orders, modes, radii, strict=True))
        for axis, sigma, order, mode, radius in reversed(configurations):
            sigma_value = float(cast("Any", sigma))
            if not one_dimensional and sigma_value <= 1e-15:
                continue
            radius_value = (
                int(truncate * sigma_value + 0.5)
                if radius is None
                else operator.index(cast("Any", radius))
            )
            result, boundary = _stencil_input_transpose(
                result,
                _gaussian_kernel(
                    sigma_value,
                    operator.index(cast("Any", order)),
                    radius_value,
                )[::-1],
                axes=(axis,),
                origins=(0,),
                modes=(mode,),
                convolution=False,
            )
            boundary_cotangent = boundary_cotangent + boundary
        return (
            _project_cotangent(result, input, output) if 0 in active else None,
            _project_cotangent(boundary_cotangent, cval, output) if 1 in active else None,
        )

    return concrete


_gaussian_filter_primitive = _install_gaussian("gaussian_filter", one_dimensional=False)
_gaussian_filter1d_primitive = _install_gaussian("gaussian_filter1d", one_dimensional=True)


def _run_uniform(
    input: Any,
    cval: Any,
    output: object,
    *,
    one_dimensional: bool,
    axes: tuple[int, ...],
    sizes: tuple[int, ...],
    origins: tuple[int, ...],
    modes: tuple[str, ...],
) -> Any:
    if one_dimensional:
        return _scipy_ndimage.uniform_filter1d(
            input,
            sizes[0],
            axis=axes[0],
            output=output,
            mode=modes[0],
            cval=cval,
            origin=origins[0],
        )
    return _scipy_ndimage.uniform_filter(
        input,
        size=sizes,
        output=output,
        mode=modes,
        cval=cval,
        origin=origins,
        axes=axes,
    )


def _install_uniform(name: str, *, one_dimensional: bool) -> Primitive[..., Any]:
    @primitive(
        name=f"scipy.ndimage.{name}",
        static_argnames=(
            "axes",
            "sizes",
            "origins",
            "modes",
            "output_dtype",
        ),
    )
    def concrete(
        input: Any,
        cval: Any,
        *,
        axes: tuple[int, ...],
        sizes: tuple[int, ...],
        origins: tuple[int, ...],
        modes: tuple[str, ...],
        output_dtype: str | None,
    ) -> Any:
        _require_numpy_values(name, input, cval)
        return _run_uniform(
            input,
            cval=cval,
            output=_runtime_output(output_dtype),
            one_dimensional=one_dimensional,
            axes=axes,
            sizes=sizes,
            origins=origins,
            modes=modes,
        )

    @concrete.def_abstract
    def abstract(
        input: AbstractValue,
        cval: AbstractValue,
        *,
        axes: tuple[int, ...],
        sizes: tuple[int, ...],
        origins: tuple[int, ...],
        modes: tuple[str, ...],
        output_dtype: str | None,
    ) -> ArraySpec:
        result = _run_uniform(
            _sample_array(input, shape=(1,) * len(input.spec.shape)),
            _sample_array(cval, shape=()),
            _runtime_output(output_dtype),
            one_dimensional=one_dimensional,
            axes=axes,
            sizes=sizes,
            origins=origins,
            modes=modes,
        )
        return _result_spec(input, result)

    @concrete.def_jvp
    def jvp_rule(
        output: Any,
        primals: tuple[Any, ...],
        tangents: tuple[Any | None, ...],
        *,
        axes: tuple[int, ...],
        sizes: tuple[int, ...],
        origins: tuple[int, ...],
        modes: tuple[str, ...],
        output_dtype: str | None,
    ) -> Any:
        del output_dtype
        input, _cval = primals
        input_tangent, cval_tangent = tangents
        if input_tangent is None and cval_tangent is None:
            return np.zeros_like(output)
        result = concrete(
            input=_zero_tangent(input, input_tangent),
            cval=0 if cval_tangent is None else cval_tangent,
            axes=axes,
            sizes=sizes,
            origins=origins,
            modes=modes,
            output_dtype=_operand_dtype(output).str,
        )
        return _cast_tangent(result, output)

    @concrete.def_transpose
    def transpose_rule(
        cotangent: Any,
        primals: tuple[Any, ...],
        output: Any,
        *,
        axes: tuple[int, ...],
        sizes: tuple[int, ...],
        origins: tuple[int, ...],
        modes: tuple[str, ...],
        output_dtype: str | None,
        active_input_indices: tuple[int, ...] | None = None,
    ) -> tuple[Any | None, Any | None]:
        del output_dtype
        input, cval = primals
        active = {0, 1} if active_input_indices is None else set(active_input_indices)
        if not active:
            return None, None
        symmetric = all(
            size % 2 == 1 and origin == 0 and _mode_name(mode) in {"constant", "reflect", "wrap"}
            for size, origin, mode in zip(sizes, origins, modes, strict=True)
        )
        if active == {0} and symmetric:
            arguments = {
                "axes": axes,
                "sizes": sizes,
                "origins": origins,
                "modes": modes,
            }
            result = (
                concrete(
                    input=cotangent,
                    cval=0,
                    output_dtype=_operand_dtype(cotangent).str,
                    **arguments,
                )
                if _is_traced_value(cotangent)
                else _run_uniform(
                    cotangent,
                    0,
                    None,
                    one_dimensional=one_dimensional,
                    **arguments,
                )
            )
            return _project_cotangent(result, input, output), None
        result = cotangent
        boundary_cotangent = np.zeros((), dtype=_operand_dtype(cotangent))
        configurations = tuple(zip(axes, sizes, origins, modes, strict=True))
        for axis, size, origin, mode in reversed(configurations):
            if not one_dimensional and size <= 1:
                continue
            result, boundary = _stencil_input_transpose(
                result,
                np.full(size, 1.0 / size),
                axes=(axis,),
                origins=(origin,),
                modes=(mode,),
                convolution=False,
            )
            boundary_cotangent = boundary_cotangent + boundary
        return (
            _project_cotangent(result, input, output) if 0 in active else None,
            _project_cotangent(boundary_cotangent, cval, output) if 1 in active else None,
        )

    return concrete


_uniform_filter_primitive = _install_uniform("uniform_filter", one_dimensional=False)
_uniform_filter1d_primitive = _install_uniform("uniform_filter1d", one_dimensional=True)


def _run_correlation(
    function: Callable[..., Any],
    input: Any,
    weights: Any,
    cval: Any,
    output: object,
    *,
    one_dimensional: bool,
    axes: tuple[int, ...],
    origins: tuple[int, ...],
    modes: tuple[str, ...],
    mode_sequence: bool,
) -> Any:
    runtime_mode: object = modes if mode_sequence else (modes[0] if modes else "reflect")
    if one_dimensional:
        return function(
            input,
            weights,
            axis=axes[0],
            output=output,
            mode=runtime_mode,
            cval=cval,
            origin=origins[0],
        )
    return function(
        input,
        weights,
        output=output,
        mode=runtime_mode,
        cval=cval,
        origin=origins,
        axes=axes,
    )


def _correlation_kernel_configuration(
    input: Any,
    *,
    axes: tuple[int, ...],
    origins: tuple[int, ...],
    modes: tuple[str, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[str, ...]]:
    kernel_axes = tuple(sorted(axes))
    if len(axes) == input.ndim:
        return kernel_axes, origins, modes
    origins_by_axis = dict(zip(axes, origins, strict=True))
    modes_by_axis = dict(zip(axes, modes, strict=True))
    return (
        kernel_axes,
        tuple(origins_by_axis[axis] for axis in kernel_axes),
        tuple(modes_by_axis[axis] for axis in kernel_axes),
    )


def _install_correlation(
    name: str,
    *,
    one_dimensional: bool,
    convolution: bool,
) -> Primitive[..., Any]:
    function = cast("Callable[..., Any]", getattr(_scipy_ndimage, name))

    @primitive(
        name=f"scipy.ndimage.{name}",
        static_argnames=(
            "axes",
            "origins",
            "modes",
            "mode_sequence",
            "output_dtype",
        ),
    )
    def concrete(
        input: Any,
        weights: Any,
        cval: Any,
        *,
        axes: tuple[int, ...],
        origins: tuple[int, ...],
        modes: tuple[str, ...],
        mode_sequence: bool,
        output_dtype: str | None,
    ) -> Any:
        _require_numpy_values(name, input, weights, cval)
        return _run_correlation(
            function,
            input,
            weights,
            cval,
            _runtime_output(output_dtype),
            one_dimensional=one_dimensional,
            axes=axes,
            origins=origins,
            modes=modes,
            mode_sequence=mode_sequence,
        )

    @concrete.def_abstract
    def abstract(
        input: AbstractValue,
        weights: AbstractValue,
        cval: AbstractValue,
        *,
        axes: tuple[int, ...],
        origins: tuple[int, ...],
        modes: tuple[str, ...],
        mode_sequence: bool,
        output_dtype: str | None,
    ) -> ArraySpec:
        result = _run_correlation(
            function,
            _sample_array(input, shape=(1,) * len(input.spec.shape)),
            _sample_array(weights),
            _sample_array(cval, shape=()),
            _runtime_output(output_dtype),
            one_dimensional=one_dimensional,
            axes=axes,
            origins=origins,
            modes=modes,
            mode_sequence=mode_sequence,
        )
        return _result_spec(input, result)

    @concrete.def_jvp
    def jvp_rule(
        output: Any,
        primals: tuple[Any, ...],
        tangents: tuple[Any | None, ...],
        *,
        axes: tuple[int, ...],
        origins: tuple[int, ...],
        modes: tuple[str, ...],
        mode_sequence: bool,
        output_dtype: str | None,
    ) -> Any:
        del output_dtype
        input, weights, cval = primals
        input_tangent, weights_tangent, cval_tangent = tangents
        result: Any | None = None
        if input_tangent is not None or cval_tangent is not None:
            result = concrete(
                input=_zero_tangent(input, input_tangent),
                weights=weights,
                cval=0 if cval_tangent is None else cval_tangent,
                axes=axes,
                origins=origins,
                modes=modes,
                mode_sequence=mode_sequence,
                output_dtype=_operand_dtype(output).str,
            )
        if weights_tangent is not None:
            weight_term = concrete(
                input=input,
                weights=weights_tangent,
                cval=cval,
                axes=axes,
                origins=origins,
                modes=modes,
                mode_sequence=mode_sequence,
                output_dtype=_operand_dtype(output).str,
            )
            result = weight_term if result is None else result + weight_term
        return _cast_tangent(np.zeros_like(output) if result is None else result, output)

    @concrete.def_transpose
    def transpose_rule(
        cotangent: Any,
        primals: tuple[Any, ...],
        output: Any,
        *,
        axes: tuple[int, ...],
        origins: tuple[int, ...],
        modes: tuple[str, ...],
        mode_sequence: bool,
        output_dtype: str | None,
        active_input_indices: tuple[int, ...] | None = None,
    ) -> tuple[Any | None, Any | None, Any | None]:
        del mode_sequence, output_dtype
        input, weights, cval = primals
        active = {0, 1, 2} if active_input_indices is None else set(active_input_indices)
        kernel_axes, kernel_origins, kernel_modes = _correlation_kernel_configuration(
            input,
            axes=axes,
            origins=origins,
            modes=modes,
        )
        input_cotangent: Any | None = None
        boundary_cotangent: Any | None = None
        if active.intersection({0, 2}):
            input_cotangent, boundary_cotangent = _stencil_input_transpose(
                cotangent,
                weights,
                axes=kernel_axes,
                origins=kernel_origins,
                modes=kernel_modes,
                convolution=convolution,
            )
        weight_cotangent = (
            _stencil_weight_transpose(
                cotangent,
                input,
                weights,
                cval,
                axes=kernel_axes,
                origins=kernel_origins,
                modes=kernel_modes,
                convolution=convolution,
            )
            if 1 in active
            else None
        )
        return (
            _project_cotangent(input_cotangent, input, output) if 0 in active else None,
            _project_cotangent(weight_cotangent, weights, output) if 1 in active else None,
            _project_cotangent(boundary_cotangent, cval, output) if 2 in active else None,
        )

    return concrete


_convolve_primitive = _install_correlation(
    "convolve",
    one_dimensional=False,
    convolution=True,
)
_correlate_primitive = _install_correlation(
    "correlate",
    one_dimensional=False,
    convolution=False,
)
_convolve1d_primitive = _install_correlation(
    "convolve1d",
    one_dimensional=True,
    convolution=True,
)
_correlate1d_primitive = _install_correlation(
    "correlate1d",
    one_dimensional=True,
    convolution=False,
)


def _correlation_call(
    name: str,
    primitive_function: Primitive[..., Any],
    input: object,
    weights: object,
    output: object,
    mode: object,
    cval: object,
    origin: object,
    axes: object,
    *,
    one_dimensional: bool,
) -> object:
    ndim = _ndim_of(input)
    normalized_axes = (
        (_normalize_axis(axes, ndim),) if one_dimensional else _normalize_axes(axes, ndim)
    )
    normalized_origins = _normalize_origins(origin, len(normalized_axes))
    mode_sequence = not isinstance(mode, str) and isinstance(mode, Iterable)
    normalized_modes = _normalize_modes(mode, len(normalized_axes))
    return _call_primitive(
        primitive_function,
        name=name,
        input=input,
        output=output,
        operands={"weights": weights, "cval": cval},
        static={
            "axes": normalized_axes,
            "origins": normalized_origins,
            "modes": normalized_modes,
            "mode_sequence": mode_sequence,
        },
    )


def _runtime_footprint(
    shape: tuple[int, ...] | None,
    values: tuple[bool, ...],
) -> np.ndarray | None:
    return None if shape is None else np.asarray(values, dtype=bool).reshape(shape)


def _resolve_rank(kind: str, value: object, filter_size: int) -> int:
    if kind == "median":
        rank = filter_size // 2
    elif kind == "percentile":
        percentile = float(cast("Any", value))
        if percentile < 0:
            percentile += 100
        if percentile < 0 or percentile > 100:
            raise RuntimeError("invalid percentile")
        rank = filter_size - 1 if percentile == 100 else int(filter_size * percentile / 100)
    else:
        rank = operator.index(cast("Any", value))
        if rank < 0:
            rank += filter_size
    if rank < 0 or rank >= filter_size:
        raise RuntimeError("rank not within filter footprint size")
    return rank


def _run_selection(
    function: Callable[..., Any],
    input: Any,
    structure: Any,
    cval: Any,
    output: object,
    *,
    sizes: tuple[int, ...] | None,
    footprint_shape: tuple[int, ...] | None,
    footprint_values: tuple[bool, ...],
    has_structure: bool,
    public_axes: tuple[int, ...],
    public_origins: tuple[int, ...],
    public_modes: tuple[str, ...],
    mode_sequence: bool,
    rank_value: object,
    one_dimensional: bool,
    grey: bool,
    rank_kind: str | None,
) -> Any:
    runtime_mode: object = (
        public_modes if mode_sequence else (public_modes[0] if public_modes else "reflect")
    )
    runtime_footprint = _runtime_footprint(footprint_shape, footprint_values)
    if one_dimensional:
        assert sizes is not None
        return function(
            input,
            sizes[0],
            axis=public_axes[0],
            output=output,
            mode=runtime_mode,
            cval=cval,
            origin=public_origins[0],
        )
    if rank_kind is not None:
        if rank_kind == "median":
            return function(
                input,
                size=sizes,
                footprint=runtime_footprint,
                output=output,
                mode=runtime_mode,
                cval=cval,
                origin=public_origins,
                axes=public_axes,
            )
        return function(
            input,
            rank_value,
            size=sizes,
            footprint=runtime_footprint,
            output=output,
            mode=runtime_mode,
            cval=cval,
            origin=public_origins,
            axes=public_axes,
        )
    if grey:
        return function(
            input,
            size=sizes,
            footprint=runtime_footprint,
            structure=structure if has_structure else None,
            output=output,
            mode=runtime_mode,
            cval=cval,
            origin=public_origins,
            axes=public_axes,
        )
    return function(
        input,
        size=sizes,
        footprint=runtime_footprint,
        output=output,
        mode=runtime_mode,
        cval=cval,
        origin=public_origins,
        axes=public_axes,
    )


def _selection_neighborhood(
    input: object,
    structure: object,
    *,
    sizes: tuple[int, ...] | None,
    footprint_shape: tuple[int, ...] | None,
    footprint_values: tuple[bool, ...],
    has_structure: bool,
    public_axes: tuple[int, ...],
    public_origins: tuple[int, ...],
    public_modes: tuple[str, ...],
    mode_sequence: bool,
    rank_filter: bool,
    dilation: bool,
) -> _Neighborhood:
    runtime_mode: object = (
        public_modes if mode_sequence else (public_modes[0] if public_modes else "reflect")
    )
    neighborhood = _neighborhood(
        input,
        size=sizes,
        footprint=_runtime_footprint(footprint_shape, footprint_values),
        structure=structure if has_structure else None,
        origin=public_origins,
        mode=runtime_mode,
        axes=public_axes,
        rank_filter=rank_filter,
    )
    if not dilation or len(public_axes) == _ndim_of(input) or neighborhood.axes == public_axes:
        return neighborhood
    public_positions = {axis: index for index, axis in enumerate(public_axes)}
    corrected_origins = tuple(
        origin_value
        + int(neighborhood.shape[public_positions[axis]] % 2 == 0)
        - int(neighborhood.shape[kernel_index] % 2 == 0)
        for kernel_index, (axis, origin_value) in enumerate(
            zip(neighborhood.axes, neighborhood.origins, strict=True)
        )
    )
    return _Neighborhood(
        axes=neighborhood.axes,
        shape=neighborhood.shape,
        footprint=neighborhood.footprint,
        origins=corrected_origins,
        modes=neighborhood.modes,
    )


def _install_selection(
    name: str,
    *,
    selection: str,
    one_dimensional: bool = False,
    grey: bool = False,
    rank_kind: str | None = None,
) -> Primitive[..., Any]:
    nondiff = () if grey else ("structure",)
    function = cast("Callable[..., Any]", getattr(_scipy_ndimage, name))

    @primitive(
        name=f"scipy.ndimage.{name}",
        static_argnames=(
            "sizes",
            "footprint_shape",
            "footprint_values",
            "has_structure",
            "public_axes",
            "public_origins",
            "public_modes",
            "mode_sequence",
            "rank_value",
            "output_dtype",
        ),
        nondiff_argnames=nondiff,
    )
    def concrete(
        input: Any,
        structure: Any,
        cval: Any,
        *,
        sizes: tuple[int, ...] | None,
        footprint_shape: tuple[int, ...] | None,
        footprint_values: tuple[bool, ...],
        has_structure: bool,
        public_axes: tuple[int, ...],
        public_origins: tuple[int, ...],
        public_modes: tuple[str, ...],
        mode_sequence: bool,
        rank_value: object,
        output_dtype: str | None,
    ) -> Any:
        _require_numpy_values(name, input, structure, cval)
        return _run_selection(
            function,
            input,
            structure,
            cval,
            _runtime_output(output_dtype),
            sizes=sizes,
            footprint_shape=footprint_shape,
            footprint_values=footprint_values,
            has_structure=has_structure,
            public_axes=public_axes,
            public_origins=public_origins,
            public_modes=public_modes,
            mode_sequence=mode_sequence,
            rank_value=rank_value,
            one_dimensional=one_dimensional,
            grey=grey,
            rank_kind=rank_kind,
        )

    @concrete.def_abstract
    def abstract(
        input: AbstractValue,
        structure: AbstractValue,
        cval: AbstractValue,
        *,
        sizes: tuple[int, ...] | None,
        footprint_shape: tuple[int, ...] | None,
        footprint_values: tuple[bool, ...],
        has_structure: bool,
        public_axes: tuple[int, ...],
        public_origins: tuple[int, ...],
        public_modes: tuple[str, ...],
        mode_sequence: bool,
        rank_value: object,
        output_dtype: str | None,
    ) -> ArraySpec:
        result = _run_selection(
            function,
            _sample_array(input, shape=(1,) * len(input.spec.shape)),
            _sample_array(structure) if has_structure else np.ones(()),
            _sample_array(cval, shape=()),
            _runtime_output(output_dtype),
            sizes=sizes,
            footprint_shape=footprint_shape,
            footprint_values=footprint_values,
            has_structure=has_structure,
            public_axes=public_axes,
            public_origins=public_origins,
            public_modes=public_modes,
            mode_sequence=mode_sequence,
            rank_value=rank_value,
            one_dimensional=one_dimensional,
            grey=grey,
            rank_kind=rank_kind,
        )
        return _result_spec(input, result)

    @concrete.def_jvp
    def jvp_rule(
        output: Any,
        primals: tuple[Any, ...],
        tangents: tuple[Any | None, ...],
        *,
        sizes: tuple[int, ...] | None,
        footprint_shape: tuple[int, ...] | None,
        footprint_values: tuple[bool, ...],
        has_structure: bool,
        public_axes: tuple[int, ...],
        public_origins: tuple[int, ...],
        public_modes: tuple[str, ...],
        mode_sequence: bool,
        rank_value: object,
        output_dtype: str | None,
    ) -> Any:
        del output_dtype
        input, structure, cval = primals
        input_tangent, structure_tangent, cval_tangent = tangents
        neighborhood = _selection_neighborhood(
            input,
            structure,
            sizes=sizes,
            footprint_shape=footprint_shape,
            footprint_values=footprint_values,
            has_structure=has_structure,
            public_axes=public_axes,
            public_origins=public_origins,
            public_modes=public_modes,
            mode_sequence=mode_sequence,
            rank_filter=rank_kind is not None,
            dilation=grey and selection == "maximum",
        )
        resolved_rank = None
        if rank_kind is not None:
            resolved_rank = _resolve_rank(
                rank_kind,
                rank_value,
                sum(neighborhood.footprint),
            )
        return _selection_jvp(
            output,
            input,
            input_tangent,
            structure,
            structure_tangent,
            cval,
            cval_tangent,
            neighborhood=neighborhood,
            selection=selection,
            dilation=grey and selection == "maximum",
            rank=resolved_rank,
            has_structure=has_structure,
        )

    @concrete.def_transpose
    def transpose_rule(
        cotangent: Any,
        primals: tuple[Any, ...],
        output: Any,
        *,
        sizes: tuple[int, ...] | None,
        footprint_shape: tuple[int, ...] | None,
        footprint_values: tuple[bool, ...],
        has_structure: bool,
        public_axes: tuple[int, ...],
        public_origins: tuple[int, ...],
        public_modes: tuple[str, ...],
        mode_sequence: bool,
        rank_value: object,
        output_dtype: str | None,
        active_input_indices: tuple[int, ...] | None = None,
    ) -> tuple[Any | None, Any | None, Any | None]:
        del output_dtype
        input, structure, cval = primals
        neighborhood = _selection_neighborhood(
            input,
            structure,
            sizes=sizes,
            footprint_shape=footprint_shape,
            footprint_values=footprint_values,
            has_structure=has_structure,
            public_axes=public_axes,
            public_origins=public_origins,
            public_modes=public_modes,
            mode_sequence=mode_sequence,
            rank_filter=rank_kind is not None,
            dilation=grey and selection == "maximum",
        )
        resolved_rank = (
            None
            if rank_kind is None
            else _resolve_rank(
                rank_kind,
                rank_value,
                sum(neighborhood.footprint),
            )
        )
        contributions = _selection_transpose(
            cotangent,
            input,
            structure,
            cval,
            output,
            neighborhood=neighborhood,
            selection=selection,
            dilation=grey and selection == "maximum",
            rank=resolved_rank,
            has_structure=has_structure,
            active_input_indices=active_input_indices,
        )
        active = {0, 1, 2} if active_input_indices is None else set(active_input_indices)
        return (
            contributions[0] if 0 in active else None,
            contributions[1] if 1 in active else None,
            contributions[2] if 2 in active else None,
        )

    return concrete


_maximum_filter_primitive = _install_selection("maximum_filter", selection="maximum")
_minimum_filter_primitive = _install_selection("minimum_filter", selection="minimum")
_maximum_filter1d_primitive = _install_selection(
    "maximum_filter1d",
    selection="maximum",
    one_dimensional=True,
)
_minimum_filter1d_primitive = _install_selection(
    "minimum_filter1d",
    selection="minimum",
    one_dimensional=True,
)
_grey_dilation_primitive = _install_selection(
    "grey_dilation",
    selection="maximum",
    grey=True,
)
_grey_erosion_primitive = _install_selection(
    "grey_erosion",
    selection="minimum",
    grey=True,
)
_median_filter_primitive = _install_selection(
    "median_filter",
    selection="minimum",
    rank_kind="median",
)
_rank_filter_primitive = _install_selection(
    "rank_filter",
    selection="minimum",
    rank_kind="rank",
)
_percentile_filter_primitive = _install_selection(
    "percentile_filter",
    selection="minimum",
    rank_kind="percentile",
)


def _selection_call(
    name: str,
    primitive_function: Primitive[..., Any],
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
    rank_value: object = None,
    one_dimensional: bool = False,
) -> object:
    ndim = _ndim_of(input)
    public_axes = (_normalize_axis(axes, ndim),) if one_dimensional else _normalize_axes(axes, ndim)
    sizes = None
    if size is not None:
        sizes = tuple(
            operator.index(cast("Any", item))
            for item in _normalize_sequence(size, len(public_axes))
        )
    public_origins = _normalize_origins(origin, len(public_axes))
    mode_sequence = not isinstance(mode, str) and isinstance(mode, Iterable)
    public_modes = _normalize_modes(mode, len(public_axes))
    static_footprint = None if footprint is None else _static_footprint(footprint)
    footprint_shape = None if static_footprint is None else static_footprint.shape
    footprint_values = (
        () if static_footprint is None else tuple(bool(item) for item in static_footprint.flat)
    )
    has_structure = structure is not None
    return _call_primitive(
        primitive_function,
        name=name,
        input=input,
        output=output,
        operands={
            "structure": np.zeros(()) if structure is None else structure,
            "cval": cval,
        },
        static={
            "sizes": sizes,
            "footprint_shape": footprint_shape,
            "footprint_values": footprint_values,
            "has_structure": has_structure,
            "public_axes": public_axes,
            "public_origins": public_origins,
            "public_modes": public_modes,
            "mode_sequence": mode_sequence,
            "rank_value": _static_scalar(rank_value),
        },
    )
