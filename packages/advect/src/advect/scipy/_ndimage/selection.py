# ruff: noqa: A001, A002, ANN401, PLR0913, PLR2004
# SciPy-compatible names/signatures and primitive rule schemas intentionally trigger these rules.
"""Differentiate nonlinear neighborhood selection with explicit tie semantics.

This module owns neighborhood construction plus winner, plateau, rank, JVP,
and transpose mechanics.  It consumes stencil boundary helpers but does not
install public filter primitives; :mod:`.morphology` owns that wiring.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy import ndimage as _scipy_ndimage

from advect.core import primitive
from advect.scipy._frontend import _is_traced_value
from advect.scipy._ndimage.common import (
    _cast_tangent,
    _mode_name,
    _ndim_of,
    _normalize_axes,
    _normalize_modes,
    _normalize_origins,
    _normalize_sequence,
    _operand_dtype,
    _project_cotangent,
    _require_numpy_values,
    _shape_of,
    _zero_tangent,
)
from advect.scipy._ndimage.stencil import (
    _axis_indices,
    _normalize_indices,
    _pad_numpy,
    _PadWidth,
    _shift,
    _stencil_input_transpose_numpy,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from advect.core import AbstractValue, ArraySpec


type _NeighborhoodEntry = tuple[int, tuple[int, ...], tuple[int, ...]]
type _NeighborhoodSlices = tuple[tuple[slice, ...], ...]
type _HomogeneousNeighborhoodData = tuple[np.ndarray, _PadWidth, str, _NeighborhoodSlices]
type _SelectionPullback = tuple[np.ndarray, np.ndarray, np.ndarray]


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
