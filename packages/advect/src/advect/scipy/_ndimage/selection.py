# ruff: noqa: A001, A002, ANN401, PLR0913, PLR2004
# SciPy-compatible names/signatures and primitive rule schemas intentionally trigger these rules.
"""Differentiate nonlinear neighborhood selection with explicit tie semantics.

This module owns neighborhood construction plus winner, plateau, rank, JVP,
and transpose mechanics.  It consumes stencil boundary helpers but does not
install public filter primitives; :mod:`.morphology` owns that wiring.
"""

from __future__ import annotations

import functools
import itertools
import math
import operator
from dataclasses import asdict, dataclass
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
    _fold_numpy,
    _pad_numpy,
    _pad_width,
    _padded_transpose_numpy,
    _shift,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from advect.core import AbstractValue, ArraySpec


type _NeighborhoodEntry = tuple[int, tuple[int, ...], tuple[int, ...]]
type _SelectionPullback = tuple[np.ndarray, np.ndarray, np.ndarray]

# Above this many window slots, sorting distinct values once beats visiting
# every slot.
_UNIQUE_VALUE_SLOTS = 25


@dataclass(frozen=True, slots=True)
class _Neighborhood:
    axes: tuple[int, ...]
    shape: tuple[int, ...]
    footprint: tuple[bool, ...]
    origins: tuple[int, ...]
    modes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Selector(_Neighborhood):
    """A neighborhood and how each output selects from its window.

    Its fields are the transpose primitive's static attributes. ``rank`` is
    ``None`` for extrema, and dilation reads the window reflected and adds
    the structure instead of subtracting it.
    """

    selection: str
    dilation: bool
    rank: int | None
    has_structure: bool


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


def _neighborhood_entries(selector: _Selector) -> Iterator[_NeighborhoodEntry]:
    if not selector.axes:
        yield 0, (), ()
        return
    centers = tuple(size // 2 for size in selector.shape)
    # Dilation reads the window reflected through its center.
    sign = -1 if selector.dilation else 1
    for flat_index, index in enumerate(np.ndindex(selector.shape)):
        if selector.footprint[flat_index]:
            offsets = tuple(
                sign * (item - center - origin)
                for item, center, origin in zip(index, centers, selector.origins, strict=True)
            )
            yield flat_index, index, offsets


def _select(values: Iterable[Any], selector: _Selector) -> Any:
    """Select each output from its window's ``values``."""
    if selector.rank is None:
        reducer = np.maximum if selector.selection == "maximum" else np.minimum
        return functools.reduce(reducer, values)
    return np.sort(np.stack(tuple(values), axis=0), axis=0)[selector.rank]


def _neighborhood_candidates(
    input: Any,
    input_tangent: Any,
    structure: Any,
    structure_tangent: Any | None,
    cval: Any,
    cval_tangent: Any,
    selector: _Selector,
) -> Iterator[tuple[Any, Any]]:
    for _flat_index, index, offsets in _neighborhood_entries(selector):
        candidate = _shift(
            input, axes=selector.axes, offsets=offsets, modes=selector.modes, cval=cval
        )
        candidate_tangent = _shift(
            input_tangent,
            axes=selector.axes,
            offsets=offsets,
            modes=selector.modes,
            cval=cval_tangent,
        )
        if selector.has_structure:
            delta = structure[index]
            tangent_delta = 0 if structure_tangent is None else structure_tangent[index]
            if selector.dilation:
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
    selector: _Selector,
) -> Any:
    operands = (
        input,
        _zero_tangent(input, input_tangent),
        structure,
        structure_tangent,
        cval,
        0 if cval_tangent is None else cval_tangent,
    )
    if not any(_is_traced_value(value) for value in (output, *operands)):
        return _selection_jvp_numpy(output, *operands, selector)
    candidates = list(_neighborhood_candidates(*operands, selector))
    if not candidates:
        return np.zeros_like(output)
    selected = _select((candidate for candidate, _ in candidates), selector)
    tangent_sum = np.zeros_like(output)
    winner_count = np.zeros_like(output)
    for candidate, candidate_tangent in candidates:
        winner = candidate == selected
        tangent_sum = tangent_sum + np.where(winner, candidate_tangent, np.zeros_like(output))
        winner_count = winner_count + np.where(winner, np.ones_like(output), np.zeros_like(output))
    return _cast_tangent(tangent_sum / winner_count, output)


def _all_finite(*values: Any) -> bool:
    return all(bool(np.isfinite(value).all()) for value in values if value is not None)


def _winning(winner: np.ndarray, value: np.ndarray, *, finite: bool) -> np.ndarray:
    """Return ``value`` at winning slots and exact zeros at losing ones.

    Weighting by the 0/1 winner mask is several times faster than ``np.where``
    but turns a non-finite value at a losing slot into NaN (``0 * inf``).
    """
    return winner * value if finite else np.where(winner, value, 0)


class _WindowSlots:
    """Every window slot of one concrete selection as a view of one padded copy.

    Padding follows each axis's boundary mode, so reflected duplicates and
    constant-boundary slots are ordinary views, and :meth:`fold` returns a
    padded cotangent's shares to the samples or ``cval`` those slots read.
    """

    def __init__(
        self,
        input: np.ndarray,
        structure: np.ndarray,
        cval: Any,
        output: np.ndarray,
        selector: _Selector,
    ) -> None:
        self.selector = selector
        self.shape = input.shape
        self.entries = tuple(_neighborhood_entries(selector))
        offsets_by_axis = tuple(zip(*(offsets for _, _, offsets in self.entries), strict=True))
        self.pad_width = _pad_width(
            input.ndim,
            selector.axes,
            tuple((min(offsets), max(offsets)) for offsets in offsets_by_axis),
        )
        self.views = tuple(self._view(offsets) for _, _, offsets in self.entries)
        self.padded = self.pad(input, cval)
        self.structure = structure if selector.has_structure else None
        self.output = self.selected = output
        if next(self.values(self.padded, self.structure)).dtype != output.dtype:
            self._reselect()

    def _view(self, offsets: tuple[int, ...]) -> tuple[slice, ...]:
        starts = [before for before, _after in self.pad_width]
        for axis, offset in zip(self.selector.axes, offsets, strict=True):
            starts[axis] += offset
        return tuple(
            slice(start, start + length) for start, length in zip(starts, self.shape, strict=True)
        )

    def _reselect(self) -> None:
        # SciPy cast or rounded its result, so select at the candidates' precision.
        self.selected = _select(self.values(self.padded, self.structure), self.selector)

    def pad(self, value: np.ndarray, cval: Any) -> np.ndarray:
        return _pad_numpy(
            value,
            self.pad_width,
            axes=self.selector.axes,
            modes=self.selector.modes,
            cval=cval,
        )

    def fold(self, padded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _fold_numpy(
            padded,
            self.pad_width,
            axes=self.selector.axes,
            modes=self.selector.modes,
            shape=self.shape,
        )

    def values(self, padded: np.ndarray, structure: np.ndarray | None) -> Iterator[np.ndarray]:
        """Yield each slot's view of ``padded``, moved by its ``structure`` entry."""
        for (_flat_index, index, _offsets), view in zip(self.entries, self.views, strict=True):
            value = padded[view]
            if structure is not None:
                value = (
                    value + structure[index] if self.selector.dilation else value - structure[index]
                )
            yield value

    def winners(self) -> Iterator[np.ndarray]:
        for value in self.values(self.padded, self.structure):
            yield value == self.selected

    def tally(
        self,
        padded: np.ndarray | None = None,
        structure: np.ndarray | None = None,
    ) -> tuple[Any, np.ndarray]:
        """Count each output's winning slots and sum their ``padded`` values.

        NaN outputs have no winning slot. Any other output without one (SciPy
        rounded it) selects every output again, once, at the candidates'
        precision.
        """
        total: Any = 0
        winner_count = np.zeros(self.shape, np.min_scalar_type(len(self.entries)))
        values = () if padded is None else self.values(padded, structure)
        finite = _all_finite(padded, structure)
        for winner, value in itertools.zip_longest(self.winners(), values):
            winner_count += winner
            if value is not None:
                total = total + _winning(winner, value, finite=finite)
        if (
            winner_count.all()
            or self.selected is not self.output
            or np.isnan(self.output[winner_count == 0]).all()
        ):
            return total, winner_count
        self._reselect()
        return self.tally(padded, structure)


def _selection_jvp_numpy(
    output: np.ndarray,
    input: np.ndarray,
    input_tangent: np.ndarray,
    structure: np.ndarray,
    structure_tangent: np.ndarray | None,
    cval: Any,
    cval_tangent: Any,
    selector: _Selector,
) -> np.ndarray:
    if input.size == 0:
        return np.zeros_like(output)
    if not selector.has_structure:
        plateau_result = _plateau_selection_jvp_numpy(
            output, input, input_tangent, cval, cval_tangent, selector
        )
        if plateau_result is not None:
            return plateau_result
        chosen = (
            _unique_value_selection_sources_numpy(input, cval, output)
            if sum(selector.footprint) > _UNIQUE_VALUE_SLOTS
            else None
        )
        if chosen is not None:
            selected = input_tangent.ravel()[np.clip(chosen, 0, input.size - 1)]
            return _cast_tangent(np.where(chosen >= 0, selected, cval_tangent), output)
    slots = _WindowSlots(input, structure, cval, output, selector)
    tangent_dtype = np.result_type(input_tangent, cval_tangent)
    tangent_sum, winner_count = slots.tally(
        slots.pad(input_tangent.astype(tangent_dtype, copy=False), cval_tangent),
        structure_tangent if selector.has_structure else None,
    )
    return _cast_tangent(tangent_sum / winner_count, output)


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
    # SciPy selects only real inputs, so only a cotangent's real part routes.
    scattered = np.bincount(
        chosen.ravel()[from_input],
        weights=np.real(flat_cotangent[from_input]),
        minlength=input.size,
    )
    return (
        _project_cotangent(scattered.reshape(input.shape), input, output),
        np.zeros_like(structure),
        _project_cotangent(np.sum(flat_cotangent[~from_input]), cval, output),
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


class _Plateau:
    """The few values a plateau-heavy selection outputs, one leading entry each.

    Box filters count each value's winning slots in every window at once.
    """

    def __init__(
        self,
        input: np.ndarray,
        cval: np.ndarray,
        output: np.ndarray,
        selector: _Selector,
        values: np.ndarray,
    ) -> None:
        self.selector = selector
        values = values.reshape((values.size, *((1,) * input.ndim)))
        self.winners = input == values
        self.output_winners = output == values
        self.boundary_winners = values == np.asarray(cval, dtype=output.dtype)
        # Dilation mirrors the window, so an even extent also shifts its origin by
        # one to count the same slots that ``_neighborhood_entries`` enumerates.
        self.origins = tuple(
            -origin - (1 - size % 2) if selector.dilation else origin
            for origin, size in zip(selector.origins, selector.shape, strict=True)
        )
        self.axes = tuple(axis + 1 for axis in selector.axes)
        # The fraction of each window that reads the constant boundary.
        self.boundary: np.ndarray | None = None
        if any(_mode_name(mode) == "constant" for mode in selector.modes):
            self.boundary = 1 - self.box(np.ones((1, *input.shape)))[0]
        # Two values that cover the input need one count; the other is its complement.
        self.pair = bool(
            values.size == 2 and self.boundary is None and np.all(self.winners[0] | self.winners[1])
        )

    def box(self, payload: np.ndarray) -> np.ndarray:
        """Average each leading entry of ``payload`` over every window."""
        return _scipy_ndimage.uniform_filter(
            payload,
            size=self.selector.shape,
            mode=self.selector.modes,
            cval=0,
            origin=self.origins,
            axes=self.axes,
        )


def _plateau_numpy(
    input: np.ndarray,
    cval: np.ndarray,
    output: np.ndarray,
    selector: _Selector,
    derivatives: tuple[Any, ...],
) -> _Plateau | None:
    # SciPy's box filters keep one running sum per line. A non-finite derivative
    # would reach every later window on its line, so the window engine handles
    # it instead. Finite results are accurate to about eps times the largest
    # derivative on the line, not entry by entry.
    if not all(selector.footprint) or input.dtype != output.dtype or not _all_finite(*derivatives):
        return None
    values = _limited_unique_values_numpy(output)
    if values is None or values.size == 0:
        return None
    return _Plateau(input, cval, output, selector, values)


def _plateau_selection_jvp_numpy(
    output: np.ndarray,
    input: np.ndarray,
    input_tangent: np.ndarray,
    cval: np.ndarray,
    cval_tangent: Any,
    selector: _Selector,
) -> np.ndarray | None:
    plateau = _plateau_numpy(input, cval, output, selector, (input_tangent, cval_tangent))
    if plateau is None:
        return None
    winner_values = plateau.winners.astype(input_tangent.dtype)
    if plateau.pair:
        first_count, first_sum, total_sum = plateau.box(
            np.stack((winner_values[0], winner_values[0] * input_tangent, input_tangent))
        )
        winner_count = np.stack((first_count, 1 - first_count))
        tangent_sum = np.stack((first_sum, total_sum - first_sum))
    else:
        winner_count, tangent_sum = np.split(
            plateau.box(np.concatenate((winner_values, winner_values * input_tangent))), 2
        )
    if plateau.boundary is not None:
        boundary = plateau.boundary_winners * plateau.boundary
        winner_count = winner_count + boundary
        tangent_sum = tangent_sum + boundary * cval_tangent
    result = np.sum(
        np.where(
            plateau.output_winners,
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
    selector: _Selector,
    active_inputs: set[int],
) -> _SelectionPullback | None:
    plateau = _plateau_numpy(input, cval, output, selector, (cotangent,))
    if plateau is None:
        return None
    filter_size = math.prod(selector.shape)
    winners = plateau.winners[:1] if plateau.pair else plateau.winners
    winner_count = plateau.box(winners.astype(cotangent.dtype))
    if plateau.pair:
        winner_count = np.concatenate((winner_count, 1 - winner_count))
    if plateau.boundary is not None:
        winner_count = winner_count + plateau.boundary_winners * plateau.boundary
    winner_count = winner_count * filter_size
    active = np.where(
        plateau.output_winners,
        cotangent / np.where(winner_count == 0, 1, winner_count),
        0,
    )
    boundary_cotangent = (
        np.sum(active * plateau.boundary_winners * plateau.boundary * filter_size)
        if plateau.boundary is not None and 2 in active_inputs
        else np.zeros_like(cval)
    )
    windows = tuple(zip(plateau.origins, selector.shape, strict=True))
    if all(
        size % 2 and not origin and _mode_name(mode) in {"constant", "reflect", "wrap"}
        for (origin, size), mode in zip(windows, selector.modes, strict=True)
    ):
        # A centered box is its own adjoint under these boundary modes.
        routed = filter_size * plateau.box(active)
    else:
        # Each counting box reads the offsets spanned by ``extents``. Its
        # adjoint is the mirrored box over a zero-padded cotangent, whose
        # padding then folds back onto the samples each boundary mode read.
        extents = tuple(
            (-(size // 2) - origin, size - 1 - size // 2 - origin) for origin, size in windows
        )
        mirrored_origins = tuple(-origin - (1 - size % 2) for origin, size in windows)
        routed, _boundary = _padded_transpose_numpy(
            active,
            lambda embedded: (
                filter_size
                * _scipy_ndimage.uniform_filter(
                    embedded,
                    size=selector.shape,
                    mode="constant",
                    cval=0,
                    origin=mirrored_origins,
                    axes=plateau.axes,
                )
            ),
            axes=plateau.axes,
            extents=extents,
            modes=selector.modes,
        )
    input_cotangent = (
        np.sum(np.where(plateau.winners, routed, 0), axis=0)
        if 0 in active_inputs
        else np.zeros_like(input)
    )
    return (
        _project_cotangent(input_cotangent, input, output),
        np.zeros_like(structure),
        _project_cotangent(boundary_cotangent, cval, output),
    )


def _selection_transpose_numpy(
    cotangent: np.ndarray,
    input: np.ndarray,
    structure: np.ndarray,
    cval: np.ndarray,
    output: np.ndarray,
    selector: _Selector,
    active_input_indices: tuple[int, ...] | None = None,
) -> _SelectionPullback:
    if input.size == 0:
        return np.zeros_like(input), np.zeros_like(structure), np.zeros_like(cval)
    active_inputs = {0, 1, 2} if active_input_indices is None else set(active_input_indices)
    if not selector.has_structure:
        fast_result = _plateau_selection_transpose_numpy(
            cotangent, input, structure, cval, output, selector, active_inputs
        )
        if fast_result is None and sum(selector.footprint) > _UNIQUE_VALUE_SLOTS:
            fast_result = _unique_value_selection_transpose_numpy(
                cotangent, input, structure, cval, output
            )
        if fast_result is not None:
            return fast_result
    slots = _WindowSlots(input, structure, cval, output, selector)
    # NaN outputs have no winning slot and route nothing.
    share = cotangent / np.maximum(slots.tally()[1], 1)
    finite = _all_finite(share)
    padded_cotangent = np.zeros(slots.padded.shape, share.dtype)
    structure_cotangent = np.zeros(structure.shape, share.dtype)
    routed = bool(active_inputs & {0, 2})
    weighted = selector.has_structure and 1 in active_inputs
    for (flat_index, _index, _offsets), view, winner in zip(
        slots.entries,
        slots.views,
        slots.winners(),
        strict=True,
    ):
        routed_share = _winning(winner, share, finite=finite)
        if routed:
            padded_cotangent[view] += routed_share
        if weighted:
            structure_cotangent.flat[flat_index] = np.sum(routed_share)
    input_cotangent, boundary_cotangent = slots.fold(padded_cotangent)
    return (
        _project_cotangent(input_cotangent, input, output)
        if 0 in active_inputs
        else np.zeros_like(input),
        _project_cotangent(
            structure_cotangent if selector.dilation else -structure_cotangent, structure, output
        )
        if weighted
        else np.zeros_like(structure),
        _project_cotangent(boundary_cotangent, cval, output)
        if 2 in active_inputs
        else np.zeros_like(cval),
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
    selector = _Selector(
        axes, shape, footprint, origins, modes, selection, dilation, rank, has_structure
    )
    return _selection_transpose_numpy(
        cotangent, input, structure, cval, output, selector, selected_input_indices
    )


@_selection_transpose_primitive.def_abstract
def _selection_transpose_abstract(
    cotangent: AbstractValue,
    input: AbstractValue,
    structure: AbstractValue,
    cval: AbstractValue,
    output: AbstractValue,
    **static: Any,
) -> tuple[ArraySpec, ArraySpec, ArraySpec]:
    del cotangent, output, static
    return input.spec, structure.spec, cval.spec


@_selection_transpose_primitive.def_jvp
def _selection_transpose_jvp(
    output: Any,
    primals: tuple[Any, ...],
    tangents: tuple[Any | None, ...],
    **static: Any,
) -> tuple[Any, Any, Any]:
    _cotangent, input, structure, cval, primal_output = primals
    if tangents[0] is None:
        return tuple(np.zeros_like(value) for value in output)
    return _selection_transpose_primitive(
        tangents[0], input, structure, cval, primal_output, **static
    )


@_selection_transpose_primitive.def_transpose
def _selection_transpose_transpose(
    output_cotangent: tuple[Any, Any, Any],
    primals: tuple[Any, ...],
    output: Any,
    *,
    selected_input_indices: tuple[int, ...] | None,
    **static: Any,
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
            _Selector(**static),
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
    selector: _Selector,
    active_input_indices: tuple[int, ...] | None = None,
) -> tuple[Any, Any, Any]:
    if not np.issubdtype(_operand_dtype(output), np.inexact):
        return np.zeros_like(input), np.zeros_like(structure), np.zeros_like(cval)
    operands = (cotangent, input, structure, cval, output)
    if not any(_is_traced_value(value) for value in operands):
        return _selection_transpose_numpy(*operands, selector, active_input_indices)
    return _selection_transpose_primitive(
        *operands, **asdict(selector), selected_input_indices=active_input_indices
    )
