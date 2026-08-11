# ruff: noqa: A001, A002, ANN401, PLR0913, PLR2004
# SciPy-compatible names/signatures and primitive rule schemas intentionally trigger these rules.
"""Install rank, extrema, and grey-morphology selection primitives.

These registrations connect public configuration to the nonlinear selection
derivative engine.  Composite morphology and all public signatures remain in
:mod:`advect.scipy.ndimage`.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy import ndimage as _scipy_ndimage

from advect.core import primitive
from advect.scipy._ndimage.common import (
    _call_primitive,
    _ndim_of,
    _normalize_axes,
    _normalize_axis,
    _normalize_modes,
    _normalize_origins,
    _normalize_sequence,
    _require_numpy_values,
    _result_spec,
    _runtime_output,
    _sample_array,
    _static_scalar,
)
from advect.scipy._ndimage.selection import (
    _Neighborhood,
    _neighborhood,
    _selection_jvp,
    _selection_transpose,
    _static_footprint,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core import AbstractValue, ArraySpec
    from advect.core._primitive import Primitive


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
        if sizes is None:
            raise AssertionError("One-dimensional selection requires a size")
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
