"""Interp JVP rules."""

from __future__ import annotations

from typing import Any, cast

from advect.autodiff.rules.array_family._backend_runtime import _array_constructor_like, xp
from advect.autodiff.rules.array_family.jvp.common import (
    _asarray_preserving_trace,
    _asarray_unwrapped,
    _infer_output_tangent_dtype,
    _zeros_output_tangent,
)

_INTERP_X_TANGENT_INDEX = 0
_INTERP_XP_TANGENT_INDEX = 1
_INTERP_FP_TANGENT_INDEX = 2

_MIN_SAMPLES_FOR_SLOPE = 2


def _tangent_at(tangents: tuple[Any | None, ...], index: int) -> Any:
    return tangents[index] if len(tangents) > index else None


def _one_hot(indices: Any, size: int, dtype: Any, *, tangent_like: Any) -> Any:
    """Build the concrete selection matrix for a data-dependent gather."""
    positions = _array_constructor_like(tangent_like, "arange", size, dtype=xp.int64)
    return xp.astype(xp.expand_dims(indices, axis=-1) == positions, dtype)


def _gather(source: Any, selection: Any) -> Any:
    """Gather from a 1-D source by contraction so the result stays traceable.

    Advanced indexing would be the natural spelling, but its pullback needs a
    scatter-add primitive that does not exist yet. Contracting against a
    concrete one-hot matrix uses only ops that already transpose, which is what
    lets the structural transpose derive reverse mode from this rule.
    """
    return xp.sum(selection * source, axis=-1)


def _interp_periodic_jvp(
    ans: xp.ndarray,
    x: xp.ndarray,
    xp_points: xp.ndarray,
    fp: xp.ndarray,
    *,
    tangents: tuple[xp.ndarray | None, ...],
    period: Any,
) -> xp.ndarray:
    period_value = abs(period)
    if period_value == 0:
        msg = "numpy.interp period must be non-zero"
        raise ValueError(msg)
    sample_count = int(_asarray_unwrapped(xp_points).shape[0])
    normalized_x = xp.remainder(_asarray_preserving_trace(x), period_value)
    normalized_positions = xp.remainder(
        _asarray_preserving_trace(xp_points),
        period_value,
    )
    order = xp.argsort(_asarray_unwrapped(normalized_positions))
    selection = _one_hot(
        order,
        sample_count,
        _asarray_unwrapped(normalized_positions).dtype,
        tangent_like=next(tangent for tangent in tangents if tangent is not None),
    )
    sorted_positions = _gather(normalized_positions, selection)
    sorted_values = _gather(_asarray_preserving_trace(fp), selection)

    def extend_positions(value: Any) -> Any:
        return xp.concatenate(
            (
                value[-1:] - period_value,
                value,
                value[:1] + period_value,
            )
        )

    def extend_values(value: Any) -> Any:
        return xp.concatenate((value[-1:], value, value[:1]))

    x_tangent = _tangent_at(tangents, _INTERP_X_TANGENT_INDEX)
    xp_tangent = _tangent_at(tangents, _INTERP_XP_TANGENT_INDEX)
    fp_tangent = _tangent_at(tangents, _INTERP_FP_TANGENT_INDEX)
    sorted_xp_tangent = (
        None if xp_tangent is None else _gather(_asarray_preserving_trace(xp_tangent), selection)
    )
    sorted_fp_tangent = (
        None if fp_tangent is None else _gather(_asarray_preserving_trace(fp_tangent), selection)
    )
    return _jvp_interp(
        ans,
        normalized_x,
        extend_positions(sorted_positions),
        extend_values(sorted_values),
        tangents=(
            x_tangent,
            None if sorted_xp_tangent is None else extend_values(sorted_xp_tangent),
            None if sorted_fp_tangent is None else extend_values(sorted_fp_tangent),
        ),
        left=None,
        right=None,
        period=None,
    )


def _jvp_interp(
    ans: xp.ndarray,
    x: xp.ndarray,
    xp_points: xp.ndarray,
    fp: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    left: Any = None,
    right: Any = None,
    period: Any = None,
    **attrs: Any,
) -> xp.ndarray:
    """Differentiate piecewise-linear interpolation in all three inputs.

    ``interp`` is linear in ``fp`` and nonlinear in ``x``/``xp``. Every output
    element depends on exactly one bracketing pair, so each tangent term is a
    gather scaled by a coefficient built from the primals. Keeping those
    coefficients traced is what preserves nested differentiation.
    """
    _ = rest, attrs
    if period is not None:
        return _interp_periodic_jvp(
            ans,
            x,
            xp_points,
            fp,
            tangents=tangents,
            period=period,
        )

    x_values = _asarray_unwrapped(x)
    sample_positions = _asarray_unwrapped(xp_points)
    if sample_positions.ndim != 1:
        msg = "numpy.interp JVP currently supports 1D xp/fp only"
        raise NotImplementedError(msg)

    x_tangent = _tangent_at(tangents, _INTERP_X_TANGENT_INDEX)
    xp_tangent = _tangent_at(tangents, _INTERP_XP_TANGENT_INDEX)
    fp_tangent = _tangent_at(tangents, _INTERP_FP_TANGENT_INDEX)
    if x_tangent is None and xp_tangent is None and fp_tangent is None:
        return _zeros_output_tangent(ans, tangents)

    dtype = _infer_output_tangent_dtype(ans, tangents)
    real_dtype = x_values.dtype
    sample_count = int(sample_positions.shape[0])

    below = x_values < sample_positions[0]
    above = x_values > sample_positions[-1]

    # Out-of-range queries take a constant: an explicit ``left``/``right``
    # contributes nothing, while the default clamps onto an endpoint of ``fp``.
    clamp_low = below if left is None else xp.zeros_like(below)
    clamp_high = above if right is None else xp.zeros_like(above)
    clamped = xp.logical_or(clamp_low, clamp_high)
    tangent_like = next(tangent for tangent in tangents if tangent is not None)
    clamp_selection = _one_hot(
        xp.where(below, 0, sample_count - 1),
        sample_count,
        real_dtype,
        tangent_like=tangent_like,
    )

    total = _zeros_output_tangent(ans, tangents)

    if sample_count < _MIN_SAMPLES_FOR_SLOPE:
        # A single sample has no interval: every resolved query returns fp[0].
        if fp_tangent is not None:
            resolved = xp.logical_or(clamped, xp.logical_not(xp.logical_or(below, above)))
            gathered = _gather(_asarray_preserving_trace(fp_tangent), clamp_selection)
            total = total + xp.astype(resolved, real_dtype) * gathered
        return cast("xp.ndarray[Any, Any]", _asarray_preserving_trace(total, dtype=dtype))

    interior = xp.astype(xp.logical_not(xp.logical_or(below, above)), real_dtype)
    clamp_mask = xp.astype(clamped, real_dtype)

    # ``side="right" - 1`` puts an exact sample hit on its own left bracket, so
    # an offset of zero reproduces fp[k] the way NumPy does.
    lower = xp.searchsorted(sample_positions, x_values, side="right") - 1
    lower = xp.minimum(xp.maximum(lower, 0), sample_count - 2)
    lower_selection = _one_hot(
        lower,
        sample_count,
        real_dtype,
        tangent_like=tangent_like,
    )
    upper_selection = _one_hot(
        lower + 1,
        sample_count,
        real_dtype,
        tangent_like=tangent_like,
    )

    x_source = _asarray_preserving_trace(x)
    positions_source = _asarray_preserving_trace(xp_points)
    values_source = _asarray_preserving_trace(fp)

    lower_position = _gather(positions_source, lower_selection)
    gap = _gather(positions_source, upper_selection) - lower_position
    offset = (x_source - lower_position) / gap
    slope = (
        _gather(values_source, upper_selection) - _gather(values_source, lower_selection)
    ) / gap

    if x_tangent is not None:
        total = total + interior * slope * _asarray_preserving_trace(x_tangent)

    if fp_tangent is not None:
        tangent_values = _asarray_preserving_trace(fp_tangent)
        total = total + interior * (
            (1 - offset) * _gather(tangent_values, lower_selection)
            + offset * _gather(tangent_values, upper_selection)
        )
        total = total + clamp_mask * _gather(tangent_values, clamp_selection)

    if xp_tangent is not None:
        tangent_positions = _asarray_preserving_trace(xp_tangent)
        total = total + interior * slope * (
            (offset - 1) * _gather(tangent_positions, lower_selection)
            - offset * _gather(tangent_positions, upper_selection)
        )

    return cast("xp.ndarray[Any, Any]", _asarray_preserving_trace(total, dtype=dtype))
