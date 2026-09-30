"""Interp JVP rules."""

from __future__ import annotations

from typing import Any, cast

from advect.autodiff.rules.array_family._backend_runtime import xp
from advect.autodiff.rules.array_family._transpose_utils import (
    infer_output_tangent_dtype as _infer_output_tangent_dtype,
)
from advect.autodiff.rules.array_family.jvp.common import (
    _asarray_preserving_trace,
    _asarray_unwrapped,
    _zeros_output_tangent,
)

_INTERP_X_TANGENT_INDEX = 0
_INTERP_XP_TANGENT_INDEX = 1
_INTERP_FP_TANGENT_INDEX = 2

_MIN_SAMPLES_FOR_SLOPE = 2


def _tangent_at(tangents: tuple[Any | None, ...], index: int) -> Any:
    return tangents[index] if len(tangents) > index else None


def _weights(mask: Any, dtype: Any) -> Any:
    # A 0-d query compares to a bool scalar, which NumPy 2.0's astype rejects.
    return xp.astype(xp.asarray(mask), dtype)


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
    normalized_x = xp.remainder(_asarray_preserving_trace(x), period_value)
    normalized_positions = xp.remainder(
        _asarray_preserving_trace(xp_points),
        period_value,
    )
    order = xp.argsort(_asarray_unwrapped(normalized_positions))
    sorted_positions = xp.take(normalized_positions, order)
    sorted_values = xp.take(_asarray_preserving_trace(fp), order)

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
        None if xp_tangent is None else xp.take(_asarray_preserving_trace(xp_tangent), order)
    )
    sorted_fp_tangent = (
        None if fp_tangent is None else xp.take(_asarray_preserving_trace(fp_tangent), order)
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
    ``take`` at concrete indices scaled by a coefficient built from the
    primals; reverse mode scatters through ``take``'s registered transpose.
    Keeping those coefficients traced is what preserves nested differentiation.
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
    clamp_index = xp.where(below, 0, sample_count - 1)

    total = _zeros_output_tangent(ans, tangents)

    if sample_count < _MIN_SAMPLES_FOR_SLOPE:
        # A single sample has no interval: every resolved query returns fp[0].
        if fp_tangent is not None:
            resolved = xp.logical_or(clamped, xp.logical_not(xp.logical_or(below, above)))
            gathered = xp.take(_asarray_preserving_trace(fp_tangent), clamp_index)
            total = total + _weights(resolved, real_dtype) * gathered
        return cast("xp.ndarray[Any, Any]", _asarray_preserving_trace(total, dtype=dtype))

    interior = _weights(xp.logical_not(xp.logical_or(below, above)), real_dtype)
    clamp_mask = _weights(clamped, real_dtype)

    # ``side="right" - 1`` puts an exact sample hit on its own left bracket, so
    # an offset of zero reproduces fp[k] the way NumPy does.
    lower = xp.searchsorted(sample_positions, x_values, side="right") - 1
    lower = xp.minimum(xp.maximum(lower, 0), sample_count - 2)
    upper = lower + 1

    x_source = _asarray_preserving_trace(x)
    positions_source = _asarray_preserving_trace(xp_points)
    values_source = _asarray_preserving_trace(fp)

    lower_position = xp.take(positions_source, lower)
    gap = xp.take(positions_source, upper) - lower_position
    offset = (x_source - lower_position) / gap
    slope = (xp.take(values_source, upper) - xp.take(values_source, lower)) / gap

    if x_tangent is not None:
        total = total + interior * slope * _asarray_preserving_trace(x_tangent)

    if fp_tangent is not None:
        tangent_values = _asarray_preserving_trace(fp_tangent)
        total = total + interior * (
            (1 - offset) * xp.take(tangent_values, lower) + offset * xp.take(tangent_values, upper)
        )
        total = total + clamp_mask * xp.take(tangent_values, clamp_index)

    if xp_tangent is not None:
        tangent_positions = _asarray_preserving_trace(xp_tangent)
        total = total + interior * slope * (
            (offset - 1) * xp.take(tangent_positions, lower)
            - offset * xp.take(tangent_positions, upper)
        )

    return cast("xp.ndarray[Any, Any]", _asarray_preserving_trace(total, dtype=dtype))
