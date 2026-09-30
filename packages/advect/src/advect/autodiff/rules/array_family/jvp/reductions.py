"""Reductions JVP rules."""

from __future__ import annotations

import math
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal, cast

from advect.autodiff.rules.array_family._backend_runtime import (
    _count_like,
    _moveaxis,
    _scalar_like,
    xp,
)
from advect.autodiff.rules.array_family.jvp.common import (
    _asarray_preserving_trace,
    _asarray_unwrapped,
    _astype_preserving_trace,
    _flatten_reduction_axes,
    _iscomplex_unwrapped,
    _maxmin_tangent,
    _normalize_axis_tuple,
    _normalize_cumulative_axis,
    _prod_jvp_last_axis,
    _reshape_reduction_result,
    _shape_unwrapped,
    linear_jvp,
)

if TYPE_CHECKING:
    from advect.autodiff.rules.array_family.jvp.common import _JVPFn


def _real_ddof(value: object, *, operation: str) -> float:
    if not isinstance(value, Real):
        msg = f"{operation} JVP requires a real-valued ddof"
        raise NotImplementedError(msg)
    return float(value)


def _valid_count(
    x: xp.ndarray,
    mask: xp.ndarray,
    *,
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
) -> xp.ndarray:
    """Count entries outside ``mask`` in a real floating dtype of ``x``.

    The mask enters only as a ``where`` condition, so an enclosing trace never
    differentiates the comparison that produced it. A floating count keeps a
    fractional ``ddof`` exact.
    """
    real = xp.real(x) if _iscomplex_unwrapped(x) else x
    zero, one = (_count_like(value, real) for value in (0, 1))
    return cast("xp.ndarray", xp.sum(xp.where(mask, zero, one), axis=axis, keepdims=keepdims))


def _nanmean_keepdims(
    x: xp.ndarray,
    mask: xp.ndarray,
    *,
    axis: int | tuple[int, ...] | None,
) -> xp.ndarray:
    valid_count = _valid_count(x, mask, axis=axis, keepdims=True)
    safe_count = xp.where(valid_count == 0, xp.ones_like(valid_count), valid_count)
    total = xp.sum(
        xp.where(mask, xp.zeros_like(x), x),
        axis=axis,
        keepdims=True,
    )
    return cast("xp.ndarray", total / safe_count)


def _linear_reduction(name: str) -> _JVPFn:
    """Return the JVP of ``sum`` or ``mean``: the same reduction of the tangent."""

    @linear_jvp
    def jvp(
        t: Any,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: object | None = None,
        keepdims: bool = False,
        **attrs: Any,
    ) -> Any:
        if any(attrs.get(control) is not None for control in ("out", "where")):
            msg = "reduction derivatives do not support where/out control operands"
            raise NotImplementedError(msg)
        typed: dict[str, object] = {} if dtype is None else {"dtype": dtype}
        return getattr(xp, name)(t, axis=axis, keepdims=keepdims, **typed)

    return jvp


_jvp_sum = _linear_reduction("sum")
_jvp_mean = _linear_reduction("mean")


@linear_jvp
def _jvp_cumsum(t: Any, axis: int | None = None, **_: Any) -> Any:
    # NumPy scans the flattened values of a missing axis or a rank-0 input; a
    # traced provider's cumulative_sum needs that vector spelled out.
    if axis is None or not _shape_unwrapped(t):
        t, axis = xp.reshape(t, (-1,)), 0
    return xp.cumsum(t, axis=axis)


def _extremum(kind: Literal["max", "min"], *, ignore_nan: bool) -> _JVPFn:
    """Return the JVP of a max or min reduction: the tangent of the winning entry."""
    reduction = f"nan{kind}" if ignore_nan else kind

    def jvp(
        ans: xp.ndarray,
        x: xp.ndarray,
        *rest: xp.ndarray,
        tangents: tuple[xp.ndarray, ...],
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        initial: object | None = None,
        **attrs: Any,
    ) -> xp.ndarray:
        _ = ans, rest, attrs
        result = _maxmin_tangent(
            x,
            tangents[0],
            axis=axis,
            keepdims=keepdims,
            reduce_kind=kind,
            ignore_nan=ignore_nan,
        )
        if initial is None:
            return cast("xp.ndarray", result)
        # Where the static initial wins, the result no longer depends on x.
        base = getattr(xp, reduction)(x, axis=axis, keepdims=keepdims)
        initial_wins = base < initial if kind == "max" else base > initial
        return cast("xp.ndarray", xp.where(initial_wins, xp.zeros_like(result), result))

    return jvp


_jvp_max = _extremum("max", ignore_nan=False)
_jvp_min = _extremum("min", ignore_nan=False)
_jvp_nanmax = _extremum("max", ignore_nan=True)
_jvp_nanmin = _extremum("min", ignore_nan=True)


def _jvp_cumprod(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray, ...],
    axis: object = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = rest, attrs
    tangent = tangents[0]
    ndim = len(_shape_unwrapped(x))
    # Like a missing axis, the axis of a rank-0 input scans its one flattened element.
    axis_norm = _normalize_cumulative_axis(axis, ndim=ndim) if ndim else None
    if axis_norm is None:
        x, tangent, axis_norm = xp.reshape(x, (-1,)), xp.reshape(tangent, (-1,)), 0
    x_last = _moveaxis(x, axis_norm, -1)
    dx_last = _moveaxis(tangent, axis_norm, -1)
    y_last = xp.cumprod(x_last, axis=-1)
    output_count = _shape_unwrapped(y_last)[-1]
    if output_count == 0:
        return xp.astype(tangent, xp.result_type(x, tangent, xp.float64)) * 0
    terms = [dx_last[..., 0]]
    for index in range(1, output_count):
        terms.append(
            xp.add(
                xp.multiply(terms[index - 1], x_last[..., index]),
                xp.multiply(y_last[..., index - 1], dx_last[..., index]),
            )
        )
    out_last = xp.stack(terms, axis=-1)
    return xp.reshape(_moveaxis(out_last, -1, axis_norm), _shape_unwrapped(ans))


def _jvp_nanmean(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray, ...],
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, rest, attrs
    tangent = tangents[0]
    mask = xp.isnan(_asarray_preserving_trace(x))
    dx_eff = xp.where(mask, xp.zeros_like(tangent), tangent)
    count = _valid_count(x, mask, axis=axis, keepdims=keepdims)
    safe_count = xp.where(count == 0, xp.ones_like(count), count)
    quotient = xp.sum(dx_eff, axis=axis, keepdims=keepdims) / safe_count
    out = xp.where(
        count == 0,
        xp.zeros_like(quotient),
        quotient,
    )
    return cast(
        "xp.ndarray[Any, Any]",
        _astype_preserving_trace(out, dtype=_asarray_unwrapped(ans).dtype),
    )


def _jvp_nansum(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray, ...],
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    dtype: Any = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, rest, attrs
    tangent = tangents[0]
    mask = xp.logical_not(xp.isnan(_asarray_preserving_trace(x)))
    out = xp.sum(
        xp.where(mask, tangent, xp.zeros_like(tangent)),
        axis=axis,
        keepdims=keepdims,
        dtype=dtype,
    )
    return cast(
        "xp.ndarray[Any, Any]",
        _astype_preserving_trace(out, dtype=_asarray_unwrapped(ans).dtype),
    )


def _product(*, ignore_nan: bool) -> _JVPFn:
    """Return the JVP of ``prod``, or of ``nanprod``, which multiplies NaNs as ones."""

    def jvp(
        ans: xp.ndarray,
        x: xp.ndarray,
        *rest: xp.ndarray,
        tangents: tuple[xp.ndarray, ...],
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        initial: object | None = None,
        **attrs: Any,
    ) -> xp.ndarray:
        _ = ans, rest, attrs
        tangent = tangents[0]
        if ignore_nan:
            mask = xp.isnan(_asarray_preserving_trace(x))
            x = xp.where(mask, xp.ones_like(x), x)
            tangent = xp.where(mask, xp.zeros_like(tangent), tangent)
        x_shape = _shape_unwrapped(x)
        axes = _normalize_axis_tuple(axis, ndim=len(x_shape))
        x_flat, _ = _flatten_reduction_axes(x, axes=axes)
        dx_flat, _ = _flatten_reduction_axes(tangent, axes=axes)
        result = _reshape_reduction_result(
            _prod_jvp_last_axis(x_flat, dx_flat),
            input_shape=x_shape,
            axes=axes,
            keepdims=keepdims,
        )
        return cast("xp.ndarray", result if initial is None else result * initial)

    return jvp


_jvp_prod = _product(ignore_nan=False)
_jvp_nanprod = _product(ignore_nan=True)


def _variance(operation: str, *, ignore_nan: bool, root: bool) -> _JVPFn:
    """Return the JVP of var, std, nanvar or nanstd.

    The shared numerator is ``sum(Re(conj(x - mean) * dx))`` over the reduced
    entries that count, so the variance derivative is twice it over the count
    less ``ddof``, and the standard deviation divides it by that count times
    the result, with a zero derivative where every deviation vanishes.
    """

    def jvp(
        ans: xp.ndarray,
        x: xp.ndarray,
        *rest: xp.ndarray,
        tangents: tuple[xp.ndarray, ...],
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        ddof: object = 0,
        **attrs: Any,
    ) -> xp.ndarray:
        _ = rest
        ddof_value = _real_ddof(attrs.get("correction", ddof), operation=operation)
        tangent = tangents[0]
        if ignore_nan:
            mask = xp.isnan(_asarray_preserving_trace(x))
            count = _valid_count(x, mask, axis=axis, keepdims=keepdims)
            denominator = count - _scalar_like(ddof_value, count)
            centered = x - _nanmean_keepdims(x, mask, axis=axis)
            centered = xp.where(mask, xp.zeros_like(centered), centered)
            tangent = xp.where(mask, xp.zeros_like(tangent), tangent)
        else:
            x_shape = _shape_unwrapped(x)
            axes = _normalize_axis_tuple(axis, ndim=len(x_shape))
            static_denominator = math.prod(x_shape[index] for index in axes) - ddof_value
            if static_denominator <= 0:
                msg = f"{operation} JVP requires count > ddof for reduced slices"
                raise NotImplementedError(msg)
            denominator = _count_like(static_denominator, ans)
            centered = x - xp.mean(x, axis=axis, keepdims=True)
        numerator = xp.sum(
            xp.real(xp.multiply(xp.conjugate(centered), tangent)),
            axis=axis,
            keepdims=keepdims,
        )
        if root:
            zero = xp.zeros_like(ans)
            safe_ans = xp.where(ans == zero, xp.ones_like(ans), ans)
            quotient = numerator / (denominator * safe_ans)
            out = xp.where(ans == zero, xp.zeros_like(quotient), quotient)
        else:
            out = xp.multiply(_scalar_like(2.0, denominator), numerator) / denominator
        return cast(
            "xp.ndarray",
            _astype_preserving_trace(out, dtype=_asarray_unwrapped(ans).dtype),
        )

    return jvp


_jvp_var = _variance("var", ignore_nan=False, root=False)
_jvp_std = _variance("std", ignore_nan=False, root=True)
_jvp_nanvar = _variance("nanvar", ignore_nan=True, root=False)
_jvp_nanstd = _variance("nanstd", ignore_nan=True, root=True)
