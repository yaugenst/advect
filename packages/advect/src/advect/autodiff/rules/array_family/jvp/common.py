"""Shared JVP helpers for canonical array-family derivative rules."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast

from advect.autodiff.rules.array_family._backend_runtime import (
    _array_constructor_like,
    _scalar_like,
    _supports_cumulative_prod,
    xp,
)
from advect.autodiff.rules.array_family._transpose_utils import (
    _normalize_axis,
    infer_tangent_dtype as _infer_tangent_dtype,
    zeros_output_tangent as _zeros_output_tangent,
)
from advect.core._protocols import _innermost, _is_traced

if TYPE_CHECKING:
    from advect.autodiff.rules.array_family.jvp.elementwise_partials import (
        ElementwisePartials,
    )

_JVPFn = Callable[..., Any]

SortKind = Literal[
    "Q",
    "quick",
    "quicksort",
    "M",
    "merge",
    "mergesort",
    "H",
    "heap",
    "heapsort",
    "S",
    "stable",
    "stablesort",
]

PartitionKind = Literal["introselect"]

_WHERE_INPUT_ARITY = 3

_MATRIX_AXIS_COUNT = 2


def _asarray_unwrapped(value: Any) -> xp.ndarray[Any, Any]:
    return xp.asarray(_innermost(value))


def _shape_unwrapped(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in _asarray_unwrapped(value).shape)


def _ndim_unwrapped(value: Any) -> int:
    return len(_shape_unwrapped(value))


def _iscomplex_unwrapped(value: Any) -> bool:
    return xp.iscomplexobj(_asarray_unwrapped(value))


def _coerce_tangent_or_zeros(
    tangent: Any | None,
    *,
    primal: Any,
    dtype: xp.dtype[Any],
) -> Any:
    if tangent is None:
        primal_value = _innermost(primal)
        return xp.zeros_like(xp.asarray(primal_value), dtype=dtype)
    if _is_traced(tangent):
        return tangent
    return xp.asarray(tangent, dtype=dtype)


def _astype_preserving_trace(value: Any, *, dtype: xp.dtype[Any]) -> Any:
    value_dtype = getattr(value, "dtype", None)
    if value_dtype is dtype or (value_dtype is not None and value_dtype == dtype):
        return value

    target_dtype = xp.dtype(dtype)
    if value_dtype is not None and value_dtype == target_dtype:
        return value
    if _is_traced(value):
        if (
            value_dtype is not None
            and xp.issubdtype(
                value_dtype,
                xp.complexfloating,
            )
            and not xp.issubdtype(target_dtype, xp.complexfloating)
        ):
            value = xp.real(value)
        return cast("Any", value).astype(target_dtype)
    value_arr = xp.asarray(value)
    if xp.iscomplexobj(value_arr) and not xp.issubdtype(target_dtype, xp.complexfloating):
        value_arr = xp.real(value_arr)
    return xp.asarray(value_arr, dtype=target_dtype)


def _asarray_preserving_trace(
    value: Any,
    *,
    dtype: xp.dtype[Any] | None = None,
) -> Any:
    """Coerce concrete values without detaching an active tangent tracer."""
    if dtype is not None:
        return _astype_preserving_trace(value, dtype=dtype)
    if _is_traced(value):
        return value
    return xp.asarray(value)


def _normalize_output_tangent(
    ans: Any,
    tangents: tuple[Any | None, ...],
    contribution: Any,
) -> xp.ndarray[Any, Any]:
    """Cast and broadcast a local tangent to the primal output contract."""
    # Most calls need neither: a contribution that already has the answer's
    # dtype and shape, from tangents of that same dtype, is its own result.
    dtype = getattr(ans, "dtype", None)
    shape = getattr(ans, "shape", None)
    if (
        dtype is not None
        and shape is not None
        and getattr(contribution, "dtype", None) == dtype
        and getattr(contribution, "shape", None) == shape
        and all(tangent is None or getattr(tangent, "dtype", None) == dtype for tangent in tangents)
    ):
        return cast("xp.ndarray[Any, Any]", contribution)
    answer = _asarray_unwrapped(ans)
    contribution_value = _asarray_unwrapped(contribution)
    target_dtype = _infer_tangent_dtype(ans, tangents)

    result = contribution
    if contribution_value.dtype != target_dtype:
        result = _astype_preserving_trace(result, dtype=target_dtype)
    if contribution_value.shape != answer.shape:
        result = xp.broadcast_to(result, answer.shape)
    return cast("xp.ndarray[Any, Any]", result)


def _copy_if_untraced_array(value: Any) -> Any:
    if _is_traced(value):
        return value
    return xp.asarray(value, copy=True)


def _validate_tangent_arity(
    *,
    op_name: str,
    inputs: tuple[Any, ...],
    tangents: tuple[Any | None, ...],
) -> None:
    if len(inputs) != len(tangents):
        msg = f"{op_name} JVP tangent arity mismatch: expected {len(inputs)}, got {len(tangents)}"
        raise RuntimeError(msg)


def linear_jvp(apply: Callable[..., Any]) -> _JVPFn:
    """Make ``apply(tangent, **attrs)`` the JVP of a one-input linear operation.

    The tape calls a JVP only when an operand's tangent is active, so the one
    tangent of a single-input rule is never ``None``.
    """

    def jvp(
        ans: Any,
        *inputs: Any,
        tangents: tuple[Any | None, ...],
        **attrs: Any,
    ) -> Any:
        del ans, inputs
        return apply(_asarray_preserving_trace(tangents[0]), **attrs)

    jvp.__name__ = jvp.__qualname__ = apply.__name__
    jvp.__doc__ = apply.__doc__
    return jvp


def product_rule(ans: Any, tangents: tuple[Any | None, ...], *terms: Callable[[Any], Any]) -> Any:
    """Sum ``term(tangent)`` over the active tangents of a multilinear operation.

    Each term applies the operation with one operand replaced by its tangent.
    The tape calls a JVP only when some tangent is active, so the sum is never
    empty; it is cast and broadcast to the output tangent contract.
    """
    out: Any = None
    for tangent, term in zip(tangents, terms, strict=False):
        if tangent is not None:
            contribution = term(_asarray_preserving_trace(tangent))
            out = contribution if out is None else xp.add(out, contribution)
    return _normalize_output_tangent(ans, tangents, out)


def scale_by_constant(value: Any, constant: float, like: Any) -> Any:
    """Multiply ``value`` by a constant partial in the dtype of ``like``."""
    if constant == 1:
        return value
    if constant == -1:
        return -value
    if type(constant) in {int, float}:
        # A weak Python scalar takes value's dtype through the operator in every
        # Array API revision, and staging reuses its one constant node instead
        # of recording a fresh literal and cast for each rule application.
        return value * constant
    return xp.multiply(_scalar_like(constant, like), value)


def make_diagonal_jvp_from_partials(
    op_name: str,
    entry: ElementwisePartials,
) -> _JVPFn:
    """Create the JVP ``sum_i p_i * t_i`` of one diagonal elementwise operation."""
    arity = len(entry.partials)

    def jvp(
        ans: Any,
        *inputs: Any,
        tangents: tuple[Any | None, ...],
        **attrs: Any,
    ) -> Any:
        del attrs
        if len(inputs) != arity or len(tangents) != arity:
            msg = (
                f"{op_name} JVP expected {arity} operands and tangents, "
                f"got {len(inputs)} and {len(tangents)}"
            )
            raise RuntimeError(msg)
        out: Any | None = None
        at = entry.bind(ans, inputs)
        for index, tangent in enumerate(tangents):
            local = None if tangent is None else at(index)
            if tangent is None or local is None:
                continue
            term = (
                scale_by_constant(tangent, local, ans)
                if isinstance(local, (int, float))
                else xp.multiply(local, tangent)
            )
            out = term if out is None else xp.add(out, term)
        if out is None:
            return _zeros_output_tangent(ans, tangents)
        return out if arity == 1 else _normalize_output_tangent(ans, tangents, out)

    jvp.__name__ = f"_jvp_{op_name.replace('.', '_')}"
    return jvp


def _normalize_axis_tuple(
    axis: int | tuple[int, ...] | None,
    *,
    ndim: int,
) -> tuple[int, ...]:
    """Map a reduction axis to distinct in-bounds axes; ``None`` selects every axis."""
    if axis is None:
        return tuple(range(ndim))
    axis_items = axis if isinstance(axis, (tuple, list)) else (axis,)
    normalized = tuple(_normalize_axis(item, ndim=ndim, op_name="reduction") for item in axis_items)
    if len(set(normalized)) != len(normalized):
        msg = f"reduction axis {axis!r} repeats an axis"
        raise ValueError(msg)
    return normalized


def _flatten_reduction_axes(
    value: Any,
    *,
    axes: tuple[int, ...],
) -> tuple[Any, tuple[int, ...]]:
    keep_axes = tuple(index for index in range(value.ndim) if index not in set(axes))
    perm = keep_axes + axes
    transposed = xp.transpose(value, perm)
    reduce_size = math.prod(value.shape[index] for index in axes)
    reshaped = xp.reshape(transposed, (*transposed.shape[: len(keep_axes)], reduce_size))
    return reshaped, keep_axes


def _reshape_reduction_result(
    reduced: Any,
    *,
    input_shape: tuple[int, ...],
    axes: tuple[int, ...],
    keepdims: bool,
) -> Any:
    keep_axes = tuple(index for index in range(len(input_shape)) if index not in set(axes))
    keep_shape = tuple(input_shape[index] for index in keep_axes)
    reduced_view = xp.reshape(reduced, keep_shape)
    if not keepdims:
        return reduced_view
    out_shape: list[int] = []
    keep_iter = iter(keep_shape)
    reduced_axes = set(axes)
    for index in range(len(input_shape)):
        if index in reduced_axes:
            out_shape.append(1)
        else:
            out_shape.append(next(keep_iter))
    return xp.reshape(reduced_view, tuple(out_shape))


def _normalize_cumulative_axis(axis: object, *, ndim: int) -> int | None:
    if axis is None:
        return None
    if not isinstance(axis, int) and hasattr(axis, "__index__"):
        axis = int(cast("Any", axis))
    if not isinstance(axis, int):
        msg = "Cumulative JVPs require axis to be an integer or None"
        raise NotImplementedError(msg)
    return axis if axis >= 0 else axis + ndim


def _prod_jvp_last_axis(x: Any, dx: Any) -> Any:
    # Keep the linear map in its declared tangent dtype. Widening only the JVP
    # makes its structurally transposed cotangent round at the primal boundary,
    # so the two maps cease to be adjoints in low precision.
    dtype = xp.result_type(x, dx)
    x_work = _asarray_preserving_trace(x, dtype=dtype)
    dx_work = _asarray_preserving_trace(dx, dtype=dtype)
    n = _shape_unwrapped(x_work)[-1]
    if n == 0:
        return xp.sum(dx_work, axis=-1)

    ones = xp.ones_like(x_work[..., :1], dtype=dtype)
    if n == 1:
        partials = ones
    elif _supports_cumulative_prod():
        prefix = xp.concatenate(
            (ones, xp.cumprod(x_work[..., :-1], axis=-1)),
            axis=-1,
        )
        suffix_tail = xp.flip(
            xp.cumprod(xp.flip(x_work[..., 1:], axis=-1), axis=-1),
            axis=-1,
        )
        suffix = xp.concatenate((suffix_tail, ones), axis=-1)
        partials = prefix * suffix
    else:
        partials = xp.concatenate(
            tuple(
                xp.prod(
                    xp.concatenate((x_work[..., :index], x_work[..., index + 1 :]), axis=-1),
                    axis=-1,
                    keepdims=True,
                )
                for index in range(n)
            ),
            axis=-1,
        )

    return xp.sum(xp.multiply(partials, dx_work), axis=-1)


def _maxmin_tangent(
    x: Any,
    dx: Any,
    *,
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
    reduce_kind: Literal["max", "min"],
    ignore_nan: bool = False,
) -> Any:
    """Select the tangent of the reduction winner; NaN loses under ``ignore_nan``."""
    x_arr = _asarray_preserving_trace(x)
    axes = _normalize_axis_tuple(axis, ndim=x_arr.ndim)
    x_flat, _ = _flatten_reduction_axes(x_arr, axes=axes)
    dx_flat, _ = _flatten_reduction_axes(dx, axes=axes)

    valid = xp.logical_not(xp.isnan(x_flat)) if ignore_nan else None
    candidate = x_flat
    if valid is not None:
        fill_value = -float("inf") if reduce_kind == "max" else float("inf")
        candidate = xp.where(valid, x_flat, xp.full_like(x_flat, fill_value))
    winner = (
        xp.argmax(candidate, axis=-1) if reduce_kind == "max" else xp.argmin(candidate, axis=-1)
    )
    winner_mask = xp.equal(
        _array_constructor_like(dx_flat, "arange", x_flat.shape[-1], dtype=xp.int64),
        winner[..., None],
    )
    gathered = xp.sum(
        xp.where(winner_mask, dx_flat, xp.zeros_like(dx_flat)),
        axis=-1,
    )
    if valid is not None:
        gathered = xp.where(xp.any(valid, axis=-1), gathered, xp.zeros_like(gathered))
    return _reshape_reduction_result(
        gathered,
        input_shape=x_arr.shape,
        axes=axes,
        keepdims=keepdims,
    )
