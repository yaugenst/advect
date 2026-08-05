"""Shared JVP helpers for canonical array-family derivative rules."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal, cast

from advect.autodiff.rules.array_family._backend_runtime import (
    _array_constructor_like,
    _scalar_like,
    _supports_cumulative_prod,
    xp,
)
from advect.autodiff.rules.array_family._transpose_utils import (
    infer_output_tangent_dtype as _transpose_infer_output_tangent_dtype,
    infer_tangent_dtype as _transpose_infer_tangent_dtype,
    zeros_output_tangent as _transpose_zeros_output_tangent,
    zeros_output_tangent_structure as _transpose_zeros_output_tangent_structure,
)
from advect.core._protocols import _snapshot_traced

_JVPReturn = xp.ndarray | tuple[xp.ndarray, ...]

_JVPFn = Callable[..., _JVPReturn]

_ElementwisePartialsFn = Callable[..., tuple[Any | None, ...]]

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

FFTNorm = Literal["backward", "ortho", "forward"]

_WHERE_INPUT_ARITY = 3

_INTERP_FP_TANGENT_INDEX = 2

_VECTOR_AXIS_COUNT = 1

_MATRIX_AXIS_COUNT = 2

_L2_NORM_ORD = 2


def _infer_output_tangent_dtype(ans: Any, tangents: tuple[Any | None, ...]) -> xp.dtype[Any]:
    return _transpose_infer_output_tangent_dtype(ans, tangents)


def _zeros_output_tangent_structure(
    ans: Any,
    tangents: tuple[Any | None, ...],
) -> tuple[xp.ndarray[Any, Any], ...] | xp.ndarray[Any, Any]:
    return cast(
        "tuple[xp.ndarray[Any, Any], ...] | xp.ndarray[Any, Any]",
        _transpose_zeros_output_tangent_structure(ans, tangents),
    )


def _infer_tangent_dtype(ans: Any, tangents: tuple[Any | None, ...]) -> xp.dtype[Any]:
    return _transpose_infer_tangent_dtype(ans, tangents)


def _zeros_output_tangent(ans: Any, tangents: tuple[Any | None, ...]) -> xp.ndarray:
    return _transpose_zeros_output_tangent(ans, tangents)


def _is_traced_leaf(value: Any) -> bool:
    return callable(getattr(value, "_advect_snapshot", None))


def _unwrap_traced_leaf(value: Any) -> Any:
    current = value
    while _is_traced_leaf(current):
        _node_id, next_value = _snapshot_traced(current)
        if next_value is current:
            break
        current = next_value
    return current


def _asarray_unwrapped(value: Any) -> xp.ndarray[Any, Any]:
    return xp.asarray(_unwrap_traced_leaf(value))


def _shape_unwrapped(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in _asarray_unwrapped(value).shape)


def _ndim_unwrapped(value: Any) -> int:
    return len(_shape_unwrapped(value))


def _iscomplex_unwrapped(value: Any) -> bool:
    return bool(xp.iscomplexobj(_asarray_unwrapped(value)))


def _positive_domain_mask(value: Any) -> xp.ndarray[Any, Any]:
    return xp.asarray(_asarray_unwrapped(value) > 0, dtype=xp.bool)


def _maximum_choice_mask(x: Any, y: Any) -> xp.ndarray[Any, Any]:
    return cast(
        "xp.ndarray[Any, Any]",
        _asarray_preserving_trace(x) >= _asarray_preserving_trace(y),
    )


def _minimum_choice_mask(x: Any, y: Any) -> xp.ndarray[Any, Any]:
    return cast(
        "xp.ndarray[Any, Any]",
        _asarray_preserving_trace(x) <= _asarray_preserving_trace(y),
    )


def _fmax_choice_mask(x: Any, y: Any) -> xp.ndarray[Any, Any]:
    x_arr = _asarray_preserving_trace(x)
    y_arr = _asarray_preserving_trace(y)
    x_nan = xp.isnan(x_arr)
    y_nan = xp.isnan(y_arr)
    choose_x = x_arr >= y_arr
    choose_x = xp.where(
        x_nan & xp.logical_not(y_nan),
        xp.zeros_like(choose_x, dtype=xp.bool),
        choose_x,
    )
    choose_x = xp.where(
        y_nan & xp.logical_not(x_nan),
        xp.ones_like(choose_x, dtype=xp.bool),
        choose_x,
    )
    return cast("xp.ndarray[Any, Any]", choose_x)


def _fmin_choice_mask(x: Any, y: Any) -> xp.ndarray[Any, Any]:
    x_arr = _asarray_preserving_trace(x)
    y_arr = _asarray_preserving_trace(y)
    x_nan = xp.isnan(x_arr)
    y_nan = xp.isnan(y_arr)
    choose_x = x_arr <= y_arr
    choose_x = xp.where(
        x_nan & xp.logical_not(y_nan),
        xp.zeros_like(choose_x, dtype=xp.bool),
        choose_x,
    )
    choose_x = xp.where(
        y_nan & xp.logical_not(x_nan),
        xp.ones_like(choose_x, dtype=xp.bool),
        choose_x,
    )
    return cast("xp.ndarray[Any, Any]", choose_x)


def _coerce_tangent_or_zeros(
    tangent: Any | None,
    *,
    primal: Any,
    dtype: xp.dtype[Any],
) -> Any:
    if tangent is None:
        primal_value = _unwrap_traced_leaf(primal)
        return xp.zeros_like(xp.asarray(primal_value), dtype=dtype)
    if _is_traced_leaf(tangent):
        return tangent
    return xp.asarray(tangent, dtype=dtype)


def _astype_preserving_trace(value: Any, *, dtype: xp.dtype[Any]) -> Any:
    value_dtype = getattr(value, "dtype", None)
    if value_dtype is dtype or (value_dtype is not None and value_dtype == dtype):
        return value

    target_dtype = xp.dtype(dtype)
    if value_dtype is not None and value_dtype == target_dtype:
        return value
    if _is_traced_leaf(value):
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
    if _is_traced_leaf(value):
        return value
    return xp.asarray(value)


def _normalize_output_tangent(
    ans: Any,
    tangents: tuple[Any | None, ...],
    contribution: Any,
    *,
    target_dtype: xp.dtype[Any] | None = None,
) -> xp.ndarray[Any, Any]:
    """Cast and broadcast a local tangent to the primal output contract."""
    answer = _asarray_unwrapped(ans)
    contribution_value = _asarray_unwrapped(contribution)
    if target_dtype is None:
        target_dtype = _infer_tangent_dtype(ans, tangents)

    result = contribution
    if contribution_value.dtype != target_dtype:
        result = _astype_preserving_trace(result, dtype=target_dtype)
    if contribution_value.shape != answer.shape:
        result = xp.broadcast_to(result, answer.shape)
    return cast("xp.ndarray[Any, Any]", result)


def _copy_if_untraced_array(value: Any) -> Any:
    if _is_traced_leaf(value):
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


def make_diagonal_jvp_from_partials(
    op_name: str,
    partials_fn: _ElementwisePartialsFn,
) -> _JVPFn:
    """Create a diagonal-style JVP from explicit local partial formulas."""

    def jvp(
        ans: Any,
        *inputs: Any,
        tangents: tuple[Any | None, ...],
        **attrs: Any,
    ) -> Any:
        _validate_tangent_arity(op_name=op_name, inputs=inputs, tangents=tangents)
        if len(inputs) == 1:
            tangent = tangents[0]
            if tangent is None:
                return _zeros_output_tangent(ans, tangents)
            partials = partials_fn(ans, inputs[0], **attrs)
            if len(partials) != 1:
                msg = (
                    f"{op_name} JVP partial arity mismatch: expected 1 partials, "
                    f"got {len(partials)}"
                )
                raise RuntimeError(msg)
            partial = partials[0]
            if partial is None:
                return _zeros_output_tangent(ans, tangents)
            if isinstance(partial, (bool, int, float, complex)):
                partial = _scalar_like(partial, ans)
            return partial * tangent

        if all(tangent is None for tangent in tangents):
            return _zeros_output_tangent(ans, tangents)

        partials = partials_fn(ans, *inputs, **attrs)
        if len(partials) != len(inputs):
            msg = (
                f"{op_name} JVP partial arity mismatch: expected {len(inputs)} partials, "
                f"got {len(partials)}"
            )
            raise RuntimeError(msg)

        out: Any | None = None
        for partial, tangent in zip(partials, tangents, strict=True):
            if partial is None or tangent is None:
                continue
            resolved_partial = (
                _scalar_like(partial, ans)
                if isinstance(partial, (bool, int, float, complex))
                else partial
            )
            term = resolved_partial * tangent
            out = term if out is None else out + term
        return (
            _zeros_output_tangent(ans, tangents)
            if out is None
            else _normalize_output_tangent(ans, tangents, out)
        )

    jvp.__name__ = f"_jvp_{op_name.replace('.', '_')}"
    return jvp


def _normalize_axis_tuple(
    axis: int | tuple[int, ...] | None,
    *,
    ndim: int,
) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    axis_items = (axis,) if isinstance(axis, int) else axis
    normalized = tuple(int(item) % ndim for item in axis_items)
    if len(set(normalized)) != len(normalized):
        msg = f"Repeated axes are not supported (axis={axis!r})"
        raise NotImplementedError(msg)
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

    return xp.sum(partials * dx_work, axis=-1)


def _maxmin_tangent(
    x: Any,
    dx: Any,
    *,
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
    reduce_kind: Literal["max", "min"],
) -> Any:
    x_arr = _asarray_preserving_trace(x)
    axes = _normalize_axis_tuple(axis, ndim=x_arr.ndim)
    x_flat, _ = _flatten_reduction_axes(x_arr, axes=axes)
    dx_flat, _ = _flatten_reduction_axes(dx, axes=axes)
    winner = xp.argmax(x_flat, axis=-1) if reduce_kind == "max" else xp.argmin(x_flat, axis=-1)
    winner_mask = xp.equal(
        _array_constructor_like(dx_flat, "arange", x_flat.shape[-1], dtype=xp.int64),
        winner[..., None],
    )
    gathered = xp.sum(
        xp.where(winner_mask, dx_flat, xp.zeros_like(dx_flat)),
        axis=-1,
    )
    return _reshape_reduction_result(
        gathered,
        input_shape=x_arr.shape,
        axes=axes,
        keepdims=keepdims,
    )


def _nan_maxmin_tangent(
    x: Any,
    dx: Any,
    *,
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
    reduce_kind: Literal["max", "min"],
) -> Any:
    x_arr = _asarray_preserving_trace(x)
    axes = _normalize_axis_tuple(axis, ndim=x_arr.ndim)
    x_flat, _ = _flatten_reduction_axes(x_arr, axes=axes)
    dx_flat, _ = _flatten_reduction_axes(dx, axes=axes)

    valid = xp.logical_not(xp.isnan(x_flat))
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
    has_valid = xp.any(valid, axis=-1)
    gathered = xp.where(has_valid, gathered, xp.zeros_like(gathered))
    return _reshape_reduction_result(
        gathered,
        input_shape=x_arr.shape,
        axes=axes,
        keepdims=keepdims,
    )
