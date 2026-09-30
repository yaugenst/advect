"""Shared helpers for linalg VJP exception rules."""

from __future__ import annotations

from typing import Any, cast

from advect.autodiff.rules.array_family._backend_runtime import (
    _array_constructor_like,
    _scalar_like,
    xp,
)
from advect.autodiff.rules.array_family._transpose_utils import (
    _conjugate_transpose as _h,
    _dtype_of,
    _normalize_axis,
)

_MIN_MATRIX_NDIM = 2

_TENSORDOT_AXES_ARITY = 2


def _merge_multioutput_cotangent(
    cotangent: Any,
    *,
    output_count: int,
    op_name: str,
) -> tuple[Any | None, ...]:
    """Merge the sparse tuples contributed by separate getoutput nodes."""
    if not isinstance(cotangent, tuple) or len(cotangent) % output_count != 0:
        msg = f"{op_name} expects cotangents in {output_count}-slot groups"
        raise TypeError(msg)

    merged: list[Any | None] = [None] * output_count
    for offset in range(0, len(cotangent), output_count):
        for index, contribution in enumerate(cotangent[offset : offset + output_count]):
            if contribution is None:
                continue
            previous = merged[index]
            merged[index] = contribution if previous is None else previous + contribution
    return tuple(merged)


def _hermitian_triangle_adjoint(x: xp.ndarray, *, uplo: str) -> xp.ndarray:
    """Transpose the map from one stored triangle to a Hermitian matrix."""
    n = int(x.shape[-1])
    eye = _array_constructor_like(x, "eye", n, dtype=_dtype_of(x))
    symmetrized = x + _h(x)
    if uplo == "L":
        off_diagonal = xp.tril(symmetrized, k=-1)
    elif uplo == "U":
        off_diagonal = xp.triu(symmetrized, k=1)
    else:
        msg = f"expected UPLO='L' or 'U', got {uplo!r}"
        raise ValueError(msg)
    diagonal = xp.real(xp.diagonal(x, axis1=-2, axis2=-1))
    return cast("xp.ndarray", off_diagonal + eye * diagonal[..., None, :])


def _broadcast_eye(
    like: object,
    *,
    n: int,
    batch_ndim: int,
    dtype: xp.dtype[Any],
) -> xp.ndarray:
    """Return a batch-broadcastable identity in the trace of ``like``."""
    eye = _array_constructor_like(like, "eye", n, dtype=dtype)
    return xp.reshape(eye, (1,) * batch_ndim + (n, n))


def _qr_skew_pullback(
    bar_omega: xp.ndarray,
    *,
    batch_dims: tuple[int, ...],
    n: int,
) -> xp.ndarray:
    eye_n = _broadcast_eye(
        bar_omega,
        n=n,
        batch_ndim=len(batch_dims),
        dtype=xp.dtype(bar_omega.dtype),
    )
    bar_c = bar_omega * eye_n
    return cast(
        "xp.ndarray",
        xp.tril(bar_omega - _h(bar_omega), k=-1)
        + (bar_c - xp.conj(bar_c)) / _scalar_like(2.0, bar_c),
    )


def _normalize_axis_sequence(axes: Any, *, ndim: int, op_name: str) -> tuple[int, ...]:
    if isinstance(axes, (tuple, list)):
        raw_axes = tuple(int(axis) for axis in axes)
    else:
        raw_axes = (int(axes),)
    normalized = tuple(_normalize_axis(axis, ndim=ndim, op_name=op_name) for axis in raw_axes)
    if len(set(normalized)) != len(normalized):
        msg = f"{op_name} received duplicate axes: {raw_axes}"
        raise ValueError(msg)
    return normalized


def _normalize_tensordot_axes(
    *,
    axes: int | tuple[Any, Any],
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    op_name: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    a_ndim = len(a_shape)
    b_ndim = len(b_shape)
    if isinstance(axes, tuple):
        if len(axes) != _TENSORDOT_AXES_ARITY:
            msg = f"{op_name} axes must be an int or a 2-tuple, got {axes!r}"
            raise TypeError(msg)
        a_axes = _normalize_axis_sequence(axes[0], ndim=a_ndim, op_name=op_name)
        b_axes = _normalize_axis_sequence(axes[1], ndim=b_ndim, op_name=op_name)
    else:
        axes_int = axes
        if axes_int < 0:
            msg = f"{op_name} does not support negative integer axes ({axes_int})"
            raise ValueError(msg)
        if axes_int > min(a_ndim, b_ndim):
            msg = f"{op_name} axes={axes_int} exceeds input ranks {a_ndim} and {b_ndim}"
            raise ValueError(msg)
        a_axes = tuple(range(a_ndim - axes_int, a_ndim))
        b_axes = tuple(range(axes_int))
    if len(a_axes) != len(b_axes):
        msg = f"{op_name} axes lengths must match: {a_axes} vs {b_axes}"
        raise ValueError(msg)
    for axis_a, axis_b in zip(a_axes, b_axes, strict=True):
        if a_shape[axis_a] != b_shape[axis_b]:
            msg = (
                f"{op_name} contraction mismatch: "
                f"a.shape[{axis_a}]={a_shape[axis_a]} != b.shape[{axis_b}]={b_shape[axis_b]}"
            )
            raise ValueError(msg)
    return a_axes, b_axes
