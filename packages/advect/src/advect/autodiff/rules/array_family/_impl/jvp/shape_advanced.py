"""Advanced shape and ordering JVP rules."""

from __future__ import annotations

from typing import Any, Literal, cast

from advect.autodiff.rules.array_family._backend_runtime import _take_along_axis, xp
from advect.autodiff.rules.array_family._impl.jvp.common import (
    PartitionKind,
    SortKind,
    _asarray_preserving_trace,
    _asarray_unwrapped,
    _coerce_tangent_or_zeros,
    _infer_tangent_dtype,
    _shape_unwrapped,
    _zeros_output_tangent,
    _zeros_output_tangent_structure,
)


def _jvp_diagonal(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return cast(
        "xp.ndarray[Any, Any]",
        xp.diagonal(
            _asarray_preserving_trace(tangent),
            offset=int(offset),
            axis1=axis1,
            axis2=axis2,
        ),
    )


def _jvp_trace(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return cast(
        "xp.ndarray[Any, Any]",
        xp.trace(
            _asarray_preserving_trace(tangent),
            offset=int(offset),
            axis1=axis1,
            axis2=axis2,
        ),
    )


def _jvp_diag(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    k: int = 0,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return cast("xp.ndarray[Any, Any]", xp.diag(_asarray_preserving_trace(tangent), k=int(k)))


def _jvp_diff(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    n: int = 1,
    axis: int = -1,
    prepend: object | None = None,
    append: object | None = None,
    **attrs: Any,
) -> xp.ndarray:
    prepend_is_input = bool(attrs.get("_advect_diff_prepend_input", False))
    append_is_input = bool(attrs.get("_advect_diff_append_input", False))
    if not any(tangent is not None for tangent in tangents):
        return _zeros_output_tangent(ans, tangents)
    dtype = _infer_tangent_dtype(ans, tangents)
    tangent_arr = _coerce_tangent_or_zeros(
        tangents[0] if tangents else None,
        primal=x,
        dtype=dtype,
    )
    if int(n) == 0:
        return cast("xp.ndarray[Any, Any]", tangent_arr)

    axis_norm = int(axis)
    x_shape = _shape_unwrapped(x)
    if axis_norm < 0:
        axis_norm += len(x_shape)
    boundary_shape = list(x_shape)
    boundary_shape[axis_norm] = 1

    def boundary_tangent(value: xp.ndarray) -> xp.ndarray:
        if len(_shape_unwrapped(value)) == 0:
            return cast("xp.ndarray", xp.broadcast_to(value, tuple(boundary_shape)))
        return value

    parts: list[xp.ndarray] = []
    cursor = 0
    if prepend_is_input:
        prepend_primal = rest[cursor]
        cursor += 1
        parts.append(
            boundary_tangent(
                _coerce_tangent_or_zeros(
                    tangents[cursor] if len(tangents) > cursor else None,
                    primal=prepend_primal,
                    dtype=dtype,
                )
            )
        )
    elif prepend is not None:
        parts.append(boundary_tangent(xp.zeros_like(xp.asarray(prepend), dtype=dtype)))
    parts.append(tangent_arr)
    if append_is_input:
        append_primal = rest[cursor]
        cursor += 1
        parts.append(
            boundary_tangent(
                _coerce_tangent_or_zeros(
                    tangents[cursor] if len(tangents) > cursor else None,
                    primal=append_primal,
                    dtype=dtype,
                )
            )
        )
    elif append is not None:
        parts.append(boundary_tangent(xp.zeros_like(xp.asarray(append), dtype=dtype)))
    joined = tangent_arr if len(parts) == 1 else xp.concatenate(parts, axis=axis_norm)
    return cast("xp.ndarray[Any, Any]", xp.diff(joined, n=int(n), axis=axis_norm))


def _jvp_repeat(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    repeats: int = 1,
    axis: int | None = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return cast(
        "xp.ndarray[Any, Any]",
        xp.repeat(_asarray_preserving_trace(tangent), int(repeats), axis=axis),
    )


def _jvp_tile(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    reps: int | tuple[int, ...] = 1,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return cast(
        "xp.ndarray[Any, Any]",
        xp.tile(_asarray_preserving_trace(tangent), reps),
    )


def _jvp_sort(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axis: int = -1,
    kind: SortKind = "quicksort",
    order: Any = None,
    descending: bool | None = None,
    stable: bool | None = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    x_arr = _asarray_unwrapped(x)
    axis_norm = int(axis)
    if axis_norm < 0:
        axis_norm += x_arr.ndim
    if descending is not None or stable is not None:
        perm = cast("Any", xp.argsort)(
            x_arr,
            axis=axis_norm,
            descending=bool(descending),
            stable=True if stable is None else bool(stable),
        )
    else:
        perm = xp.argsort(x_arr, axis=axis_norm, kind=kind, order=order)
    return _take_along_axis(tangent, perm, axis=axis_norm)


def _jvp_partition(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    kth: int | tuple[int, ...] = 0,
    axis: int = -1,
    kind: PartitionKind = "introselect",
    order: Any = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    x_arr = _asarray_unwrapped(x)
    axis_norm = int(axis)
    if axis_norm < 0:
        axis_norm += x_arr.ndim
    perm = xp.argpartition(x_arr, kth=kth, axis=axis_norm, kind=kind, order=order)
    return _take_along_axis(tangent, perm, axis=axis_norm)


def _jvp_pad(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    pad_width: tuple[tuple[int, int], ...] | tuple[int, int] | int = 0,
    mode: str = "constant",
    constant_values: Any = 0,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, x, rest, attrs
    _ = constant_values
    if mode != "constant":
        msg = f"numpy.pad JVP only supports mode='constant' (got {mode!r})"
        raise NotImplementedError(msg)
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    pad_mode = cast("Literal['constant']", mode)
    return cast(
        "xp.ndarray[Any, Any]",
        xp.pad(
            _asarray_preserving_trace(tangent),
            pad_width,
            mode=pad_mode,
            constant_values=0,
        ),
    )


def _jvp_gradient(
    ans: tuple[xp.ndarray, ...] | xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axis: int | tuple[int, ...] | None = None,
    edge_order: Literal[1, 2] = 1,
    **attrs: Any,
) -> tuple[xp.ndarray, ...] | xp.ndarray:
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return cast(
            "tuple[xp.ndarray, ...] | xp.ndarray", _zeros_output_tangent_structure(ans, tangents)
        )
    return cast(
        "tuple[xp.ndarray, ...] | xp.ndarray",
        xp.gradient(tangent, axis=axis, edge_order=edge_order),
    )
