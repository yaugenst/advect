"""Advanced shape and ordering JVP rules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from advect.autodiff.rules.array_family._backend_runtime import _take_along_axis, xp
from advect.autodiff.rules.array_family._transpose_utils import (
    _axis_slice,
    _difference,
    _tile_layout,
)
from advect.autodiff.rules.array_family.jvp.common import (
    _asarray_unwrapped,
    _coerce_tangent_or_zeros,
    _infer_tangent_dtype,
    _shape_unwrapped,
    linear_jvp,
)

if TYPE_CHECKING:
    from advect.autodiff.rules.array_family.jvp.common import PartitionKind, SortKind


@linear_jvp
def _jvp_diagonal(t: Any, offset: int = 0, axis1: int = 0, axis2: int = 1, **_: Any) -> Any:
    return xp.diagonal(t, offset=offset, axis1=axis1, axis2=axis2)


@linear_jvp
def _jvp_trace(
    t: Any, offset: int = 0, axis1: int = 0, axis2: int = 1, dtype: Any = None, **_: Any
) -> Any:
    return xp.trace(t, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype)


@linear_jvp
def _jvp_diag(t: Any, k: int = 0, **_: Any) -> Any:
    return xp.diag(t, k=k)


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
    dtype = _infer_tangent_dtype(ans, tangents)
    tangent_arr = _coerce_tangent_or_zeros(
        tangents[0] if tangents else None,
        primal=x,
        dtype=dtype,
    )
    if n == 0:
        return cast("xp.ndarray[Any, Any]", tangent_arr)

    axis_norm = axis
    x_shape = _shape_unwrapped(x)
    if axis_norm < 0:
        axis_norm += len(x_shape)
    boundary_shape = list(x_shape)
    boundary_shape[axis_norm] = 1

    def boundary_tangent(value: xp.ndarray) -> xp.ndarray:
        if len(_shape_unwrapped(value)) == 0:
            return xp.broadcast_to(value, tuple(boundary_shape))
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
    return cast("xp.ndarray[Any, Any]", _difference(joined, order=n, axis=axis_norm))


@linear_jvp
def _jvp_repeat(t: Any, repeats: int = 1, axis: int | None = None, **_: Any) -> Any:
    # Broadcast a new axis: repeat entered the Array API in 2023.12. NumPy
    # repeats a rank-0 source along axis 0 or -1 as its one-element flattening.
    if axis is None or not _shape_unwrapped(t):
        t, axis = xp.reshape(t, (-1,)), 0
    shape = _shape_unwrapped(t)
    axis %= len(shape)
    copies = xp.broadcast_to(
        xp.expand_dims(t, axis=axis + 1),
        (*shape[: axis + 1], repeats, *shape[axis + 1 :]),
    )
    return xp.reshape(copies, (*shape[:axis], shape[axis] * repeats, *shape[axis + 1 :]))


@linear_jvp
def _jvp_tile(t: Any, reps: int | tuple[int, ...] = 1, **_: Any) -> Any:
    # Broadcast interleaved axes: tile entered the Array API in 2023.12.
    source, pairs = _tile_layout(_shape_unwrapped(t), reps)
    single = xp.reshape(t, tuple(extent for size in source for extent in (1, size)))
    copies = xp.broadcast_to(single, pairs)
    return xp.reshape(copies, tuple(c * size for c, size in zip(pairs[::2], source, strict=True)))


def _jvp_sort(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray, ...],
    axis: int = -1,
    kind: SortKind = "quicksort",
    order: Any = None,
    descending: bool | None = None,
    stable: bool | None = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, rest, attrs
    tangent = tangents[0]
    x_arr = _asarray_unwrapped(x)
    axis_norm = axis
    if axis_norm < 0:
        axis_norm += x_arr.ndim
    if descending is not None or stable is not None:
        # NumPy before 2.5 has no descending keyword; an ascending sort omits it.
        reverse: dict[str, bool] = {"descending": True} if descending else {}
        perm = cast("Any", xp.argsort)(
            x_arr,
            axis=axis_norm,
            stable=True if stable is None else stable,
            **reverse,
        )
    else:
        perm = xp.argsort(x_arr, axis=axis_norm, kind=kind, order=order)
    return _take_along_axis(tangent, perm, axis=axis_norm)


def _jvp_partition(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray, ...],
    kth: int | tuple[int, ...] = 0,
    axis: int = -1,
    kind: PartitionKind = "introselect",
    order: Any = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, rest, attrs
    tangent = tangents[0]
    x_arr = _asarray_unwrapped(x)
    axis_norm = axis
    if axis_norm < 0:
        axis_norm += x_arr.ndim
    perm = xp.argpartition(x_arr, kth=kth, axis=axis_norm, kind=kind, order=order)
    return _take_along_axis(tangent, perm, axis=axis_norm)


@linear_jvp
def _jvp_pad(
    t: Any,
    pad_width: tuple[tuple[int, int], ...] | tuple[int, int] | int = 0,
    mode: str = "constant",
    **_: Any,
) -> Any:
    if mode != "constant":
        msg = f"numpy.pad JVP only supports mode='constant' (got {mode!r})"
        raise NotImplementedError(msg)
    return xp.pad(t, pad_width, mode="constant", constant_values=0)


@linear_jvp
def _jvp_gradient(t: Any, axis: int = 0, edge_order: Literal[1, 2] = 1, **_: Any) -> Any:
    """Apply NumPy's unit-spacing stencil along one axis by slicing.

    ``gradient`` is not an Array API function. The frontend emits one node per
    gradient axis, and NumPy's evenly spaced formulas with a unit step leave
    only these constant weights.
    """
    rank = len(_shape_unwrapped(t))

    def along(start: int | None, stop: int | None) -> Any:
        return t[_axis_slice(ndim=rank, axis=axis, start=start, stop=stop)]

    interior = (along(2, None) - along(None, -2)) / 2.0
    if edge_order == 1:
        head = along(1, 2) - along(0, 1)
        tail = along(-1, None) - along(-2, -1)
    else:
        head = -1.5 * along(0, 1) + 2.0 * along(1, 2) + -0.5 * along(2, 3)
        tail = 0.5 * along(-3, -2) + -2.0 * along(-2, -1) + 1.5 * along(-1, None)
    return xp.concatenate((head, interior, tail), axis=axis)
