"""Indexing JVP rules."""

from __future__ import annotations

from typing import Any, cast

from advect.autodiff.rules.array_family._backend_runtime import (
    _zeros_in_trace,
    decode_array_index,
    xp,
)
from advect.autodiff.rules.array_family.jvp.common import (
    _asarray_preserving_trace,
    _asarray_unwrapped,
    _astype_preserving_trace,
    _zeros_output_tangent,
)
from advect.core._protocols import _is_traced
from advect.core._scatter_add import scatter_add


def _jvp_getitem(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray, ...],
    index: Any = None,
    **attrs: Any,
) -> xp.ndarray:
    """JVP for advect.getitem."""
    _ = ans, x, rest, attrs
    idx = cast("Any", decode_array_index(index))
    return cast(
        "xp.ndarray",
        cast("Any", _asarray_preserving_trace(tangents[0]))[idx],
    )


def _updatable_copy(value: Any) -> Any:
    """Copy a tangent before an in-place update without rebinding the caller's.

    NumPy arrays and their traces copy through ``copy()``; the Array API has
    no such method and copies through ``asarray``.
    """
    copy = getattr(value, "copy", None)
    return copy() if callable(copy) else xp.asarray(value, copy=True)


def _updatable_tangent(ans: Any, base_tangent: Any, replacement_tangent: Any, dtype: Any) -> Any:
    """Return a writable output tangent that can hold ``replacement_tangent``.

    Under nested forward mode the replacement tangent may belong to an outer
    trace that a constant base tangent does not: the update is then written
    into zeros of that trace, so the write stays traced.
    """
    base = None if base_tangent is None else _astype_preserving_trace(base_tangent, dtype=dtype)
    if _is_traced(replacement_tangent) and not _is_traced(base):
        shape = tuple(int(size) for size in _asarray_unwrapped(ans).shape)
        zeros = _zeros_in_trace(replacement_tangent, shape, dtype)
        return zeros if base is None else _updatable_copy(zeros + base)
    return _updatable_copy(xp.zeros_like(ans, dtype=dtype) if base is None else base)


def _jvp_index_update(
    ans: xp.ndarray,
    base: xp.ndarray,
    replacement: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    index: Any = None,
    mode: str = "set",
    **attrs: Any,
) -> xp.ndarray:
    """JVP for a pure basic-index set or additive update.

    Set mode overwrites the selected base tangent. Add mode preserves the base
    tangent and adds the replacement tangent at the selected index.
    """
    _ = base, replacement, rest, attrs
    base_tangent = tangents[0] if tangents else None
    replacement_tangent = tangents[1] if len(tangents) > 1 else None
    output_dtype = _asarray_unwrapped(ans).dtype
    idx = cast("Any", decode_array_index(index))
    if mode == "add":
        if replacement_tangent is None:
            if base_tangent is None:
                return xp.zeros_like(ans, dtype=output_dtype)
            return cast("xp.ndarray", _astype_preserving_trace(base_tangent, dtype=output_dtype))
        result = _updatable_tangent(ans, base_tangent, replacement_tangent, output_dtype)
        result[idx] += _astype_preserving_trace(
            replacement_tangent,
            dtype=output_dtype,
        )
        return cast("xp.ndarray", result)
    if mode != "set":
        msg = f"Unsupported index_update mode {mode!r}"
        raise ValueError(msg)

    result = _updatable_tangent(ans, base_tangent, replacement_tangent, output_dtype)
    if replacement_tangent is None:
        result[idx] = 0
    else:
        result[idx] = _astype_preserving_trace(
            replacement_tangent,
            dtype=output_dtype,
        )
    return cast("xp.ndarray", result)


def _jvp_scatter_add(
    ans: xp.ndarray,
    values: xp.ndarray,
    indices: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axis: int,
    size: int,
    **attrs: Any,
) -> xp.ndarray:
    """Scatter the values tangent to the same positions; the indices carry none."""
    _ = values, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    return cast("xp.ndarray", scatter_add(tangent, indices, axis=axis, size=size))
