"""Indexing JVP rules."""

from __future__ import annotations

from typing import Any, cast

from advect.autodiff.rules.array_family._backend_runtime import (
    decode_array_index,
    xp,
)
from advect.autodiff.rules.array_family.jvp.common import (
    _asarray_preserving_trace,
    _asarray_unwrapped,
    _astype_preserving_trace,
    _zeros_output_tangent,
)


def _jvp_getitem(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    index: Any = None,
    **attrs: Any,
) -> xp.ndarray:
    """JVP for advect.getitem."""
    _ = ans, x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    idx = cast("Any", decode_array_index(index))
    return cast(
        "xp.ndarray",
        cast("Any", _asarray_preserving_trace(tangent))[idx],
    )


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
    if base_tangent is None and replacement_tangent is None:
        return _zeros_output_tangent(ans, tangents)

    output_dtype = _asarray_unwrapped(ans).dtype
    idx = cast("Any", decode_array_index(index))
    if mode == "add":
        if base_tangent is None:
            result = xp.zeros_like(ans, dtype=output_dtype)
        else:
            result = _astype_preserving_trace(base_tangent, dtype=output_dtype)
        if replacement_tangent is None:
            return cast("xp.ndarray", result)
        result = cast("Any", result).copy()
        result[idx] += _astype_preserving_trace(
            replacement_tangent,
            dtype=output_dtype,
        )
        return cast("xp.ndarray", result)
    if mode != "set":
        msg = f"Unsupported index_update mode {mode!r}"
        raise ValueError(msg)

    if base_tangent is None:
        result = xp.zeros_like(ans, dtype=output_dtype)
    else:
        result = _astype_preserving_trace(base_tangent, dtype=output_dtype)
    result = cast("Any", result).copy()

    if replacement_tangent is None:
        result[idx] = 0
    else:
        result[idx] = _astype_preserving_trace(
            replacement_tangent,
            dtype=output_dtype,
        )
    return cast("xp.ndarray", result)
