"""Bounded value diagnostics used by debug mode."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

from advect.core._array_api.providers import _get_array_namespace
from advect.core._context import _get_numerics_context
from advect.core._errors import NumericsError
from advect.core._pytree import format_path, tree_flatten_with_paths

_PREVIEW_LIMIT = 96

if TYPE_CHECKING:
    from advect.core._native import DynamicTape


def _shape(value: object) -> tuple[int, ...]:
    try:
        return tuple(int(dimension) for dimension in getattr(value, "shape", ()))
    except (TypeError, ValueError):
        return ()


def _dtype(value: object) -> str:
    dtype = getattr(value, "dtype", None)
    return type(value).__name__ if dtype is None else str(dtype)


def _is_tracer(value: object) -> bool:
    return callable(getattr(value, "_advect_snapshot", None)) or bool(
        getattr(type(value), "__advect_abstract_array__", False)
    )


def _finite_counts(value: object) -> tuple[int, int] | None:
    if _is_tracer(value):
        return None
    if isinstance(value, (bool, int, float, complex)):
        try:
            return (int(math.isfinite(cast("Any", value))), 1)
        except TypeError:
            return None

    namespace = _get_array_namespace(value)
    isfinite = getattr(namespace, "isfinite", None) if namespace is not None else None
    if not callable(isfinite):
        return None
    try:
        mask = isfinite(value)
        total = math.prod(_shape(value)) if _shape(value) else 1
        sum_function = getattr(namespace, "sum", None)
        finite = sum_function(mask) if callable(sum_function) else cast("Any", mask).sum()
        return int(cast("Any", finite)), total
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _preview(value: object) -> str:
    try:
        rendered = repr(value).replace("\n", " ")
    except Exception:  # noqa: BLE001 - diagnostics must not mask the result
        rendered = f"<{type(value).__name__}>"
    return rendered if len(rendered) <= _PREVIEW_LIMIT else f"{rendered[:95]}…"


def summarize_value(value: object) -> str:
    """Summarize one concrete value without exposing a payload API."""
    parts = [f"shape={_shape(value)}", f"dtype={_dtype(value)}"]
    counts = _finite_counts(value)
    if counts is not None:
        parts.append(f"finite={counts[0]}/{counts[1]}")
    parts.append(f"values={_preview(value)}")
    return ", ".join(parts)


def raise_if_nonfinite(
    value: object,
    *,
    phase: str,
    op: str,
    source_location: str | None,
) -> None:
    """Raise for the first non-finite leaf in a result pytree."""
    paths, leaves, _treedef = tree_flatten_with_paths(value)
    for path, leaf in zip(paths, leaves, strict=True):
        counts = _finite_counts(leaf)
        if counts is None or counts[0] == counts[1]:
            continue
        summary = summarize_value(leaf)
        if path:
            summary = f"leaf{format_path(path)}: {summary}"
        raise NumericsError(
            phase=phase,
            op=op,
            summary=summary,
            source_location=source_location,
        )


def check_tape_numerics(tape: DynamicTape) -> None:
    """Report the first non-finite value on a live dynamic tape."""
    trace_level, _frame_id = tape.runtime_trace_identity()
    if trace_level != 0:
        return
    phase, origin = _get_numerics_context()
    rows = tape._diagnostic_snapshot()  # noqa: SLF001 - private native debug ABI
    for op, source_location, value in rows:
        raise_if_nonfinite(
            value,
            phase=phase,
            op=op,
            source_location=origin or source_location,
        )
