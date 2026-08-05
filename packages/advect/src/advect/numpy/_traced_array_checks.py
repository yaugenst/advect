"""Common validation helpers for TracedArray operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from advect.core._context import _trace_use_status
from advect.core._errors import EscapedTracerError, TracingError

if TYPE_CHECKING:
    from advect.core._native import DynamicTape


@overload
def require_active_trace(
    *,
    recorder: DynamicTape,
    allow_pending: Literal[False] = False,
    take_pending: Literal[False] = False,
) -> None: ...


@overload
def require_active_trace(
    *,
    recorder: DynamicTape,
    allow_pending: Literal[True],
    take_pending: bool = False,
) -> object | None: ...


def require_active_trace(
    *,
    recorder: DynamicTape,
    allow_pending: bool = False,
    take_pending: bool = False,
) -> object | None:
    """Ensure the array still belongs to the active Advect transform.

    ``__setitem__`` can also consume and return the pending indexed-update
    acknowledgement from this same authoritative frame lookup.
    """
    is_active, contains_recorder, pending = _trace_use_status(
        recorder,
        take_pending=take_pending or not allow_pending,
    )
    if not is_active:
        msg = (
            "This TracedArray escaped its Advect transform; the creating trace has already exited."
        )
        raise EscapedTracerError(msg)

    if not contains_recorder:
        msg = (
            "Cannot use a TracedArray from a different trace context. "
            "This TracedArray escaped its creating trace, which has already exited."
        )
        raise EscapedTracerError(msg)

    if allow_pending:
        return pending
    if pending is None or bool(getattr(pending, "complete_without_setitem", False)):
        return None

    message = getattr(pending, "unconsumed_message", None)
    if not isinstance(message, str):
        message = (
            "A traced augmented assignment through a view was not followed by "
            "its matching subscript assignment."
        )
    raise TracingError(message)
