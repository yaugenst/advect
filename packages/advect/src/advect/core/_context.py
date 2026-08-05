"""Tracing context state for Advect.

This module manages per-thread trace frame state. A single thread may hold
multiple active trace frames (nested traces), where the top-most frame is the
active frame for new traced operations.
"""

from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from advect.core._errors import TracingError

# Thread-local storage for trace frame state
_thread_local = threading.local()

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


@dataclass(slots=True)
class TraceFrame:
    """Runtime state for a single active trace frame."""

    recorder: Any
    trace_level: int
    trace_kind: str
    frame_id: int
    array_api_version: str | None = None
    pending_update: object | None = None


def _get_trace_frames() -> list[TraceFrame]:
    frames = cast("list[TraceFrame] | None", getattr(_thread_local, "trace_frames", None))
    if frames is None:
        new_frames: list[TraceFrame] = []
        _thread_local.trace_frames = new_frames
        return new_frames
    return frames


def _next_trace_frame_id() -> int:
    current = getattr(_thread_local, "trace_frame_counter", 0)
    _thread_local.trace_frame_counter = int(current) + 1
    return int(current)


def _get_active_trace_frame() -> TraceFrame | None:
    frames = _get_trace_frames()
    if not frames:
        return None
    return frames[-1]


def _is_recorder_in_active_trace_stack(recorder: Any) -> bool:  # noqa: ANN401
    """Return whether ``recorder`` belongs to any currently active trace frame."""
    frames = _get_trace_frames()
    if not frames:
        return False
    if frames[-1].recorder is recorder:
        return True
    return any(frame.recorder is recorder for frame in frames[:-1])


def _trace_use_status(
    recorder: Any,  # noqa: ANN401
    *,
    take_pending: bool,
) -> tuple[bool, bool, object | None]:
    """Validate one tracer use and optionally consume its pending update.

    Trace membership and pending-update state share the same frame. Reading
    both from one stack snapshot keeps the common operation path cheap and
    avoids races between independent lookups in nested traces.
    """
    frames = _get_trace_frames()
    if not frames:
        return False, False, None

    current = frames[-1]
    if current.recorder is recorder:
        frame = current
    else:
        candidate = next(
            (candidate for candidate in reversed(frames[:-1]) if candidate.recorder is recorder),
            None,
        )
        if candidate is None:
            return True, False, None
        frame = candidate

    pending = frame.pending_update
    if take_pending:
        frame.pending_update = None
    return True, True, pending


def _get_active_recorder() -> Any | None:  # noqa: ANN401 - recorders have two lifetimes
    """Return the currently active recorder, or ``None`` outside tracing."""
    frame = _get_active_trace_frame()
    if frame is None:
        return None
    return frame.recorder


def _select_deepest_active_recorder(recorders: Iterable[object]) -> object:
    """Select the innermost active recorder represented by operation operands."""
    candidates = tuple(recorders)
    frames = _get_trace_frames()
    selected: object | None = None
    selected_depth = -1
    for recorder in candidates:
        depth = next(
            (
                index
                for index in range(len(frames) - 1, -1, -1)
                if frames[index].recorder is recorder
            ),
            None,
        )
        if depth is None:
            msg = "Cannot use a tracer from an unrelated or expired trace recorder"
            raise TracingError(msg)
        if depth > selected_depth:
            selected = recorder
            selected_depth = depth
    if selected is None:
        msg = "A traced operation requires at least one active recorder"
        raise TracingError(msg)
    return selected


def _get_operation_recorder() -> object | None:
    recorders = _get_operation_recorders()
    return None if not recorders else recorders[-1]


def _get_operation_recorders() -> list[object]:
    recorders = cast(
        "list[object] | None",
        getattr(_thread_local, "operation_recorders", None),
    )
    if recorders is None:
        new_recorders: list[object] = []
        _thread_local.operation_recorders = new_recorders
        return new_recorders
    return recorders


@contextmanager
def _use_operation_recorder(recorder: object) -> Iterator[None]:
    """Expose one selected recorder while a backend handler evaluates operands."""
    recorders = _get_operation_recorders()
    recorders.append(recorder)
    try:
        yield
    finally:
        recorders.pop()


@contextmanager
def _suspend_tracing() -> Iterator[None]:
    """Temporarily hide active recorders while an atomic provider executes."""
    frames = _get_trace_frames()
    operation_recorders = _get_operation_recorders()
    suspended_frames = tuple(frames)
    suspended_operation_recorders = tuple(operation_recorders)
    frames.clear()
    operation_recorders.clear()
    try:
        yield
    finally:
        frames[:] = suspended_frames
        operation_recorders[:] = suspended_operation_recorders


def _get_active_trace_kind() -> str | None:
    """Return the active trace mode without exposing the mutable frame."""
    frame = _get_active_trace_frame()
    return None if frame is None else frame.trace_kind


def _has_active_trace_kind(trace_kind: str) -> bool:
    """Return whether any current-thread trace frame has ``trace_kind``."""
    frames = _get_trace_frames()
    if not frames:
        return False
    if frames[-1].trace_kind == trace_kind:
        return True
    return any(frame.trace_kind == trace_kind for frame in frames[:-1])


@contextmanager
def _use_array_api_version(array_api_version: str) -> Iterator[None]:
    """Retain a trace's selected Array API revision during replay."""
    versions = cast(
        "list[str] | None",
        getattr(_thread_local, "array_api_version_overrides", None),
    )
    if versions is None:
        new_versions: list[str] = []
        _thread_local.array_api_version_overrides = new_versions
        versions = new_versions
    versions.append(array_api_version)
    try:
        yield
    finally:
        versions.pop()


@contextmanager
def _rematerialization_region() -> Iterator[None]:
    """Mark execution whose intermediates will be replayed during autodiff."""
    depth = int(getattr(_thread_local, "rematerialization_depth", 0))
    _thread_local.rematerialization_depth = depth + 1
    try:
        yield
    finally:
        _thread_local.rematerialization_depth = depth


def _is_rematerializing() -> bool:
    """Return whether the current call is inside a checkpointed region."""
    return bool(getattr(_thread_local, "rematerialization_depth", 0))


def _set_active_recorder(
    recorder: Any | None,  # noqa: ANN401 - recorders have two lifetimes
    *,
    trace_kind: str = "trace",
    array_api_version: str | None = None,
) -> None:
    """Push or pop active recorder frames.

    Passing a recorder pushes a frame; passing ``None`` pops the top frame.
    """
    frames = _get_trace_frames()
    if recorder is None:
        if not frames:
            return
        frame = frames.pop()
        pending = frame.pending_update
        if pending is not None and not bool(getattr(pending, "complete_without_setitem", False)):
            message = getattr(pending, "unconsumed_message", None)
            if not isinstance(message, str):
                message = (
                    "A traced augmented assignment through a view was not completed. "
                    "Rewrite it as an explicit functional update."
                )
            raise TracingError(message)
        return

    frame = TraceFrame(
        recorder=recorder,
        trace_level=len(frames),
        trace_kind=trace_kind,
        frame_id=_next_trace_frame_id(),
        array_api_version=array_api_version,
    )
    frames.append(frame)
    bind_trace_frame = getattr(recorder, "bind_trace_frame", None)
    if callable(bind_trace_frame):
        bind_trace_frame(trace_level=frame.trace_level, trace_frame_id=frame.frame_id)


def _trace_frame_for_recorder(recorder: Any) -> TraceFrame | None:  # noqa: ANN401
    """Return the active frame owning ``recorder``, if present."""
    for frame in reversed(_get_trace_frames()):
        if frame.recorder is recorder:
            return frame
    return None


def _peek_pending_update(recorder: Any) -> object | None:  # noqa: ANN401
    """Return a recorder's pending augmented-view update without consuming it."""
    frame = _trace_frame_for_recorder(recorder)
    return None if frame is None else frame.pending_update


def _set_pending_update(recorder: Any, pending: object) -> None:  # noqa: ANN401
    """Register the sole pending augmented-view update for one trace frame."""
    frame = _trace_frame_for_recorder(recorder)
    if frame is None:
        msg = "Pending updates require an active trace frame"
        raise RuntimeError(msg)
    if frame.pending_update is not None:
        msg = (
            "A traced augmented assignment through a view is already pending. "
            "Complete the matching subscript assignment before another traced operation."
        )
        raise RuntimeError(msg)
    frame.pending_update = pending


def _take_pending_update(recorder: Any) -> object | None:  # noqa: ANN401
    """Consume and return the pending update for ``recorder``."""
    frame = _trace_frame_for_recorder(recorder)
    if frame is None:
        return None
    pending = frame.pending_update
    frame.pending_update = None
    return pending


def is_tracing() -> bool:
    """Return whether an Advect transform is currently tracing.

    Returns
    -------
    bool
        True while a dynamic transform or staging trace is active.
    """
    return _get_active_trace_frame() is not None


def _get_active_trace_level() -> int | None:
    """Return the currently active trace nesting level."""
    frame = _get_active_trace_frame()
    if frame is None:
        return None
    return frame.trace_level


def _get_active_array_api_version() -> str | None:
    """Return the Array API contract selected by the active trace frame."""
    frame = _get_active_trace_frame()
    if frame is not None and frame.array_api_version is not None:
        return frame.array_api_version
    versions = cast(
        "list[str] | None",
        getattr(_thread_local, "array_api_version_overrides", None),
    )
    return None if not versions else versions[-1]


def is_debug() -> bool:
    """Check if debug mode is enabled.

    Debug mode enables additional trace diagnostics.

    Returns
    -------
    bool
        True if debug mode is enabled, False otherwise.

    See Also
    --------
    set_debug : Enable or disable debug mode.

    """
    return getattr(_thread_local, "debug", False)


def _is_numerics_debug() -> bool:
    """Return whether first-nonfinite diagnostics are enabled."""
    return is_debug() and bool(getattr(_thread_local, "debug_numerics", False))


def _get_numerics_context() -> tuple[str, str | None]:
    return cast(
        "tuple[str, str | None]",
        getattr(_thread_local, "numerics_context", ("primal evaluation", None)),
    )


@contextmanager
def _numerics_context(phase: str, source_location: str | None) -> Iterator[None]:
    if not _is_numerics_debug():
        yield
        return
    previous = _get_numerics_context()
    if previous[0] != "primal evaluation":
        yield
        return
    _thread_local.numerics_context = (phase, source_location)
    try:
        yield
    finally:
        _thread_local.numerics_context = previous


def set_debug(*, enabled: bool) -> None:
    """Enable or disable debug mode.

    Debug mode enables additional trace diagnostics.

    Parameters
    ----------
    enabled
        True to enable debug mode, False to disable.

    Notes
    -----
    Debug mode adds diagnostic overhead and should normally be disabled in hot
    paths.

    """
    _thread_local.debug = enabled


@contextmanager
def debug(*, numerics: bool = False) -> Iterator[None]:
    """Enable scoped trace diagnostics.

    Debug mode records per-operation user locations and gives live tracers a
    bounded concrete-value summary. ``numerics=True`` additionally raises at
    the first non-finite primal, JVP, or VJP value found by a dynamic transform.
    State is thread-local and restored exactly when the scope exits.
    """
    previous = (
        bool(getattr(_thread_local, "debug", False)),
        bool(getattr(_thread_local, "debug_numerics", False)),
    )
    set_debug(enabled=True)
    _thread_local.debug_numerics = bool(numerics)
    try:
        yield
    finally:
        set_debug(enabled=previous[0])
        _thread_local.debug_numerics = previous[1]


def get_source_location() -> str | None:
    """Get the source location of the caller.

    Only captures location in debug mode for performance.

    Returns
    -------
    str or None
        Source location string "file:line in function()", or None
        if not in debug mode.
    """
    if not is_debug():
        return None

    frame = sys._getframe(1)  # noqa: SLF001 - frame walking is the diagnostic boundary
    while frame is not None:
        module = str(frame.f_globals.get("__name__", ""))
        internal = any(
            module == prefix or module.startswith(f"{prefix}.")
            for prefix in ("advect", "numpy", "array_api_compat", "contextlib")
        )
        if not internal:
            code = frame.f_code
            return f"{code.co_filename}:{frame.f_lineno} in {code.co_name}()"
        frame = frame.f_back
    return None
