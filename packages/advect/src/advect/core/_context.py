"""Tracing context state for Advect.

This module manages per-thread trace frame state. A single thread may hold
multiple active trace frames (nested traces), where the top-most frame is the
active frame for new traced operations.
"""

from __future__ import annotations

import sys
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from advect.core._errors import TracingError

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable
    from contextlib import AbstractContextManager
    from types import FrameType


@dataclass(slots=True)
class TraceFrame:
    """Runtime state for a single active trace frame."""

    recorder: Any
    trace_level: int
    trace_kind: str
    frame_id: int
    array_api_version: str | None = None
    pending_update: object | None = None
    transform_state: dict[object, object] | None = None
    require_jvp: bool = False


_PRIMAL_PHASE = "primal evaluation"


class _TraceState(threading.local):
    """Per-thread trace state; ``threading.local`` runs ``__init__`` per thread."""

    def __init__(self) -> None:
        self.frames: list[TraceFrame] = []
        self.operation_recorders: list[object] = []
        self.next_frame_id = 0
        self.array_api_versions: list[str] = []
        self.rematerialization_depth = 0
        self.debug = False
        self.debug_numerics = False
        self.numerics_context: tuple[str, str | None] = (_PRIMAL_PHASE, None)


_state = _TraceState()
_NO_NUMERICS_SCOPE: AbstractContextManager[None] = nullcontext()


def _get_active_trace_frame() -> TraceFrame | None:
    frames = _state.frames
    return frames[-1] if frames else None


def _is_recorder_in_active_trace_stack(recorder: Any) -> bool:  # noqa: ANN401
    """Return whether ``recorder`` belongs to any currently active trace frame."""
    frames = _state.frames
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
    frames = _state.frames
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
    frames = _state.frames
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
    recorders = _state.operation_recorders
    return recorders[-1] if recorders else None


@contextmanager
def _use_operation_recorder(recorder: object) -> Generator[None]:
    """Expose one selected recorder while a backend handler evaluates operands."""
    recorders = _state.operation_recorders
    recorders.append(recorder)
    try:
        yield
    finally:
        recorders.pop()


@contextmanager
def _suspend_tracing() -> Generator[None]:
    """Temporarily hide active recorders while an atomic provider executes."""
    frames = _state.frames
    operation_recorders = _state.operation_recorders
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


def _active_trace_requires_jvp() -> bool:
    """Return whether the active transform requires forward-mode rules."""
    frame = _get_active_trace_frame()
    return frame is not None and frame.require_jvp


def _has_active_trace_kind(trace_kind: str) -> bool:
    """Return whether any current-thread trace frame has ``trace_kind``."""
    frames = _state.frames
    if not frames:
        return False
    if frames[-1].trace_kind == trace_kind:
        return True
    return any(frame.trace_kind == trace_kind for frame in frames[:-1])


@contextmanager
def _use_array_api_version(array_api_version: str) -> Generator[None]:
    """Retain a trace's selected Array API revision during replay."""
    versions = _state.array_api_versions
    versions.append(array_api_version)
    try:
        yield
    finally:
        versions.pop()


@contextmanager
def _rematerialization_region() -> Generator[None]:
    """Mark execution whose intermediates will be replayed during autodiff."""
    state = _state
    depth = state.rematerialization_depth
    state.rematerialization_depth = depth + 1
    try:
        yield
    finally:
        state.rematerialization_depth = depth


def _is_rematerializing() -> bool:
    """Return whether the current call is inside a checkpointed region."""
    return _state.rematerialization_depth > 0


def _set_active_recorder(
    recorder: Any | None,  # noqa: ANN401 - recorders have two lifetimes
    *,
    trace_kind: str = "trace",
    array_api_version: str | None = None,
    require_jvp: bool = False,
) -> None:
    """Push or pop active recorder frames.

    Passing a recorder pushes a frame; passing ``None`` pops the top frame.
    """
    state = _state
    frames = state.frames
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

    frame_id = state.next_frame_id
    state.next_frame_id = frame_id + 1
    frame = TraceFrame(
        recorder=recorder,
        trace_level=len(frames),
        trace_kind=trace_kind,
        frame_id=frame_id,
        array_api_version=array_api_version,
        require_jvp=require_jvp,
    )
    frames.append(frame)
    bind_trace_frame = getattr(recorder, "bind_trace_frame", None)
    if callable(bind_trace_frame):
        bind_trace_frame(trace_level=frame.trace_level, trace_frame_id=frame.frame_id)


def _trace_frame_for_recorder(recorder: Any) -> TraceFrame | None:  # noqa: ANN401
    """Return the active frame owning ``recorder``, if present."""
    for frame in reversed(_state.frames):
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


def transform_state[T](
    namespace: object,
    factory: Callable[[], T],
) -> T | None:
    """Return namespaced state owned by the active dynamic transform.

    Libraries can use this to retain ordinary Python bookkeeping for exactly
    one define-by-run transform invocation without wrapping the transform or
    keeping process-global state. Repeated calls with the same namespace return
    the same object. Nested transforms have independent state, and Advect drops
    the state when its owning trace exits, including on exceptions.

    State is not a hidden differentiable input or a backward residual. Pass
    active leaves explicitly to primitives, and retain backward data with
    ``PrimitiveResult``.

    Outside a transform this returns ``None``. Abstract staging rejects the
    operation because staged programs cannot retain invocation-local Python
    state.

    Parameters
    ----------
    namespace
        Hashable library-owned key identifying the state.
    factory
        Zero-argument callable used once to create the state.

    Returns
    -------
    object or None
        The invocation-local state, or ``None`` outside dynamic tracing.
    """
    frame = _get_active_trace_frame()
    if frame is None:
        return None
    if frame.trace_kind != "autodiff_dynamic":
        msg = "transform_state is available only during concrete dynamic transforms"
        raise TracingError(msg)
    state = frame.transform_state
    if state is None:
        state = {}
        frame.transform_state = state
    if namespace not in state:
        state[namespace] = factory()
    return cast("T", state[namespace])


def transform_states[T](namespace: object) -> tuple[T, ...]:
    """Return existing namespaced states from inner to outer dynamic transforms.

    This lets a library resolve state owned by an enclosing transform while a
    nested transform is active. It never creates state. Outside a transform it
    returns an empty tuple; abstract staging rejects the operation.
    """
    frame = _get_active_trace_frame()
    if frame is not None and frame.trace_kind != "autodiff_dynamic":
        msg = "transform_states is available only during concrete dynamic transforms"
        raise TracingError(msg)
    states = []
    for frame in reversed(_state.frames):
        if frame.trace_kind != "autodiff_dynamic" or frame.transform_state is None:
            continue
        if namespace in frame.transform_state:
            states.append(frame.transform_state[namespace])
    return cast("tuple[T, ...]", tuple(states))


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
    versions = _state.array_api_versions
    return versions[-1] if versions else None


def is_debug() -> bool:
    """Check if debug mode is enabled.

    Debug mode enables additional trace diagnostics.

    Returns
    -------
    bool
        True if debug mode is enabled, False otherwise.

    """
    return _state.debug


def _is_numerics_debug() -> bool:
    """Return whether first-nonfinite diagnostics are enabled."""
    state = _state
    return state.debug and state.debug_numerics


def _get_numerics_context() -> tuple[str, str | None]:
    return _state.numerics_context


def _numerics_context(phase: str, source_location: str | None) -> AbstractContextManager[None]:
    """Attribute non-finite values to ``phase`` unless an outer phase already owns them."""
    state = _state
    if not (state.debug and state.debug_numerics) or state.numerics_context[0] != _PRIMAL_PHASE:
        return _NO_NUMERICS_SCOPE
    return _numerics_phase(phase, source_location)


@contextmanager
def _numerics_phase(phase: str, source_location: str | None) -> Generator[None]:
    state = _state
    previous = state.numerics_context
    state.numerics_context = (phase, source_location)
    try:
        yield
    finally:
        state.numerics_context = previous


@contextmanager
def debug(*, numerics: bool = False) -> Generator[None]:
    """Enable scoped trace diagnostics.

    Debug mode records per-operation user locations and gives live tracers a
    bounded concrete-value summary. ``numerics=True`` additionally raises at
    the first non-finite primal, JVP, or VJP value found by a dynamic transform.
    State is thread-local and restored exactly when the scope exits.
    """
    state = _state
    previous = (state.debug, state.debug_numerics)
    state.debug = True
    state.debug_numerics = numerics
    try:
        yield
    finally:
        state.debug, state.debug_numerics = previous


_ADVECT_PACKAGES = frozenset({"advect"})
# Debug diagnostics also skip the array libraries and helpers that dispatch into Advect.
_DIAGNOSTIC_PACKAGES = frozenset({"advect", "numpy", "array_api_compat", "contextlib"})


def external_frame(
    frame: FrameType | None,
    internal: frozenset[str] = _ADVECT_PACKAGES,
) -> FrameType | None:
    """Return *frame* or its innermost caller outside the *internal* top-level packages."""
    while frame is not None:
        module = frame.f_globals.get("__name__")
        if not isinstance(module, str) or module.partition(".")[0] not in internal:
            return frame
        frame = frame.f_back
    return None


def caller_location(internal: frozenset[str] = _ADVECT_PACKAGES) -> str | None:
    """Return ``"file:line in function()"`` for the innermost caller outside *internal*."""
    # Frame walking is the diagnostic boundary.
    frame = external_frame(sys._getframe(1), internal)  # noqa: SLF001
    if frame is None:
        return None
    code = frame.f_code
    return f"{code.co_filename}:{frame.f_lineno} in {code.co_name}()"


def get_source_location() -> str | None:
    """Get the source location of the caller.

    Only captures location in debug mode for performance.

    Returns
    -------
    str or None
        Source location string "file:line in function()", or None
        if not in debug mode.
    """
    return caller_location(_DIAGNOSTIC_PACKAGES) if is_debug() else None
