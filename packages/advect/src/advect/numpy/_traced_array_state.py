"""Ephemeral source-level mutation state for the NumPy tracer.

The compute graph remains immutable SSA.  These objects exist only while
executing Python user code and are never serialized into graph attributes.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import FrameType

    from advect.numpy._traced_array import TracedArray


@dataclass(frozen=True, slots=True)
class SourceLocation:
    """Lightweight source reference formatted only when a diagnostic needs it."""

    filename: str
    lineno: int
    function: str

    def __str__(self) -> str:
        return f"{self.filename}:{self.lineno} in {self.function}"


def _source_location(frame: FrameType) -> SourceLocation:
    return SourceLocation(frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name)


def user_location(*, depth: int | None = None) -> SourceLocation | None:
    """Capture one user frame without materializing or formatting a traceback."""
    try:
        frame = vars(sys)["_getframe"](1 if depth is None else depth)
    except (KeyError, ValueError):  # Python implementations without frames.
        return None

    if depth is not None:
        return _source_location(frame)

    while frame is not None:
        module = frame.f_globals.get("__name__")
        if not isinstance(module, str) or (module != "advect" and not module.startswith("advect.")):
            return _source_location(frame)
        frame = frame.f_back
    return None


@dataclass(frozen=True, slots=True)
class ViewState:
    """Tracer-only description of one conservative alias relationship."""

    root: TracedArray
    epoch: int
    index_spec: object | None
    location: SourceLocation | None


@dataclass(frozen=True, slots=True)
class PendingIndexUpdate:
    """Acknowledgement for Python's getitem/iadd/setitem protocol."""

    root: TracedArray
    root_epoch: int
    index_spec: object
    replacement: TracedArray

    @property
    def complete_without_setitem(self) -> bool:
        """Mark an already-applied update whose generated setitem is optional."""
        return True
