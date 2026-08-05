# ruff: noqa: ANN401, FBT001
# ANN401: Protocols use Any for backend-agnostic dtype property.
# FBT001: __array__ follows NumPy's positional copy protocol.
"""Protocols for Advect.

This module defines protocols (structural types) that allow Advect to work with
different array backends while maintaining type safety.
"""

from __future__ import annotations

from typing import Any, Protocol, cast, runtime_checkable


@runtime_checkable
class ArrayLike(Protocol):
    """Minimal metadata shared by arrays from supported providers.

    Satisfying this protocol alone does not make a type traceable. A provider
    must also enter through an input handler and supply an array namespace and
    evaluator.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the array."""
        ...

    @property
    def dtype(self) -> Any:
        """Return the data type of the array."""
        ...

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        ...

    @property
    def size(self) -> int:
        """Return the total number of elements."""
        ...


@runtime_checkable
class TracedArrayLike(ArrayLike, Protocol):
    """Protocol for traced array objects used in dispatch.

    This protocol defines the minimal interface that array protocol runtimes
    need from traced arrays. It decouples those runtimes from the concrete
    TracedArray implementation and avoids circular imports.

    The protocol requires an internal ``_advect_snapshot()`` method that returns
    the current SSA identifier and concrete payload together.  Implementations
    validate trace and view lifetimes before returning either value; concrete
    payloads are intentionally not part of the public tracer surface.

    Notes
    -----
    This protocol is runtime-checkable to support isinstance() checks
    in dispatch logic. The dispatch modules use this to identify traced
    inputs and atomically snapshot their values and node IDs.

    """

    def _advect_snapshot(self) -> tuple[int, ArrayLike]:
        """Return a lifetime-validated internal SSA/value pair."""
        ...

    def __array__(self, dtype: Any | None = None, copy: bool | None = None) -> Any:
        """Expose the provider protocol signature.

        Concrete tracers intentionally raise from this method so providers
        cannot silently detach them.  Declaring it here lets backend-neutral
        lowering code type-check calls that NumPy will route through
        ``__array_function__`` before coercion.
        """
        ...

    def __getitem__(self, key: object) -> Any:
        """Return a traced indexed value."""
        ...

    @property
    def node_id(self) -> int:
        """Return the node ID in the computation graph."""
        ...


def _snapshot_traced(value: object) -> tuple[int, Any]:
    """Invoke Advect's internal tracer snapshot protocol."""
    snapshot = getattr(value, "_advect_snapshot", None)
    if not callable(snapshot):
        msg = f"{type(value).__name__} does not implement the Advect tracer snapshot protocol"
        raise TypeError(msg)
    return cast("tuple[int, Any]", snapshot())


def _snapshot_traced_in_active_trace(value: object) -> tuple[int, Any]:
    """Use a frontend fast path after dispatch validated the active trace."""
    snapshot = getattr(value, "_advect_snapshot_in_active_trace", None)
    if callable(snapshot):
        return cast("tuple[int, Any]", snapshot())
    return _snapshot_traced(value)
