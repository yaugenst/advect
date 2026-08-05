# ruff: noqa: ANN401
"""NumPy-owned lifecycle policy for abstract staging."""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from threading import Lock
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, override

import numpy as np

from advect.core._context import _has_active_trace_kind

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from contextlib import AbstractContextManager


_RNG_ENTRYPOINTS = tuple(
    (name, getattr(np.random, name))
    for name in np.random.__all__
    if callable(getattr(np.random, name))
)
_RNG_NAMES = tuple(name for name, _entrypoint in _RNG_ENTRYPOINTS)
_RNG_TYPES: tuple[type[Any], ...] = tuple(
    entrypoint for _name, entrypoint in _RNG_ENTRYPOINTS if isinstance(entrypoint, type)
)
_rng_entrypoint_names: dict[int, str] = {}
for _name, _entrypoint in _RNG_ENTRYPOINTS:
    _rng_entrypoint_names.setdefault(id(_entrypoint), _name)
_RNG_ENTRYPOINT_NAMES: Mapping[int, str] = MappingProxyType(_rng_entrypoint_names)
del _rng_entrypoint_names, _name, _entrypoint
_AMBIENT_RNG_ERROR = (
    "Ambient random-number generation is not allowed while staging. "
    "Pass explicit random state/key data as an input."
)


class _RNGTypeGuardMeta(type):
    """Preserve type checks while a public NumPy RNG type is guarded."""

    _advect_original: type[Any]

    @override
    def __instancecheck__(cls, instance: object) -> bool:
        return isinstance(instance, cls._advect_original)

    @override
    def __subclasscheck__(cls, subclass: type[Any]) -> bool:
        return issubclass(subclass, cls._advect_original)


def _guarded_rng_type(original: type[Any]) -> type[Any]:
    """Return a staging-aware subclass proxy for one public RNG type."""

    def guarded_new(_cls: type[Any], *args: Any, **kwargs: Any) -> Any:
        if _has_active_trace_kind("stage_abstract"):
            raise RuntimeError(_AMBIENT_RNG_ERROR)
        return original(*args, **kwargs)

    return _RNGTypeGuardMeta(
        original.__name__,
        (original,),
        {
            "__doc__": original.__doc__,
            "__module__": original.__module__,
            "__new__": guarded_new,
            "__qualname__": original.__qualname__,
            "_advect_original": original,
        },
    )


def _guarded_rng_callable(original: Any) -> Any:
    """Return a staging-aware wrapper for one public RNG callable."""

    @wraps(original)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        if _has_active_trace_kind("stage_abstract"):
            raise RuntimeError(_AMBIENT_RNG_ERROR)
        return original(*args, **kwargs)

    return guarded


def _guarded_rng_entrypoint(original: Any) -> Any:
    """Return the appropriate staging guard for one public RNG entry point."""
    return (
        _guarded_rng_type(original)
        if isinstance(original, type)
        else _guarded_rng_callable(original)
    )


_RNG_GUARDS = tuple(
    (name, entrypoint, _guarded_rng_entrypoint(entrypoint)) for name, entrypoint in _RNG_ENTRYPOINTS
)
_RNG_GUARD_BY_NAME: Mapping[str, Any] = MappingProxyType(
    {name: guard for name, _entrypoint, guard in _RNG_GUARDS}
)
_RNG_PATCH_LOCK = Lock()
_RNG_PATCH_DEPTH = 0
_RNG_ORIGINALS: dict[str, Any] = {}


def _validate_stage_capture(name: str, value: object) -> None:
    """Reject NumPy RNG state captured by a staged callable."""
    entrypoint_name = _RNG_ENTRYPOINT_NAMES.get(id(value))
    if entrypoint_name is not None:
        msg = (
            f"Captured NumPy random entry point {entrypoint_name!r} "
            f"{name!r} is ambient mutable random state and cannot be staged. "
            "Pass explicit random state/key data as an array input."
        )
        raise RuntimeError(msg)
    owner = getattr(value, "__self__", None)
    candidate = owner if owner is not None else value
    if not isinstance(candidate, _RNG_TYPES):
        return
    msg = (
        f"Captured NumPy {type(candidate).__name__} "
        f"{name!r} is ambient mutable random state and cannot be staged. "
        "Pass explicit random state/key data as an array input."
    )
    raise RuntimeError(msg)


def stage_context(
    captures: Sequence[tuple[str, object]],
) -> AbstractContextManager[None]:
    """Validate captures and return NumPy's scoped ambient-RNG tripwire."""
    for name, value in captures:
        _validate_stage_capture(name, value)
    return _ambient_rng_tripwire()


@contextmanager
def _ambient_rng_tripwire() -> Iterator[None]:
    """Tripwire NumPy's ambient RNG only in threads that are staging."""
    global _RNG_PATCH_DEPTH  # noqa: PLW0603
    with _RNG_PATCH_LOCK:
        if _RNG_PATCH_DEPTH == 0:
            canonical = all(
                getattr(np.random, name, None) is entrypoint
                for name, entrypoint in _RNG_ENTRYPOINTS
            )
            if canonical:
                _RNG_ORIGINALS.update(_RNG_ENTRYPOINTS)
                vars(np.random).update(_RNG_GUARD_BY_NAME)
            else:
                for name, entrypoint, canonical_guard in _RNG_GUARDS:
                    try:
                        original = getattr(np.random, name)
                    except AttributeError:
                        continue
                    if not callable(original):
                        continue
                    _RNG_ORIGINALS[name] = original
                    guard = (
                        canonical_guard
                        if original is entrypoint
                        else _guarded_rng_entrypoint(original)
                    )
                    setattr(np.random, name, guard)
        _RNG_PATCH_DEPTH += 1
    try:
        yield
    finally:
        with _RNG_PATCH_LOCK:
            _RNG_PATCH_DEPTH -= 1
            if _RNG_PATCH_DEPTH == 0:
                vars(np.random).update(_RNG_ORIGINALS)
                _RNG_ORIGINALS.clear()
