"""Shared backend hook resolution helpers.

This module centralizes op/backend hook lookup used by both graph execution
and autodiff forward evaluation. Lookups are cached and invalidated whenever
a new single-assignment backend hook is registered.
"""

from __future__ import annotations

from typing import Any

from advect.core._array_namespace import (
    _get_array_namespace,
    _get_backend_key_from_namespace,
)
from advect.core._backends import get_hook

type HookPair = tuple[Any, Any | None]

_NAMESPACE_HOOK_CACHE: dict[str, HookPair | None] = {}
_BACKEND_HOOK_CACHE: dict[str, HookPair | None] = {}

__all__ = ["resolve_backend_hooks"]


def clear_backend_hook_cache() -> None:
    """Clear memoized backend hook lookups."""
    _NAMESPACE_HOOK_CACHE.clear()
    _BACKEND_HOOK_CACHE.clear()


def _hooks_for_namespace(namespace: str) -> HookPair | None:
    cached = _NAMESPACE_HOOK_CACHE.get(namespace)
    if cached is not None or namespace in _NAMESPACE_HOOK_CACHE:
        return cached

    evaluate_op = get_hook(f"{namespace}.evaluate_op")
    if evaluate_op is None:
        _NAMESPACE_HOOK_CACHE[namespace] = None
        return None

    hooks = (evaluate_op, get_hook(f"{namespace}.decode_attrs"))
    _NAMESPACE_HOOK_CACHE[namespace] = hooks
    return hooks


def _hooks_for_backend(backend: str) -> HookPair | None:
    cached = _BACKEND_HOOK_CACHE.get(backend)
    if cached is not None or backend in _BACKEND_HOOK_CACHE:
        return cached

    evaluate_op = get_hook(f"{backend}.evaluate_op")
    if evaluate_op is None:
        _BACKEND_HOOK_CACHE[backend] = None
        return None

    hooks = (evaluate_op, get_hook(f"{backend}.decode_attrs"))
    _BACKEND_HOOK_CACHE[backend] = hooks
    return hooks


def resolve_backend_hooks(op: str, inputs: tuple[Any, ...]) -> HookPair:
    """Resolve ``(evaluate_op, decode_attrs)`` hooks for an op and inputs."""
    if "." in op:
        namespace = op.split(".", 1)[0]
        hooks = _hooks_for_namespace(namespace)
        if hooks is not None:
            return hooks

    for value in inputs:
        xp = _get_array_namespace(value)
        if xp is None:
            continue
        backend = _get_backend_key_from_namespace(xp)
        if backend is None:
            continue
        hooks = _hooks_for_backend(backend)
        if hooks is not None:
            return hooks

    msg = (
        f"No backend evaluator available for operation '{op}'. "
        "Install and load an array backend plugin."
    )
    raise RuntimeError(msg)
