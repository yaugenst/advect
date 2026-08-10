"""Shared backend hook resolution helpers."""

from __future__ import annotations

from typing import Any

from advect.core._array_api.providers import (
    _get_array_namespace,
    _get_backend_key_from_namespace,
)
from advect.core._backends import get_hook

type HookPair = tuple[Any, Any | None]

__all__ = ["resolve_backend_hooks"]


def _hooks_for(name: str) -> HookPair | None:
    evaluate_op = get_hook(f"{name}.evaluate_op")
    if evaluate_op is None:
        return None
    return evaluate_op, get_hook(f"{name}.decode_attrs")


def resolve_backend_hooks(op: str, inputs: tuple[Any, ...]) -> HookPair:
    """Resolve ``(evaluate_op, decode_attrs)`` hooks for an op and inputs."""
    if "." in op:
        namespace = op.split(".", 1)[0]
        hooks = _hooks_for(namespace)
        if hooks is not None:
            return hooks

    for value in inputs:
        xp = _get_array_namespace(value)
        if xp is None:
            continue
        backend = _get_backend_key_from_namespace(xp)
        if backend is None:
            continue
        hooks = _hooks_for(backend)
        if hooks is not None:
            return hooks

    msg = (
        f"No backend evaluator available for operation '{op}'. "
        "Install and load an array backend plugin."
    )
    raise RuntimeError(msg)
