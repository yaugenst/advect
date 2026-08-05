"""Test-owned isolation for process-global backend registration state."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from advect.core import _backend_hooks, _backends

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def isolated_backend_state() -> Iterator[None]:
    """Restore import-time backend registration after an isolated test mutation."""
    input_handlers = list(_backends._input_handlers)
    exact_input_handlers = dict(_backends._exact_input_handlers)
    hooks = dict(_backends._hooks)
    core_handlers_loaded = _backends._state.core_handlers_loaded
    namespace_hook_cache = dict(_backend_hooks._NAMESPACE_HOOK_CACHE)
    backend_hook_cache = dict(_backend_hooks._BACKEND_HOOK_CACHE)

    try:
        yield
    finally:
        _backends._input_handlers[:] = input_handlers
        _backends._exact_input_handlers.clear()
        _backends._exact_input_handlers.update(exact_input_handlers)
        _backends._hooks.clear()
        _backends._hooks.update(hooks)
        _backends._state.core_handlers_loaded = core_handlers_loaded
        _backend_hooks._NAMESPACE_HOOK_CACHE.clear()
        _backend_hooks._NAMESPACE_HOOK_CACHE.update(namespace_hook_cache)
        _backend_hooks._BACKEND_HOOK_CACHE.clear()
        _backend_hooks._BACKEND_HOOK_CACHE.update(backend_hook_cache)
