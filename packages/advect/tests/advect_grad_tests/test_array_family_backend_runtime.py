"""Tests for array-family backend runtime wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import advect.autodiff.rules.array_family._backend_runtime as backend_runtime
from advect.autodiff.rules.array_family.providers import (
    ArrayFamilyBackendProvider,
    register_array_family_backend_provider,
)

if TYPE_CHECKING:
    import pytest


@dataclass(frozen=True, slots=True)
class _TestProvider(ArrayFamilyBackendProvider):
    backend: str
    namespace: Any
    ext: Any | None = None


def test_wrap_vjp_uses_active_provider_without_re_resolving(
    monkeypatch: pytest.MonkeyPatch,
    isolated_provider_registry: None,
) -> None:
    provider = _TestProvider(backend="numpy", namespace=SimpleNamespace(__name__="numpy"))
    register_array_family_backend_provider(provider)

    def _fail_resolve(*_values: object) -> _TestProvider:
        msg = "wrapper should not resolve provider when context is already active"
        raise AssertionError(msg)

    monkeypatch.setattr(backend_runtime, "_resolve_provider_for_call", _fail_resolve)
    token = backend_runtime._CURRENT_ARRAY_FAMILY_PROVIDER.set(provider)
    try:
        seen: dict[str, object] = {}

        def _rule(ans: object, *inputs: object, g: object, **attrs: object) -> tuple[object]:
            _ = ans, inputs, attrs
            seen["provider"] = backend_runtime.current_array_backend_provider()
            return (g,)

        wrapped = backend_runtime.wrap_array_family_vjp_rule(_rule)
        assert wrapped(1.0, g=2.0) == (2.0,)
        assert seen["provider"] is provider
    finally:
        backend_runtime._CURRENT_ARRAY_FAMILY_PROVIDER.reset(token)


def test_wrap_jvp_uses_active_provider_without_re_resolving(
    monkeypatch: pytest.MonkeyPatch,
    isolated_provider_registry: None,
) -> None:
    provider = _TestProvider(backend="numpy", namespace=SimpleNamespace(__name__="numpy"))
    register_array_family_backend_provider(provider)

    def _fail_resolve(*_values: object) -> _TestProvider:
        msg = "wrapper should not resolve provider when context is already active"
        raise AssertionError(msg)

    monkeypatch.setattr(backend_runtime, "_resolve_provider_for_call", _fail_resolve)
    token = backend_runtime._CURRENT_ARRAY_FAMILY_PROVIDER.set(provider)
    try:
        seen: dict[str, object] = {}

        def _rule(
            ans: object, *inputs: object, tangents: tuple[object | None, ...], **attrs: object
        ) -> object:
            _ = inputs, tangents, attrs
            seen["provider"] = backend_runtime.current_array_backend_provider()
            return ans

        wrapped = backend_runtime.wrap_array_family_jvp_rule(_rule)
        assert wrapped(1.0, tangents=(2.0,)) == 1.0
        assert seen["provider"] is provider
    finally:
        backend_runtime._CURRENT_ARRAY_FAMILY_PROVIDER.reset(token)


def test_wrap_jvp_resolves_provider_and_restores_context(
    monkeypatch: pytest.MonkeyPatch,
    isolated_provider_registry: None,
) -> None:
    provider = _TestProvider(backend="numpy", namespace=SimpleNamespace(__name__="numpy"))
    register_array_family_backend_provider(provider)
    monkeypatch.setattr(
        backend_runtime,
        "resolve_array_family_backend_provider",
        lambda *_values: provider,
    )
    assert backend_runtime.current_array_backend_provider() is None

    def _rule(
        ans: object, *inputs: object, tangents: tuple[object | None, ...], **attrs: object
    ) -> object:
        _ = inputs, tangents, attrs
        assert backend_runtime.current_array_backend_provider() is provider
        return ans

    wrapped = backend_runtime.wrap_array_family_jvp_rule(_rule)
    assert wrapped(3.0, tangents=(1.0,)) == 3.0
    assert backend_runtime.current_array_backend_provider() is None


def test_wrap_vjp_exposes_unwrapped_rule_for_specialization() -> None:
    def _rule(ans: object, *inputs: object, g: object, **attrs: object) -> tuple[object]:
        _ = ans, inputs, attrs
        return (g,)

    wrapped = backend_runtime.wrap_array_family_vjp_rule(_rule)
    unwrapped = backend_runtime._maybe_unwrap_array_family_vjp_rule(wrapped)
    assert unwrapped is _rule
    assert backend_runtime._maybe_unwrap_array_family_vjp_rule(_rule) is None


def test_wrap_jvp_exposes_unwrapped_rule_for_specialization() -> None:
    def _rule(
        ans: object, *inputs: object, tangents: tuple[object | None, ...], **attrs: object
    ) -> object:
        _ = inputs, tangents, attrs
        return ans

    wrapped = backend_runtime.wrap_array_family_jvp_rule(_rule)
    unwrapped = backend_runtime._maybe_unwrap_array_family_jvp_rule(wrapped)
    assert unwrapped is _rule
    assert backend_runtime._maybe_unwrap_array_family_jvp_rule(_rule) is None
