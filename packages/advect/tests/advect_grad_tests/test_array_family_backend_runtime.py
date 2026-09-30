"""Contracts for the array-family backend-provider scope and ``xp`` proxy."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

import advect.autodiff.rules.array_family._backend_runtime as backend_runtime
from advect.autodiff.rules.array_family.providers import ArrayFamilyBackendProvider


@dataclass(frozen=True, slots=True)
class _Provider(ArrayFamilyBackendProvider):
    backend: str
    namespace: Any
    ext: Any | None = None


def _run(provider: _Provider, fn: Any, /, *args: object, **kwargs: object) -> object:
    return backend_runtime.run_with_array_family_backend_provider(provider, fn, *args, **kwargs)


def test_wrap_jvp_resolves_a_provider_only_outside_an_active_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = _Provider(backend="numpy", namespace=SimpleNamespace(__name__="numpy"))
    resolved = _Provider(backend="numpy", namespace=SimpleNamespace(__name__="numpy"))
    resolutions: list[tuple[object, ...]] = []

    def resolve(*values: object) -> _Provider:
        resolutions.append(values)
        return resolved

    monkeypatch.setattr(backend_runtime, "resolve_array_family_backend_provider", resolve)
    seen: list[ArrayFamilyBackendProvider | None] = []

    def rule(ans: object, *inputs: object, tangents: tuple[object | None, ...]) -> object:
        del inputs, tangents
        seen.append(backend_runtime.current_array_backend_provider())
        return ans

    wrapped = backend_runtime.wrap_array_family_jvp_rule(rule)

    assert _run(active, wrapped, 1.0, 5.0, tangents=(2.0,)) == 1.0
    assert wrapped(3.0, 5.0, tangents=(4.0,)) == 3.0
    assert seen == [active, resolved]
    assert resolutions == [(3.0, 5.0, 4.0)]
    assert backend_runtime.current_array_backend_provider() is None
    assert backend_runtime._maybe_unwrap_array_family_jvp_rule(wrapped) is rule
    assert backend_runtime._maybe_unwrap_array_family_jvp_rule(rule) is None


def test_namespace_proxy_falls_back_to_the_provider_extension() -> None:
    provider = _Provider(
        backend="test-extension-fallback",
        namespace=SimpleNamespace(),
        ext=SimpleNamespace(extension_only="extension-value"),
    )

    assert _run(provider, lambda: backend_runtime.xp.extension_only) == "extension-value"


def test_namespace_proxy_reports_an_unknown_attribute() -> None:
    provider = _Provider(
        backend="test-missing-attribute",
        namespace=SimpleNamespace(),
        ext=SimpleNamespace(),
    )

    with pytest.raises(AttributeError, match=r"test-missing-attribute.*unknown"):
        _run(provider, lambda: backend_runtime.xp.unknown)


def test_namespace_proxy_routes_functions_but_not_classes_of_a_standard_namespace() -> None:
    """Dtype-category classes such as ``complexfloating`` are compared by identity."""

    class Category:
        pass

    def function() -> str:
        return "called"

    provider = _Provider(
        backend="test-standard",
        namespace=SimpleNamespace(__array_api_version__="2024.12", function=function),
        ext=SimpleNamespace(Category=Category),
    )
    category, routed = _run(
        provider, lambda: (backend_runtime.xp.Category, backend_runtime.xp.function)
    )

    assert category is Category
    assert routed is not function
    assert routed() == "called"


def test_namespace_proxy_resolves_attributes_once_per_scope() -> None:
    namespace = SimpleNamespace(value="first")
    provider = _Provider(backend="test-scope-cache", namespace=namespace)

    def read_twice() -> tuple[object, object]:
        first = backend_runtime.xp.value
        namespace.value = "second"
        return first, backend_runtime.xp.value

    assert _run(provider, read_twice) == ("first", "first")
    assert _run(provider, lambda: backend_runtime.xp.value) == "second"


def test_namespace_proxy_requires_an_active_provider() -> None:
    with pytest.raises(RuntimeError, match="active backend provider"):
        _ = backend_runtime.xp.sum


def test_namespace_proxy_supports_standard_callable_introspection() -> None:
    assert inspect.unwrap(backend_runtime.xp) is backend_runtime.xp
