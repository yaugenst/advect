"""Edge contracts for array-family namespace and selective-rule dispatch."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from advect.autodiff.rules.array_family import _backend_runtime as backend_runtime
from advect.autodiff.rules.array_family.providers import ArrayFamilyBackendProvider


@dataclass(frozen=True, slots=True)
class _Provider(ArrayFamilyBackendProvider):
    backend: str
    namespace: Any
    ext: Any | None = None


def test_active_namespace_provider_falls_back_to_extension() -> None:
    provider = _Provider(
        backend="test-extension-fallback",
        namespace=SimpleNamespace(),
        ext=SimpleNamespace(extension_only="extension-value"),
    )

    assert (
        backend_runtime.run_with_array_family_backend_provider(
            provider,
            lambda: backend_runtime.xp.extension_only,
        )
        == "extension-value"
    )


def test_active_namespace_provider_reports_an_unknown_attribute() -> None:
    provider = _Provider(
        backend="test-missing-attribute",
        namespace=SimpleNamespace(),
        ext=SimpleNamespace(),
    )

    with pytest.raises(AttributeError, match=r"test-missing-attribute.*unknown"):
        backend_runtime.run_with_array_family_backend_provider(
            provider,
            lambda: backend_runtime.xp.unknown,
        )


def test_namespace_proxy_reraises_provider_resolution_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_resolution(*_values: object) -> ArrayFamilyBackendProvider:
        msg = "no backend provider"
        raise RuntimeError(msg)

    monkeypatch.setattr(
        backend_runtime,
        "resolve_array_family_backend_provider",
        fail_resolution,
    )

    with pytest.raises(RuntimeError, match="no backend provider"):
        _ = backend_runtime.xp.attribute_without_type_fallback


def test_namespace_proxy_supports_standard_callable_introspection() -> None:
    assert inspect.unwrap(backend_runtime.xp) is backend_runtime.xp


def test_selective_vjp_wrapper_uses_the_active_provider() -> None:
    provider = _Provider(
        backend="test-selective-active",
        namespace=SimpleNamespace(),
    )
    seen: list[ArrayFamilyBackendProvider | None] = []

    def full_vjp(
        _ans: object,
        _x: object,
        *,
        g: object,
    ) -> tuple[object]:
        return (g,)

    def selective_vjp(
        _ans: object,
        _x: object,
        *,
        g: object,
        active_input_indices: tuple[int, ...],
    ) -> tuple[object | None]:
        seen.append(backend_runtime.current_array_backend_provider())
        return (g if active_input_indices else None,)

    full_vjp.__advect_vjp_for_input_indices__ = selective_vjp
    wrapped = backend_runtime.wrap_array_family_vjp_rule(full_vjp)
    wrapped_selective = wrapped.__advect_vjp_for_input_indices__

    actual = backend_runtime.run_with_array_family_backend_provider(
        provider,
        wrapped_selective,
        2.0,
        2.0,
        g=3.0,
        active_input_indices=(0,),
    )

    assert actual == (3.0,)
    assert seen == [provider]
