"""Shared state isolation for array-family provider tests."""

from __future__ import annotations

import pytest

from advect.autodiff.rules.array_family import providers as provider_module
from advect.core import _array_namespace as array_namespace_module


@pytest.fixture()
def isolated_provider_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give each provider contract an empty runtime and namespace registry."""
    monkeypatch.setattr(provider_module, "_ARRAY_FAMILY_BACKEND_PROVIDERS", {})
    monkeypatch.setattr(provider_module, "_RUNTIME_ARRAY_API_PROVIDERS", {})
    monkeypatch.setattr(array_namespace_module, "_NAMESPACE_BY_TYPE", {})
    monkeypatch.setattr(array_namespace_module, "_WRAPPED_NAMESPACE_BY_TYPES", {})
