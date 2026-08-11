"""Tests for runtime array-family provider resolution."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from advect.autodiff.rules.array_family.providers import (
    resolve_array_family_backend_provider,
    try_resolve_array_family_backend_provider,
)


class _ArrayWithNamespace:
    def __init__(self, namespace: Any) -> None:
        self._namespace = namespace

    def __array_namespace__(self, *, api_version: str | None = None) -> Any:
        assert api_version == "2024.12"
        return self._namespace


def _nonstandard_runtime_namespace(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        __name__=name,
        __array_api_version__="2024.12",
        __array_namespace_info__=object,
        asarray=lambda value: value,
    )


def test_numpy_resolves_directly_from_its_runtime_namespace() -> None:
    value = np.asarray([1.0, 2.0])

    provider = resolve_array_family_backend_provider(value)

    assert provider.backend == "numpy"
    assert provider.namespace is np


def test_module_provider_is_reused() -> None:
    first = resolve_array_family_backend_provider(np.asarray([1.0]))
    second = try_resolve_array_family_backend_provider(np.asarray([2.0]))

    assert second is first


def test_nonstandard_namespace_error_names_protocol() -> None:
    runtime_value = _ArrayWithNamespace(_nonstandard_runtime_namespace("unsupported"))

    with pytest.raises(RuntimeError, match="Python Array API"):
        resolve_array_family_backend_provider(runtime_value)
    assert try_resolve_array_family_backend_provider(runtime_value) is None


def test_provider_resolution_does_not_guess_from_python_scalars() -> None:
    assert try_resolve_array_family_backend_provider(1.0) is None
    with pytest.raises(RuntimeError, match="Could not resolve"):
        resolve_array_family_backend_provider(1.0)
