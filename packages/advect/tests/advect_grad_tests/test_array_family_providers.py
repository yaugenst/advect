"""Tests for runtime array-family provider resolution."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import array_api_strict as strict
import numpy as np
import pytest

from advect.autodiff.rules.array_family.providers import (
    _runtime_array_api_provider,
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


@pytest.mark.parametrize(
    ("value", "backend", "namespace"),
    [
        (np.asarray([1.0, 2.0]), "numpy", np),
        (strict.asarray([1.0, 2.0], dtype=strict.float64), "array_api_strict", strict),
    ],
    ids=["numpy", "array-api-strict"],
)
def test_runtime_value_resolves_its_own_namespace(value: Any, backend: str, namespace: Any) -> None:
    provider = resolve_array_family_backend_provider(value)

    assert provider.backend == backend
    assert provider.namespace is namespace


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


@pytest.mark.parametrize(
    ("reported", "accepted"),
    [("2024.12", True), ("2025.12", True), ("2023.12", False), ("latest", False)],
)
def test_derivative_provider_requires_the_negotiated_revision(
    reported: str,
    accepted: bool,  # noqa: FBT001 - parametrized expectation
) -> None:
    # Rules and negotiation share one predicate, so a derivative provider is
    # never built for a namespace that could not have negotiated the revision.
    namespace = _nonstandard_runtime_namespace("provider")
    namespace.__array_api_version__ = reported
    namespace.zeros_like = namespace.ones_like = lambda value: value

    provider = _runtime_array_api_provider("provider", namespace, array_api_version="2024.12")

    assert (provider is not None) is accepted
