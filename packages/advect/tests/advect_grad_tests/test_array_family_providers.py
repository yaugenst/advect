"""Tests for array-family backend-provider registration and resolution."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from advect.autodiff.rules.array_family.providers import (
    ArrayFamilyBackendProvider,
    get_array_family_backend_provider,
    register_array_family_backend_provider,
    resolve_array_family_backend_provider,
    try_resolve_array_family_backend_provider,
)


@dataclass(frozen=True, slots=True)
class _TestProvider(ArrayFamilyBackendProvider):
    backend: str
    namespace: Any
    ext: Any | None = None


class _ArrayWithNamespace:
    def __init__(self, namespace: Any) -> None:
        self._namespace = namespace

    def __array_namespace__(self, *, api_version: str | None = None) -> Any:
        assert api_version == "2024.12"
        return self._namespace


def _runtime_namespace(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        __name__=name,
        __array_api_version__="2024.12",
        __array_namespace_info__=lambda: object(),
        __version__="2.3.5",
        asarray=lambda value: value,
    )


def test_register_and_get_provider_by_backend_key(isolated_provider_registry: None) -> None:
    namespace = _runtime_namespace("numpy")
    provider = _TestProvider(backend="numpy", namespace=namespace)

    register_array_family_backend_provider(provider)

    resolved = get_array_family_backend_provider("numpy")
    assert resolved is provider


@pytest.mark.parametrize("extension_first", [False, True])
def test_provider_registration_composes_extensions_independent_of_order(
    isolated_provider_registry: None,
    *,
    extension_first: bool,
) -> None:
    namespace = SimpleNamespace(__name__="numpy")
    extension = SimpleNamespace(scientific_marker=object())
    base = _TestProvider(backend="numpy", namespace=namespace, ext=namespace)
    scientific = _TestProvider(backend="numpy", namespace=namespace, ext=extension)

    first, second = (scientific, base) if extension_first else (base, scientific)
    register_array_family_backend_provider(first)
    register_array_family_backend_provider(second)

    resolved = get_array_family_backend_provider("numpy")
    assert resolved.namespace is namespace
    assert resolved.ext is not None
    assert cast("Any", resolved.ext).scientific_marker is extension.scientific_marker


def test_provider_registration_chains_multiple_extension_packs(
    isolated_provider_registry: None,
) -> None:
    namespace = SimpleNamespace(__name__="numpy")
    first = SimpleNamespace(first_marker=object())
    second = SimpleNamespace(second_marker=object())

    register_array_family_backend_provider(
        _TestProvider(backend="numpy", namespace=namespace, ext=first)
    )
    register_array_family_backend_provider(
        _TestProvider(backend="numpy", namespace=namespace, ext=second)
    )

    extension = get_array_family_backend_provider("numpy").ext
    assert extension is not None
    assert cast("Any", extension).first_marker is first.first_marker
    assert cast("Any", extension).second_marker is second.second_marker


def test_resolve_provider_from_runtime_values(isolated_provider_registry: None) -> None:
    namespace = _runtime_namespace("numpy")
    provider = _TestProvider(backend="numpy", namespace=namespace)
    register_array_family_backend_provider(provider)

    runtime_value = _ArrayWithNamespace(namespace)
    resolved = resolve_array_family_backend_provider(runtime_value)

    assert resolved is provider


def test_resolve_provider_accepts_dotted_namespace_backend_keys(
    isolated_provider_registry: None,
) -> None:
    provider = _TestProvider(backend="numpy", namespace=SimpleNamespace(__name__="numpy"))
    register_array_family_backend_provider(provider)

    runtime_namespace = _runtime_namespace("numpy.array_api")
    runtime_value = _ArrayWithNamespace(runtime_namespace)

    resolved = resolve_array_family_backend_provider(runtime_value)
    assert resolved is provider


def test_numpy_resolves_directly_from_its_runtime_namespace(
    isolated_provider_registry: None,
) -> None:
    value = np.asarray([1.0, 2.0])

    provider = resolve_array_family_backend_provider(value)

    assert provider.backend == "numpy"
    assert provider.namespace is np


def test_nonstandard_missing_provider_error_names_protocol(
    isolated_provider_registry: None,
) -> None:
    runtime_namespace = _runtime_namespace("cupy")
    runtime_value = _ArrayWithNamespace(runtime_namespace)

    with pytest.raises(RuntimeError, match="Python Array API"):
        resolve_array_family_backend_provider(runtime_value)


def test_try_resolve_provider_returns_none_when_unregistered(
    isolated_provider_registry: None,
) -> None:
    runtime_namespace = _runtime_namespace("cupy")

    assert try_resolve_array_family_backend_provider(_ArrayWithNamespace(runtime_namespace)) is None


def test_try_resolve_provider_does_not_guess_from_python_scalars(
    isolated_provider_registry: None,
) -> None:
    namespace = SimpleNamespace(__name__="numpy")
    register_array_family_backend_provider(_TestProvider(backend="numpy", namespace=namespace))

    assert try_resolve_array_family_backend_provider(1.0) is None
    assert resolve_array_family_backend_provider(1.0).namespace is namespace


def test_scalar_backend_hint_is_explicit_and_array_resolution_stays_dynamic(
    isolated_provider_registry: None,
) -> None:
    numpy_namespace = SimpleNamespace(__name__="numpy")
    dummy_namespace = _runtime_namespace("dummy")
    numpy_provider = _TestProvider(backend="numpy", namespace=numpy_namespace)
    dummy_provider = _TestProvider(backend="dummy", namespace=dummy_namespace)
    register_array_family_backend_provider(numpy_provider)
    register_array_family_backend_provider(dummy_provider)

    with pytest.raises(RuntimeError, match="Could not resolve"):
        resolve_array_family_backend_provider(1.0)
    assert resolve_array_family_backend_provider(1.0, scalar_backend_hint="numpy") is numpy_provider
    assert (
        resolve_array_family_backend_provider(
            _ArrayWithNamespace(dummy_namespace),
            scalar_backend_hint="numpy",
        )
        is dummy_provider
    )


@pytest.mark.parametrize("reverse_order", [False, True])
def test_extension_namespace_public_collisions_are_rejected_independent_of_order(
    isolated_provider_registry: None,
    *,
    reverse_order: bool,
) -> None:
    namespace = SimpleNamespace(__name__="numpy")
    first_extension = SimpleNamespace(shared=object())
    second_extension = SimpleNamespace(shared=object())
    first = _TestProvider(backend="numpy", namespace=namespace, ext=first_extension)
    second = _TestProvider(backend="numpy", namespace=namespace, ext=second_extension)
    first, second = (second, first) if reverse_order else (first, second)

    register_array_family_backend_provider(first)
    with pytest.raises(ValueError, match=r"overlapping public attributes.*shared"):
        register_array_family_backend_provider(second)
    assert get_array_family_backend_provider("numpy") is first


def test_extension_registration_and_reregistration_are_idempotent(
    isolated_provider_registry: None,
) -> None:
    namespace = SimpleNamespace(__name__="numpy")
    first = _TestProvider(
        backend="numpy",
        namespace=namespace,
        ext=SimpleNamespace(first_marker=object()),
    )
    second = _TestProvider(
        backend="numpy",
        namespace=namespace,
        ext=SimpleNamespace(second_marker=object()),
    )
    register_array_family_backend_provider(first)
    register_array_family_backend_provider(second)
    composed = get_array_family_backend_provider("numpy")

    register_array_family_backend_provider(first)
    register_array_family_backend_provider(second)

    resolved = get_array_family_backend_provider("numpy")
    assert resolved is composed
    assert cast("Any", resolved.ext).first_marker is cast("Any", first.ext).first_marker
    assert cast("Any", resolved.ext).second_marker is cast("Any", second.ext).second_marker
