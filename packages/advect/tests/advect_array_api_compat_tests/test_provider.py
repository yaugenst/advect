"""Focused provider-resolution tests."""

from __future__ import annotations

import warnings
from importlib.util import find_spec
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

import advect._array_api_compat as provider
from advect.core._array_api.frontend import ArrayAPINamespace
from advect.core._array_api.profiles import SUPPORTED_ARRAY_API_VERSIONS
from advect.core._array_api.providers import _get_array_namespace, _version_key
from advect.core._eval_dispatch import _can_donate_array
from advect_core_tests._backend_state import isolated_backend_state

if TYPE_CHECKING:
    from collections.abc import Iterator


def test_compatibility_bridge_has_no_public_registration_module() -> None:
    assert find_spec("advect.array_api_compat") is None


@pytest.fixture(autouse=True)
def _restore_backends() -> Iterator[None]:
    with isolated_backend_state():
        yield


@pytest.fixture()
def fake_cupy(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SimpleNamespace, list[str | None]]:
    """Resolve every non-NumPy value to one future-revision compatibility namespace."""
    namespace = SimpleNamespace(
        __name__="array_api_compat.cupy",
        __array_api_version__="2025.12",
        __array_namespace_info__=object,
        asarray=lambda item: item,
    )
    requests: list[str | None] = []

    def resolve(*_args: object, api_version: str | None = None, **_kwargs: object) -> object:
        requests.append(api_version)
        return namespace

    monkeypatch.setattr(provider.array_api_compat, "is_numpy_array", lambda _value: False)
    monkeypatch.setattr(provider.array_api_compat, "array_namespace", resolve)
    return namespace, requests


@pytest.mark.parametrize("array_api_version", SUPPORTED_ARRAY_API_VERSIONS)
def test_upstream_dependency_accepts_each_supported_revision_request(
    array_api_version: str,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        namespace = provider.array_api_compat.array_namespace(
            np.asarray([1.0]),
            api_version=array_api_version,
            use_compat=True,
        )

    assert namespace.__name__ == "array_api_compat.numpy"
    reported = _version_key(namespace.__array_api_version__)
    assert reported is not None
    assert reported >= cast("tuple[int, int]", _version_key(array_api_version))


@pytest.mark.parametrize("array_api_version", SUPPORTED_ARRAY_API_VERSIONS)
def test_fallback_requests_each_supported_revision_without_relabeling(
    fake_cupy: tuple[SimpleNamespace, list[str | None]],
    array_api_version: str,
) -> None:
    namespace, requests = fake_cupy

    resolved = _get_array_namespace(
        SimpleNamespace(shape=(2,), dtype="float32"), api_version=array_api_version
    )

    assert resolved is namespace
    assert requests == [array_api_version]
    assert resolved.__array_api_version__ == "2025.12"


@pytest.mark.parametrize(
    ("dtype", "expected_2022"),
    [("float32", "float64"), ("complex64", "complex128")],
)
@pytest.mark.parametrize("operation", ["sum", "prod"])
def test_generic_frontend_applies_requested_2022_accumulation_dtype(
    fake_cupy: tuple[SimpleNamespace, list[str | None]],
    dtype: str,
    expected_2022: str,
    operation: str,
) -> None:
    namespace, _requests = fake_cupy
    value = SimpleNamespace(shape=(2,), dtype=dtype)
    calls: list[object | None] = []

    def reduction(
        _value: object,
        *,
        axis: object = None,
        dtype: object | None = None,
        keepdims: bool = False,
    ) -> object:
        del axis, keepdims
        calls.append(dtype)
        return dtype

    for name in ("float32", "float64", "complex64", "complex128"):
        setattr(namespace, name, name)
    namespace.sum = namespace.prod = reduction

    resolved_2022 = _get_array_namespace(value, api_version="2022.12")
    resolved_2024 = _get_array_namespace(value, api_version="2024.12")
    selected_2022 = ArrayAPINamespace(resolved_2022, array_api_version="2022.12")
    selected_2024 = ArrayAPINamespace(resolved_2024, array_api_version="2024.12")

    assert resolved_2022 is namespace
    assert resolved_2024 is namespace
    assert getattr(selected_2022, operation)(value) == expected_2022
    assert getattr(selected_2024, operation)(value) is None
    assert calls == [expected_2022, None]


@pytest.mark.parametrize("is_cupy", [True, False])
def test_only_cupy_arrays_qualify_for_donation_without_a_namespace_proxy(
    fake_cupy: tuple[SimpleNamespace, list[str | None]],
    monkeypatch: pytest.MonkeyPatch,
    is_cupy: bool,  # noqa: FBT001
) -> None:
    namespace, requests = fake_cupy
    value = SimpleNamespace(
        shape=(2,),
        dtype="float32",
        flags=SimpleNamespace(owndata=True),
        base=None,
    )
    monkeypatch.setattr(provider.array_api_compat, "is_cupy_array", lambda _value: is_cupy)

    # Without an explicit request the fallback asks for the latest revision.
    assert _get_array_namespace(value) is namespace
    assert requests == [SUPPORTED_ARRAY_API_VERSIONS[-1]]
    assert _can_donate_array(value) is is_cupy


def test_fallback_supplies_the_native_numpy_namespace_for_scalars() -> None:
    assert _get_array_namespace(np.float32(1.0)) is np


def test_fallback_rejects_values_attached_to_an_external_autodiff_tape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = SimpleNamespace(requires_grad=True)
    monkeypatch.setattr(provider.array_api_compat, "is_numpy_array", lambda _value: False)

    with pytest.raises(TypeError, match="active autodiff tape"):
        _get_array_namespace(value)


def test_unsupported_values_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject(*_args: object, **_kwargs: object) -> object:
        msg = "unsupported"
        raise TypeError(msg)

    monkeypatch.setattr(provider.array_api_compat, "is_numpy_array", lambda _value: False)
    monkeypatch.setattr(provider.array_api_compat, "array_namespace", reject)

    assert _get_array_namespace(object()) is None
