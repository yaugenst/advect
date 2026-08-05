"""Focused provider-resolution tests."""

from __future__ import annotations

import warnings
from importlib.util import find_spec
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

import advect._array_api_compat as provider
from advect.core._array_api import ArrayAPINamespace
from advect.core._array_namespace import _get_array_namespace
from advect.core._eval_dispatch import _can_donate_array
from advect_core_tests._backend_state import isolated_backend_state

if TYPE_CHECKING:
    from collections.abc import Iterator


def _revision_key(version: str) -> tuple[int, int]:
    year, month = version.split(".")
    return int(year), int(month)


def test_compatibility_bridge_has_no_public_registration_module() -> None:
    assert find_spec("advect.array_api_compat") is None


@pytest.fixture(autouse=True)
def _restore_backends() -> Iterator[None]:
    with isolated_backend_state():
        yield


def test_fallback_preserves_a_compatible_future_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = SimpleNamespace(shape=(2,), dtype="float32")
    namespace = SimpleNamespace(
        __name__="array_api_compat.cupy",
        __array_api_version__="2025.12",
        __array_namespace_info__=lambda: object(),
        asarray=lambda item: item,
    )
    monkeypatch.setattr(provider.array_api_compat, "is_numpy_array", lambda _value: False)
    monkeypatch.setattr(
        provider.array_api_compat,
        "array_namespace",
        lambda *_args, **_kwargs: namespace,
    )

    resolved = _get_array_namespace(value)

    assert resolved is namespace
    assert resolved.__array_api_version__ == "2025.12"
    assert resolved.__name__ == "array_api_compat.cupy"
    assert resolved.asarray(3) == 3


@pytest.mark.parametrize("array_api_version", ["2022.12", "2023.12", "2024.12"])
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
    assert _revision_key(namespace.__array_api_version__) >= _revision_key(array_api_version)


@pytest.mark.parametrize("array_api_version", ["2022.12", "2023.12", "2024.12"])
def test_fallback_requests_each_supported_revision_without_relabeling(
    monkeypatch: pytest.MonkeyPatch,
    array_api_version: str,
) -> None:
    value = SimpleNamespace(shape=(2,), dtype="float32")
    namespace = SimpleNamespace(
        __name__="array_api_compat.cupy",
        __array_api_version__="2025.12",
        __array_namespace_info__=lambda: object(),
        asarray=lambda item: item,
    )
    requests: list[str | None] = []

    def resolve(*_args: object, api_version: str | None, **_kwargs: object) -> object:
        requests.append(api_version)
        return namespace

    monkeypatch.setattr(provider.array_api_compat, "is_numpy_array", lambda _value: False)
    monkeypatch.setattr(provider.array_api_compat, "array_namespace", resolve)

    resolved = _get_array_namespace(value, api_version=array_api_version)

    assert resolved is namespace
    assert requests == [array_api_version]
    assert resolved.__array_api_version__ == "2025.12"


@pytest.mark.parametrize(
    ("dtype", "expected_2022"),
    [("float32", "float64"), ("complex64", "complex128")],
)
@pytest.mark.parametrize("operation", ["sum", "prod"])
def test_generic_frontend_applies_requested_2022_accumulation_dtype(
    monkeypatch: pytest.MonkeyPatch,
    dtype: str,
    expected_2022: str,
    operation: str,
) -> None:
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

    namespace = SimpleNamespace(
        __name__="array_api_compat.cupy",
        __array_api_version__="2025.12",
        __array_namespace_info__=lambda: object(),
        asarray=lambda item: item,
        float32="float32",
        float64="float64",
        complex64="complex64",
        complex128="complex128",
        sum=reduction,
        prod=reduction,
    )
    monkeypatch.setattr(provider.array_api_compat, "is_numpy_array", lambda _value: False)
    monkeypatch.setattr(
        provider.array_api_compat,
        "array_namespace",
        lambda *_args, **_kwargs: namespace,
    )

    resolved_2022 = _get_array_namespace(value, api_version="2022.12")
    resolved_2024 = _get_array_namespace(value, api_version="2024.12")
    selected_2022 = ArrayAPINamespace(resolved_2022, array_api_version="2022.12")
    selected_2024 = ArrayAPINamespace(resolved_2024, array_api_version="2024.12")

    assert resolved_2022 is namespace
    assert resolved_2024 is namespace
    assert getattr(selected_2022, operation)(value) == expected_2022
    assert getattr(selected_2024, operation)(value) is None
    assert calls == [expected_2022, None]


def test_cupy_arrays_are_qualified_for_donation_without_a_namespace_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = SimpleNamespace(
        shape=(2,),
        dtype="float32",
        flags=SimpleNamespace(owndata=True),
        base=None,
    )
    namespace = SimpleNamespace(
        __name__="array_api_compat.cupy",
        __array_api_version__="2025.12",
        __array_namespace_info__=lambda: object(),
        asarray=lambda item: item,
    )
    monkeypatch.setattr(provider.array_api_compat, "is_numpy_array", lambda _value: False)
    monkeypatch.setattr(provider.array_api_compat, "is_cupy_array", lambda _value: True)
    monkeypatch.setattr(
        provider.array_api_compat,
        "array_namespace",
        lambda *_args, **_kwargs: namespace,
    )

    resolved = _get_array_namespace(value)

    assert resolved is namespace
    assert _can_donate_array(value)


def test_non_cupy_arrays_do_not_qualify_for_donation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = SimpleNamespace(
        shape=(2,),
        dtype="float32",
        flags=SimpleNamespace(owndata=True),
        base=None,
    )
    monkeypatch.setattr(provider.array_api_compat, "is_cupy_array", lambda _value: False)

    assert not _can_donate_array(value)


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
