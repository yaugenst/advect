"""Contract tests for automatic array-api-strict integration."""

from __future__ import annotations

import pytest

from advect.autodiff.api.common import _require_array_namespace_for_higher_order
from advect.autodiff.rules.array_family.providers import (
    resolve_array_family_backend_provider,
)
from advect.core._errors import HigherOrderNotSupportedError

xp = pytest.importorskip("array_api_strict")


def test_array_api_strict_provider_resolves_from_runtime_value() -> None:
    runtime_value = xp.asarray([1.0, 2.0], dtype=xp.float64)

    provider = resolve_array_family_backend_provider(runtime_value)

    assert provider.backend == "array_api_strict"
    assert provider.namespace is xp


def test_array_api_strict_missing_namespace_capabilities_raise_typed_error() -> None:
    runtime_value = xp.asarray([1.0, 2.0], dtype=xp.float64)

    with pytest.raises(HigherOrderNotSupportedError) as exc_info:
        _ = _require_array_namespace_for_higher_order(args=(runtime_value,), kwargs={})
    message = str(exc_info.value)
    assert "diag" in message
    assert "zeros_like" in message
