"""Tracer payload privacy contracts for the NumPy frontend."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import advect as ad
import advect.numpy
from advect.core._errors import EscapedTracerError, TracingError
from advect.core._protocols import _snapshot_traced


def test_numpy_payload_is_private_without_breaking_functionalized_mutation() -> None:
    primal = np.arange(5.0)
    escaped: list[Any] = []

    def update(traced: Any) -> Any:
        with pytest.raises(TracingError, match="payloads are private"):
            _ = traced.value
        escaped.append(traced)
        updated = traced.copy()
        updated[1:-1] += 2.0
        return updated

    value, tangent = ad.jvp(update)(primal, tangents=np.ones_like(primal))

    expected = primal.copy()
    expected[1:-1] += 2.0
    np.testing.assert_array_equal(value, expected)
    np.testing.assert_array_equal(tangent, np.ones_like(primal))

    with pytest.raises(EscapedTracerError, match="escaped"):
        _snapshot_traced(escaped[0])


def test_same_dtype_astype_copy_creates_owned_mutable_value() -> None:
    def update(traced: Any) -> Any:
        copied = traced.astype(traced.dtype, copy=True)
        assert copied is not traced
        copied += 2.0
        return copied

    primal = np.arange(4.0, dtype=np.float32)
    value, tangent = ad.jvp(update)(primal, tangents=np.ones_like(primal))

    np.testing.assert_array_equal(value, primal + 2.0)
    np.testing.assert_array_equal(tangent, np.ones_like(primal))
