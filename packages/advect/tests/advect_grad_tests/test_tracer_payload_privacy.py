"""Tracer payload privacy contracts for the generic Array API frontend."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import advect as ad
from advect.core._errors import EscapedTracerError, TracingError
from advect.core._protocols import _snapshot_traced

xp = pytest.importorskip("array_api_strict")


def test_scalar_payload_is_private_and_internal_snapshot_is_lifetime_checked() -> None:
    escaped: list[Any] = []

    def objective(value: Any) -> Any:
        with pytest.raises(TracingError, match="payloads are private"):
            _ = value.value
        node_id, payload = _snapshot_traced(value)
        assert node_id == value.node_id
        assert payload.shape == ()
        assert payload.dtype == np.dtype("float64")
        np.testing.assert_allclose(payload, 4.0)
        escaped.append(value)
        return value * 3.0

    value, tangent = ad.jvp(objective)(4.0, tangents=1.0)

    assert value == 12.0
    assert tangent == 3.0
    with pytest.raises(EscapedTracerError, match="creating trace has already exited"):
        _snapshot_traced(escaped[0])


def test_scalar_repr_is_payload_free_and_lifetime_checked() -> None:
    escaped: list[Any] = []
    representations: list[str] = []

    def objective(value: Any) -> Any:
        representations.append(repr(value))
        escaped.append(value)
        return value * value

    gradient = ad.grad(objective)(12_345.678)

    assert gradient == pytest.approx(24_691.356)
    assert representations[0].startswith("TracedArray(node=")
    assert "12345.678" not in representations[0]
    with pytest.raises(EscapedTracerError, match="creating trace has already exited"):
        repr(escaped[0])


def test_array_api_payload_is_private_and_dynamic_transforms_still_compose() -> None:
    privacy_checks = 0

    def objective(value: Any) -> Any:
        nonlocal privacy_checks
        with pytest.raises(TracingError, match="payloads are private"):
            _ = value.value
        privacy_checks += 1
        namespace = value.__array_namespace__()
        return namespace.sum(value * value)

    primal = xp.asarray([1.0, -2.0], dtype=xp.float32)
    tangent = xp.asarray([0.5, 0.25], dtype=xp.float32)

    gradient = ad.grad(objective)(primal)
    value, directional = ad.jvp(objective)(primal, tangents=tangent)

    np.testing.assert_allclose(np.asarray(gradient), np.asarray(2.0 * primal))
    np.testing.assert_allclose(np.asarray(value), np.asarray(xp.asarray(5.0, dtype=xp.float32)))
    np.testing.assert_allclose(
        np.asarray(directional),
        np.asarray(xp.asarray(0.0, dtype=xp.float32)),
    )
    assert privacy_checks == 2


def test_staging_generic_array_api_code_never_needs_a_tracer_payload() -> None:
    def objective(value: Any) -> Any:
        namespace = value.__array_namespace__()
        return namespace.sum(value * value)

    program = ad.stage(objective, specs=(ad.ArraySpec((2,), "float32"),))
    primal = xp.asarray([1.0, -2.0], dtype=xp.float32)

    np.testing.assert_allclose(
        np.asarray(program(primal)),
        np.asarray(xp.asarray(5.0, dtype=xp.float32)),
    )
