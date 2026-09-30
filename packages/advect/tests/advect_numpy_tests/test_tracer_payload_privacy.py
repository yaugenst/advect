"""Tracer payload privacy contracts for the NumPy frontend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import advect as ad
import advect.numpy
from advect.core._errors import EscapedTracerError, TracingError
from advect.core._protocols import _snapshot_traced

if TYPE_CHECKING:
    from collections.abc import Callable


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


@pytest.mark.parametrize(
    "use",
    [
        lambda value: value * 2.0,
        lambda value: 2.0 * value,
        lambda value: -value,
        lambda value: value.real,
        lambda value: value.__iadd__(1.0),
    ],
    ids=["operator", "reflected", "unary", "real", "augmented"],
)
def test_escaped_rank_zero_tracers_report_the_escape_through_operators(
    use: Callable[[Any], Any],
) -> None:
    # Operators read a rank-zero tracer's weak-scalar category, which a
    # released tape no longer holds; they still report the escape.
    escaped: list[Any] = []

    def objective(value: Any) -> Any:
        escaped.extend((value, value * 1.0, np.sin(value)))
        return value * value

    ad.grad(objective)(3.0)

    for tracer in escaped:
        with pytest.raises(EscapedTracerError, match="escaped"):
            use(tracer)


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
