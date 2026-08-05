"""Public dynamic forward-mode contracts."""

from __future__ import annotations

import math

import numpy as np
import pytest

import advect as ad


def test_jvp_scalar_square() -> None:
    value, tangent = ad.jvp(lambda x: x * x)(3.0, tangents=2.0)

    assert value == pytest.approx(9.0)
    assert tangent == pytest.approx(12.0)


def test_jvp_multiple_arguments() -> None:
    def function(x: float, y: float) -> float:
        return x * y + np.sin(y)

    value, tangent = ad.jvp(function, argnums=(0, 1))(
        2.0,
        0.5,
        tangents=(3.0, 4.0),
    )

    expected_tangent = 3.0 * 0.5 + 4.0 * (2.0 + math.cos(0.5))
    assert value == pytest.approx(2.0 * 0.5 + math.sin(0.5))
    assert tangent == pytest.approx(expected_tangent)


def test_jvp_disconnected_argument_returns_zero_tangent() -> None:
    def function(x: float, _y: float) -> float:
        return x * x

    value, tangent = ad.jvp(function, argnums=1)(3.0, 4.0, tangents=5.0)

    assert value == pytest.approx(9.0)
    assert tangent == pytest.approx(0.0)


def test_jvp_multi_output_preserves_pytree_structure() -> None:
    value, tangent = ad.jvp(lambda x: (x, x * x))(2.0, tangents=3.0)

    assert value == (2.0, 4.0)
    assert tangent == pytest.approx((3.0, 12.0))


def test_jvp_rejects_a_tangent_with_the_wrong_pytree() -> None:
    function = ad.jvp(lambda params: params["x"] * params["x"])

    with pytest.raises(ValueError, match="pytree structure"):
        function({"x": 2.0}, tangents=2.0)
