from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import advect as ad
from advect.testing import check_gradient


def test_check_gradient_validates_a_composed_function() -> None:
    def objective(inputs: tuple[object, object]) -> object:
        x, y = inputs
        return np.sum(np.sin(x * y))

    x = np.array([0.2, 0.4, 0.7])
    y = np.array([1.1, -0.3, 0.8])
    check_gradient(
        objective,
        (x, y),
        tangent=(np.array([0.3, -0.2, 0.5]), np.array([-0.1, 0.4, 0.2])),
    )


def test_check_gradient_names_custom_primitives_on_the_failing_path() -> None:
    @ad.primitive(name="tests.debugging.wrong_gradient")
    def wrong_gradient(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return x * x

    @wrong_gradient.def_jvp
    def wrong_jvp(
        output: np.ndarray[Any, Any],
        primals: tuple[object, ...],
        tangents: tuple[object | None, ...],
    ) -> np.ndarray[Any, Any]:
        del primals, tangents
        return np.zeros_like(output)

    @wrong_gradient.def_transpose
    def wrong_transpose(
        cotangent: object,
        primals: tuple[object, ...],
        output: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any]]:
        del cotangent, primals
        return (np.zeros_like(output),)

    def objective(x: object) -> object:
        return np.sum(wrong_gradient(x))

    with pytest.raises(AssertionError) as caught:
        check_gradient(objective, np.array([1.0, 2.0]))

    message = str(caught.value)
    assert "central finite differences" in message
    assert "Custom primitives on this path: tests.debugging.wrong_gradient" in message
    assert "run check_primitive" in message
