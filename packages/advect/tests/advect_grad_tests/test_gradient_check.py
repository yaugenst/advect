"""Tests for author-facing composed-gradient validation."""

from __future__ import annotations

from typing import Any

import array_api_strict as strict
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


@pytest.mark.parametrize(
    "value",
    [
        strict.asarray([1.0, -2.0], dtype=strict.float64),
        strict.asarray([1.0 + 0.5j, -2.0 - 1.5j], dtype=strict.complex128),
    ],
)
def test_check_gradient_accepts_an_array_api_provider(value: object) -> None:
    def energy(field):
        namespace = field.__array_namespace__()
        return namespace.sum(namespace.abs(field) ** 2)

    check_gradient(energy, value)


def test_check_gradient_stops_at_the_first_agreeing_step() -> None:
    shifts: list[float] = []

    def energy(x: Any) -> Any:
        if not ad.is_traced(x):
            shifts.append(abs(float(x[0]) - 2.0))
        return np.sum(x * x)

    # Central differences are exact for a quadratic, so the first step agrees
    # and the second is never evaluated.
    check_gradient(energy, np.array([2.0]), epsilons=(1e-2, 1e-3))
    assert shifts == pytest.approx([1e-2, 1e-2])


def test_check_gradient_validates_steps_and_the_reverse_adjoint() -> None:
    with pytest.raises(ValueError, match="epsilons must be a non-empty sequence"):
        check_gradient(lambda x: x * x, 2.0, epsilons=())

    wrong_adjoint = ad.primitive(name="tests.testing.wrong_adjoint")(lambda x: x * x)
    wrong_adjoint.def_jvp(lambda _output, primals, tangents: 2 * primals[0] * tangents[0])
    wrong_adjoint.def_transpose(lambda cotangent, _primals, _output: (np.zeros_like(cotangent),))
    with pytest.raises(AssertionError, match="reverse gradient disagreed with the JVP"):
        check_gradient(lambda x: np.sum(wrong_adjoint(x)), np.array([1.0, 2.0]))


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
