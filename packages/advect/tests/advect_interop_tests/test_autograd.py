"""HIPS Autograd bridge qualification."""

from __future__ import annotations

import autograd
import autograd.numpy as anp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from advect.interop.autograd import wrap


def test_autograd_bridge_preserves_pytree_outputs_and_multi_argument_gradients() -> None:
    calls = 0

    def operation(parameters: dict[str, np.ndarray], scale: np.ndarray):
        nonlocal calls
        calls += 1
        field = parameters["field"] * scale
        return {"field": field, "energy": np.sum(field * field)}

    bridged = wrap(operation)
    parameters = {"field": anp.asarray([1.0, -2.0, 0.5])}
    scale = anp.asarray(1.5)

    direct = bridged(parameters, scale)
    assert_allclose(direct["energy"], 11.8125)
    calls = 0

    def objective(params, factor):
        return bridged(params, factor)["energy"]

    parameter_gradient, scale_gradient = autograd.grad(objective, argnum=(0, 1))(
        parameters,
        scale,
    )

    assert calls == 1
    assert_allclose(parameter_gradient["field"], 2 * parameters["field"] * scale**2)
    assert_allclose(
        scale_gradient,
        2 * scale * np.sum(parameters["field"] * parameters["field"]),
    )


def test_autograd_bridge_translates_the_complex_adjoint_convention() -> None:
    coefficient = 2.0 + 3.0j
    bridged = wrap(lambda value: coefficient * value)
    sample = np.asarray(1.5 + 2.0j)

    bridged_gradient = autograd.grad(
        lambda value: anp.real(bridged(value) * anp.conj(bridged(value)))
    )(sample)
    native_gradient = autograd.grad(
        lambda value: anp.real((coefficient * value) * anp.conj(coefficient * value))
    )(sample)

    assert_allclose(bridged_gradient, native_gradient)
    assert_allclose(bridged_gradient, 39.0 - 52.0j)


def test_autograd_bridge_matches_nonholomorphic_complex_outputs() -> None:
    bridged = wrap(
        lambda value: (
            np.conjugate(value),
            np.real(value),
            value.imag,
            np.abs(value) ** 2,
        )
    )
    sample = np.asarray(1.5 + 2.0j)

    def bridged_loss(value):
        conjugate, real, imag, power = bridged(value)
        return anp.real((1.0 + 2.0j) * conjugate) + 0.3 * real - 0.7 * imag + 0.2 * power

    def native_loss(value):
        return (
            anp.real((1.0 + 2.0j) * anp.conj(value))
            + 0.3 * anp.real(value)
            - 0.7 * anp.imag(value)
            + 0.2 * anp.abs(value) ** 2
        )

    assert_allclose(autograd.grad(bridged_loss)(sample), autograd.grad(native_loss)(sample))


def test_autograd_bridge_supplies_none_for_an_unused_output_cotangent() -> None:
    bridged = wrap(lambda value: {"used": value * value, "unused": value + 1})
    gradient = autograd.grad(lambda value: anp.sum(bridged(value)["used"]))(
        anp.asarray([2.0, -3.0])
    )
    assert_allclose(gradient, [4.0, -6.0])


def test_autograd_bridge_rejects_higher_order_differentiation() -> None:
    bridged = wrap(lambda value: np.sum(value * value))
    second = autograd.grad(autograd.grad(bridged))

    with pytest.raises(
        NotImplementedError,
        match=r"first-order VJPs only.*higher-order differentiation",
    ):
        second(anp.asarray(2.0))


def test_autograd_bridge_rejects_integer_input_leaves() -> None:
    bridged = wrap(lambda value: np.sum(value))
    with pytest.raises(TypeError, match="only NumPy floating and complex leaves"):
        bridged(np.asarray([1, 2], dtype=np.int64))


def test_autograd_bridge_accepts_python_scalars() -> None:
    gradient = autograd.grad(wrap(lambda value: value * value))(2.0)
    assert gradient == 4.0
