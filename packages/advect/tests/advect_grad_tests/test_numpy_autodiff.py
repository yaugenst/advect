"""Focused end-to-end autodiff contracts for the NumPy frontend."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad


def test_grad_composes_rules_and_supports_multiple_arguments() -> None:
    x = np.array([0.2, -0.4, 0.7])
    y = np.array([1.5, -0.25, 2.0])

    def loss(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.sum(np.sin(np.exp(left)) + left * right)

    dx, dy = ad.grad(loss, argnums=(0, 1))(x, y)

    assert_allclose(dx, np.cos(np.exp(x)) * np.exp(x) + y)
    assert_allclose(dy, x)


def test_value_and_grad_preserves_static_auxiliary_data() -> None:
    x = np.array([1.0, 2.0, 3.0])

    def loss(value: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        return np.sum(value * value), {"shape": value.shape, "label": "loss"}

    value, gradient, auxiliary = ad.value_and_grad(loss, has_aux=True)(x)

    assert_allclose(value, 14.0)
    assert_allclose(gradient, 2.0 * x)
    assert auxiliary == {"shape": (3,), "label": "loss"}


def test_vjp_and_jacobian_keep_numpy_values() -> None:
    matrix = np.array([[1.0, 2.0], [-0.5, 1.5], [2.0, -1.0]])
    x = np.array([0.3, -0.7])

    value, pullback = ad.vjp(lambda arg: matrix @ arg)(x)
    cotangent = np.array([0.5, -1.0, 2.0])

    assert isinstance(value, np.ndarray)
    assert_allclose(pullback(cotangent), matrix.T @ cotangent)
    assert_allclose(ad.jacobian(lambda arg: matrix @ arg)(x), matrix)


def test_grad_accumulates_reused_and_broadcast_inputs() -> None:
    x = np.array([[1.0], [2.0]])
    coefficient = np.array([1.0, 2.0, 3.0])

    gradient = ad.grad(lambda arg: np.sum(arg * arg + arg * coefficient))(x)

    assert_allclose(gradient, 2.0 * x * coefficient.size + np.sum(coefficient))


@pytest.mark.parametrize(
    ("operation", "expected_value", "expected_tangent"),
    [
        (np.maximum, np.array([0.0, 2.0, 1.0]), np.array([0.0, -0.3, 0.0])),
        (np.minimum, np.array([-1.0, 1.0, 0.5]), np.array([0.2, 0.0, 0.4])),
    ],
    ids=["maximum", "minimum"],
)
def test_elementwise_extrema_use_the_ephemeral_ufunc_path(
    operation: Any,
    expected_value: np.ndarray,
    expected_tangent: np.ndarray,
) -> None:
    x = np.array([-1.0, 2.0, 0.5])
    bound = np.array([0.0, 1.0, 1.0])
    tangent = np.array([0.2, -0.3, 0.4])

    value, output_tangent = ad.jvp(lambda argument: operation(argument, bound))(
        x,
        tangents=tangent,
    )

    assert_allclose(value, expected_value)
    assert_allclose(output_tangent, expected_tangent)


def _extreme_ldexp_values() -> tuple[np.ndarray, np.ndarray]:
    values = np.array(
        [
            np.ldexp(1.0, 100),
            np.ldexp(1.0, -100),
            2.0,
            0.5,
        ]
    )
    exponents = np.array([-1100, 1100, -1075, 1024], dtype=np.int32)
    return values, exponents


def test_ldexp_jvp_scales_extreme_tangents_without_intermediate_overflow() -> None:
    value, exponent = _extreme_ldexp_values()

    primal, tangent = ad.jvp(lambda argument: np.ldexp(argument, exponent))(
        value,
        tangents=value,
    )
    expected = np.ldexp(value, exponent)

    assert np.all(np.isfinite(expected))
    assert np.all(expected != 0)
    assert_allclose(primal, expected)
    assert_allclose(tangent, expected)


def test_ldexp_vjp_transposes_extreme_scaling_without_intermediate_overflow() -> None:
    value, exponent = _extreme_ldexp_values()
    cotangent = value.copy()

    primal, pullback = ad.vjp(lambda argument: np.ldexp(argument, exponent))(value)
    try:
        actual = pullback(cotangent)
    finally:
        pullback.close()
    expected = np.ldexp(cotangent, exponent)

    assert np.all(np.isfinite(expected))
    assert np.all(expected != 0)
    assert_allclose(primal, np.ldexp(value, exponent))
    assert_allclose(actual, expected)


def test_numpy_pytree_inputs_and_outputs_round_trip() -> None:
    params = {
        "weight": np.array([1.0, 2.0]),
        "bias": np.array([0.25, -0.5]),
    }

    gradients = ad.grad(lambda tree: np.sum(tree["weight"] * tree["weight"] + tree["bias"]))(params)
    assert_allclose(gradients["weight"], 2.0 * params["weight"])
    assert_allclose(gradients["bias"], np.ones_like(params["bias"]))

    value, pullback = ad.vjp(lambda arg: {"square": arg * arg, "shift": arg + 1})(params["weight"])
    assert set(value) == {"square", "shift"}
    assert_allclose(
        pullback({"square": np.ones_like(params["weight"]), "shift": None}),
        2.0 * params["weight"],
    )


def test_vjp_accumulates_cotangents_for_repeated_output_leaf() -> None:
    value = np.array([1.0, -2.0])
    output, pullback = ad.vjp(lambda argument: (argument, argument))(value)

    assert_allclose(output[0], value)
    assert_allclose(output[1], value)
    assert_allclose(
        pullback((np.array([2.0, 3.0]), np.array([-0.5, 4.0]))),
        np.array([1.5, 7.0]),
    )


def test_true_keyword_arguments_can_be_selected() -> None:
    x = np.array([1.0, 2.0, 3.0])

    def loss(value: np.ndarray, *, scale: float) -> np.ndarray:
        return np.sum(value * scale)

    gradients = ad.grad(loss, argnums=(0,), argnames=("scale",))(x, scale=0.5)
    positional, named = gradients

    assert_allclose(positional[0], np.full_like(x, 0.5))
    assert named["scale"] == pytest.approx(float(np.sum(x)))


def test_einsum_out_is_functionalized_for_gradients() -> None:
    def loss(a: np.ndarray, b: np.ndarray) -> np.floating[Any]:
        out = np.zeros_like(np.sum(a))
        np.einsum("ij,ji->", a, b, out=out)
        return cast("np.floating[Any]", out)

    a = np.array([[0.2, -0.1], [0.4, 0.3]])
    b = np.array([[1.1, 0.7], [0.6, 1.4]])
    grad_a, grad_b = ad.grad(loss, argnums=(0, 1))(a, b)

    assert_allclose(grad_a, b.T)
    assert_allclose(grad_b, a.T)


@pytest.mark.parametrize("contraction", ["dot", "tensordot"])
def test_contraction_hessian_preserves_enclosing_trace(contraction: str) -> None:
    matrix = np.array([[1.0, 2.0], [-0.5, 1.5]])
    x = np.array([0.3, -0.7])

    def loss(value: Any) -> Any:
        projected = (
            np.dot(matrix, value)
            if contraction == "dot"
            else np.tensordot(matrix, value, axes=([1], [0]))
        )
        return np.sum(projected * projected)

    assert_allclose(ad.hessian(loss)(x), 2.0 * matrix.T @ matrix)


def test_shape_jvps_preserve_nested_tangent_tracers() -> None:
    x = np.array([1.0, 2.0, 3.0])
    weights = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def loss(value: np.ndarray) -> np.ndarray:
        columns = np.broadcast_to(value[:, None], (3, 2))
        return np.sum(columns.T * weights)

    assert_allclose(ad.grad(loss)(x), np.array([5.0, 7.0, 9.0]))


def test_fft_jvp_accepts_numpy_default_n() -> None:
    x = np.array([1.0, 2.0, 3.0])
    tangent = np.ones_like(x)

    value, output_tangent = ad.jvp(lambda arg: np.fft.fft(arg))(x, tangents=tangent)

    assert_allclose(value, np.fft.fft(x))
    assert_allclose(output_tangent, np.fft.fft(tangent))


@pytest.mark.parametrize(
    ("loss", "first", "second"),
    [
        (
            lambda value: np.sum(np.log(value)),
            lambda value: 1.0 / value,
            lambda value: -1.0 / value**2,
        ),
        (
            lambda value: np.sum(np.sqrt(value)),
            lambda value: 0.5 / np.sqrt(value),
            lambda value: -0.25 / value**1.5,
        ),
    ],
    ids=["log", "sqrt"],
)
def test_division_based_jvps_transpose_and_nest(
    loss: Any,
    first: Any,
    second: Any,
) -> None:
    x = np.array([0.7, 1.2, 1.8])
    gradient = ad.grad(loss)

    assert_allclose(gradient(x), first(x))
    assert_allclose(ad.grad(lambda value: np.sum(gradient(value)))(x), second(x))


def test_nested_grad_retains_captured_outer_array() -> None:
    x = np.array([1.0, -2.0, 3.0])
    ones = np.ones_like(x)
    derivative = ad.grad(lambda outer: np.sum(ad.grad(lambda inner: np.sum(outer * inner))(ones)))

    assert_allclose(derivative(x), ones)


def test_nested_array_grad_retains_captured_outer_scalar() -> None:
    x = np.array([1.0, 2.0])

    def objective(outer: float) -> np.ndarray:
        def inner_loss(inner: np.ndarray) -> np.ndarray:
            return np.sum(outer * inner * inner)

        return np.sum(ad.grad(inner_loss)(x))

    assert_allclose(ad.grad(objective)(2.0), 6.0)
