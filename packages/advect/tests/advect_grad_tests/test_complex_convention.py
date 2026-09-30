"""Complex autodiff convention and weak-scalar regression tests."""

from __future__ import annotations

from typing import Any

import array_api_strict as strict
import numpy as np
import pytest
from numpy.testing import assert_allclose

from advect import grad, hessian, hvp, jvp, value_and_grad, vjp


def test_grad_absolute_squared_is_two_z() -> None:
    z = np.array([1.0 + 2.0j, -0.5 + 0.25j], dtype=np.complex64)

    result = grad(lambda value: np.sum(np.abs(value) ** 2))(z)

    assert result.dtype == z.dtype
    assert_allclose(result, 2 * z, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "transform",
    [
        lambda loss, z: grad(loss)(z),
        lambda loss, z: value_and_grad(loss)(z),
        lambda loss, z: jvp(loss)(z, tangents=1.0 + 0.0j),
        lambda loss, z: vjp(loss)(z),
    ],
)
def test_selected_python_complex_scalar_requires_zero_dimensional_array(transform) -> None:
    with pytest.raises(TypeError, match=r"Python complex scalar.*0-D array"):
        transform(lambda z: abs(z) ** 2, 1.0 + 2.0j)


def test_grad_imaginary_part_is_positive_i() -> None:
    z = strict.asarray([1.0 + 2.0j, -0.5 + 0.25j], dtype=strict.complex64)

    def imaginary_sum(value: object) -> object:
        xp = value.__array_namespace__()  # type: ignore[attr-defined]
        return xp.sum(xp.imag(value))

    result = grad(imaginary_sum)(z)

    assert result.dtype == z.dtype
    assert_allclose(np.asarray(result), np.full(2, 1j, dtype=np.complex64), rtol=1e-6, atol=1e-6)


def test_real_input_projects_complex_pullback_and_preserves_float32() -> None:
    x = np.array([1.0, 2.0], dtype=np.float32)
    cotangent = np.array([1.0 + 3.0j, 2.0 - 4.0j], dtype=np.complex64)
    _value, pullback = vjp(lambda value: (1.0 + 2.0j) * value)(x)

    result = pullback(cotangent)

    assert result.dtype == x.dtype
    assert_allclose(result, np.array([7.0, -6.0], dtype=np.float32))


@pytest.mark.parametrize("operation", [np.add, np.subtract, np.multiply, np.divide])
@pytest.mark.parametrize(
    "dtypes",
    [(np.float32, np.float64), (np.complex64, np.complex128), (np.float32, np.complex128)],
)
def test_mixed_precision_cotangents_take_each_input_dtype(operation, dtypes) -> None:
    inputs = tuple(np.array([0.75, 1.5], dtype=dtype) for dtype in dtypes)

    def loss(x, y):
        return np.real(np.sum(operation(x, y)))

    gradients = grad(loss, argnums=(0, 1))(*inputs)
    references = grad(loss, argnums=(0, 1))(*(value.astype(np.complex128) for value in inputs))

    for gradient, value, reference in zip(gradients, inputs, references, strict=True):
        assert gradient.dtype == value.dtype
        expected = reference if np.iscomplexobj(value) else np.real(reference)
        assert_allclose(gradient, expected, rtol=1e-6)


@pytest.mark.parametrize("dtype", [strict.float64, strict.float32], ids=["float64", "float32"])
def test_foreign_provider_cotangent_passes_through_a_lifted_python_scalar(dtype) -> None:
    # A Python-float primal lifts with NumPy; an Array API cotangent pulled back
    # through it keeps its own provider instead of taking the NumPy dtype.
    def pulled_back(cotangent):
        return vjp(lambda value: value + 2.0)(3.0)[1](cotangent)

    cotangent = strict.asarray(2.0, dtype=dtype)
    value, tangent = jvp(pulled_back)(cotangent, tangents=cotangent)
    gradient = grad(pulled_back)(cotangent)

    assert (value, tangent) == (2.0, 2.0)
    assert gradient.dtype == dtype
    assert float(gradient) == 1.0


def test_complex_product_and_unary_coefficients_are_conjugated() -> None:
    z = np.array([0.25 + 0.5j, -0.4 + 0.2j], dtype=np.complex64)
    coefficient = np.complex64(1.5 + 0.75j)

    product_grad = grad(lambda value: np.real(np.sum(coefficient * value)))(z)
    sine_grad = grad(lambda value: np.real(np.sum(np.sin(value))))(z)

    assert_allclose(product_grad, np.full_like(z, np.conjugate(coefficient)), rtol=1e-6)
    assert_allclose(sine_grad, np.conjugate(np.cos(z)), rtol=1e-6)


def test_complex_division_transpose_conjugates_both_partials() -> None:
    numerator = np.array([0.25 + 0.5j, -0.4 + 0.2j], dtype=np.complex64)
    denominator = np.array([1.5 - 0.25j, 0.75 + 0.5j], dtype=np.complex64)

    numerator_grad, denominator_grad = grad(
        lambda x, y: np.real(np.sum(x / y)),
        argnums=(0, 1),
    )(numerator, denominator)

    assert_allclose(numerator_grad, np.conjugate(1 / denominator), rtol=1e-6)
    assert_allclose(
        denominator_grad,
        np.conjugate(-numerator / denominator**2),
        rtol=1e-6,
    )


def test_python_complex_scalar_remains_weak_in_jvp() -> None:
    x = np.array([1.0, 2.0], dtype=np.float32)
    tangent = np.array([0.5, -0.25], dtype=np.float32)

    value, output_tangent = jvp(lambda arg: 1j * arg)(x, tangents=tangent)

    assert value.dtype == np.dtype(np.complex64)
    assert output_tangent.dtype == np.dtype(np.complex64)
    assert_allclose(value, 1j * x)
    assert_allclose(output_tangent, 1j * tangent)


def test_grad_rejects_complex_scalar_output() -> None:
    z = np.array([1.0 + 2.0j, -0.5 + 0.25j], dtype=np.complex64)

    with pytest.raises(ValueError, match="real scalar output"):
        grad(lambda value: np.sum(value))(z)  # noqa: PLW0108 - grad boundary


def test_negative_real_power_higher_order_is_warning_clean() -> None:
    x = np.array([-1.2, 0.5, 2.0], dtype=np.float64)
    direction = np.array([0.25, -0.5, 0.75], dtype=np.float64)

    def loss(value: np.ndarray) -> np.ndarray:
        return np.sum(value**3)

    with np.errstate(all="raise"):
        _value, product = hvp(loss)(x, vectors=direction)
        matrix = hessian(loss)(x)

    assert_allclose(product, 6.0 * x * direction)
    assert_allclose(matrix, np.diag(6.0 * x))


def test_complex_absolute_forward_over_forward_keeps_the_direction_traced() -> None:
    z = np.array([0.3 + 0.4j, -1.2 + 0.5j])
    direction = np.array([1.0 + 0.5j, -0.3 + 0.2j])

    def directional(value: np.ndarray) -> np.ndarray:
        return jvp(np.abs)(value, tangents=direction)[1]

    _value, curvature = jvp(directional)(z, tangents=direction)

    # d^2|z|[v, v] = |v|^2 / |z| - Re(conj(z) v)^2 / |z|^3.
    magnitude = np.abs(z)
    projection = np.real(np.conjugate(z) * direction)
    expected = np.abs(direction) ** 2 / magnitude - projection**2 / magnitude**3
    assert_allclose(curvature, expected, rtol=1e-12)


def test_array_api_complex_cast_keeps_the_imaginary_part_of_its_derivatives() -> None:
    """A complex cast to another precision matches the NumPy frontend.

    The cast recognizes a complex target through the provider's
    ``complexfloating`` class, which a standard namespace must expose unwrapped.
    """
    value = np.array([0.3 + 0.1j, -1.2 + 0.5j])
    tangent = np.array([1.0 + 2.0j, -0.5 + 1.5j])

    def cast(v: Any) -> Any:
        xp = v.__array_namespace__()
        return xp.astype(v, xp.complex64)

    def loss(v: Any) -> Any:
        xp = v.__array_namespace__()
        return xp.sum(xp.imag(cast(v) * cast(v)))

    _output, output_tangent = jvp(cast)(strict.asarray(value), tangents=strict.asarray(tangent))
    _output, expected_tangent = jvp(lambda v: v.astype(np.complex64))(value, tangents=tangent)
    gradient = grad(loss)(strict.asarray(value))
    expected_gradient = grad(lambda v: np.sum(np.imag(v.astype(np.complex64) ** 2)))(value)

    assert np.asarray(output_tangent).dtype == np.complex64
    assert_allclose(np.asarray(output_tangent), expected_tangent)
    assert_allclose(np.asarray(gradient), expected_gradient)
