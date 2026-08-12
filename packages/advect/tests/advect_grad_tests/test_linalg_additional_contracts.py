"""Additional public transform contracts for supported linalg norm and UPLO variants."""

from __future__ import annotations

from typing import Any

import array_api_strict as strict
import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad


def _real_inner(left: Any, right: Any) -> float:
    if isinstance(left, tuple):
        assert isinstance(right, tuple)
        return sum(_real_inner(a, b) for a, b in zip(left, right, strict=True))
    return float(np.real(np.vdot(np.asarray(left), np.asarray(right))))


def _assert_adjoint(
    function: Any,
    value: Any,
    tangent: Any,
    cotangent: Any,
    *,
    rtol: float = 2e-9,
    atol: float = 2e-10,
) -> Any:
    output, output_tangent = ad.jvp(function)(value, tangents=tangent)
    reverse_output, pullback = ad.vjp(function)(value)
    try:
        input_cotangent = pullback(cotangent)
    finally:
        pullback.close()

    if isinstance(output, tuple):
        for actual, expected in zip(reverse_output, output, strict=True):
            assert_allclose(np.asarray(actual), np.asarray(expected), rtol=rtol, atol=atol)
    else:
        assert_allclose(np.asarray(reverse_output), np.asarray(output), rtol=rtol, atol=atol)
    assert_allclose(
        _real_inner(cotangent, output_tangent),
        _real_inner(input_cotangent, tangent),
        rtol=rtol,
        atol=atol,
    )
    return input_cotangent


def _assert_jvp_matches_difference(
    function: Any,
    value: Any,
    tangent: Any,
    *,
    rtol: float = 4e-6,
    atol: float = 4e-7,
) -> None:
    _output, output_tangent = ad.jvp(function)(value, tangents=tangent)
    step = 1e-6
    positive = function(value + step * tangent)
    negative = function(value - step * tangent)

    if isinstance(output_tangent, tuple):
        for actual, positive_leaf, negative_leaf in zip(
            output_tangent,
            positive,
            negative,
            strict=True,
        ):
            assert_allclose(
                np.asarray(actual),
                np.asarray((positive_leaf - negative_leaf) / (2 * step)),
                rtol=rtol,
                atol=atol,
            )
        return
    assert_allclose(
        np.asarray(output_tangent),
        np.asarray((positive - negative) / (2 * step)),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("order", [0, 1.5, np.inf, -np.inf])
def test_numpy_vector_norm_orders_have_consistent_public_transforms(order: float) -> None:
    value = np.array(
        [
            [0.4 + 0.2j, -1.3 + 0.1j, 2.2 - 0.5j],
            [1.8 - 0.3j, 0.7 + 0.4j, -0.6 + 0.8j],
        ]
    )
    tangent = np.array(
        [
            [0.2 - 0.1j, 0.3 + 0.2j, -0.4 + 0.1j],
            [-0.1 + 0.3j, 0.5 - 0.2j, 0.2 + 0.4j],
        ]
    )

    def function(x: Any) -> Any:
        return np.linalg.norm(x, ord=order, axis=1, keepdims=True)

    _assert_jvp_matches_difference(function, value, tangent)
    _assert_adjoint(function, value, tangent, np.array([[0.7], [-0.4]]))


def test_numpy_default_norm_flattens_all_axes_for_public_transforms() -> None:
    value = np.arange(1.0, 13.0).reshape(2, 3, 2) / 4
    tangent = np.linspace(-0.4, 0.7, value.size).reshape(value.shape)

    def function(x: Any) -> Any:
        return np.linalg.norm(x, keepdims=True)

    output, _output_tangent = ad.jvp(function)(value, tangents=tangent)
    assert output.shape == (1, 1, 1)
    _assert_jvp_matches_difference(function, value, tangent)
    _assert_adjoint(function, value, tangent, np.array([[[1.3]]]))


@pytest.mark.parametrize("order", ["nuc", 2, -2, 1, -1, np.inf, -np.inf])
def test_numpy_matrix_norm_orders_support_nonfinal_axes(order: str | float) -> None:
    value = np.array(
        [
            [[1.2 + 0.1j, -0.4 + 0.2j], [0.3 - 0.2j, 1.5 + 0.4j], [2.2, 0.8 - 0.3j]],
            [[0.5 - 0.2j, 2.0 + 0.1j], [-1.1 + 0.3j, 0.9 - 0.2j], [1.3 + 0.1j, -0.6 + 0.4j]],
        ]
    )
    tangent = np.array(
        [
            [[0.1, -0.2j], [0.3 + 0.1j, -0.2], [0.1 - 0.3j, 0.4]],
            [[-0.2 + 0.1j, 0.3], [0.2, -0.1 + 0.2j], [-0.4, 0.2 + 0.1j]],
        ]
    )

    def function(x: Any) -> Any:
        return np.linalg.norm(x, ord=order, axis=(0, 2), keepdims=True)

    _assert_jvp_matches_difference(function, value, tangent)
    _assert_adjoint(function, value, tangent, np.array([[[0.6], [-0.8], [0.3]]]))


def test_array_api_norms_preserve_provider_and_multi_axis_semantics() -> None:
    value = strict.asarray(
        np.arange(1.0, 13.0).reshape(2, 2, 3) / 5,
        dtype=strict.float64,
    )
    tangent = strict.asarray(
        np.linspace(-0.3, 0.5, 12).reshape(2, 2, 3),
        dtype=strict.float64,
    )

    def function(x: Any) -> tuple[Any, Any]:
        namespace = x.__array_namespace__()
        return (
            namespace.linalg.matrix_norm(x, ord="nuc", keepdims=True),
            namespace.linalg.vector_norm(x, ord=3, axis=(0, 2), keepdims=True),
        )

    output, output_tangent = ad.jvp(function)(value, tangents=tangent)
    assert all(type(leaf) is type(value) for leaf in (*output, *output_tangent))
    assert_allclose(
        np.asarray(output[0]),
        np.linalg.norm(np.asarray(value), ord="nuc", axis=(-2, -1), keepdims=True),
    )
    assert_allclose(
        np.asarray(output[1]),
        np.sum(np.abs(np.asarray(value)) ** 3, axis=(0, 2), keepdims=True) ** (1 / 3),
    )

    cotangent = (
        strict.asarray([[[0.4]], [[-0.7]]], dtype=strict.float64),
        strict.asarray([[[0.2], [-0.5]]], dtype=strict.float64),
    )
    gradient = _assert_adjoint(function, value, tangent, cotangent)
    assert type(gradient) is type(value)


def _upper_triangle_inputs() -> tuple[np.ndarray, np.ndarray]:
    value = np.array(
        [
            [3.0, 0.4 + 0.2j, -0.1 + 0.3j],
            [9.0 + 7.0j, 1.8, 0.25 - 0.1j],
            [-8.0 + 6.0j, 5.0 - 4.0j, 0.9],
        ]
    )
    tangent = np.array(
        [
            [0.2, -0.1 + 0.3j, 0.25 - 0.2j],
            [3.0 + 2.0j, -0.15, 0.1 + 0.05j],
            [-4.0 + 1.0j, 2.0 - 3.0j, 0.3],
        ]
    )
    return value, tangent


def test_upper_triangle_eigh_ignores_the_inactive_lower_triangle() -> None:
    value, tangent = _upper_triangle_inputs()

    def function(x: Any) -> Any:
        return np.linalg.eigh(x, UPLO="U")

    (eigenvalues, eigenvectors), (d_values, d_vectors) = ad.jvp(function)(
        value,
        tangents=tangent,
    )
    reconstructed = (
        d_vectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
        + eigenvectors @ np.diag(d_values) @ eigenvectors.conj().T
        + eigenvectors @ np.diag(eigenvalues) @ d_vectors.conj().T
    )
    active_tangent = np.triu(tangent) + np.triu(tangent, 1).conj().T
    assert_allclose(reconstructed, active_tangent, rtol=2e-9, atol=2e-9)

    cotangent = (
        np.array([0.4, -0.2, 0.3]),
        np.conjugate(eigenvectors) * (0.15 - 0.1j),
    )
    gradient = _assert_adjoint(function, value, tangent, cotangent, rtol=3e-8, atol=3e-9)
    assert_allclose(np.tril(gradient, -1), 0.0)


def test_upper_triangle_eigvalsh_ignores_the_inactive_lower_triangle() -> None:
    value, tangent = _upper_triangle_inputs()

    def function(x: Any) -> Any:
        return np.linalg.eigvalsh(x, UPLO="U")

    _assert_jvp_matches_difference(function, value, tangent)
    gradient = _assert_adjoint(function, value, tangent, np.array([0.4, -0.2, 0.3]))
    assert_allclose(np.tril(gradient, -1), 0.0)
