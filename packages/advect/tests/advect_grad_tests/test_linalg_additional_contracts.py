"""Public upper-triangle regressions for NumPy linear-algebra rules."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.testing import assert_allclose

import advect as ad


def _upper_triangle_inputs() -> tuple[
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
]:
    value = np.array(
        [
            [4.0, 1.0 + 0.4j, -0.3 + 0.2j],
            [11.0 - 7.0j, 2.0, 0.5 - 0.6j],
            [-9.0 + 3.0j, 8.0 + 2.0j, -1.0],
        ],
        dtype=complex,
    )
    tangent = np.array(
        [
            [0.4, -0.2 + 0.1j, 0.3 - 0.05j],
            [13.0 - 2.0j, -0.1, 0.25 + 0.2j],
            [-5.0 + 6.0j, 7.0 - 4.0j, 0.2],
        ],
        dtype=complex,
    )
    return value, tangent


def test_upper_triangle_eigh_uses_the_selected_triangle() -> None:
    """``eigh(..., UPLO='U')`` ignores the inactive lower triangle."""
    value, tangent = _upper_triangle_inputs()

    def function(x: np.ndarray[Any, Any]) -> tuple[Any, Any]:
        return np.linalg.eigh(x, UPLO="U")

    (eigenvalues, eigenvectors), (d_values, d_vectors) = ad.jvp(function)(value, tangents=tangent)
    active_tangent = np.triu(tangent) + np.triu(tangent, 1).conj().T
    reconstruction = (
        d_vectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
        + eigenvectors @ np.diag(d_values) @ eigenvectors.conj().T
        + eigenvectors @ np.diag(eigenvalues) @ d_vectors.conj().T
    )
    assert_allclose(reconstruction, active_tangent, rtol=2e-6, atol=2e-7)

    cotangent = (
        np.array([0.5, -0.7, 0.2]),
        np.conjugate(eigenvectors) * (0.15 - 0.1j),
    )
    _output, pullback = ad.vjp(function)(value)
    try:
        gradient = pullback(cotangent)
    finally:
        pullback.close()

    assert_allclose(
        np.real(np.vdot(cotangent[0], d_values) + np.vdot(cotangent[1], d_vectors)),
        np.real(np.vdot(gradient, tangent)),
        rtol=2e-6,
        atol=2e-7,
    )
    assert_allclose(np.tril(gradient, -1), 0.0, atol=2e-10)


def test_upper_triangle_eigvalsh_uses_the_selected_triangle() -> None:
    """Lowercase ``UPLO='u'`` selects the upper triangle."""
    value, tangent = _upper_triangle_inputs()

    def function(x: np.ndarray[Any, Any]) -> Any:
        return np.linalg.eigvalsh(x, UPLO="u")

    expected, eigenvectors = np.linalg.eigh(value, UPLO="U")
    output, directional = ad.jvp(function)(value, tangents=tangent)
    active_tangent = np.triu(tangent) + np.triu(tangent, 1).conj().T
    expected_directional = np.real(np.diag(eigenvectors.conj().T @ active_tangent @ eigenvectors))
    assert_allclose(output, expected, rtol=2e-9, atol=2e-10)
    assert_allclose(directional, expected_directional, rtol=2e-6, atol=2e-7)

    cotangent = np.array([0.5, -0.7, 0.2])
    _output, pullback = ad.vjp(function)(value)
    try:
        gradient = pullback(cotangent)
    finally:
        pullback.close()

    assert_allclose(
        np.real(np.vdot(cotangent, directional)),
        np.real(np.vdot(gradient, tangent)),
        rtol=2e-6,
        atol=2e-7,
    )
    assert_allclose(np.tril(gradient, -1), 0.0, atol=2e-10)
