"""Signature-level NumPy linalg contracts that differ materially by flags."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import advect as ad
from advect_numpy_tests._assertions import (
    assert_jvp_matches_central_difference,
    assert_staged_round_trip,
)


def test_rectangular_full_svd_rejects_singular_vector_derivatives() -> None:
    value = np.array([[1.5, -0.2], [0.4, 0.8], [-0.6, 0.3]])

    with pytest.raises(NotImplementedError, match="full_matrices=False"):
        ad.jvp(np.linalg.svd)(value, tangents=np.ones_like(value))

    result, pullback = ad.vjp(np.linalg.svd)(value)
    cotangent = type(result)(
        np.ones_like(result[0]),
        np.zeros_like(result[1]),
        np.zeros_like(result[2]),
    )
    with pytest.raises(NotImplementedError, match="full_matrices=False"):
        pullback(cotangent)


def test_svd_singular_values_support_the_hermitian_algorithm_flag() -> None:
    value = np.array([[2.0, 0.3], [0.3, -1.2]])
    direction = np.array([[0.2, -0.1], [-0.1, 0.4]])

    def singular_values(x: Any) -> Any:
        return np.linalg.svd(
            x,
            full_matrices=False,
            compute_uv=False,
            hermitian=True,
        )

    assert_jvp_matches_central_difference(
        singular_values, (value,), (direction,), rtol=2e-6, atol=2e-6
    )
    assert_staged_round_trip(singular_values, value)


@pytest.mark.parametrize(
    "singular_values",
    [
        lambda x: np.linalg.svd(x, compute_uv=0),
        lambda x: np.linalg.svd(x, compute_uv=np.False_),
        lambda x: np.linalg.svd(x, False, False),  # noqa: FBT003
    ],
    ids=("integer-flag", "numpy-bool-flag", "positional-flags"),
)
@pytest.mark.parametrize("shape", [(2, 2), (3, 2)], ids=("square", "rectangular"))
def test_svd_reads_compute_uv_by_truthiness_in_every_lifetime(
    singular_values: Any,
    shape: tuple[int, int],
) -> None:
    value = np.arange(1.0, 1.0 + np.prod(shape)).reshape(shape) + np.eye(*shape)
    expected = singular_values(value)

    primal, tangent = ad.jvp(singular_values)(value, tangents=np.ones_like(value))
    assert primal.shape == tangent.shape == expected.shape
    np.testing.assert_allclose(primal, expected)
    assert_staged_round_trip(singular_values, value)


def test_linalg_required_operands_accept_keyword_spelling_when_staged() -> None:
    matrix = np.array([[3.0, 1.0], [1.0, 2.0]])
    right = np.array([1.0, 4.0])

    def solve(a: Any, b: Any) -> Any:
        return np.linalg.solve(a=a, b=b)

    expected = np.linalg.solve(matrix, right)
    primal, tangent = ad.jvp(solve, argnums=(0, 1))(
        matrix,
        right,
        tangents=(np.zeros_like(matrix), np.zeros_like(right)),
    )
    np.testing.assert_allclose(primal, expected)
    np.testing.assert_array_equal(tangent, np.zeros_like(expected))

    assert_staged_round_trip(solve, matrix, right)
