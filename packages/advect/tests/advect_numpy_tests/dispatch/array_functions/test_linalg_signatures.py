"""Signature-level NumPy linalg contracts that differ materially by flags."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import advect as ad


def test_rectangular_full_svd_rejects_singular_vector_derivatives() -> None:
    value = np.array([[1.5, -0.2], [0.4, 0.8], [-0.6, 0.3]])

    with pytest.raises(NotImplementedError, match="full_matrices=False"):
        ad.jvp(np.linalg.svd)(value, tangents=np.ones_like(value))

    result, pullback = ad.vjp(np.linalg.svd)(value)
    cotangent = (
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

    primal, tangent = ad.jvp(singular_values)(value, tangents=direction)
    epsilon = 1e-6
    finite_difference = (
        singular_values(value + epsilon * direction) - singular_values(value - epsilon * direction)
    ) / (2 * epsilon)

    np.testing.assert_allclose(primal, singular_values(value))
    np.testing.assert_allclose(tangent, finite_difference, rtol=2e-6, atol=2e-6)

    program = ad.stage(
        singular_values,
        specs=(ad.ArraySpec(value.shape, value.dtype),),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())
    for staged in (program, restored):
        np.testing.assert_allclose(staged(value), singular_values(value))


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

    program = ad.stage(
        solve,
        specs=(
            ad.ArraySpec(matrix.shape, matrix.dtype),
            ad.ArraySpec(right.shape, right.dtype),
        ),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())
    for staged in (program, restored):
        np.testing.assert_allclose(staged(matrix, right), expected)


def test_slogdet_preserves_named_outputs_across_program_lifetimes() -> None:
    value = np.array([[3.0, 1.0], [1.0, 2.0]])
    expected = np.linalg.slogdet(value)

    primal, tangent = ad.jvp(np.linalg.slogdet)(
        value,
        tangents=np.full_like(value, 0.1),
    )
    assert primal._fields == ("sign", "logabsdet")
    assert tangent._fields == ("sign", "logabsdet")
    np.testing.assert_allclose(primal.sign, expected.sign)
    np.testing.assert_allclose(primal.logabsdet, expected.logabsdet)

    program = ad.stage(
        np.linalg.slogdet,
        specs=(ad.ArraySpec(value.shape, value.dtype),),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())
    for staged in (program, restored):
        result = staged(value)
        assert result._fields == ("sign", "logabsdet")
        np.testing.assert_allclose(result.sign, expected.sign)
        np.testing.assert_allclose(result.logabsdet, expected.logabsdet)
