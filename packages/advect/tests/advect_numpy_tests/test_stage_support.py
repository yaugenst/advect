"""Completeness and representative execution for NumPy staging support."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.lib import scimath

import advect as ad


def test_scimath_real_to_complex_boundary_is_dynamic_only() -> None:
    positive = np.array([4.0])
    negative = np.array([-4.0])

    positive_primal, positive_tangent = ad.jvp(scimath.sqrt)(
        positive,
        tangents=np.ones_like(positive),
    )
    negative_primal, negative_tangent = ad.jvp(scimath.sqrt)(
        negative,
        tangents=np.ones_like(negative),
    )

    np.testing.assert_allclose(positive_primal, scimath.sqrt(positive))
    np.testing.assert_allclose(positive_tangent, np.array([0.25]))
    assert positive_primal.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(negative_primal, scimath.sqrt(negative))
    np.testing.assert_allclose(negative_tangent, np.array([-0.25j]))
    assert negative_primal.dtype == np.dtype(np.complex128)

    with pytest.raises(ad.TracingError, match=r"dynamic-only.*output dtype"):
        ad.stage(
            scimath.sqrt,
            specs=(ad.ArraySpec(positive.shape, positive.dtype),),
        )


def test_special_abstract_composites_round_trip() -> None:
    matrix = np.array([[2.0, 0.2], [0.1, 1.5]])
    vector = np.arange(6.0).reshape(2, 3)

    matrix_program = ad.stage(
        lambda value: np.linalg.matrix_power(value, -2),
        specs=(ad.ArraySpec(matrix.shape, matrix.dtype),),
    )
    average_program = ad.stage(
        lambda value: (
            value * np.size(value, axis=1)
            + np.average(value, axis=1, weights=np.array([1.0, 2.0, 3.0]))[:, None]
        ),
        specs=(ad.ArraySpec(vector.shape, vector.dtype),),
    )
    cumulative_program = ad.stage(
        lambda value: np.cumulative_sum(value, axis=1, include_initial=True),
        specs=(ad.ArraySpec(vector.shape, vector.dtype),),
    )

    np.testing.assert_allclose(matrix_program(matrix), np.linalg.matrix_power(matrix, -2))
    np.testing.assert_allclose(
        average_program(vector),
        vector * np.size(vector, axis=1)
        + np.average(vector, axis=1, weights=np.array([1.0, 2.0, 3.0]))[:, None],
    )
    np.testing.assert_allclose(
        cumulative_program(vector),
        np.cumulative_sum(vector, axis=1, include_initial=True),
    )


def test_controlled_float32_reductions_stage_without_dtype_creep() -> None:
    value = np.arange(6, dtype=np.float32).reshape(2, 3)
    mask = np.array([[True, False, True], [True, True, False]])
    functions = (
        lambda array: np.mean(array, axis=1, keepdims=True, where=mask),
        lambda array: np.var(
            array,
            axis=1,
            correction=1,
            keepdims=True,
            where=mask,
        ),
    )

    for function in functions:
        program = ad.stage(
            function,
            specs=(ad.ArraySpec(value.shape, value.dtype),),
        )
        result = program(value)
        reference = function(value)
        assert result.dtype == np.dtype(np.float32)
        np.testing.assert_allclose(result, reference)
