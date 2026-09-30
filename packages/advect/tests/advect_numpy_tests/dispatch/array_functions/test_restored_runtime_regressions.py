"""Concrete-runtime regressions found while qualifying restored NumPy forms."""

from __future__ import annotations

import numpy as np

import advect as ad
from advect_numpy_tests._assertions import assert_staged_round_trip


def test_arange_preserves_the_consumed_like_dispatch_anchor() -> None:
    anchor = np.array([2.0, 4.0])

    def call(value: np.ndarray) -> np.ndarray:
        return np.arange(4, dtype=np.float64, like=value)

    primal, tangent = ad.jvp(call)(anchor, tangents=np.ones_like(anchor))
    np.testing.assert_array_equal(primal, np.arange(4, dtype=np.float64))
    np.testing.assert_array_equal(tangent, np.zeros(4, dtype=np.float64))

    assert_staged_round_trip(call, anchor, rtol=0.0)


def test_lstsq_rank_dtype_matches_numpy_2_4() -> None:
    matrix = np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])
    right = np.array([1.0, 2.0, 3.0])
    expected = np.linalg.lstsq(matrix, right, rcond=None)

    actual = ad.jvp(lambda a: np.linalg.lstsq(a, right, rcond=None))(
        matrix,
        tangents=np.ones_like(matrix),
    )[0]

    assert np.asarray(actual[2]).dtype == np.asarray(expected[2]).dtype
    assert int(actual[2]) == int(expected[2])
