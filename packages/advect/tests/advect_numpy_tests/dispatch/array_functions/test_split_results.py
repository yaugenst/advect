"""NumPy split-family result-container contracts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import advect as ad


@pytest.mark.parametrize(
    ("operation", "value"),
    [
        (np.split, np.arange(8.0)),
        (np.array_split, np.arange(7.0)),
        (np.hsplit, np.arange(8.0).reshape(2, 4)),
        (np.vsplit, np.arange(8.0).reshape(4, 2)),
        (np.dsplit, np.arange(8.0).reshape(1, 2, 4)),
    ],
    ids=("split", "array_split", "hsplit", "vsplit", "dsplit"),
)
def test_split_family_preserves_numpy_list_results(
    operation: Any,
    value: np.ndarray[Any, Any],
) -> None:
    expected = operation(value, 2)
    primal, tangent = ad.jvp(lambda x: operation(x, 2))(
        value,
        tangents=np.ones_like(value),
    )

    assert isinstance(primal, list)
    assert isinstance(tangent, list)
    assert len(primal) == len(expected) == len(tangent)
    for actual, reference, direction in zip(primal, expected, tangent, strict=True):
        np.testing.assert_array_equal(actual, reference)
        np.testing.assert_array_equal(direction, np.ones_like(reference))
