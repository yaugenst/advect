"""Evaluation and abstract contracts of the internal ``advect.scatter_add``."""

from __future__ import annotations

from typing import Any

import array_api_strict as strict
import numpy as np
import pytest
from hypothesis import example, given, strategies as st
from hypothesis.extra import numpy as hnp

from advect.core._abstract_model import ArraySpec
from advect.core._eval_dispatch import bind_node_evaluator
from advect.core._registry import get_registry

_ELEMENTS = {
    "float32": st.floats(-1e3, 1e3, width=32),
    "float64": st.floats(-1e3, 1e3),
    "complex128": st.complex_numbers(max_magnitude=1e3, allow_nan=False, allow_infinity=False),
}


@st.composite
def _scatters(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Draw values, in-range positions, an axis, and an output length.

    A short output makes repeated positions, and so runs to sum, likely.
    """
    rank = draw(st.integers(1, 3))
    axis = draw(st.integers(0, rank - 1))
    count = draw(st.integers(0, 24))
    size = draw(st.integers(1 if count else 0, 8))
    shape = list(draw(hnp.array_shapes(min_dims=rank, max_dims=rank, min_side=0, max_side=3)))
    shape[axis] = count
    dtype = draw(st.sampled_from(tuple(_ELEMENTS)))
    values = draw(hnp.arrays(dtype, tuple(shape), elements=_ELEMENTS[dtype]))
    positions = draw(hnp.arrays(np.int64, (count,), elements=st.integers(0, max(size - 1, 0))))
    # A loaded artifact may spell the axis from the end.
    return values, positions, axis - draw(st.sampled_from((0, rank))), size


def _add_at(values: np.ndarray, positions: np.ndarray, axis: int, size: int) -> np.ndarray:
    """Accumulate slices one by one, independently of any provider's ``add.at``."""
    slices = np.moveaxis(values, axis, 0)
    result = np.zeros((size, *slices.shape[1:]), dtype=values.dtype)
    for position, piece in zip(positions, slices, strict=True):
        result[position] += piece
    return np.moveaxis(result, 0, axis)


@given(case=_scatters())
@example(case=(np.arange(6.0).reshape(2, 3), np.zeros(3, dtype=np.int64), 1, 1))
@example(case=(np.arange(6.0).reshape(2, 3), np.array([1, 0, 1]), -1, 2))
@example(case=(np.ones((0, 2)), np.zeros(0, dtype=np.int64), 0, 0))
@example(case=(np.array([1e16, 1.0, -1e16]), np.array([0, 1, 0]), 0, 2))
def test_every_provider_scatters_as_numpy_add_at(case: tuple[Any, ...]) -> None:
    """NumPy scatters with ``add.at``; a provider without it sums sorted runs.

    Both orders round a run's sum within the bound of sequential summation,
    so a small total never cancels against a large prefix.
    """
    values, positions, axis, size = case
    evaluate = bind_node_evaluator("advect.scatter_add", {"axis": axis, "size": size})
    reference = _add_at(values, positions, axis, size)

    np.testing.assert_array_equal(evaluate((values, positions), None, None), reference)

    portable = np.asarray(evaluate((strict.asarray(values), strict.asarray(positions)), None, None))
    assert portable.dtype == reference.dtype
    assert portable.shape == reference.shape
    epsilon = np.finfo(values.dtype).eps
    bound = (positions.size + 1) * epsilon * _add_at(np.abs(values), positions, axis, size)
    assert np.all(np.abs(portable - reference) <= bound)


def test_abstract_scatter_places_the_positions_along_the_axis() -> None:
    evaluator = get_registry().get("advect.scatter_add").abstract_evaluator
    assert evaluator is not None
    (result,) = evaluator(
        (ArraySpec((2, 5, 3), "float32"), ArraySpec((5,), "int64")),
        {"axis": -2, "size": 7},
    )
    assert (result.shape, result.dtype) == ((2, 7, 3), "float32")


@pytest.mark.parametrize(
    ("values", "positions", "attrs", "message"),
    [
        ((2, 3), ((3,), "int64"), {"axis": 0, "size": 4}, "disagree along"),
        ((3,), ((3, 1), "int64"), {"axis": 0, "size": 4}, "one-dimensional integer"),
        ((3,), ((3,), "float64"), {"axis": 0, "size": 4}, "one-dimensional integer"),
        ((3,), ((3,), "int64"), {"axis": 0, "size": -1}, "non-negative integer"),
        ((3,), ((3,), "int64"), {"axis": 1, "size": 4}, "out of bounds"),
    ],
)
def test_abstract_scatter_rejects_a_malformed_signature(
    values: tuple[int, ...],
    positions: tuple[tuple[int, ...], str],
    attrs: dict[str, int],
    message: str,
) -> None:
    evaluator = get_registry().get("advect.scatter_add").abstract_evaluator
    assert evaluator is not None
    specs = (ArraySpec(values, "float64"), ArraySpec(*positions))
    with pytest.raises(ValueError, match=message):
        evaluator(specs, attrs)
