"""NumPy split-family result-container contracts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import example, given, strategies as st

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
    # A traced section count alone does not own the split.
    with pytest.raises(ad.TracingError, match="requires a traced first argument"):
        ad.jvp(lambda sections: operation(value, sections))(
            np.asarray(2),
            tangents=np.asarray(0),
        )


@st.composite
def _split_calls(draw: st.DrawFn) -> tuple[Any, np.ndarray[Any, Any], tuple[Any, ...]]:
    shape = tuple(draw(st.lists(st.integers(0, 4), min_size=1, max_size=3)))
    operation = draw(st.sampled_from((np.split, np.array_split, np.hsplit, np.vsplit, np.dsplit)))
    if draw(st.booleans()):
        sections: object = draw(st.integers(1, 4))
    else:
        sections = sorted(draw(st.lists(st.integers(0, 5), max_size=3)))
    extra: tuple[Any, ...] = ()
    if operation in {np.split, np.array_split}:
        extra = (draw(st.integers(-len(shape), len(shape) - 1)),)
    value = np.arange(float(np.prod(shape))).reshape(shape)
    return operation, value, (sections, *extra)


@given(_split_calls())
@example((np.split, np.arange(12.0).reshape(3, 4), (2, 1)))
@example((np.array_split, np.arange(12.0).reshape(3, 4), (2, 1)))
# Empty split axes once defeated an axis-inference shortcut in the handler.
@example((np.split, np.zeros((0, 2)), (1, 0)))
@example((np.split, np.zeros((3, 0)), (3, 0)))
@example((np.array_split, np.zeros((2, 0)), (3, 1)))
@example((np.hsplit, np.zeros((3, 0)), ([1, 2],)))
@example((np.dsplit, np.zeros((1, 2, 0)), (2,)))
def test_split_family_matches_numpy_for_any_axis_and_sections(
    call: tuple[Any, np.ndarray[Any, Any], tuple[Any, ...]],
) -> None:
    operation, value, arguments = call
    try:
        expected = operation(value, *arguments)
    except (ValueError, IndexError) as error:
        with pytest.raises(type(error)):
            ad.jvp(lambda x: operation(x, *arguments))(value, tangents=np.ones_like(value))
        return
    direction = value + 0.5
    primal, tangent = ad.jvp(lambda x: operation(x, *arguments))(value, tangents=direction)

    assert len(primal) == len(expected) == len(tangent)
    for actual, reference, derivative, piece in zip(
        primal, expected, tangent, operation(direction, *arguments), strict=True
    ):
        np.testing.assert_array_equal(actual, reference)
        np.testing.assert_array_equal(derivative, piece)
