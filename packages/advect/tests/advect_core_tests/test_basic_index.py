"""Tests for the canonical serialized index representation and staged basic indexing."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

import advect as ad
from advect.core._basic_index import (
    decode_basic_index,
    decode_index,
    encode_basic_index,
    normalize_basic_index,
)
from advect.core._errors import TracingError
from advect.numpy._traced_array_indexing import index_to_attrs, normalize_integer_scalars

if TYPE_CHECKING:
    from collections.abc import Callable

    from hypothesis.strategies import DataObject


def _assert_staged_equals(
    function: Callable[..., object],
    expected: np.ndarray,
    *values: np.ndarray,
) -> None:
    program = ad.stage(
        function,
        specs=tuple(ad.ArraySpec(value.shape, value.dtype.name) for value in values),
    )
    for staged in (program, ad.StagedProgram.from_dict(program.to_dict())):
        actual = np.asarray(staged(*values))
        assert (actual.shape, actual.dtype) == (expected.shape, expected.dtype)
        np.testing.assert_array_equal(actual, expected)


@given(data=st.data())
def test_basic_index_codec_and_staging_match_numpy(data: DataObject) -> None:
    shape = data.draw(hnp.array_shapes(min_dims=0, max_dims=4, min_side=0, max_side=5))
    index = data.draw(hnp.basic_indices(shape, allow_newaxis=True, allow_ellipsis=True))
    components = index if isinstance(index, tuple) else (index,)
    encoded = encode_basic_index(index)
    assert decode_basic_index(encoded) == components
    assert index_to_attrs(components) == encoded

    # The NumPy frontend reads NumPy integer scalars as the integers they index with.
    if data.draw(st.booleans(), label="numpy integers"):
        index = tuple(np.int64(item) if type(item) is int else item for item in components)
    value = np.arange(math.prod(shape), dtype=np.int64).reshape(shape)
    selected = np.asarray(value[index])
    _assert_staged_equals(lambda array: array[index], selected, value)

    kept = data.draw(st.integers(0, selected.ndim), label="replacement rank")
    replacement_shape = tuple(
        data.draw(st.sampled_from((1, size))) for size in selected.shape[selected.ndim - kept :]
    )
    replacement = -1 - np.arange(math.prod(replacement_shape), dtype=np.int64)
    replacement = replacement.reshape(replacement_shape)
    assigned = value.copy()
    assigned[index] = replacement

    def assign(array: np.ndarray, update: np.ndarray) -> np.ndarray:
        result = array.copy()
        result[index] = update
        return result

    _assert_staged_equals(assign, assigned, value, replacement)


def test_basic_index_codec_reads_slice_bounds_through_index() -> None:
    index = (1, slice(np.int8(0), np.array(-1), np.int32(2)), None, Ellipsis)
    expected = (1, slice(0, -1, 2), None, Ellipsis)

    normalized = normalize_basic_index(index)
    bounds = normalized[1]

    assert normalized == expected
    assert isinstance(bounds, slice)
    assert {type(bounds.start), type(bounds.stop), type(bounds.step)} == {int}
    assert decode_basic_index(encode_basic_index(index)) == expected
    with pytest.raises(TypeError, match="slice indices must be integers"):
        encode_basic_index(slice(1.5, None))
    with pytest.raises(TypeError, match="slice indices must be integers"):
        normalize_integer_scalars(slice(None, np.float64(1.0)))


@pytest.mark.parametrize(
    "item",
    [np.int64(1), np.array(1), 1.0],
    ids=["numpy-integer", "zero-dimensional-array", "float"],
)
def test_basic_index_codec_requires_python_integer_components(item: object) -> None:
    with pytest.raises(TracingError, match="Basic indexing supports only"):
        encode_basic_index((0, item))


def test_numpy_frontend_reads_integer_scalars_but_not_zero_dimensional_arrays() -> None:
    array_index = np.array(1)

    integer, bounds, array = normalize_integer_scalars(
        (np.int64(1), slice(np.intp(-1), None), array_index)
    )

    assert (integer, bounds) == (1, slice(-1, None))
    assert {type(integer), type(bounds.start)} == {int}
    assert array is array_index
    assert index_to_attrs((integer, bounds)) == encode_basic_index((1, slice(-1, None)))


@pytest.mark.parametrize(
    "index",
    [
        np.int64(1),
        (np.intp(-1), Ellipsis),
        (0, np.int32(2)),
        slice(np.int64(1), None),
        (slice(None), slice(None, None, np.int8(2))),
        slice(np.array(1), None),
    ],
    ids=["integer", "ellipsis", "tuple", "slice-start", "slice-step", "array-slice-start"],
)
def test_staged_indexing_reads_numpy_integers_as_python_integers(index: object) -> None:
    def update(value: Any) -> tuple[Any, Any]:
        result = value.copy()
        result[index] += 1.0
        return value[index], result

    value = np.arange(6.0, dtype=np.float32).reshape(2, 3)
    program = ad.stage(update, specs=(ad.ArraySpec(value.shape, "float32"),))

    for staged in (program, ad.StagedProgram.from_dict(program.to_dict())):
        for actual, expected in zip(staged(value), update(value), strict=True):
            np.testing.assert_array_equal(actual, expected)


def test_numpy_indexing_with_a_zero_dimensional_array_never_aliases() -> None:
    # test_staged_indexing_reports_invalid_basic_indices owns its staged rejection.
    def scale_row(value: Any) -> Any:
        result = value.copy()
        row = result[np.array(1)]
        row *= 3.0
        return np.sum(result)

    value = np.arange(6.0).reshape(2, 3)

    assert scale_row(value) == 15.0
    np.testing.assert_array_equal(ad.grad(scale_row)(value), np.ones_like(value))


@pytest.mark.parametrize(
    ("index", "error", "message"),
    [
        ((Ellipsis, Ellipsis), IndexError, "Only one ellipsis"),
        ((0, 0, 0), IndexError, "Too many indices"),
        (3, IndexError, "out of bounds"),
        ("row", TracingError, "Basic indexing supports only"),
        (np.array(1), TracingError, "Basic indexing supports only"),
        ((0, np.array(1)), TracingError, "Basic indexing supports only"),
        (slice(None, None, 0), ValueError, "slice step cannot be zero"),
        (slice(0.5, None), TypeError, "slice indices must be integers"),
    ],
    ids=[
        "multiple-ellipsis",
        "too-many",
        "integer-out-of-bounds",
        "string",
        "zero-dimensional-array",
        "zero-dimensional-array-component",
        "zero-step",
        "float-slice-bound",
    ],
)
def test_staged_indexing_reports_invalid_basic_indices(
    index: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        ad.stage(lambda value: value[index], specs=(ad.ArraySpec((2, 3), "float32"),))


def test_concrete_frontend_extends_only_array_index_encoding() -> None:
    key = (slice(None, None, -1), np.array([1, 3], dtype=np.int32), None)

    encoded = index_to_attrs(key)

    assert encoded[0] == encode_basic_index((key[0],))[0]
    assert encoded[1] == {
        "type": "array",
        "dtype": "int64",
        "shape": (2,),
        "values": [1, 3],
    }
    assert encoded[2] == encode_basic_index((None,))[0]
    decoded = decode_index(
        encoded,
        array_decoder=lambda values, dtype, shape: np.asarray(values, dtype=dtype).reshape(shape),
    )
    np.testing.assert_array_equal(decoded[1], np.array([1, 3], dtype=np.int64))


def test_boolean_scalar_index_is_rejected_at_the_canonical_boundary() -> None:
    with pytest.raises(TracingError, match="Boolean scalar indexing"):
        encode_basic_index((True,))

    with pytest.raises(TracingError, match="Boolean scalar indexing"):
        index_to_attrs((True,))

    with pytest.raises(TracingError, match="Basic indexing supports only"):
        encode_basic_index((np.True_,))


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"type": "int", "value": True}, "integer index"),
        ({"type": "slice", "start": None, "stop": None}, "slice index"),
        (
            {"type": "slice", "start": "0", "stop": None, "step": None},
            "slice bounds",
        ),
        ({"type": "newaxis", "extra": None}, "new-axis index"),
        ({"type": "ellipsis", "extra": None}, "ellipsis index"),
        ({"type": "unknown"}, "Unknown serialized index component"),
        (
            {"type": "array", "dtype": "int64", "shape": [2]},
            "Invalid serialized array index",
        ),
        (
            {"type": "array", "dtype": 1, "shape": [2], "values": [0, 1]},
            "dtype must be a string",
        ),
        (
            {"type": "array", "dtype": "int64", "shape": "2", "values": [0, 1]},
            "shape must be a sequence",
        ),
        (
            {"type": "array", "dtype": "int64", "shape": [-1], "values": []},
            "dimensions must be nonnegative integers",
        ),
        (
            {"type": "array", "dtype": "int64", "shape": [2], "values": [0, 1]},
            "Array indices are not supported",
        ),
        ([{"type": "ellipsis"}, {"type": "ellipsis"}], "at most one ellipsis"),
        (object(), "Invalid serialized index component object"),
    ],
)
def test_serialized_index_validation_rejects_malformed_components(
    payload: object,
    match: str,
) -> None:
    with pytest.raises(TypeError, match=match):
        decode_index(payload)


def test_staged_basic_index_validation_requires_a_sequence() -> None:
    with pytest.raises(TypeError, match="metadata must be a sequence"):
        decode_basic_index({"type": "int", "value": 1})
