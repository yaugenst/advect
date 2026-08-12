"""Tests for the canonical serialized index representation."""

from __future__ import annotations

import numpy as np
import pytest

from advect.core._basic_index import decode_basic_index, decode_index, encode_basic_index
from advect.core._errors import TracingError
from advect.numpy._traced_array_indexing import index_to_attrs


def test_basic_index_codec_round_trips_every_component() -> None:
    index = (slice(1, -1, 2), 3, None, Ellipsis)

    encoded = encode_basic_index(index)

    assert decode_basic_index(encoded) == index
    assert index_to_attrs(index) == encoded


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
