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
