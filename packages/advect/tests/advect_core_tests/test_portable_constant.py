"""Portable staged-constant codec contracts."""

from __future__ import annotations

import copy
import json
import struct
from typing import cast

import numpy as np
import pytest

from advect import _native_core as advect_native
from advect.core._portable_constant import (
    _constant_payload,
    iter_constant_values,
    portable_constant_from_payload,
    snapshot_constant_parts,
    validate_constant,
)
from advect.core._stage import _coerce_constant


def _snapshot_payload(value: object, *, shape: tuple[int, ...], dtype: str) -> dict[str, object]:
    constant = snapshot_constant_parts(value, shape=shape, dtype=dtype)
    builder = advect_native.GraphBuilder()
    node_id, digest = builder.append_constant(
        constant.data,
        list(constant.shape),
        constant.dtype,
        kind=constant.kind,
    )
    assert digest == constant.digest
    builder.append_output(node_id)
    store, old_to_new, _report, _trace = builder.finish()
    mapped_id = old_to_new[node_id]
    assert mapped_id is not None
    payload = json.loads(store._to_json())["constants"][str(mapped_id)]
    return cast("dict[str, object]", payload)


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        ("bool", [False, True]),
        ("int8", [-2, 7]),
        ("int16", [-300, 400]),
        ("int32", [-70_000, 80_000]),
        ("int64", [-5_000_000_000, 6_000_000_000]),
        ("uint8", [2, 7]),
        ("uint16", [300, 400]),
        ("uint32", [70_000, 80_000]),
        ("uint64", [5_000_000_000, 6_000_000_000]),
        ("float16", [1.25, -2.5]),
        ("float32", [1.25, -2.5]),
        ("float64", [1.25, -2.5]),
        ("complex64", [1 + 2j, -3 + 0.5j]),
        ("complex128", [1 + 2j, -3 + 0.5j]),
    ],
)
def test_portable_constants_round_trip_standard_dtypes(
    dtype: str,
    values: list[object],
) -> None:
    source = np.asarray(values, dtype=dtype)

    payload = _snapshot_payload(source, shape=(2,), dtype=dtype)
    constant = portable_constant_from_payload(payload)
    decoded = tuple(iter_constant_values(constant))

    assert constant.kind == "array"
    assert constant.dtype == dtype
    assert constant.shape == (2,)
    np.testing.assert_array_equal(np.asarray(decoded, dtype=dtype), source)


def test_portable_constant_wire_format_is_fixed_little_endian_bytes() -> None:
    source = np.asarray([1.0, -2.0], dtype=np.float32)

    payload = _snapshot_payload(source, shape=(2,), dtype="float32")

    assert payload == {
        "format": "advect.numeric-constant",
        "version": 2,
        "kind": "array",
        "dtype": "float32",
        "shape": [2],
        "layout": "C",
        "byte_order": "little",
        "data": "0000803f000000c0",
        "digest": "0255a880d670a37afb400442395d880ac054e0ddaa0fa4ddf78b54b65c2a1e27",
    }


def test_array_constant_uses_bulk_c_order_bytes_when_available() -> None:
    class BulkArray:
        dtype = type("_DType", (), {"byteorder": "<"})()

        def __init__(self) -> None:
            self.calls: list[str] = []

        def tobytes(self, *, order: str) -> bytes:
            self.calls.append(order)
            return struct.pack("<2f", 1.0, 2.0)

        def __getitem__(self, _index: object) -> object:
            msg = "bulk snapshot unexpectedly indexed the provider array"
            raise AssertionError(msg)

    value = BulkArray()

    constant = snapshot_constant_parts(value, shape=(2,), dtype="float32")

    assert constant.data.hex() == "0000803f00000040"
    assert value.calls == ["C"]


def test_portable_scalar_preserves_scalar_materialization() -> None:
    payload = _snapshot_payload(1 + 2j, shape=(), dtype="complex64")
    constant = portable_constant_from_payload(payload)
    values = tuple(iter_constant_values(constant))

    assert (constant.kind, constant.dtype, constant.shape) == ("scalar", "complex64", ())
    assert values == (1 + 2j,)


def test_portable_constant_validation_is_transactional() -> None:
    payload = _snapshot_payload(
        np.asarray([1, 2], dtype=np.int32),
        shape=(2,),
        dtype="int32",
    )
    corrupt = copy.deepcopy(payload)
    corrupt["data"] = "00000000"

    with pytest.raises(ValueError, match="require 8 bytes"):
        validate_constant(corrupt)
    assert validate_constant(payload) == payload


def test_portable_constant_rejects_invalid_scalar_and_boolean_encodings() -> None:
    scalar = snapshot_constant_parts(1, shape=(), dtype="int64")
    with pytest.raises(ValueError, match="scalar constant must have rank zero"):
        validate_constant({**_constant_payload(scalar), "shape": [1]})

    boolean = snapshot_constant_parts([True], shape=(1,), dtype="bool")
    with pytest.raises(ValueError, match="bytes must be zero or one"):
        validate_constant({**_constant_payload(boolean), "data": "02"})


def test_portable_constant_rejects_nonstandard_dtype() -> None:
    with pytest.raises(TypeError, match="Unsupported staged constant dtype"):
        snapshot_constant_parts(
            np.asarray([1.0], dtype=np.longdouble),
            shape=(1,),
            dtype="float128",
        )


def test_byte_materialization_moves_to_the_selected_device() -> None:
    class FakeArray:
        shape = (2,)
        dtype = "float32"

        def __init__(self, device: str) -> None:
            self.device = device

    class FakeRawNamespace:
        @staticmethod
        def frombuffer(_data: bytes, *, dtype: object) -> FakeArray:
            assert dtype == "float32"
            return FakeArray("cuda:0")

    class FakeNamespace:
        raw_namespace = FakeRawNamespace()
        float32 = "float32"

        def __init__(self) -> None:
            self.requests: list[str] = []

        def asarray(
            self,
            value: FakeArray,
            *,
            dtype: object,
            device: object,
        ) -> FakeArray:
            assert value.device == "cuda:0"
            assert dtype == "float32"
            self.requests.append(str(device))
            return FakeArray(str(device))

    namespace = FakeNamespace()
    constant = snapshot_constant_parts(
        np.asarray([1.0, 2.0], dtype=np.float32),
        shape=(2,),
        dtype="float32",
    )

    result = _coerce_constant(constant, namespace, device="cuda:1")

    assert result.device == "cuda:1"
    assert namespace.requests == ["cuda:1"]
