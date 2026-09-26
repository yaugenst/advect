"""Portable staged-constant codec contracts."""

from __future__ import annotations

import json
import struct
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from advect import _native_core as advect_native
from advect.core._portable_constant import (
    iter_constant_values,
    portable_constant_from_native,
    snapshot_constant_parts,
)
from advect.core._stage import _coerce_constant

if TYPE_CHECKING:
    from advect.core._portable_constant import _PortableConstant


def _snapshot_graph(value: object, *, shape: tuple[int, ...], dtype: str) -> dict[str, Any]:
    constant = snapshot_constant_parts(value, shape=shape, dtype=dtype)
    builder = advect_native.GraphBuilder()
    node_id, _digest = builder.append_constant(
        constant.data,
        list(constant.shape),
        constant.dtype,
        kind=constant.kind,
    )
    builder.append_output(node_id)
    store, _old_to_new, _report, _trace = builder.finish()
    return json.loads(store._to_json())


def _load_constant(graph: dict[str, Any]) -> _PortableConstant:
    """Round-trip the durable graph and decode its only constant."""
    store = advect_native.deserialize_graph_json(json.dumps(graph))
    (constant_id,) = store.constant_ids()
    return portable_constant_from_native(store._constant_parts(constant_id))


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

    constant = _load_constant(_snapshot_graph(source, shape=(2,), dtype=dtype))
    decoded = tuple(iter_constant_values(constant))

    assert constant.kind == "array"
    assert constant.dtype == dtype
    assert constant.shape == (2,)
    np.testing.assert_array_equal(np.asarray(decoded, dtype=dtype), source)


def test_portable_constant_wire_format_is_fixed_little_endian_bytes() -> None:
    source = np.asarray([1.0, -2.0], dtype=np.float32)

    graph = _snapshot_graph(source, shape=(2,), dtype="float32")

    assert graph["constants"]["0"] == {
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
    constant = _load_constant(_snapshot_graph(1 + 2j, shape=(), dtype="complex64"))
    values = tuple(iter_constant_values(constant))

    assert (constant.kind, constant.dtype, constant.shape) == ("scalar", "complex64", ())
    assert values == (1 + 2j,)


@pytest.mark.parametrize(
    ("value", "dtype", "corruption", "message"),
    [
        (np.asarray([1, 2], dtype=np.int32), "int32", {"data": "00000000"}, "require 8 bytes"),
        (1, "int64", {"shape": [1]}, "scalar constant must have rank zero"),
        (np.asarray([True]), "bool", {"data": "02"}, "bytes must be exactly 0 or 1"),
        (np.asarray([7], dtype=np.int8), "int8", {"data": "08"}, "digest does not match"),
    ],
    ids=["byte-count", "scalar-rank", "bool-byte", "digest"],
)
def test_runtime_rejects_corrupt_constant_payloads(
    value: object,
    dtype: str,
    corruption: dict[str, object],
    message: str,
) -> None:
    shape = tuple(np.shape(value))
    graph = _snapshot_graph(value, shape=shape, dtype=dtype)
    graph["constants"]["0"].update(corruption)

    with pytest.raises(ValueError, match=message):
        _load_constant(graph)


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

    class FakeNamespace:
        float32 = "float32"

        def __init__(self) -> None:
            self.requests: list[str] = []

        @staticmethod
        def frombuffer(_data: bytes, *, dtype: object) -> FakeArray:
            assert dtype == "float32"
            return FakeArray("cuda:0")

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
