"""Portable staged-constant codec contracts."""

from __future__ import annotations

import hashlib
import json
import struct
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from hypothesis import example, given, strategies as st
from hypothesis.extra import numpy as hnp

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


_DTYPES = (
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
)


def _assert_durable_record(constant: _PortableConstant, value: object) -> None:
    """Check the runtime record digests its canonical JSON body and loads back unchanged."""
    graph = _snapshot_graph(value, shape=constant.shape, dtype=constant.dtype)
    record = graph["constants"]["0"]
    body = json.dumps(
        {key: item for key, item in record.items() if key != "digest"},
        sort_keys=True,
        separators=(",", ":"),
    )
    assert record["data"] == constant.data.hex()
    assert record["digest"] == hashlib.sha256(body.encode()).hexdigest()
    assert _load_constant(graph) == constant


@st.composite
def _provider_arrays(draw: st.DrawFn) -> np.ndarray[Any, Any]:
    dtype = np.dtype(draw(st.sampled_from(_DTYPES))).newbyteorder(draw(st.sampled_from("<>")))
    shape = draw(hnp.array_shapes(min_dims=0, max_dims=3, min_side=0, max_side=3))
    source = draw(hnp.arrays(dtype, shape))
    return source.T if draw(st.booleans()) else source


@given(_provider_arrays())
@example(np.array([0x7C01, 0xFE01], dtype=">u2").view(">f2")).via(
    "discovered failure: big-endian NaN payloads were quieted"
)
@example(np.array([0x7F800001, 0x7FA00000], dtype=">u4").view(">c8")).via(
    "discovered failure: big-endian NaN payloads were quieted"
)
def test_portable_array_constants_round_trip_bit_exactly(source: np.ndarray[Any, Any]) -> None:
    dtype = source.dtype.name

    constant = snapshot_constant_parts(source, shape=source.shape, dtype=dtype)

    little_endian = np.ascontiguousarray(source, dtype=source.dtype.newbyteorder("<"))
    assert (constant.kind, constant.dtype, constant.shape) == ("array", dtype, source.shape)
    assert constant.data == little_endian.tobytes()
    _assert_durable_record(constant, source)
    decoded = np.asarray(list(iter_constant_values(constant)), dtype=dtype)
    np.testing.assert_array_equal(decoded.reshape(source.shape), source)


@given(
    st.one_of(
        st.booleans(),
        st.integers(min_value=-(2**63), max_value=2**63 - 1),
        st.floats(),
        st.complex_numbers(),
    )
)
def test_portable_python_scalars_round_trip_in_their_own_type(value: complex) -> None:
    dtype = {bool: "bool", int: "int64", float: "float64", complex: "complex128"}[type(value)]

    constant = snapshot_constant_parts(value, shape=(), dtype=dtype)

    assert (constant.kind, constant.dtype, constant.shape) == ("scalar", dtype, ())
    _assert_durable_record(constant, value)
    (decoded,) = iter_constant_values(constant)
    assert type(decoded) is type(value)
    assert repr(decoded) == repr(value)


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
