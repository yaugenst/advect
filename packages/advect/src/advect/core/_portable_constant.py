# ruff: noqa: ANN401, C901, EM101, EM102, PLR0912, TRY003
"""Portable numeric constants at the Python/native graph boundary."""

from __future__ import annotations

import binascii
import hashlib
import json
import math
import struct
import sys
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Iterator

_FORMAT = "advect.numeric-constant"
_VERSION = 2
_LAYOUT = "C"
_BYTE_ORDER = "little"
_PAYLOAD_KEYS = {
    "byte_order",
    "data",
    "digest",
    "dtype",
    "format",
    "kind",
    "layout",
    "shape",
    "version",
}

_DTYPE_ALIASES = {
    "bool_": "bool",
    "byte": "int8",
    "ubyte": "uint8",
    "short": "int16",
    "ushort": "uint16",
    "int": "int64",
    "intp": "int64",
    "long": "int64",
    "uint": "uint64",
    "uintp": "uint64",
    "ulong": "uint64",
    "half": "float16",
    "single": "float32",
    "double": "float64",
    "float": "float64",
    "csingle": "complex64",
    "cdouble": "complex128",
    "complex": "complex128",
}
_SCALAR_FORMATS = {
    "bool": "?",
    "int8": "b",
    "int16": "h",
    "int32": "i",
    "int64": "q",
    "uint8": "B",
    "uint16": "H",
    "uint32": "I",
    "uint64": "Q",
    "float16": "e",
    "float32": "f",
    "float64": "d",
}
_COMPLEX_FORMATS = {
    "complex64": "ff",
    "complex128": "dd",
}


@dataclass(frozen=True, slots=True)
class _PortableConstant:
    kind: Literal["scalar", "array"]
    dtype: str
    shape: tuple[int, ...]
    data: bytes
    digest: str


def normalize_constant_dtype(dtype: str) -> str:
    """Return the closed portable name for one supported numeric dtype."""
    normalized = _DTYPE_ALIASES.get(dtype.strip().lower(), dtype.strip().lower())
    if normalized not in _SCALAR_FORMATS and normalized not in _COMPLEX_FORMATS:
        raise TypeError(
            f"Unsupported staged constant dtype {dtype!r}; expected bool, "
            "int8/16/32/64, uint8/16/32/64, float16/32/64, or complex64/128"
        )
    return normalized


def snapshot_constant_parts(
    value: Any,
    *,
    shape: tuple[int, ...],
    dtype: str,
) -> _PortableConstant:
    """Detach one provider value without constructing its textual artifact form."""
    normalized_dtype = normalize_constant_dtype(dtype)
    kind = "scalar" if isinstance(value, (bool, int, float, complex)) else "array"
    if kind == "scalar" and shape:
        raise ValueError("A staged scalar constant must have rank zero")
    expected_bytes = _element_count(shape) * _item_size(normalized_dtype)
    raw = _snapshot_constant_bytes(
        value,
        shape=shape,
        dtype=normalized_dtype,
        kind=kind,
        expected_bytes=expected_bytes,
    )
    data = bytes(raw)
    return _PortableConstant(
        kind=kind,
        dtype=normalized_dtype,
        shape=shape,
        data=data,
        digest=_constant_digest_bytes(
            kind=kind,
            dtype=normalized_dtype,
            shape=shape,
            data=data,
        ),
    )


def _snapshot_constant_bytes(
    value: Any,
    *,
    shape: tuple[int, ...],
    dtype: str,
    kind: str,
    expected_bytes: int,
) -> bytes | bytearray:
    tobytes = getattr(value, "tobytes", None)
    value_dtype = getattr(value, "dtype", None)
    byte_order = getattr(value_dtype, "byteorder", "=")
    native_is_little = sys.byteorder == "little"
    provider_bytes_are_little = byte_order in {"<", "|"} or (byte_order == "=" and native_is_little)
    if kind == "array" and callable(tobytes) and provider_bytes_are_little:
        try:
            raw = tobytes(order="C")
        except TypeError:
            raw = tobytes()
        if isinstance(raw, (bytes, bytearray)) and len(raw) == expected_bytes:
            return raw

    item_size = _item_size(dtype)
    raw = bytearray(expected_bytes)
    captured_elements = 0
    for captured_elements, item in enumerate(
        _constant_elements(value, shape=shape, kind=kind),
        start=1,
    ):
        _pack_element_into(
            raw,
            offset=(captured_elements - 1) * item_size,
            value=item,
            dtype=dtype,
        )
    captured_bytes = captured_elements * item_size
    if captured_bytes != expected_bytes:
        raise ValueError(
            f"Staged constant shape {shape} and dtype {dtype!r} "
            f"require {expected_bytes} bytes; captured {captured_bytes}"
        )
    return raw


def validate_constant(
    payload: object,
    *,
    shape: tuple[int, ...] | None = None,
    dtype: str | None = None,
    byte_count: int | None = None,
) -> dict[str, object]:
    """Validate and detach one portable constant mapping."""
    if not isinstance(payload, dict):
        raise TypeError("Staged constant payload must be a mapping")
    if set(payload) != _PAYLOAD_KEYS:
        raise ValueError("Staged constant payload has invalid fields")
    if payload["format"] != _FORMAT:
        raise ValueError(f"Unknown staged constant format {payload['format']!r}")
    if payload["version"] != _VERSION:
        raise ValueError(f"Unsupported staged constant version {payload['version']!r}")
    if payload["layout"] != _LAYOUT:
        raise ValueError(f"Unsupported staged constant layout {payload['layout']!r}")
    if payload["byte_order"] != _BYTE_ORDER:
        raise ValueError(f"Unsupported staged constant byte order {payload['byte_order']!r}")

    kind = payload["kind"]
    payload_dtype = payload["dtype"]
    payload_shape = payload["shape"]
    encoded_data = payload["data"]
    digest = payload["digest"]
    if kind not in {"scalar", "array"}:
        raise ValueError(f"Unknown staged constant kind {kind!r}")
    if not isinstance(payload_dtype, str):
        raise TypeError("Staged constant dtype must be a string")
    normalized_dtype = normalize_constant_dtype(payload_dtype)
    if payload_dtype != normalized_dtype:
        raise ValueError("Staged constant dtype must use its canonical portable name")
    if not isinstance(payload_shape, list) or any(
        type(size) is not int or size < 0 for size in payload_shape
    ):
        raise TypeError("Staged constant shape must contain non-negative integers")
    normalized_shape = tuple(payload_shape)
    if kind == "scalar" and normalized_shape:
        raise ValueError("A staged scalar constant must have rank zero")
    if not isinstance(encoded_data, str):
        raise TypeError("Staged constant data must be a hexadecimal string")
    if encoded_data != encoded_data.lower() or any(
        character not in "0123456789abcdef" for character in encoded_data
    ):
        raise ValueError("Staged constant data must be lowercase hexadecimal")
    if len(encoded_data) % 2:
        raise ValueError("Staged constant data must contain complete bytes")
    raw = bytes.fromhex(encoded_data)
    expected_bytes = _element_count(normalized_shape) * _item_size(normalized_dtype)
    if len(raw) != expected_bytes:
        raise ValueError(
            f"Staged constant shape {normalized_shape} and dtype {normalized_dtype!r} "
            f"require {expected_bytes} bytes; payload has {len(raw)}"
        )
    if normalized_dtype == "bool" and any(item not in {0, 1} for item in raw):
        raise ValueError("Staged bool constant bytes must be zero or one")
    expected_digest = _constant_digest_bytes(
        kind=kind,
        dtype=normalized_dtype,
        shape=normalized_shape,
        data=raw,
    )
    if not isinstance(digest, str) or digest != expected_digest:
        raise ValueError("Staged constant payload digest does not match its contents")
    if shape is not None and normalized_shape != shape:
        raise ValueError("Staged constant payload shape does not match its graph metadata")
    if dtype is not None and normalized_dtype != normalize_constant_dtype(dtype):
        raise ValueError("Staged constant payload dtype does not match its graph metadata")
    if byte_count is not None and len(raw) != byte_count:
        raise ValueError("Staged constant payload byte count does not match its manifest")
    return _constant_payload(
        _PortableConstant(
            kind=kind,
            dtype=normalized_dtype,
            shape=normalized_shape,
            data=raw,
            digest=digest,
        )
    )


def portable_constant_from_payload(
    payload: object,
    *,
    shape: tuple[int, ...] | None = None,
    dtype: str | None = None,
    byte_count: int | None = None,
) -> _PortableConstant:
    """Decode one validated textual payload into its normal runtime form."""
    validated = validate_constant(
        payload,
        shape=shape,
        dtype=dtype,
        byte_count=byte_count,
    )
    return _PortableConstant(
        kind=cast("Literal['scalar', 'array']", validated["kind"]),
        dtype=str(validated["dtype"]),
        shape=tuple(cast("list[int]", validated["shape"])),
        data=bytes.fromhex(str(validated["data"])),
        digest=str(validated["digest"]),
    )


def portable_constant_from_native(
    kind: str,
    dtype: str,
    shape: list[int],
    data: bytes,
    digest: str,
) -> _PortableConstant:
    """Validate raw parts returned by the native portable graph store."""
    normalized_dtype = normalize_constant_dtype(dtype)
    normalized_shape = tuple(shape)
    if kind not in {"scalar", "array"}:
        raise ValueError(f"Unknown staged constant kind {kind!r}")
    if kind == "scalar" and normalized_shape:
        raise ValueError("A staged scalar constant must have rank zero")
    expected_bytes = _element_count(normalized_shape) * _item_size(normalized_dtype)
    if len(data) != expected_bytes:
        raise ValueError(
            f"Staged constant shape {normalized_shape} and dtype {normalized_dtype!r} "
            f"require {expected_bytes} bytes; native store has {len(data)}"
        )
    expected_digest = _constant_digest_bytes(
        kind=kind,
        dtype=normalized_dtype,
        shape=normalized_shape,
        data=data,
    )
    if digest != expected_digest:
        raise ValueError("Native staged constant digest does not match its contents")
    return _PortableConstant(kind, normalized_dtype, normalized_shape, data, digest)


def iter_constant_values(
    constant: _PortableConstant,
) -> Iterator[bool | int | float | complex]:
    """Iterate decoded values only for providers without a byte materializer."""
    return _unpack_elements(constant.data, constant.dtype)


def _constant_payload(constant: _PortableConstant) -> dict[str, object]:
    body = _constant_body(
        kind=constant.kind,
        dtype=constant.dtype,
        shape=constant.shape,
        data=constant.data.hex(),
    )
    return {**body, "digest": constant.digest}


def _constant_body(
    *,
    kind: str,
    dtype: str,
    shape: tuple[int, ...],
    data: str,
) -> dict[str, object]:
    return {
        "format": _FORMAT,
        "version": _VERSION,
        "kind": kind,
        "dtype": dtype,
        "shape": list(shape),
        "layout": _LAYOUT,
        "byte_order": _BYTE_ORDER,
        "data": data,
    }


def _constant_digest_bytes(
    *,
    kind: str,
    dtype: str,
    shape: tuple[int, ...],
    data: bytes,
) -> str:
    digest = hashlib.sha256()
    digest.update(b'{"byte_order":"little","data":"')
    view = memoryview(data)
    chunk_size = 1024 * 1024
    for offset in range(0, len(view), chunk_size):
        digest.update(binascii.hexlify(view[offset : offset + chunk_size]))
    suffix = (
        f'","dtype":{json.dumps(dtype, ensure_ascii=True)},'
        f'"format":"{_FORMAT}","kind":{json.dumps(kind, ensure_ascii=True)},'
        f'"layout":"{_LAYOUT}","shape":'
        f"{json.dumps(list(shape), ensure_ascii=True, separators=(',', ':'))},"
        f'"version":{_VERSION}}}'
    )
    digest.update(suffix.encode("ascii"))
    return digest.hexdigest()


def _constant_elements(
    value: Any,
    *,
    shape: tuple[int, ...],
    kind: str,
) -> Iterator[object]:
    if kind == "scalar":
        yield value
        return
    for index in product(*(range(size) for size in shape)):
        try:
            if isinstance(value, (tuple, list)):
                item = value
                for coordinate in index:
                    item = item[coordinate]
                yield item
            else:
                yield value[index]
        except (IndexError, KeyError, TypeError) as error:
            raise TypeError(
                f"Could not snapshot staged constant element {index} from {type(value).__name__}"
            ) from error


def _pack_element_into(
    buffer: bytearray,
    *,
    offset: int,
    value: Any,
    dtype: str,
) -> None:
    try:
        if dtype in _COMPLEX_FORMATS:
            normalized = complex(value)
            struct.pack_into(
                f"<{_COMPLEX_FORMATS[dtype]}",
                buffer,
                offset,
                normalized.real,
                normalized.imag,
            )
        else:
            if dtype == "bool":
                normalized = bool(value)
            elif dtype.startswith(("int", "uint")):
                normalized = int(value)
            else:
                normalized = float(value)
            struct.pack_into(f"<{_SCALAR_FORMATS[dtype]}", buffer, offset, normalized)
    except (OverflowError, struct.error, TypeError, ValueError) as error:
        raise TypeError(
            f"Could not encode staged {dtype!r} constant element from {type(value).__name__}"
        ) from error


def _unpack_elements(
    raw: bytes,
    dtype: str,
) -> Iterator[bool | int | float | complex]:
    if dtype in _COMPLEX_FORMATS:
        for real, imag in struct.iter_unpack(f"<{_COMPLEX_FORMATS[dtype]}", raw):
            yield complex(real, imag)
        return
    for (value,) in struct.iter_unpack(f"<{_SCALAR_FORMATS[dtype]}", raw):
        yield value


def _item_size(dtype: str) -> int:
    format_code = _COMPLEX_FORMATS.get(dtype, _SCALAR_FORMATS.get(dtype))
    if format_code is None:
        raise TypeError(f"Unsupported staged constant dtype {dtype!r}")
    return struct.calcsize(f"<{format_code}")


def _element_count(shape: tuple[int, ...]) -> int:
    return math.prod(shape)
