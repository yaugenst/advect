# ruff: noqa: ANN401
"""Portable numeric constants at the Python/native graph boundary."""

from __future__ import annotations

import math
import struct
import sys
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Iterator

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
    """Canonical constant bytes; advect-runtime owns the wire format and digest."""

    kind: Literal["scalar", "array"]
    dtype: str
    shape: tuple[int, ...]
    data: bytes


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
    expected_bytes = math.prod(shape) * _item_size(normalized_dtype)
    raw = _snapshot_constant_bytes(
        value,
        shape=shape,
        dtype=normalized_dtype,
        kind=kind,
        expected_bytes=expected_bytes,
    )
    return _PortableConstant(kind=kind, dtype=normalized_dtype, shape=shape, data=bytes(raw))


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


def portable_constant_from_native(
    parts: tuple[str, str, list[int], bytes, str],
) -> _PortableConstant:
    """Wrap one payload the native graph store has already validated."""
    kind, dtype, shape, data, _digest = parts
    return _PortableConstant(cast("Literal['scalar', 'array']", kind), dtype, tuple(shape), data)


def iter_constant_values(
    constant: _PortableConstant,
) -> Iterator[bool | int | float | complex]:
    """Iterate decoded values only for providers without a byte materializer."""
    return _unpack_elements(constant.data, constant.dtype)


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
    return struct.calcsize(f"<{_COMPLEX_FORMATS.get(dtype) or _SCALAR_FORMATS[dtype]}")
