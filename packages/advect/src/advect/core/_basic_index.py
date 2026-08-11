"""Canonical wire representation for array indices."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from advect.core._errors import TracingError

if TYPE_CHECKING:
    from collections.abc import Callable

    type ArrayIndexDecoder = Callable[[object, str, tuple[int, ...]], object]


def encode_basic_index(index: object) -> list[dict[str, object]]:
    """Encode integers, slices, new axes, and ellipses as graph attributes."""
    items = index if isinstance(index, tuple) else (index,)
    encoded: list[dict[str, object]] = []
    for item in items:
        if isinstance(item, bool):
            msg = "Boolean scalar indexing is not supported"
            raise TracingError(msg)
        if isinstance(item, int):
            encoded.append({"type": "int", "value": item})
        elif isinstance(item, slice):
            encoded.append(
                {
                    "type": "slice",
                    "start": item.start,
                    "stop": item.stop,
                    "step": item.step,
                }
            )
        elif item is None:
            encoded.append({"type": "newaxis"})
        elif item is Ellipsis:
            encoded.append({"type": "ellipsis"})
        else:
            msg = "Basic indexing supports only integers, slices, new axes, and one ellipsis"
            raise TracingError(msg)
    return encoded


def _decode_mapping(
    payload: Mapping[object, object],
    *,
    array_decoder: ArrayIndexDecoder | None,
) -> object:
    kind = payload.get("type")
    if kind == "int":
        if set(payload) != {"type", "value"} or type(payload.get("value")) is not int:
            msg = "Invalid serialized integer index"
            raise TypeError(msg)
        return payload["value"]
    if kind == "slice":
        if set(payload) != {"type", "start", "stop", "step"}:
            msg = "Invalid serialized slice index"
            raise TypeError(msg)
        parts = (payload["start"], payload["stop"], payload["step"])
        if any(part is not None and type(part) is not int for part in parts):
            msg = "Serialized slice bounds must be integers or None"
            raise TypeError(msg)
        return slice(*parts)
    if kind == "newaxis":
        if set(payload) != {"type"}:
            msg = "Invalid serialized new-axis index"
            raise TypeError(msg)
        return None
    if kind == "ellipsis":
        if set(payload) != {"type"}:
            msg = "Invalid serialized ellipsis index"
            raise TypeError(msg)
        return Ellipsis
    if kind == "array":
        return _decode_array_mapping(payload, array_decoder=array_decoder)
    msg = f"Unknown serialized index component {kind!r}"
    raise TypeError(msg)


def _decode_array_mapping(
    payload: Mapping[object, object],
    *,
    array_decoder: ArrayIndexDecoder | None,
) -> object:
    if set(payload) != {"type", "dtype", "shape", "values"}:
        msg = "Invalid serialized array index"
        raise TypeError(msg)
    dtype = payload["dtype"]
    shape = payload["shape"]
    if not isinstance(dtype, str):
        msg = "Serialized array-index dtype must be a string"
        raise TypeError(msg)
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes, bytearray)):
        msg = "Serialized array-index shape must be a sequence"
        raise TypeError(msg)
    if any(type(dimension) is not int or dimension < 0 for dimension in shape):
        msg = "Serialized array-index dimensions must be nonnegative integers"
        raise TypeError(msg)
    if array_decoder is None:
        msg = "Array indices are not supported at this boundary"
        raise TypeError(msg)
    dimensions = tuple(int(dimension) for dimension in shape)
    return array_decoder(payload["values"], dtype, dimensions)


def decode_index(
    payload: object,
    *,
    array_decoder: ArrayIndexDecoder | None = None,
) -> object:
    """Decode and validate canonical index metadata.

    ``array_decoder`` is supplied only by a concrete array provider. Keeping
    provider materialization behind that callback lets this codec remain
    standard-library-only.
    """
    if payload is None or type(payload) is int:
        return payload
    if isinstance(payload, Mapping):
        return _decode_mapping(payload, array_decoder=array_decoder)
    if isinstance(payload, (list, tuple)):
        decoded = tuple(decode_index(item, array_decoder=array_decoder) for item in payload)
        if sum(item is Ellipsis for item in decoded) > 1:
            msg = "A serialized index may contain at most one ellipsis"
            raise TypeError(msg)
        return decoded
    msg = f"Invalid serialized index component {type(payload).__name__}"
    raise TypeError(msg)


def decode_basic_index(payload: object) -> tuple[object, ...]:
    """Decode the closed staged basic-index representation."""
    if not isinstance(payload, (list, tuple)):
        msg = "Staged basic-index metadata must be a sequence"
        raise TypeError(msg)
    decoded = decode_index(payload)
    if not isinstance(decoded, tuple):
        msg = "Staged basic-index metadata must decode to a tuple"
        raise TypeError(msg)
    return decoded


__all__ = ["decode_basic_index", "decode_index", "encode_basic_index"]
