# ruff: noqa: ANN401
"""Equality-safe snapshots for static xarray pytree metadata."""

from __future__ import annotations

import datetime as dt
import struct
from typing import Any

import numpy as np

_PREFIX = "advect.xarray."
_ARRAY = f"{_PREFIX}array"
_COMPLEX = f"{_PREFIX}complex"
_DICT = f"{_PREFIX}dict"
_FLOAT = f"{_PREFIX}float"
_LIST = f"{_PREFIX}list"
_NUMPY_OBJECT_SCALAR = f"{_PREFIX}numpy-object-scalar"
_NUMPY_SCALAR = f"{_PREFIX}numpy-scalar"
_SLICE = f"{_PREFIX}slice"
_TUPLE = f"{_PREFIX}tuple"
_TAGS = {
    _ARRAY,
    _COMPLEX,
    _DICT,
    _FLOAT,
    _LIST,
    _NUMPY_OBJECT_SCALAR,
    _NUMPY_SCALAR,
    _SLICE,
    _TUPLE,
}
_NOT_STATIC_SCALAR = object()


def contains_tracer(value: Any) -> bool:
    """Return whether static metadata contains an Advect tracer."""
    if callable(getattr(value, "_advect_snapshot", None)):
        return True
    if isinstance(value, np.ndarray):
        return value.dtype.hasobject and any(contains_tracer(item) for item in value.flat)
    if isinstance(value, dict):
        return any(contains_tracer(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_tracer(item) for item in value)
    return False


def _freeze_scalar(value: Any, *, path: str) -> Any:
    if value is None or isinstance(
        value,
        (bool, int, str, bytes, dt.date, dt.datetime, dt.timedelta),
    ):
        return value
    if isinstance(value, float):
        return (_FLOAT, struct.pack("!d", value).hex())
    if isinstance(value, complex):
        return (
            _COMPLEX,
            struct.pack("!d", value.real).hex(),
            struct.pack("!d", value.imag).hex(),
        )
    if isinstance(value, np.generic):
        array = np.asarray(value)
        if array.dtype.hasobject:
            return (
                _NUMPY_OBJECT_SCALAR,
                array.dtype.str,
                freeze(value.item(), path=f"{path}.item()"),
            )
        return (_NUMPY_SCALAR, array.dtype.str, array.tobytes().hex())
    return _NOT_STATIC_SCALAR


def _freeze_mapping(value: dict[Any, Any], *, path: str) -> tuple[str, tuple[Any, ...]]:
    items: list[tuple[str, Any]] = []
    for key, item in value.items():
        if not isinstance(key, str):
            msg = f"xarray attribute keys must be strings; got {type(key).__name__} at {path}"
            raise TypeError(msg)
        items.append((key, freeze(item, path=f"{path}[{key!r}]")))
    return (_DICT, tuple(sorted(items)))


def _freeze_array(value: np.ndarray[Any, Any], *, path: str) -> tuple[Any, ...]:
    if value.dtype.fields is not None:
        msg = f"xarray structured metadata arrays are not supported at {path}"
        raise TypeError(msg)
    shape = tuple(int(size) for size in value.shape)
    if value.dtype.hasobject:
        items = tuple(
            freeze(item, path=f"{path}.flat[{index}]") for index, item in enumerate(value.flat)
        )
        return (_ARRAY, value.dtype.str, shape, items)
    data = np.ascontiguousarray(value).tobytes().hex()
    return (_ARRAY, value.dtype.str, shape, data)


def freeze(value: Any, *, path: str) -> Any:
    """Copy metadata into an immutable representation with scalar equality."""
    if callable(getattr(value, "_advect_snapshot", None)):
        msg = (
            "xarray coordinates, dimensions, names, and attributes are static; "
            f"found a traced value at {path}. Pass differentiable values as data "
            "or as a separate argument."
        )
        raise TypeError(msg)

    scalar = _freeze_scalar(value, path=path)
    if scalar is not _NOT_STATIC_SCALAR:
        return scalar
    if isinstance(value, slice):
        return (
            _SLICE,
            freeze(value.start, path=f"{path}.start"),
            freeze(value.stop, path=f"{path}.stop"),
            freeze(value.step, path=f"{path}.step"),
        )
    if isinstance(value, tuple):
        return (
            _TUPLE,
            tuple(freeze(item, path=f"{path}[{index}]") for index, item in enumerate(value)),
        )
    if isinstance(value, list):
        return (
            _LIST,
            tuple(freeze(item, path=f"{path}[{index}]") for index, item in enumerate(value)),
        )
    if isinstance(value, dict):
        return _freeze_mapping(value, path=path)
    if isinstance(value, np.ndarray):
        return _freeze_array(value, path=path)

    msg = f"xarray static metadata at {path} has unsupported type {type(value).__name__}"
    raise TypeError(msg)


def _thaw_scalar(value: tuple[Any, ...]) -> Any:
    tag = value[0]
    if tag == _FLOAT:
        return struct.unpack("!d", bytes.fromhex(value[1]))[0]
    if tag == _COMPLEX:
        real = struct.unpack("!d", bytes.fromhex(value[1]))[0]
        imag = struct.unpack("!d", bytes.fromhex(value[2]))[0]
        return complex(real, imag)
    if tag == _NUMPY_SCALAR:
        return np.frombuffer(bytes.fromhex(value[2]), dtype=np.dtype(value[1]))[0]
    if tag == _NUMPY_OBJECT_SCALAR:
        return np.asarray(thaw(value[2]), dtype=np.dtype(value[1]))[()]
    msg = f"Unknown scalar metadata tag {tag!r}"
    raise ValueError(msg)


def _thaw_container(value: tuple[Any, ...]) -> Any:
    tag = value[0]
    if tag == _SLICE:
        return slice(thaw(value[1]), thaw(value[2]), thaw(value[3]))
    if tag == _TUPLE:
        return tuple(thaw(item) for item in value[1])
    if tag == _LIST:
        return [thaw(item) for item in value[1]]
    if tag == _DICT:
        return {key: thaw(item) for key, item in value[1]}
    msg = f"Unknown container metadata tag {tag!r}"
    raise ValueError(msg)


def _thaw_array(value: tuple[Any, ...]) -> np.ndarray[Any, Any]:
    _tag, dtype_string, shape, payload = value
    dtype = np.dtype(dtype_string)
    if dtype.hasobject:
        result = np.empty(shape, dtype=object)
        for index, item in enumerate(payload):
            result.flat[index] = thaw(item)
        return result
    return np.frombuffer(bytes.fromhex(payload), dtype=dtype).copy().reshape(shape)


def thaw(value: Any) -> Any:
    """Rebuild one xarray metadata value from :func:`freeze` output."""
    if not isinstance(value, tuple) or not value or value[0] not in _TAGS:
        return value
    if value[0] in {_FLOAT, _COMPLEX, _NUMPY_SCALAR, _NUMPY_OBJECT_SCALAR}:
        return _thaw_scalar(value)
    if value[0] in {_SLICE, _TUPLE, _LIST, _DICT}:
        return _thaw_container(value)
    return _thaw_array(value)
