"""JSON-safe encoding helpers for static array-valued attrs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np

_STATIC_ARRAY_TAG = "__advect_static_array__"


def is_static_array_attr(value: object) -> bool:
    """Return whether ``value`` is an encoded static-array attr payload."""
    return isinstance(value, Mapping) and value.get(_STATIC_ARRAY_TAG) is True


def encode_static_array_attr(value: object) -> dict[str, Any]:
    """Encode array-like static attr values into a JSON-safe payload."""
    arr = np.asarray(value)
    payload: dict[str, Any] = {_STATIC_ARRAY_TAG: True, "dtype": str(arr.dtype)}
    if np.issubdtype(arr.dtype, np.complexfloating):
        payload["real"] = np.real(arr).tolist()
        payload["imag"] = np.imag(arr).tolist()
    else:
        payload["value"] = arr.tolist()
    return payload


def decode_static_array_attr(value: object) -> object:
    """Decode static-array payloads back to ``np.ndarray`` values."""
    if not is_static_array_attr(value):
        return value

    payload = cast("Mapping[str, Any]", value)
    dtype = np.dtype(payload["dtype"]) if isinstance(payload.get("dtype"), str) else None
    if "real" in payload and "imag" in payload:
        real = np.asarray(payload["real"], dtype=np.float64)
        imag = np.asarray(payload["imag"], dtype=np.float64)
        arr = real + 1j * imag
        return arr.astype(dtype if dtype is not None else np.complex128, copy=False)

    data = payload.get("value")
    return np.asarray(data) if dtype is None else np.asarray(data, dtype=dtype)
