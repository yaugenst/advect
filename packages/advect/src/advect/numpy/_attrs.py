"""Attribute codec utilities for the NumPy backend.

The backend decodes its static array attrs here so graph attrs stay
backend-agnostic and JSON-serializable.
"""

from __future__ import annotations

from typing import Any

from advect.numpy._op_bindings import decanonicalize_array_op
from advect.numpy._static_attr_arrays import decode_static_array_attr

__all__ = ["decode_attrs"]


def _decode_clip(attrs: dict[str, Any]) -> dict[str, Any]:
    decoded = dict(attrs)
    decoded["a_min"] = decode_static_array_attr(attrs.get("a_min"))
    decoded["a_max"] = decode_static_array_attr(attrs.get("a_max"))
    return decoded


def _decode_diff(attrs: dict[str, Any]) -> dict[str, Any]:
    decoded = dict(attrs)
    if "prepend" in attrs:
        decoded["prepend"] = decode_static_array_attr(attrs.get("prepend"))
    if "append" in attrs:
        decoded["append"] = decode_static_array_attr(attrs.get("append"))
    return decoded


_ATTR_DECODERS = {"numpy.clip": _decode_clip, "numpy.diff": _decode_diff}


def decode_attrs(op: str, attrs: dict[str, Any]) -> dict[str, Any]:
    """Decode attrs for an op using the registered decoder (if any)."""
    decoder = _ATTR_DECODERS.get(op) or _ATTR_DECODERS.get(decanonicalize_array_op(op))
    return decoder(attrs) if decoder is not None else dict(attrs)
