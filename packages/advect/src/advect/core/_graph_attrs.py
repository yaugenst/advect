"""Closed codec boundary for structured graph attributes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

_PRIMITIVE_CALL_KEY = "__advect_primitive_call__"
_EMPTY_NATIVE_ATTRS: dict[str, Any] = {}


def encode_graph_attrs_for_native(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """Encode attrs for immediate transfer into native-owned typed storage.

    The native boundary recursively copies every supported value, so a second
    Python copy here would only add trace-time allocation.
    """
    if not attrs:
        return _EMPTY_NATIVE_ATTRS
    if _PRIMITIVE_CALL_KEY not in attrs and isinstance(attrs, dict):
        return attrs
    result = dict(attrs)
    if _PRIMITIVE_CALL_KEY in attrs:
        from advect.core._primitive_call import _encode_primitive_call_meta  # noqa: PLC0415

        result[_PRIMITIVE_CALL_KEY] = _encode_primitive_call_meta(attrs[_PRIMITIVE_CALL_KEY])
    return result


def decode_graph_attrs_from_native(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """Decode one fresh native snapshot without redundantly deep-copying it."""
    result = dict(attrs)
    if _PRIMITIVE_CALL_KEY in attrs:
        from advect.core._primitive_call import _decode_primitive_call_meta  # noqa: PLC0415

        result[_PRIMITIVE_CALL_KEY] = _decode_primitive_call_meta(attrs[_PRIMITIVE_CALL_KEY])
    return result
