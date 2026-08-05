"""Attribute codec utilities for the NumPy backend.

This module lets the backend register per-op attr decoders so graph attrs
stay backend-agnostic/JSON-serializable while backends interpret them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from advect.core._basic_index import decode_index
from advect.numpy._op_bindings import decanonicalize_array_op
from advect.numpy._static_attr_arrays import decode_static_array_attr

AttrDecoder = Callable[[dict[str, Any]], dict[str, Any]]

_ATTR_DECODERS: dict[str, AttrDecoder] = {}

__all__ = ["decode_attrs"]


def _attr_decoder(op: str) -> Callable[[AttrDecoder], AttrDecoder]:
    """Co-locate an attribute decoder with the operation it handles."""

    def register(decoder: AttrDecoder) -> AttrDecoder:
        _ATTR_DECODERS[op] = decoder
        return decoder

    return register


def decode_attrs(op: str, attrs: dict[str, Any]) -> dict[str, Any]:
    """Decode attrs for an op using the registered decoder (if any)."""
    decoder = _ATTR_DECODERS.get(op)
    if decoder is None:
        decoder = _ATTR_DECODERS.get(decanonicalize_array_op(op))
    return decoder(attrs) if decoder is not None else dict(attrs)


def _decode_index(payload: object) -> object:
    def materialize(values: object, dtype: str, shape: tuple[int, ...]) -> object:
        return np.asarray(values, dtype=np.dtype(dtype)).reshape(shape)

    return decode_index(payload, array_decoder=materialize)


@_attr_decoder("advect.getitem")
def _decode_getitem(attrs: dict[str, Any]) -> dict[str, Any]:
    decoded = dict(attrs)
    payload = attrs.get("index")
    decoded["index"] = slice(None) if payload is None else _decode_index(payload)
    return decoded


@_attr_decoder("advect.index_update")
def _decode_index_update(attrs: dict[str, Any]) -> dict[str, Any]:
    decoded = dict(attrs)
    payload = attrs.get("index")
    decoded["index"] = slice(None) if payload is None else _decode_index(payload)
    return decoded


@_attr_decoder("numpy.clip")
def _decode_clip(attrs: dict[str, Any]) -> dict[str, Any]:
    decoded = dict(attrs)
    decoded["a_min"] = decode_static_array_attr(attrs.get("a_min"))
    decoded["a_max"] = decode_static_array_attr(attrs.get("a_max"))
    return decoded


@_attr_decoder("numpy.diff")
def _decode_diff(attrs: dict[str, Any]) -> dict[str, Any]:
    decoded = dict(attrs)
    if "prepend" in attrs:
        decoded["prepend"] = decode_static_array_attr(attrs.get("prepend"))
    if "append" in attrs:
        decoded["append"] = decode_static_array_attr(attrs.get("append"))
    return decoded


@_attr_decoder("numpy.full")
def _decode_full(attrs: dict[str, Any]) -> dict[str, Any]:
    decoded = dict(attrs)
    if "like" in attrs:
        decoded["like"] = decode_static_array_attr(attrs.get("like"))
    return decoded
