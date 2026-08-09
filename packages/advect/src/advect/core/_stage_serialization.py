"""JSON encoding for staged-program values and pytree definitions."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any, cast

from advect.core._array_api.results import (
    _RESULT_TYPE_TAGS,
    _SERIALIZED_RESULT_TYPES,
)
from advect.core._pytree import Static, TreeDef

_VALUE_KIND = "kind"
_VALUE_PAYLOAD = "value"
_KEY_VALUE_PAIR_LEN = 2
_ENCODED_VALUE_FIELDS = {_VALUE_KIND, _VALUE_PAYLOAD}
_TREEDEF_FIELDS = {"type", "aux", "children", "num_leaves"}


def _encode_value(value: object) -> dict[str, object]:
    if value is None or type(value) in (bool, int, str):
        return {_VALUE_KIND: "scalar", _VALUE_PAYLOAD: value}
    if type(value) is float:
        if not math.isfinite(value):
            msg = "Staged metadata must contain only finite floats"
            raise TypeError(msg)
        return {_VALUE_KIND: "scalar", _VALUE_PAYLOAD: value}
    if type(value) is bytes:
        return {_VALUE_KIND: "bytes", _VALUE_PAYLOAD: value.hex()}
    if type(value) is list:
        return {_VALUE_KIND: "list", _VALUE_PAYLOAD: [_encode_value(item) for item in value]}
    if type(value) is tuple:
        return {_VALUE_KIND: "tuple", _VALUE_PAYLOAD: [_encode_value(item) for item in value]}
    if type(value) is dict:
        entries = [[_encode_value(key), _encode_value(item)] for key, item in value.items()]
        entries.sort(key=lambda entry: json.dumps(entry[0], sort_keys=True, separators=(",", ":")))
        return {_VALUE_KIND: "dict", _VALUE_PAYLOAD: entries}
    msg = f"Staged metadata is not JSON serializable: {type(value).__name__}"
    raise TypeError(msg)


def _decode_scalar(value: object) -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    msg = "Encoded staged scalar metadata is invalid"
    raise TypeError(msg)


def _decode_bytes(value: object) -> bytes:
    if not isinstance(value, str):
        msg = "Encoded staged bytes metadata must be a string"
        raise TypeError(msg)
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        msg = "Encoded staged bytes metadata is invalid"
        raise ValueError(msg) from exc


def _decode_sequence(kind: str, value: object) -> object:
    if not isinstance(value, list):
        msg = f"Encoded staged {kind} metadata must be a list"
        raise TypeError(msg)
    items = [_decode_value(item) for item in value]
    return items if kind == "list" else tuple(items)


def _decode_dict(value: object) -> dict[object, object]:
    if not isinstance(value, list):
        msg = "Encoded staged dict metadata must be a list"
        raise TypeError(msg)
    decoded: dict[object, object] = {}
    for entry in value:
        if not isinstance(entry, list) or len(entry) != _KEY_VALUE_PAIR_LEN:
            msg = "Encoded staged dict entries must be key/value pairs"
            raise TypeError(msg)
        key = _decode_value(entry[0])
        item = _decode_value(entry[1])
        try:
            if key in decoded:
                msg = "Encoded staged dict metadata contains duplicate keys"
                raise ValueError(msg)
            decoded[key] = item
        except TypeError as exc:
            msg = "Decoded staged dict keys must be hashable"
            raise TypeError(msg) from exc
    return decoded


def _decode_value(payload: object) -> object:
    if not isinstance(payload, Mapping):
        msg = "Encoded staged metadata must be a mapping"
        raise TypeError(msg)
    if set(payload) != _ENCODED_VALUE_FIELDS:
        msg = "Encoded staged metadata has invalid fields"
        raise ValueError(msg)
    kind = payload.get(_VALUE_KIND)
    value = payload.get(_VALUE_PAYLOAD)
    if kind == "scalar":
        return _decode_scalar(value)
    if kind == "bytes":
        return _decode_bytes(value)
    if kind in {"list", "tuple"}:
        return _decode_sequence(kind, value)
    if kind == "dict":
        return _decode_dict(value)
    msg = f"Unknown staged metadata kind: {kind!r}"
    raise ValueError(msg)


def _treedef_tag(node_type: type[Any] | None) -> str:
    if node_type is None:
        return "leaf"
    tags = {
        dict: "dict",
        list: "list",
        tuple: "tuple",
        Static: "static",
        **_RESULT_TYPE_TAGS,
    }
    try:
        return tags[node_type]
    except KeyError as exc:
        msg = (
            "Staged serialization supports pytrees made from dict, list, tuple, "
            "standard Array API result containers, and advect.pytree.Static; "
            f"got {node_type.__qualname__}. "
            "Stage the numerical leaves or raw-array kernel and reconstruct "
            "the custom container outside the staged program."
        )
        raise TypeError(msg) from exc


def _encode_treedef(treedef: TreeDef) -> dict[str, object]:
    tag = _treedef_tag(treedef.node_type)
    aux: object
    if tag == "dict":
        aux = [_encode_value(key) for key in cast("tuple[Any, ...]", treedef.aux_data)]
    elif tag == "static":
        aux = _encode_value(treedef.aux_data)
    elif tag in _SERIALIZED_RESULT_TYPES:
        aux = len(treedef.children)
    else:
        aux = treedef.aux_data
    return {
        "type": tag,
        "aux": aux,
        "children": [_encode_treedef(child) for child in treedef.children],
        "num_leaves": treedef.num_leaves,
    }


def _decode_leaf_treedef(
    aux: object,
    children: tuple[TreeDef, ...],
) -> tuple[type[Any] | None, object, int]:
    if aux is not None:
        msg = "Encoded staged leaf treedef aux must be None"
        raise ValueError(msg)
    if children:
        msg = "Encoded staged leaf treedef cannot have children"
        raise ValueError(msg)
    return None, None, 1


def _decode_dict_treedef(
    aux: object,
    children: tuple[TreeDef, ...],
) -> tuple[type[Any], object, int]:
    if not isinstance(aux, list):
        msg = "Encoded staged dict treedef aux must be a list"
        raise TypeError(msg)
    decoded_keys = tuple(_decode_value(key) for key in aux)
    seen_keys: dict[object, None] = {}
    for key in decoded_keys:
        try:
            if key in seen_keys:
                msg = "Encoded staged dict treedef contains duplicate keys"
                raise ValueError(msg)
            seen_keys[key] = None
        except TypeError as exc:
            msg = "Encoded staged dict treedef keys must be hashable"
            raise TypeError(msg) from exc
    if len(decoded_keys) != len(children):
        msg = "Encoded staged dict treedef keys must match its children"
        raise ValueError(msg)
    return dict, decoded_keys, sum(child.num_leaves for child in children)


def _decode_sequence_treedef(
    tag: str,
    aux: object,
    children: tuple[TreeDef, ...],
) -> tuple[type[Any], object, int]:
    if isinstance(aux, bool) or not isinstance(aux, int):
        msg = f"Encoded staged {tag} treedef aux must be an integer"
        raise TypeError(msg)
    if aux != len(children):
        msg = f"Encoded staged {tag} treedef length must match its children"
        raise ValueError(msg)
    node_type = list if tag == "list" else _SERIALIZED_RESULT_TYPES.get(tag, tuple)
    node_aux = aux if node_type in {list, tuple} else None
    return node_type, node_aux, sum(child.num_leaves for child in children)


def _decode_static_treedef(
    aux: object,
    children: tuple[TreeDef, ...],
) -> tuple[type[Any], object, int]:
    if children:
        msg = "Encoded staged Static treedef cannot have children"
        raise ValueError(msg)
    return Static, _decode_value(aux), 0


def _decode_treedef_node(
    tag: object,
    aux: object,
    children: tuple[TreeDef, ...],
) -> tuple[type[Any] | None, object, int]:
    if tag == "leaf":
        return _decode_leaf_treedef(aux, children)
    if tag == "dict":
        return _decode_dict_treedef(aux, children)
    if tag in {"list", "tuple", *_SERIALIZED_RESULT_TYPES}:
        return _decode_sequence_treedef(cast("str", tag), aux, children)
    if tag == "static":
        return _decode_static_treedef(aux, children)
    msg = f"Unknown staged treedef type: {tag!r}"
    raise ValueError(msg)


def _decode_treedef(payload: object) -> TreeDef:
    if not isinstance(payload, Mapping):
        msg = "Encoded staged treedef must be a mapping"
        raise TypeError(msg)
    if set(payload) != _TREEDEF_FIELDS:
        msg = "Encoded staged treedef has invalid fields"
        raise ValueError(msg)
    tag = payload.get("type")
    aux = payload.get("aux")
    children_data = payload.get("children")
    num_leaves = payload.get("num_leaves")
    if not isinstance(children_data, list):
        msg = "Encoded staged treedef children must be a list"
        raise TypeError(msg)
    if isinstance(num_leaves, bool) or not isinstance(num_leaves, int):
        msg = "Encoded staged treedef num_leaves must be an integer"
        raise TypeError(msg)
    children = tuple(_decode_treedef(child) for child in children_data)
    node_type, normalized_aux, expected_leaves = _decode_treedef_node(
        tag,
        aux,
        children,
    )

    if num_leaves != expected_leaves:
        msg = "Encoded staged treedef has an inconsistent leaf count"
        raise ValueError(msg)
    return TreeDef(
        node_type=node_type,
        aux_data=normalized_aux,
        children=children,
        num_leaves=num_leaves,
    )
