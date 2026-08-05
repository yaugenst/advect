"""Focused contracts for staged-program value and pytree codecs."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from advect.core._pytree import TreeDef, static, tree_flatten
from advect.core._stage_serialization import (
    _decode_scalar,
    _decode_treedef,
    _decode_value,
    _encode_treedef,
    _encode_value,
)


def _scalar(value: object) -> dict[str, object]:
    return {"kind": "scalar", "value": value}


def _leaf_payload() -> dict[str, object]:
    return {
        "type": "leaf",
        "aux": None,
        "children": [],
        "num_leaves": 1,
    }


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(1.25, id="finite-float"),
        pytest.param(b"\x00\xff", id="bytes"),
        pytest.param([1, "two"], id="list"),
        pytest.param((True, 3), id="tuple"),
        pytest.param({"name": [1, 2], ("key", 1): b"value"}, id="dict"),
    ],
)
def test_static_value_codec_round_trips_supported_values(value: object) -> None:
    assert _decode_value(_encode_value(value)) == value


def test_static_value_codec_canonicalizes_dict_order() -> None:
    assert _encode_value({"left": 1, "right": 2}) == _encode_value({"right": 2, "left": 1})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_static_value_encoder_rejects_non_finite_floats(value: float) -> None:
    with pytest.raises(TypeError, match="finite floats"):
        _encode_value(value)


def test_static_value_encoder_rejects_arbitrary_objects() -> None:
    with pytest.raises(TypeError, match="not JSON serializable"):
        _encode_value(object())


@pytest.mark.parametrize("value", [1.5, float("nan"), object()])
def test_scalar_decoder_accepts_only_finite_scalar_metadata(value: object) -> None:
    if value == 1.5:
        assert _decode_scalar(value) == value
        return
    with pytest.raises(TypeError, match="scalar metadata is invalid"):
        _decode_scalar(value)


@pytest.mark.parametrize(
    ("payload", "error", "match"),
    [
        pytest.param(
            {"kind": "bytes", "value": 1},
            TypeError,
            "must be a string",
            id="bytes-not-string",
        ),
        pytest.param(
            {"kind": "bytes", "value": "not-hex"},
            ValueError,
            "bytes metadata is invalid",
            id="invalid-hex",
        ),
        pytest.param(
            {"kind": "list", "value": "not-a-list"},
            TypeError,
            "list metadata must be a list",
            id="sequence-not-list",
        ),
        pytest.param(
            {"kind": "dict", "value": "not-a-list"},
            TypeError,
            "dict metadata must be a list",
            id="dict-not-list",
        ),
        pytest.param(
            {"kind": "dict", "value": [[_scalar("key")]]},
            TypeError,
            "key/value pairs",
            id="dict-entry-not-pair",
        ),
        pytest.param(
            {
                "kind": "dict",
                "value": [
                    [
                        {"kind": "list", "value": []},
                        _scalar("value"),
                    ]
                ],
            },
            TypeError,
            "keys must be hashable",
            id="unhashable-dict-key",
        ),
        pytest.param([], TypeError, "must be a mapping", id="payload-not-mapping"),
        pytest.param(
            {"kind": "scalar", "value": 1, "extra": True},
            ValueError,
            "invalid fields",
            id="invalid-fields",
        ),
        pytest.param(
            {"kind": "unknown", "value": None},
            ValueError,
            "Unknown staged metadata kind",
            id="unknown-kind",
        ),
    ],
)
def test_static_value_decoder_rejects_malformed_payloads(
    payload: object,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        _decode_value(payload)


@dataclass
class _UnsupportedPytreeNode:
    value: object


@pytest.mark.parametrize(
    "tree",
    [
        pytest.param(1.0, id="leaf"),
        pytest.param({"x": 1.0}, id="dict"),
        pytest.param([1.0, 2.0], id="list"),
        pytest.param((1.0, 2.0), id="tuple"),
        pytest.param(static({"solver": "native"}), id="static"),
    ],
)
def test_treedef_codec_round_trips_supported_nodes(tree: object) -> None:
    _leaves, treedef = tree_flatten(tree)
    assert _decode_treedef(_encode_treedef(treedef)) == treedef


def test_treedef_encoder_rejects_unregistered_node_types() -> None:
    treedef = TreeDef(
        node_type=_UnsupportedPytreeNode,
        aux_data=None,
        children=(),
        num_leaves=0,
    )
    with pytest.raises(TypeError, match="Staged serialization supports pytrees"):
        _encode_treedef(treedef)


def _malformed_treedef_cases() -> list[tuple[object, type[Exception], str]]:
    leaf = _leaf_payload()
    return [
        (
            {"type": "leaf", "aux": 1, "children": [], "num_leaves": 1},
            ValueError,
            "leaf treedef aux",
        ),
        (
            {"type": "leaf", "aux": None, "children": [leaf], "num_leaves": 1},
            ValueError,
            "leaf treedef cannot have children",
        ),
        (
            {"type": "dict", "aux": "x", "children": [], "num_leaves": 0},
            TypeError,
            "dict treedef aux",
        ),
        (
            {
                "type": "dict",
                "aux": [_scalar(1), {"kind": "scalar", "value": True}],
                "children": [leaf, leaf],
                "num_leaves": 2,
            },
            ValueError,
            "duplicate keys",
        ),
        (
            {
                "type": "dict",
                "aux": [{"kind": "list", "value": []}],
                "children": [leaf],
                "num_leaves": 1,
            },
            TypeError,
            "keys must be hashable",
        ),
        (
            {"type": "dict", "aux": [], "children": [leaf], "num_leaves": 1},
            ValueError,
            "keys must match its children",
        ),
        (
            {"type": "list", "aux": True, "children": [], "num_leaves": 0},
            TypeError,
            "aux must be an integer",
        ),
        (
            {"type": "tuple", "aux": 2, "children": [leaf], "num_leaves": 1},
            ValueError,
            "length must match its children",
        ),
        (
            {"type": "static", "aux": _scalar("x"), "children": [leaf], "num_leaves": 1},
            ValueError,
            "Static treedef cannot have children",
        ),
        (
            {"type": "unknown", "aux": None, "children": [], "num_leaves": 0},
            ValueError,
            "Unknown staged treedef type",
        ),
        ([], TypeError, "treedef must be a mapping"),
        (
            {"type": "leaf", "aux": None, "children": []},
            ValueError,
            "invalid fields",
        ),
        (
            {"type": "leaf", "aux": None, "children": (), "num_leaves": 1},
            TypeError,
            "children must be a list",
        ),
        (
            {"type": "leaf", "aux": None, "children": [], "num_leaves": True},
            TypeError,
            "num_leaves must be an integer",
        ),
        (
            {"type": "leaf", "aux": None, "children": [], "num_leaves": 2},
            ValueError,
            "inconsistent leaf count",
        ),
    ]


@pytest.mark.parametrize(("payload", "error", "match"), _malformed_treedef_cases())
def test_treedef_decoder_rejects_malformed_payloads(
    payload: object,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        _decode_treedef(payload)
