"""Focused contracts for staged-program value and pytree codecs."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from hypothesis import example, given, strategies as st

from advect.core._array_api.results import _SERIALIZED_RESULT_TYPES
from advect.core._pytree import TreeDef, static, tree_flatten
from advect.core._stage import _same_static_value
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


_STATIC_SCALARS = st.one_of(
    st.sampled_from([None, False, True, 0, 1, 0.0, -0.0, 1.0, "1", b"1"]),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=3),
    st.binary(max_size=3),
)
_STATIC_VALUES = st.recursive(
    _STATIC_SCALARS,
    lambda children: st.one_of(
        st.lists(children, max_size=3),
        st.lists(children, max_size=3).map(tuple),
        st.dictionaries(
            st.one_of(_STATIC_SCALARS, st.tuples(_STATIC_SCALARS)), children, max_size=3
        ),
    ),
    max_leaves=8,
)


def _codec_identity(value: object) -> str:
    return json.dumps(_encode_value(value))


def _reversed_dicts(value: object) -> object:
    """Rebuild ``value`` with every dict's insertion order reversed."""
    if type(value) is dict:
        return {key: _reversed_dicts(item) for key, item in reversed(value.items())}
    if type(value) in (list, tuple):
        return type(value)(map(_reversed_dicts, value))
    return value


@given(_STATIC_VALUES, _STATIC_VALUES)
@example(1, 1.0)
@example(left=1, right=True)
@example(0.0, -0.0)
@example((1,), [1])
@example({1: "one"}, {1.0: "one"})
@example({(1,): 0}, {(True,): 0})
@example({"a": 1, "b": [2.0]}, {"b": [2.0], "a": 1})
def test_static_value_comparison_matches_codec_identity(left: object, right: object) -> None:
    """Static identity is codec identity, which JSON and dict order preserve."""
    assert _same_static_value(left, right) == (_codec_identity(left) == _codec_identity(right))
    assert _same_static_value(_decode_value(json.loads(_codec_identity(left))), left)
    assert _codec_identity(_reversed_dicts(left)) == _codec_identity(left)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_static_value_encoder_rejects_non_finite_floats(value: float) -> None:
    with pytest.raises(TypeError, match="finite floats"):
        _encode_value(value)


def test_static_value_encoder_rejects_arbitrary_objects() -> None:
    with pytest.raises(TypeError, match="not JSON serializable"):
        _encode_value(object())


@pytest.mark.parametrize("value", [float("nan"), object()])
def test_scalar_decoder_accepts_only_finite_scalar_metadata(value: object) -> None:
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


def _result_nodes(children: st.SearchStrategy[object]) -> st.SearchStrategy[object]:
    return st.sampled_from(list(_SERIALIZED_RESULT_TYPES.values())).flatmap(
        lambda result: st.tuples(*[children] * len(result._fields)).map(
            lambda fields: result(*fields)
        )
    )


_SERIALIZABLE_TREES = st.recursive(
    st.one_of(st.just(0.0), st.builds(static, _STATIC_VALUES)),
    lambda children: st.one_of(
        st.lists(children, max_size=3),
        st.lists(children, max_size=3).map(tuple),
        st.dictionaries(_STATIC_SCALARS, children, max_size=3),
        _result_nodes(children),
    ),
    max_leaves=8,
)


@given(_SERIALIZABLE_TREES)
def test_treedef_codec_round_trips_serializable_trees(tree: object) -> None:
    _leaves, treedef = tree_flatten(tree)
    encoded = _encode_treedef(treedef)
    decoded = _decode_treedef(json.loads(json.dumps(encoded)))
    assert decoded == treedef
    # TreeDef equality conflates 1 and 1.0 keys or aux; the encoding does not.
    assert _encode_treedef(decoded) == encoded


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
