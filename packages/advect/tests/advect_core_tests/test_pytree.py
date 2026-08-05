"""Tests for pytree utilities."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad


class _ProtocolBase:
    def __init__(self, left: object, right: object, *, tag: str) -> None:
        self.left = left
        self.right = right
        self.tag = tag

    def __advect_tree_flatten__(self) -> tuple[tuple[object, ...], object]:
        return (self.left, self.right), self.tag

    @classmethod
    def __advect_tree_unflatten__(
        cls,
        aux_data: object,
        children: tuple[object, ...],
    ) -> _ProtocolBase:
        assert isinstance(aux_data, str)
        return cls(children[0], children[1], tag=aux_data)


class _ProtocolChild(_ProtocolBase):
    pass


def test_tree_flatten_unflatten_roundtrip() -> None:
    tree = {"a": [1, 2.0], "b": (3.0,)}
    leaves, treedef = ad.pytree.tree_flatten(tree)
    assert leaves == [1, 2.0, 3.0]
    assert ad.pytree.tree_unflatten(treedef, leaves) == tree


def test_tree_map_single_tree() -> None:
    tree = {"x": 1.0, "y": [2.0, 3.0]}
    mapped = ad.pytree.tree_map(lambda v: v * 2, tree)
    assert mapped == {"x": 2.0, "y": [4.0, 6.0]}


def test_tree_map_multi_tree() -> None:
    a = {"x": 1.0, "y": 2.0}
    b = {"x": 3.0, "y": 4.0}
    mapped = ad.pytree.tree_map(lambda x, y: x + y, a, b)
    assert mapped == {"x": 4.0, "y": 6.0}


def test_static_wrapper_has_no_leaves_and_is_preserved() -> None:
    tree = {"cfg": ad.pytree.static({"foo": 1}), "x": 1.0}
    leaves, treedef = ad.pytree.tree_flatten(tree)
    assert leaves == [1.0]

    mapped = ad.pytree.tree_map(lambda v: v + 1.0, tree)
    assert isinstance(mapped["cfg"], ad.pytree.Static)
    assert mapped["cfg"].value == {"foo": 1}
    assert mapped["x"] == 2.0
    assert ad.pytree.tree_unflatten(treedef, leaves) == tree


def test_tree_flatten_with_paths_returns_typed_entries() -> None:
    tree = {"a": [1.0]}
    paths, leaves, treedef = ad.pytree.tree_flatten_with_paths(tree)

    assert leaves == [1.0]
    assert treedef.num_leaves == 1
    assert len(paths) == 1

    path = paths[0]
    assert isinstance(path[0], ad.pytree.DictKey)
    assert isinstance(path[1], ad.pytree.SequenceKey)
    assert ad.pytree.format_path(path) == "['a'][0]"


def test_tree_flatten_leaf_roundtrip() -> None:
    leaf = 3.14
    leaves, treedef = ad.pytree.tree_flatten(leaf)
    paths, leaves_with_paths, treedef_with_paths = ad.pytree.tree_flatten_with_paths(leaf)

    assert leaves == [leaf]
    assert leaves_with_paths == [leaf]
    assert paths == [()]
    assert treedef == treedef_with_paths
    assert ad.pytree.tree_unflatten(treedef, leaves) == leaf


def test_tree_flatten_treats_subclassed_builtin_container_as_leaf() -> None:
    class _DictSubclass(dict[object, object]):
        pass

    tree = _DictSubclass({"a": 1.0})
    leaves, treedef = ad.pytree.tree_flatten(tree)

    assert leaves == [tree]
    assert treedef.node_type is None


def test_inherited_pytree_protocol_preserves_the_concrete_subclass() -> None:
    tree = _ProtocolChild(1.0, 2.0, tag="parameters")

    leaves, treedef = ad.pytree.tree_flatten(tree)
    restored = ad.pytree.tree_unflatten(treedef, leaves)

    assert leaves == [1.0, 2.0]
    assert treedef.node_type is _ProtocolChild
    assert type(restored) is _ProtocolChild
    assert restored.left == 1.0
    assert restored.right == 2.0
    assert restored.tag == "parameters"


def test_inherited_pytree_protocol_participates_in_autodiff() -> None:
    tree = _ProtocolChild(
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        tag="parameters",
    )

    gradient = ad.grad(lambda pair: np.sum(pair.left * pair.right))(tree)

    assert type(gradient) is _ProtocolChild
    assert gradient.tag == tree.tag
    assert_allclose(gradient.left, tree.right)
    assert_allclose(gradient.right, tree.left)


def test_incomplete_pytree_protocol_fails_at_the_structural_boundary() -> None:
    class _Incomplete:
        def __advect_tree_flatten__(self) -> tuple[tuple[object, ...], object]:
            return (), None

    with pytest.raises(TypeError, match="requires both"):
        ad.pytree.tree_flatten(_Incomplete())
