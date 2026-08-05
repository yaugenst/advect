"""Property tests for pytree utilities.

These tests focus on structural invariants (flatten/unflatten round-trip, typed
leaf paths, and tree_map behavior).
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from hypothesis import given, settings

import advect as ad
from _pytree_strategies import (
    Pair,
    pytree_nonempty_scalar_tree,
    pytree_numeric_tree,
    pytree_scalar_tree,
)


def _dict_key_orders(tree: Any) -> list[tuple[Any, ...]]:
    if isinstance(tree, dict):
        dict_orders = [tuple(tree.keys())]
        for value in tree.values():
            dict_orders.extend(_dict_key_orders(value))
        return dict_orders
    if isinstance(tree, (list, tuple)):
        seq_orders: list[tuple[Any, ...]] = []
        for value in tree:
            seq_orders.extend(_dict_key_orders(value))
        return seq_orders
    if isinstance(tree, Pair):
        return [*_dict_key_orders(tree.left), *_dict_key_orders(tree.right)]
    if isinstance(tree, ad.pytree.Static):
        return []
    return []


def _get_by_path(tree: Any, path: ad.pytree.TreePath) -> Any:
    value = tree
    for entry in path:
        value = value[entry.key] if isinstance(entry, ad.pytree.DictKey) else value[entry.index]
    return value


class TestPytreeCoreProperties:
    """Core pytree invariants."""

    @given(tree=pytree_scalar_tree())
    @settings(max_examples=50)
    def test_flatten_unflatten_roundtrip(self, tree: Any) -> None:
        """Flatten/unflatten round-trips and preserves dict key order."""
        leaves, treedef = ad.pytree.tree_flatten(tree)
        assert len(leaves) == treedef.num_leaves

        rebuilt = ad.pytree.tree_unflatten(treedef, leaves)
        assert rebuilt == tree
        assert _dict_key_orders(rebuilt) == _dict_key_orders(tree)

    @given(tree=pytree_nonempty_scalar_tree())
    @settings(max_examples=50)
    def test_unflatten_raises_on_leaf_count_mismatch(self, tree: Any) -> None:
        """tree_unflatten raises if the number of leaves does not match the treedef."""
        leaves, treedef = ad.pytree.tree_flatten(tree)
        assert treedef.num_leaves > 0

        with pytest.raises(ValueError, match=r"treedef expects"):
            ad.pytree.tree_unflatten(treedef, leaves[:-1])

        with pytest.raises(ValueError, match=r"treedef expects"):
            ad.pytree.tree_unflatten(treedef, [*leaves, None])

    @given(tree=pytree_scalar_tree())
    def test_flatten_with_paths_is_consistent(self, tree: Any) -> None:
        """tree_flatten_with_paths matches tree_flatten and paths locate leaves."""
        leaves0, treedef0 = ad.pytree.tree_flatten(tree)
        paths, leaves1, treedef1 = ad.pytree.tree_flatten_with_paths(tree)

        assert leaves1 == leaves0
        assert treedef1 == treedef0
        assert len(paths) == len(leaves1)

        for path, leaf in zip(paths, leaves1, strict=True):
            for entry in path:
                assert isinstance(entry, (ad.pytree.DictKey, ad.pytree.SequenceKey))
            assert _get_by_path(tree, path) == leaf

    @given(tree=pytree_scalar_tree())
    @settings(max_examples=50)
    def test_tree_map_raises_on_structure_mismatch(self, tree: Any) -> None:
        """tree_map raises on mismatched treedefs without calling f."""
        other = (tree,)
        with pytest.raises(ValueError, match=r"tree_map requires .* same structure"):
            ad.pytree.tree_map(lambda a, _b: a, tree, other)

    @given(tree=pytree_numeric_tree())
    def test_tree_map_identity(self, tree: Any) -> None:
        """tree_map with identity preserves the input tree."""
        assert ad.pytree.tree_map(lambda v: v, tree) == tree

    @given(tree=pytree_numeric_tree())
    def test_tree_map_multi_tree_is_leafwise(self, tree: Any) -> None:
        """tree_map over two trees is leaf-wise and structure-preserving."""
        leaves, treedef = ad.pytree.tree_flatten(tree)
        other = ad.pytree.tree_unflatten(treedef, [v + 1 for v in leaves])

        summed = ad.pytree.tree_map(lambda a, b: a + b, tree, other)

        summed_leaves, summed_def = ad.pytree.tree_flatten(summed)
        assert summed_def == treedef

        for base, out in zip(leaves, summed_leaves, strict=True):
            assert math.isclose(out, base + (base + 1), rel_tol=0.0, abs_tol=1e-12)
