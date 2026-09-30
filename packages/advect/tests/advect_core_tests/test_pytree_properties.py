"""Property tests for pytree utilities.

Structural invariants compare sentinel leaves by identity, so a swapped or
misplaced leaf cannot hide behind an equal value.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import hypothesis.strategies as st
import pytest
from hypothesis import example, given, settings

import advect as ad
from _pytree_strategies import (
    Fresh,
    LabeledBox,
    Pair,
    Point,
    pytree_nonempty_scalar_tree,
    pytree_scalar_tree,
)
from advect.core._pytree import _tree_contains_tracer

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


class _FakeTracer:
    """Any value with a callable ``_advect_snapshot`` counts as a tracer."""

    def _advect_snapshot(self) -> None:
        msg = "the tracer scan must not snapshot values"
        raise AssertionError(msg)


def _children(tree: Any) -> tuple[Any, ...]:
    """Independent oracle for the flatten order of the strategy's node kinds."""
    if isinstance(tree, Pair):
        return (tree.left, tree.right)
    if isinstance(tree, Fresh):
        return ([tree.left, tree.right],)
    if isinstance(tree, LabeledBox):
        return (tree.value,)
    return tuple(tree)


def _get_by_path(tree: Any, path: ad.pytree.TreePath) -> Any:
    value = tree
    for entry in path:
        if isinstance(entry, ad.pytree.DictKey):
            assert type(value) is dict
            value = value[entry.key]
        else:
            assert isinstance(entry, ad.pytree.SequenceKey)
            value = _children(value)[entry.index]
    return value


def _dict_key_orders(tree: Any) -> list[tuple[Any, ...]]:
    if not isinstance(tree, (dict, list, tuple, Pair, Fresh, LabeledBox)):
        return []
    if isinstance(tree, dict):
        return [
            tuple(tree),
            *(order for child in tree.values() for order in _dict_key_orders(child)),
        ]
    return [order for child in _children(tree) for order in _dict_key_orders(child)]


_REBUILD: dict[type[Any], Callable[[Any, list[Any]], Any]] = {
    list: lambda _tree, fields: fields,
    tuple: lambda _tree, fields: tuple(fields),
    dict: lambda tree, fields: dict(zip(tree, fields, strict=True)),
    Point: lambda _tree, fields: Point(*fields),
    Pair: lambda tree, fields: Pair(*fields, tag=tree.tag),
    Fresh: lambda _tree, fields: Fresh(*fields),
    LabeledBox: lambda _tree, fields: LabeledBox(*fields),
}


def _fields(tree: Any) -> list[Any]:
    if type(tree) is dict:
        return list(tree.values())
    if isinstance(tree, (Pair, Fresh)):
        return [tree.left, tree.right]
    if isinstance(tree, LabeledBox):
        return [tree.value]
    return list(tree)


def _container_edits(tree: Any) -> Iterator[Any]:
    """Yield the edits of one container itself: its type, arity, or key order."""
    kind = type(tree)
    if kind in (list, tuple):
        yield (tuple if kind is list else list)(tree)
        if tree:
            yield kind(tree[:-1])
    elif kind is dict:
        if len(tree) > 1:
            yield dict(reversed(tree.items()))
        if tree:
            yield dict(list(tree.items())[:-1])


def _structural_variants(tree: Any) -> Iterator[Any]:
    """Yield every tree that differs from ``tree`` by one container edit."""
    rebuild = _REBUILD.get(type(tree))
    if rebuild is None:
        return
    yield from _container_edits(tree)
    fields = _fields(tree)
    for index, child in enumerate(fields):
        for variant in _structural_variants(child):
            yield rebuild(tree, [*fields[:index], variant, *fields[index + 1 :]])


class TestPytreeCoreProperties:
    """Core pytree invariants."""

    @given(tree=pytree_scalar_tree())
    @settings(max_examples=50)
    def test_flatten_unflatten_roundtrip(self, tree: Any) -> None:
        """Flatten/unflatten round-trips and preserves dict key order."""
        leaves, treedef = ad.pytree.tree_flatten(tree)
        assert len(leaves) == treedef.num_leaves
        assert ad.pytree.tree_leaves(tree) == leaves

        rebuilt = ad.pytree.tree_unflatten(treedef, leaves)
        assert rebuilt == tree
        assert _dict_key_orders(rebuilt) == _dict_key_orders(tree)

    @given(tree=pytree_scalar_tree())
    def test_paths_locate_every_leaf_by_identity(self, tree: Any) -> None:
        """Paths agree with flatten, are distinct, and address each leaf exactly."""
        leaves, treedef = ad.pytree.tree_flatten(tree)
        paths, path_leaves, path_treedef = ad.pytree.tree_flatten_with_paths(tree)
        assert path_treedef == treedef
        assert len(paths) == len(set(paths)) == len(path_leaves) == len(leaves)
        for path, leaf, path_leaf in zip(paths, leaves, path_leaves, strict=True):
            assert path_leaf is leaf
            assert _get_by_path(tree, path) is leaf

        sentinels = [object() for _ in leaves]
        rebuilt = ad.pytree.tree_unflatten(treedef, sentinels)
        rebuilt_paths, rebuilt_leaves, rebuilt_treedef = ad.pytree.tree_flatten_with_paths(rebuilt)
        assert rebuilt_treedef == treedef
        assert rebuilt_paths == paths
        assert _dict_key_orders(rebuilt) == _dict_key_orders(tree)
        for path, sentinel, rebuilt_leaf in zip(paths, sentinels, rebuilt_leaves, strict=True):
            assert rebuilt_leaf is sentinel
            assert _get_by_path(rebuilt, path) is sentinel

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

    @given(tree=pytree_scalar_tree(), data=st.data())
    @settings(max_examples=50)
    def test_tree_map_raises_on_structure_mismatch(self, tree: Any, data: st.DataObject) -> None:
        """Any one-container structural edit is rejected before ``f`` runs."""
        other = data.draw(st.sampled_from([(tree,), *_structural_variants(tree)]))
        assert ad.pytree.tree_flatten(other)[1] != ad.pytree.tree_flatten(tree)[1]
        calls: list[tuple[Any, ...]] = []

        with pytest.raises(ValueError, match=r"tree_map requires .* same structure"):
            ad.pytree.tree_map(lambda *values: calls.append(values), tree, other)
        assert not calls

    @given(tree=pytree_scalar_tree())
    def test_tree_map_is_leafwise_in_flatten_order(self, tree: Any) -> None:
        """tree_map pairs leaves positionally and rebuilds the first tree's structure."""
        leaves, treedef = ad.pytree.tree_flatten(tree)
        partners = [object() for _ in leaves]
        other = ad.pytree.tree_unflatten(treedef, partners)
        outputs = [object() for _ in leaves]
        calls: list[tuple[Any, Any]] = []

        def record(leaf: Any, partner: Any) -> object:
            calls.append((leaf, partner))
            return outputs[len(calls) - 1]

        mapped = ad.pytree.tree_map(record, tree, other)
        mapped_leaves, mapped_treedef = ad.pytree.tree_flatten(mapped)

        assert mapped_treedef == treedef
        assert len(calls) == len(leaves)
        for (leaf, partner), expected_leaf, expected_partner in zip(
            calls, leaves, partners, strict=True
        ):
            assert leaf is expected_leaf
            assert partner is expected_partner
        assert all(got is want for got, want in zip(mapped_leaves, outputs, strict=True))

    @given(tree=pytree_scalar_tree(), position=st.integers(min_value=0))
    @example(
        tree=Pair(Fresh(0, 0), Point(Fresh(0, 0), 0), tag="a"),
        position=2,
    ).via("discovered failure: reused temporary addresses hid a tracer")
    def test_tracer_scan_finds_a_tracer_at_any_leaf(self, tree: Any, position: int) -> None:
        """The tracer scan is False for concrete trees and True once any leaf is traced."""
        assert not _tree_contains_tracer(tree)
        leaves, treedef = ad.pytree.tree_flatten(tree)
        if not leaves:
            return
        traced_position = position % len(leaves)
        marked = ad.pytree.tree_unflatten(
            treedef,
            [
                _FakeTracer() if index == traced_position else leaf
                for index, leaf in enumerate(leaves)
            ],
        )
        assert _tree_contains_tracer(marked)
