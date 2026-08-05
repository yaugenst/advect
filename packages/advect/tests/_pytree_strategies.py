"""Shared Hypothesis strategies for core and autodiff pytree contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import hypothesis.strategies as st

import advect as ad


@dataclass(frozen=True, slots=True)
class Pair:
    """Small custom pytree used by structural and transform properties."""

    left: Any
    right: Any
    tag: str

    def __getitem__(self, index: int) -> Any:
        if index == 0:
            return self.left
        if index == 1:
            return self.right
        raise IndexError(index)


def _pair_flatten(tree: Pair) -> tuple[tuple[Any, ...], Any]:
    return (tree.left, tree.right), tree.tag


def _pair_unflatten(aux_data: Any, children: tuple[Any, ...]) -> Pair:
    if not isinstance(aux_data, str):
        msg = f"Invalid aux_data for Pair: expected str, got {type(aux_data).__name__}"
        raise TypeError(msg)
    if len(children) != 2:
        msg = f"Invalid children for Pair: expected 2 children, got {len(children)}"
        raise ValueError(msg)
    left, right = children
    return Pair(left=left, right=right, tag=aux_data)


ad.pytree.register_pytree_node(
    Pair,
    flatten_fn=_pair_flatten,
    unflatten_fn=_pair_unflatten,
)

_SIMPLE_KEY = st.text(
    alphabet=st.characters(min_codepoint=97, max_codepoint=122),
    min_size=1,
    max_size=3,
)
_STATIC_VALUE = st.dictionaries(
    _SIMPLE_KEY,
    st.integers(min_value=-3, max_value=3),
    max_size=3,
)
_NUMERIC_LEAF = st.one_of(
    st.integers(min_value=-5, max_value=5),
    st.floats(
        min_value=-10.0,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
_STATIC_LEAF = st.builds(ad.pytree.static, _STATIC_VALUE)
_SCALAR_LEAF = st.one_of(
    _NUMERIC_LEAF,
    st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), max_size=6),
    st.none(),
    _STATIC_LEAF,
)


def _recursive_tree(
    leaf: st.SearchStrategy[Any],
    *,
    max_leaves: int,
) -> st.SearchStrategy[Any]:
    return st.recursive(
        leaf,
        lambda children: st.one_of(
            st.lists(children, max_size=4),
            st.lists(children, max_size=4).map(tuple),
            st.dictionaries(_SIMPLE_KEY, children, max_size=4),
            st.builds(Pair, children, children, tag=_SIMPLE_KEY),
        ),
        max_leaves=max_leaves,
    )


def _with_required_leaf(
    tree: st.SearchStrategy[Any],
    required_leaf: st.SearchStrategy[Any],
) -> st.SearchStrategy[Any]:
    """Add one guaranteed leaf without filtering an otherwise recursive tree."""
    required_then_tree = st.tuples(required_leaf, tree)
    return st.one_of(
        required_leaf,
        required_then_tree.map(list),
        st.tuples(tree, required_leaf),
        required_then_tree.map(lambda values: {"required": values[0], "tree": values[1]}),
        st.builds(Pair, required_leaf, tree, tag=_SIMPLE_KEY),
    )


def pytree_scalar_tree(*, max_leaves: int = 25) -> st.SearchStrategy[Any]:
    """Build recursive pytrees over scalar, static, text, and null leaves."""
    return _recursive_tree(_SCALAR_LEAF, max_leaves=max_leaves)


def pytree_nonempty_scalar_tree(*, max_leaves: int = 25) -> st.SearchStrategy[Any]:
    """Build scalar pytrees with at least one ordinary, non-static leaf."""
    return _with_required_leaf(
        pytree_scalar_tree(max_leaves=max_leaves),
        _NUMERIC_LEAF,
    )


def pytree_numeric_tree(*, max_leaves: int = 25) -> st.SearchStrategy[Any]:
    """Build recursive pytrees over Python numeric and static leaves."""
    return _recursive_tree(
        st.one_of(_NUMERIC_LEAF, _STATIC_LEAF),
        max_leaves=max_leaves,
    )


def pytree_numeric_tree_with_scalar(*, max_leaves: int = 25) -> st.SearchStrategy[Any]:
    """Build numeric pytrees with at least one differentiable Python scalar."""
    return _with_required_leaf(
        pytree_numeric_tree(max_leaves=max_leaves),
        _NUMERIC_LEAF,
    )


def pytree_array_mode_tree(
    *,
    array_leaf: st.SearchStrategy[Any],
    max_leaves: int = 20,
) -> st.SearchStrategy[Any]:
    """Build mixed pytrees with at least one NumPy array leaf."""
    tree = _recursive_tree(
        st.one_of(_NUMERIC_LEAF, array_leaf, _STATIC_LEAF),
        max_leaves=max_leaves,
    )
    return _with_required_leaf(tree, array_leaf)
