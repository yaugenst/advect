"""Tests for pytree utilities."""

from __future__ import annotations

from typing import cast

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


class _FreshContainers:
    """Protocol node whose flatten builds new child and metadata containers."""

    def __init__(self, left: object, right: object) -> None:
        self.left = left
        self.right = right

    def __advect_tree_flatten__(self) -> tuple[tuple[object, ...], object]:
        return ((self.left, self.right), {"fresh": 0}), None

    @classmethod
    def __advect_tree_unflatten__(
        cls, aux_data: object, children: tuple[object, ...]
    ) -> _FreshContainers:
        del aux_data
        left, right = cast("tuple[object, object]", children[0])
        return cls(left, right)


def test_tree_flatten_treats_subclassed_builtin_container_as_leaf() -> None:
    class _DictSubclass(dict[object, object]):
        pass

    tree = _DictSubclass({"a": 1.0})
    leaves, treedef = ad.pytree.tree_flatten(tree)

    assert leaves == [tree]
    assert treedef.node_type is None


def test_opt_in_pytree_registration_uses_nearest_registered_base() -> None:
    class _Base:
        def __init__(self, value: object) -> None:
            self.value = value

    class _Nearer(_Base):
        pass

    class _Child(_Nearer):
        pass

    def register(cls: type[_Base], tag: str) -> None:
        ad.pytree.register_pytree_node(
            cls,
            flatten_fn=lambda tree: ((tree.value,), (type(tree), tag)),
            unflatten_fn=lambda metadata, children: metadata[0](children[0]),
            include_subclasses=True,
        )

    register(_Base, "base")
    register(_Nearer, "nearer")
    leaves, treedef = ad.pytree.tree_flatten(_Child(1.0))
    restored = ad.pytree.tree_unflatten(treedef, leaves)

    assert treedef.aux_data[1] == "nearer"
    assert type(restored) is _Child


def test_pytree_protocol_overrides_inherited_registration() -> None:
    class _Base:
        pass

    class _Child(_Base):
        def __advect_tree_flatten__(self) -> tuple[tuple[object, ...], object]:
            return (2.0,), "protocol"

        @classmethod
        def __advect_tree_unflatten__(
            cls, aux_data: object, children: tuple[object, ...]
        ) -> _Child:
            assert aux_data == "protocol"
            return cls()

    ad.pytree.register_pytree_node(
        _Base,
        flatten_fn=lambda _tree: ((), "base"),
        unflatten_fn=lambda _metadata, _children: _Base(),
        include_subclasses=True,
    )

    leaves, treedef = ad.pytree.tree_flatten(_Child())

    assert leaves == [2.0]
    assert treedef.aux_data == "protocol"
    assert type(ad.pytree.tree_unflatten(treedef, leaves)) is _Child


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


def test_static_rejects_a_tracer_behind_freshly_flattened_containers() -> None:
    # Regression: temporaries freed after one node's scan reused their
    # addresses for the next node, so the cycle guard skipped the tracer.
    def function(x: np.ndarray) -> object:
        ad.pytree.static([_FreshContainers(1.0, 2.0), _FreshContainers(3.0, x)])
        return np.sum(x)

    with pytest.raises(TypeError, match="Static pytree metadata cannot contain"):
        ad.grad(function)(np.ones(2))


def test_static_primitive_argument_rejects_a_tracer_behind_fresh_containers() -> None:
    @ad.primitive(name="tests.pytree.fresh_static_config", static_argnames=("config",))
    def primitive(x: np.ndarray, config: object) -> np.ndarray:
        del config
        return x

    def function(x: np.ndarray) -> object:
        config = [_FreshContainers(1.0, 2.0), _FreshContainers(3.0, x)]
        return np.sum(primitive(x, config))

    with pytest.raises(TypeError, match=r"declared static.*received a traced value"):
        ad.grad(function)(np.ones(2))


_LEAF = ad.pytree.TreeDef(node_type=None, aux_data=None, children=(), num_leaves=1)


@pytest.mark.parametrize(
    ("children", "declared", "message"),
    [
        pytest.param((_LEAF, _LEAF), 1, "needs more leaves", id="undercounted"),
        pytest.param((_LEAF,), 2, "did not consume all leaves", id="overcounted"),
    ],
)
def test_unflatten_rejects_a_hand_built_treedef_with_a_wrong_leaf_count(
    children: tuple[ad.pytree.TreeDef, ...], declared: int, message: str
) -> None:
    treedef = ad.pytree.TreeDef(
        node_type=tuple, aux_data=len(children), children=children, num_leaves=declared
    )
    with pytest.raises(ValueError, match=message):
        ad.pytree.tree_unflatten(treedef, list(range(declared)))


def test_incomplete_pytree_protocol_fails_at_the_structural_boundary() -> None:
    class _Incomplete:
        def __advect_tree_flatten__(self) -> tuple[tuple[object, ...], object]:
            return (), None

    with pytest.raises(TypeError, match="requires both"):
        ad.pytree.tree_flatten(_Incomplete())
