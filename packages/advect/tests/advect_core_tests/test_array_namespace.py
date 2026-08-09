"""Tests for array namespace detection helpers."""

from __future__ import annotations

import numpy as np
from hypothesis import given, strategies as st

from advect.core._array_api import providers as namespace_helpers
from advect.core._pytree import register_pytree_node


class _NamespaceLeaf:
    def __array_namespace__(self, *, api_version: str | None = None):
        assert api_version == "2024.12"
        return np


def test_get_array_namespace_direct() -> None:
    assert namespace_helpers._get_array_namespace(_NamespaceLeaf()) is np


def test_get_array_namespace_caches_type_level_namespace_resolution() -> None:
    namespace_helpers.__dict__["_NAMESPACE_BY_TYPE"].clear()

    class _CountingLeaf:
        calls = 0

        def __array_namespace__(self, *, api_version: str | None = None):
            assert api_version == "2024.12"
            _CountingLeaf.calls += 1
            return np

    leaf = _CountingLeaf()
    assert namespace_helpers._get_array_namespace(leaf) is np
    assert namespace_helpers._get_array_namespace(leaf) is np
    assert _CountingLeaf.calls == 1


def test_get_array_namespace_respects_instance_level_override() -> None:
    namespace_helpers.__dict__["_NAMESPACE_BY_TYPE"].clear()
    custom_namespace = object()

    class _Leaf:
        def __array_namespace__(self, *, api_version: str | None = None):
            assert api_version == "2024.12"
            return np

    regular = _Leaf()
    overridden = _Leaf()
    override_attr = "__array_namespace__"

    def instance_namespace(*, api_version: str | None = None) -> object:
        assert api_version == "2024.12"
        return custom_namespace

    setattr(overridden, override_attr, instance_namespace)

    assert namespace_helpers._get_array_namespace(regular) is np
    assert namespace_helpers._get_array_namespace(overridden) is custom_namespace


def test_get_array_namespace_respects_instance_specific_class_marker() -> None:
    first_namespace = object()
    second_namespace = object()

    class InstanceSpecific:
        __slots__ = ("namespace",)
        __advect_namespace_is_instance_specific__ = True

        def __init__(self, namespace: object) -> None:
            self.namespace = namespace

        def __array_namespace__(self, *, api_version: str | None = None) -> object:
            assert api_version == "2024.12"
            return self.namespace

    first = InstanceSpecific(first_namespace)
    second = InstanceSpecific(second_namespace)

    assert namespace_helpers._get_array_namespace(first) is first_namespace
    assert namespace_helpers._get_array_namespace(second) is second_namespace


def test_get_array_namespace_primitives_return_none() -> None:
    bool_leaf = True
    assert namespace_helpers._get_array_namespace(1) is None
    assert namespace_helpers._get_array_namespace(1.0) is None
    assert namespace_helpers._get_array_namespace(bool_leaf) is None
    assert namespace_helpers._get_array_namespace("x") is None


def test_infer_array_namespace_scans_args_and_kwargs() -> None:
    tree = {"left": [1.0, _NamespaceLeaf()]}
    assert namespace_helpers._infer_array_namespace_for_call(args=(tree,), kwargs={}) is np
    assert (
        namespace_helpers._infer_array_namespace_for_call(args=(), kwargs={"payload": tree}) is np
    )


def test_infer_array_namespace_handles_non_pytree_leaf() -> None:
    assert namespace_helpers._infer_array_namespace_for_call(args=(1.0,), kwargs={}) is None


def test_infer_array_namespace_validates_later_pytree_branches() -> None:
    calls = {"count": 0}

    class _LaterBranch:
        pass

    def _flatten_later_branch(tree: object) -> tuple[tuple[object, ...], None]:
        del tree
        calls["count"] += 1
        return (), None

    def _unflatten_later_branch(
        aux_data: object,
        children: tuple[object, ...],
    ) -> _LaterBranch:
        del aux_data, children
        return _LaterBranch()

    register_pytree_node(
        _LaterBranch,
        flatten_fn=_flatten_later_branch,
        unflatten_fn=_unflatten_later_branch,
    )
    tree = [_NamespaceLeaf(), _LaterBranch()]
    assert namespace_helpers._infer_array_namespace_for_call(args=(tree,), kwargs={}) is np
    assert calls["count"] == 1


_STATIC_TREE = st.recursive(
    st.one_of(st.none(), st.booleans(), st.integers(), st.text(max_size=8)),
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.lists(children, max_size=4).map(tuple),
        st.dictionaries(st.text(max_size=4), children, max_size=4),
    ),
    max_leaves=12,
)


@given(before=_STATIC_TREE, after=_STATIC_TREE)
def test_infer_array_namespace_finds_array_after_arbitrary_static_trees(
    before: object,
    after: object,
) -> None:
    assert (
        namespace_helpers._infer_array_namespace_for_call(
            args=(before,),
            kwargs={"payload": [after, {"array": _NamespaceLeaf()}]},
        )
        is np
    )


@given(tree=_STATIC_TREE)
def test_infer_array_namespace_returns_none_for_static_pytrees(tree: object) -> None:
    assert namespace_helpers._infer_array_namespace_for_call(args=(tree,), kwargs={}) is None
