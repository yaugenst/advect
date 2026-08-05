"""Property tests for scalar-boundary pytrees."""

from __future__ import annotations

import math
from typing import Any

from hypothesis import given, settings

import advect as ad
from _pytree_strategies import pytree_numeric_tree_with_scalar


class TestPytreeAutodiffProperties:
    """Autodiff invariants for pytrees containing real Python scalars."""

    @given(tree=pytree_numeric_tree_with_scalar())
    @settings(max_examples=50)
    def test_grad_preserves_structure_for_python_scalars(self, tree: Any) -> None:
        """Grad preserves structure and differentiates float leaves."""
        assert any(isinstance(v, (int, float)) for v in ad.pytree.tree_leaves(tree))

        def f(params: Any) -> Any:
            total = 0.0
            for leaf in ad.pytree.tree_leaves(params):
                total = total + leaf * leaf
            return total

        grads = ad.grad(f)(tree)
        expected = ad.pytree.tree_map(
            lambda v: 2.0 * v if isinstance(v, (int, float)) else None,
            tree,
        )

        leaves_got, def_got = ad.pytree.tree_flatten(grads)
        leaves_exp, def_exp = ad.pytree.tree_flatten(expected)
        assert def_got == def_exp

        for got, exp in zip(leaves_got, leaves_exp, strict=True):
            assert got == exp

    @given(tree=pytree_numeric_tree_with_scalar())
    @settings(max_examples=50)
    def test_grad_returns_zeros_for_disconnected_float_leaves(self, tree: Any) -> None:
        """Disconnected traced float leaves get zero gradients (not None)."""
        leaves, treedef = ad.pytree.tree_flatten(tree)
        scalar_positions = [i for i, value in enumerate(leaves) if isinstance(value, (int, float))]
        assert scalar_positions

        def f(params: Any) -> Any:
            for leaf in ad.pytree.tree_leaves(params):
                if callable(getattr(leaf, "_advect_snapshot", None)):
                    return leaf * leaf
            return 0.0

        grads = ad.grad(f)(tree)

        first_scalar = scalar_positions[0]
        expected_leaves: list[Any] = []
        for i, value in enumerate(leaves):
            if isinstance(value, (int, float)):
                expected_leaves.append(2.0 * value if i == first_scalar else 0.0)
            else:
                expected_leaves.append(None)
        expected = ad.pytree.tree_unflatten(treedef, expected_leaves)

        got_leaves, got_def = ad.pytree.tree_flatten(grads)
        exp_leaves, exp_def = ad.pytree.tree_flatten(expected)
        assert got_def == exp_def

        for got, exp in zip(got_leaves, exp_leaves, strict=True):
            if exp is None:
                assert got is None
            else:
                assert isinstance(got, float)
                assert math.isclose(got, exp, rel_tol=0.0, abs_tol=1e-12)
