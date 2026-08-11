"""Property tests for pytree integration in autodiff (NumPy array mode)."""

from __future__ import annotations

import math
from typing import Any

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings
from numpy.testing import assert_allclose

import advect as ad
from _pytree_strategies import pytree_array_mode_tree

_ARRAY_LEAF = hnp.arrays(
    dtype=st.sampled_from([np.dtype(np.float32), np.dtype(np.float64)]),
    shape=hnp.array_shapes(min_dims=0, max_dims=2, min_side=0, max_side=3),
    elements=st.floats(
        min_value=-10.0,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    ),
)
_PYTREE_ARRAY_MODE_TREE = pytree_array_mode_tree(array_leaf=_ARRAY_LEAF)


class TestPytreeArrayModeProperties:
    """Autodiff invariants for pytree inputs/outputs (NumPy array mode)."""

    @given(tree=_PYTREE_ARRAY_MODE_TREE)
    @settings(max_examples=50)
    def test_grad_preserves_structure_for_array_mode(self, tree: Any) -> None:
        """Grad preserves structure in array mode and restores float leaf grads."""
        assert any(isinstance(v, np.ndarray) for v in ad.pytree.tree_leaves(tree))

        def f(params: Any) -> Any:
            total: Any = 0.0
            for leaf in ad.pytree.tree_leaves(params):
                if isinstance(leaf, (int, float)) and not isinstance(leaf, bool):
                    total = total + leaf * leaf
                else:
                    total = total + np.sum(leaf * leaf)
            return total

        grads = ad.grad(f)(tree)
        expected = ad.pytree.tree_map(
            lambda v: (
                2.0 * v
                if isinstance(v, (int, float)) and not isinstance(v, bool)
                else (2 * v if isinstance(v, np.ndarray) else None)
            ),
            tree,
        )

        got_leaves, got_def = ad.pytree.tree_flatten(grads)
        exp_leaves, exp_def = ad.pytree.tree_flatten(expected)
        assert got_def == exp_def

        for got, exp in zip(got_leaves, exp_leaves, strict=True):
            if exp is None:
                assert got is None
            elif isinstance(exp, (np.ndarray, np.generic)):
                assert_allclose(np.asarray(got), np.asarray(exp), rtol=1e-6, atol=1e-6)
            else:
                assert isinstance(got, float)
                assert math.isclose(got, exp, rel_tol=0.0, abs_tol=1e-12)
