"""Properties relating the public transforms to one another."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from hypothesis import example, given, settings
from numpy.testing import assert_allclose, assert_array_equal

import advect as ad
from _pytree_strategies import pytree_array_mode_tree, pytree_numeric_tree_with_scalar

if TYPE_CHECKING:
    from collections.abc import Callable


def _sine_sum(x: Any) -> Any:
    return np.sum(np.sin(x))


_SCALAR_LOSSES = (
    lambda x: np.sum(np.sin(x) * x),
    lambda x: 0.5 * np.sum(np.tanh(x) ** 2),
    lambda x: np.sum(np.exp(0.1 * x)),
    # Identity partials pass the output seed through to a rank-zero input.
    np.sum,
    lambda x: np.sum(x + 1.0),
    _sine_sum,
)


@st.composite
def _real_scalar_loss_inputs(draw: st.DrawFn) -> Any:
    dtype = draw(st.sampled_from([np.float32, np.float64]))
    value = draw(
        hnp.arrays(
            dtype,
            hnp.array_shapes(min_dims=0, max_dims=2, min_side=0, max_side=4),
            elements=st.floats(min_value=-2.0, max_value=2.0, width=32),
        )
    )
    if value.shape != ():
        return value
    return draw(st.sampled_from([value, value[()], float(value)]))


@given(loss=st.sampled_from(_SCALAR_LOSSES), value=_real_scalar_loss_inputs())
@example(loss=_sine_sum, value=np.array(0.0))
@settings(deadline=None)
def test_reverse_transforms_agree_with_the_unary_grad_fast_path(
    loss: Callable[[Any], Any],
    value: Any,
) -> None:
    # An integer argnums takes the unary fast path for array inputs; a tuple
    # selection, vjp and linearize pull back through the general LinearMap.
    gradient = ad.grad(loss)(value)
    primal, value_gradient = ad.value_and_grad(loss)(value)
    _output, pullback = ad.vjp(loss)(value)
    _output, linear = ad.linearize(loss, value)
    with linear:
        linear_gradient = linear.pullback(1.0)

    for other in (
        value_gradient,
        ad.grad(loss, argnums=(0,))(value)[0],
        pullback(1.0),
        linear_gradient,
    ):
        assert type(other) is type(gradient)
        assert np.asarray(other).dtype == np.asarray(gradient).dtype
        assert_array_equal(other, gradient)
    np.testing.assert_array_max_ulp(np.asarray(primal), np.asarray(loss(value)), maxulp=1)

    # A square Jacobian may be assembled in forward mode, which rounds differently.
    jacobian = ad.jacobian(loss)(value)
    tolerance = 8 * np.finfo(np.asarray(gradient).dtype).eps
    assert np.asarray(jacobian).dtype == np.asarray(gradient).dtype
    assert np.shape(jacobian) == np.shape(gradient)
    assert_allclose(jacobian, gradient, rtol=tolerance, atol=tolerance)


_PYTREES = st.one_of(
    pytree_numeric_tree_with_scalar(),
    pytree_array_mode_tree(
        array_leaf=hnp.arrays(
            st.sampled_from([np.float32, np.float64]),
            hnp.array_shapes(min_dims=0, max_dims=2, min_side=0, max_side=3),
            elements=st.floats(min_value=-10.0, max_value=10.0, width=32),
        )
    ),
)


def _weight(position: int) -> float:
    """Weight leaves by flatten position; every third leaf is disconnected."""
    return 0.0 if position % 3 == 2 else 1.0 + position / 4


def _weighted_sines(tree: Any) -> Any:
    total: Any = 0.0
    for position, leaf in enumerate(ad.pytree.tree_leaves(tree)):
        if _weight(position):
            total = total + _weight(position) * np.sum(np.sin(leaf))
    return total


@given(tree=_PYTREES)
@settings(deadline=None)
def test_pytree_gradients_agree_across_transforms(tree: Any) -> None:
    # Position weights and position-dependent tangents expose a gradient or
    # tangent tree flattened in another order than the primal.
    leaves, treedef = ad.pytree.tree_flatten(tree)
    tangents = [
        np.linspace(-1.0, 1.0, leaf.size).reshape(leaf.shape).astype(leaf.dtype) * (1 + position)
        if isinstance(leaf, np.ndarray)
        else 0.5 - position / 8
        for position, leaf in enumerate(leaves)
    ]
    gradient = ad.grad(_weighted_sines)(tree)
    _value, pullback = ad.vjp(_weighted_sines)(tree)
    value, directional = ad.jvp(_weighted_sines)(
        tree, tangents=ad.pytree.tree_unflatten(treedef, tangents)
    )
    assert np.asarray(directional).dtype == np.asarray(value).dtype

    gradients, gradient_treedef = ad.pytree.tree_flatten(gradient)
    pulled, pulled_treedef = ad.pytree.tree_flatten(pullback(1.0))
    assert gradient_treedef == pulled_treedef == treedef
    pairing = magnitude = 0.0
    for position, (leaf, leaf_gradient, leaf_pulled, tangent) in enumerate(
        zip(leaves, gradients, pulled, tangents, strict=True)
    ):
        assert_array_equal(leaf_pulled, leaf_gradient, strict=True)
        if isinstance(leaf, np.ndarray):
            assert np.shape(leaf_gradient) == leaf.shape
            assert np.asarray(leaf_gradient).dtype == leaf.dtype
        else:
            # Disconnected Python scalars receive 0.0, never None.
            assert type(leaf_gradient) is float
        expected = _weight(position) * np.cos(np.asarray(leaf, dtype=np.float64))
        eps = np.finfo(np.asarray(leaf_gradient).dtype).eps
        assert_allclose(leaf_gradient, expected, rtol=0, atol=4 * eps * _weight(position))
        products = np.asarray(leaf_gradient, dtype=np.float64) * tangent
        pairing += float(np.sum(products))
        magnitude += float(np.sum(np.abs(products)))
    eps = max(np.finfo(getattr(leaf, "dtype", float)).eps for leaf in leaves)
    assert abs(float(directional) - pairing) <= 16 * eps * magnitude
