"""Edge contracts for backend-neutral cotangent value helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from advect.autodiff.api import _pullback_values as values
from advect.core._pytree import tree_flatten


class _MethodSum:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, bool]] = []

    def sum(self, *, axis: Any, keepdims: bool) -> tuple[Any, bool]:
        self.calls.append((axis, keepdims))
        return axis, keepdims


def test_sum_uses_value_method_without_array_namespace() -> None:
    value = _MethodSum()

    assert values._sum(value, axis=(0, 2), keepdims=True) == ((0, 2), True)
    assert value.calls == [((0, 2), True)]


def test_sum_rejects_values_without_namespace_or_method() -> None:
    with pytest.raises(TypeError, match="Cannot sum value of type object"):
        values._sum(object(), axis=None, keepdims=False)


def test_unbroadcast_preserves_non_array_and_matching_shape_identity() -> None:
    scalar = object()
    matching = np.ones((2, 3), dtype=np.float64)

    assert values._unbroadcast(scalar, (2, 3)) is scalar
    assert values._unbroadcast(matching, (2, 3)) is matching


def test_unbroadcast_restores_missing_leading_singleton_dimensions() -> None:
    gradient = np.arange(3.0)

    restored = values._unbroadcast(gradient, (1, 3))

    assert restored.shape == (1, 3)
    np.testing.assert_array_equal(restored, gradient[None, :])


def test_unbroadcast_rejects_missing_non_singleton_dimensions() -> None:
    with pytest.raises(ValueError, match="Cotangent rank cannot be restored"):
        values._unbroadcast(np.arange(3.0), (2, 3))


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: values._ones_like(object()), "Cannot construct cotangent"),
        (lambda: values._zeros_like(object()), "Cannot construct zero gradient"),
    ],
)
def test_array_constructors_require_an_array_namespace(
    factory: Any,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        factory()


def test_zero_like_uses_the_value_array_namespace() -> None:
    value = np.array([1.0, 2.0], dtype=np.float64)

    np.testing.assert_array_equal(values._zeros_like(value), np.zeros_like(value))


def test_output_cotangent_flattening_validates_pytree_structure() -> None:
    _leaves, leaf_def = tree_flatten(0.0)
    marker = object()
    assert values._flatten_output_cotangents(leaf_def, marker) == [marker]

    _leaves, pair_def = tree_flatten((0.0, 0.0))
    assert values._flatten_output_cotangents(pair_def, (2.0, 3.0)) == [2.0, 3.0]
    with pytest.raises(ValueError, match="Cotangent pytree structure"):
        values._flatten_output_cotangents(pair_def, [2.0, 3.0])


def test_gradient_tree_preserves_none_for_missing_none_primal() -> None:
    _leaves, treedef = tree_flatten(0.0)
    spec = SimpleNamespace(
        treedef=treedef,
        leaf_specs=(
            SimpleNamespace(
                node_id=7,
                primal=None,
                restore_python_scalar=False,
            ),
        ),
    )

    assert values._build_grad_tree(spec, grads={}) is None
