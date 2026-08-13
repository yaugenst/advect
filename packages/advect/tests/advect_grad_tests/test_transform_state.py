"""Invocation-local transform state for library integrations."""

from __future__ import annotations

import numpy as np
import pytest

import advect as ad


def test_transform_state_is_scoped_to_one_dynamic_invocation() -> None:
    namespace = object()
    observations: list[tuple[list[float], bool]] = []

    def loss(value: np.ndarray) -> np.ndarray:
        state = ad.transform_state(namespace, list)
        assert state is not None
        same_state = ad.transform_state(namespace, list)
        state.append(float(ad.stop_gradient(value)[0]))
        observations.append((state.copy(), same_state is state))
        return np.sum(value**2)

    assert ad.transform_state(namespace, list) is None
    np.testing.assert_allclose(ad.grad(loss)(np.array([2.0])), np.array([4.0]))
    np.testing.assert_allclose(ad.grad(loss)(np.array([3.0])), np.array([6.0]))
    assert observations == [([2.0], True), ([3.0], True)]


def test_nested_transforms_have_distinct_state_and_restore_the_outer_state() -> None:
    namespace = object()
    identities: list[tuple[int, int, int]] = []

    def outer(value: np.ndarray) -> np.ndarray:
        outer_state = ad.transform_state(namespace, list)
        assert outer_state is not None
        outer_state.append("outer")

        def inner(inner_value: np.ndarray) -> np.ndarray:
            inner_state = ad.transform_state(namespace, list)
            assert inner_state is not None
            assert inner_state == []
            assert ad.transform_states(namespace) == (inner_state, outer_state)
            identities.append((id(outer_state), id(inner_state), 0))
            return np.sum(inner_value**2)

        _ = ad.grad(inner)(value)
        restored = ad.transform_state(namespace, list)
        assert restored is outer_state
        identities[-1] = (identities[-1][0], identities[-1][1], id(restored))
        return np.sum(value**2)

    np.testing.assert_allclose(ad.grad(outer)(np.array([2.0])), np.array([4.0]))
    outer_id, inner_id, restored_id = identities[0]
    assert outer_id == restored_id
    assert inner_id != outer_id
    assert ad.transform_states(namespace) == ()


def test_transform_state_is_released_after_failure() -> None:
    namespace = object()
    states: list[list[str]] = []

    def failing(value: np.ndarray) -> np.ndarray:
        del value
        state = ad.transform_state(namespace, list)
        assert state is not None
        state.append("failed")
        states.append(state)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        ad.grad(failing)(np.array([1.0]))

    def succeeding(value: np.ndarray) -> np.ndarray:
        state = ad.transform_state(namespace, list)
        assert state is not None
        states.append(state)
        return np.sum(value)

    np.testing.assert_allclose(ad.grad(succeeding)(np.array([1.0])), np.array([1.0]))
    assert states[0] == ["failed"]
    assert states[1] == []
    assert states[1] is not states[0]


def test_transform_state_rejects_abstract_staging() -> None:
    with pytest.raises(ad.TracingError, match="only during concrete dynamic transforms"):
        ad.stage(
            lambda value: (ad.transform_state(object(), dict), value)[1],
            specs=(ad.ArraySpec((1,), "float64"),),
        )

    with pytest.raises(ad.TracingError, match="only during concrete dynamic transforms"):
        ad.stage(
            lambda value: (ad.transform_states(object()), value)[1],
            specs=(ad.ArraySpec((1,), "float64"),),
        )
