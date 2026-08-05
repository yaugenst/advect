"""Mutation and alias-safety contracts for NumPy array functions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import advect as ad


def test_nan_to_num_copy_false_is_rejected_without_mutating_the_input() -> None:
    primal = np.array([np.nan, 2.0])

    def objective(value: Any) -> Any:
        return np.sum(np.nan_to_num(value, copy=False))

    with pytest.raises(
        ad.TracingError,
        match=r"nan_to_num\(copy=False\).*copy=True",
    ):
        ad.grad(objective)(primal)

    assert np.isnan(primal[0])
    assert primal[1] == 2.0


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda value: np.broadcast_arrays(value)[0], id="broadcast_arrays"),
        pytest.param(np.real_if_close, id="real_if_close"),
    ],
)
def test_supported_array_functions_retain_alias_provenance(operation: Any) -> None:
    primal = np.array([1.0, 2.0])
    tangent = np.array([0.25, -0.5])

    value, directional = ad.jvp(operation)(primal, tangents=tangent)

    np.testing.assert_array_equal(value, operation(primal))
    np.testing.assert_array_equal(directional, operation(tangent))

    def use_stale_view(traced: Any) -> Any:
        owned = traced.copy()
        view = operation(owned)
        owned += 1.0
        return np.sum(view)

    with pytest.raises(ad.StaleViewError, match="functionally updated"):
        ad.jvp(use_stale_view)(primal, tangents=tangent)


@pytest.mark.parametrize("operation", [np.atleast_1d, np.atleast_2d, np.atleast_3d])
def test_atleast_functions_reject_multi_input_alias_results(operation: Any) -> None:
    primal = np.array([1.0, 2.0])

    def objective(value: Any) -> Any:
        first, _second = operation(value, value + 1.0)
        return np.sum(first)

    with pytest.raises(ad.TracingError, match="Call it separately for each input"):
        ad.grad(objective)(primal)


def test_single_input_atleast_function_retains_alias_provenance() -> None:
    primal = np.array([1.0, 2.0])
    tangent = np.ones_like(primal)

    value, directional = ad.jvp(lambda traced: np.atleast_2d(traced))(
        primal,
        tangents=tangent,
    )

    np.testing.assert_array_equal(value, np.atleast_2d(primal))
    np.testing.assert_array_equal(directional, np.atleast_2d(tangent))

    def use_stale_view(traced: Any) -> Any:
        owned = traced.copy()
        view = np.atleast_2d(owned)
        owned += 1.0
        return np.sum(view)

    with pytest.raises(ad.StaleViewError, match="functionally updated"):
        ad.jvp(use_stale_view)(primal, tangents=tangent)
