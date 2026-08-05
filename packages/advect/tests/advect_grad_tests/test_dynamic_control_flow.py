"""Define-by-run control-flow tests for dynamic autodiff calls."""

from __future__ import annotations

import numpy as np

import advect as ad


def _branched_loss(x: np.ndarray) -> np.ndarray:
    if x > 0:
        return x * x
    return -x


def test_default_grad_retraces_data_dependent_array_branch() -> None:
    grad_fn = ad.grad(_branched_loss)

    positive_grad = grad_fn(np.array(2.5))
    negative_grad = grad_fn(np.array(-2.5))

    np.testing.assert_allclose(positive_grad, 5.0)
    np.testing.assert_allclose(negative_grad, -1.0)


def test_python_scalar_truth_is_define_by_run() -> None:
    def loss(x: float) -> float:
        if x:
            return x * x
        return x + 1.0

    grad_fn = ad.grad(loss)

    assert grad_fn(3.0) == 6.0
    assert grad_fn(0.0) == 1.0
