"""Tests for public higher-order autodiff APIs."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad


def _cubic_loss(x: np.ndarray[Any, Any]) -> np.floating[Any]:
    return cast("np.floating[Any]", np.sum(x**3))


def _two_arg_loss(x: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> np.floating[Any]:
    return cast("np.floating[Any]", np.sum((x * y) ** 2))


def test_hvp_matches_expected_quadratic_form() -> None:
    x = np.array([0.5, -1.2, 2.0], dtype=np.float64)
    v = np.array([1.0, -0.25, 0.7], dtype=np.float64)

    value, hvp_val = ad.hvp(_cubic_loss)(x, vectors=v)

    assert value == pytest.approx(float(np.sum(x**3)), rel=1e-10, abs=1e-10)
    expected = 6.0 * x * v
    assert_allclose(hvp_val, expected, atol=2e-4, rtol=2e-4)


def test_hvp_evaluates_stateful_function_once() -> None:
    calls = 0

    def stateful_loss(x: np.ndarray[Any, Any]) -> np.floating[Any]:
        nonlocal calls
        calls += 1
        return cast("np.floating[Any]", calls * np.sum(x**3))

    x = np.array([0.5, -1.2, 2.0], dtype=np.float64)
    v = np.array([1.0, -0.25, 0.7], dtype=np.float64)

    value, product = ad.hvp(stateful_loss)(x, vectors=v)

    assert calls == 1
    assert value == pytest.approx(float(np.sum(x**3)), rel=1e-10, abs=1e-10)
    assert_allclose(product, 6.0 * x * v, atol=2e-4, rtol=2e-4)


def test_hessian_matches_expected_matrix() -> None:
    x = np.array([0.5, -1.2, 2.0], dtype=np.float64)

    hess = ad.hessian(_cubic_loss)(x)

    expected = np.diag(6.0 * x)
    assert_allclose(hess, expected, atol=2e-4, rtol=2e-4)


def test_hessian_diag_matches_expected_vector() -> None:
    x = np.array([0.5, -1.2, 2.0], dtype=np.float64)

    diag = ad.hessian_diag(_cubic_loss)(x)

    expected = 6.0 * x
    assert_allclose(diag, expected, atol=2e-4, rtol=2e-4)


def test_higher_order_exports_available_on_top_level() -> None:
    assert hasattr(ad, "hvp")
    assert hasattr(ad, "hessian")
    assert hasattr(ad, "hessian_diag")


def test_hvp_requires_vectors_keyword_argument() -> None:
    x = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(TypeError):
        _ = ad.hvp(_cubic_loss)(x)


def test_hessian_supports_multi_argnums_with_block_structure() -> None:
    x = np.array([1.0, 2.0], dtype=np.float64)
    y = np.array([3.0, 4.0], dtype=np.float64)

    hessian_blocks = ad.hessian(_two_arg_loss, argnums=(0, 1))(x, y)

    assert isinstance(hessian_blocks, tuple)
    assert len(hessian_blocks) == 2
    assert isinstance(hessian_blocks[0], tuple)
    assert isinstance(hessian_blocks[1], tuple)
    assert len(hessian_blocks[0]) == 2
    assert len(hessian_blocks[1]) == 2

    expected_xx = np.diag(2.0 * y * y)
    expected_xy = np.diag(4.0 * x * y)
    expected_yx = np.diag(4.0 * x * y)
    expected_yy = np.diag(2.0 * x * x)

    assert_allclose(hessian_blocks[0][0], expected_xx, atol=2e-4, rtol=2e-4)
    assert_allclose(hessian_blocks[0][1], expected_xy, atol=2e-4, rtol=2e-4)
    assert_allclose(hessian_blocks[1][0], expected_yx, atol=2e-4, rtol=2e-4)
    assert_allclose(hessian_blocks[1][1], expected_yy, atol=2e-4, rtol=2e-4)


def test_hessian_diag_supports_multi_argnums() -> None:
    x = np.array([1.0, 2.0], dtype=np.float64)
    y = np.array([3.0, 4.0], dtype=np.float64)

    hessian_diag = ad.hessian_diag(_two_arg_loss, argnums=(0, 1))(x, y)

    assert isinstance(hessian_diag, tuple)
    assert len(hessian_diag) == 2
    expected_x = 2.0 * y * y
    expected_y = 2.0 * x * x
    assert_allclose(hessian_diag[0], expected_x, atol=2e-4, rtol=2e-4)
    assert_allclose(hessian_diag[1], expected_y, atol=2e-4, rtol=2e-4)
