"""Focused regressions for array-family derivative rules."""

from __future__ import annotations

import warnings
from typing import Any

import array_api_strict as strict
import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad


@pytest.mark.filterwarnings("ignore:`axes` should not be `None`:DeprecationWarning")
def test_array_api_irfftn_explicit_shape_uses_runtime_output_shape() -> None:
    """Explicit inverse lengths determine the Array API result shape."""
    value = strict.asarray([[1.0 + 0.0j, 0.4 - 0.2j, -0.3 + 0.0j]], dtype=strict.complex128)
    tangent = strict.asarray(
        [[0.2 + 0.1j, -0.5 + 0.3j, 0.7 - 0.4j]],
        dtype=strict.complex128,
    )
    cotangent = strict.asarray([[0.6, -0.4, 0.2, 0.9]], dtype=strict.float64)

    def function(x: Any) -> Any:
        return x.__array_namespace__().fft.irfftn(x, s=(4,), axes=None, norm="ortho")

    output, output_tangent = ad.jvp(function)(value, tangents=tangent)
    assert type(output) is type(value)
    assert output.shape == (1, 4)
    assert_allclose(
        np.asarray(output),
        np.fft.irfftn(np.asarray(value), s=(4,), axes=None, norm="ortho"),
        rtol=2e-9,
        atol=2e-10,
    )

    reverse_output, pullback = ad.vjp(function)(value)
    try:
        gradient = pullback(cotangent)
    finally:
        pullback.close()

    assert type(gradient) is type(value)
    assert_allclose(np.asarray(reverse_output), np.asarray(output), rtol=2e-9, atol=2e-10)
    assert_allclose(
        np.real(np.vdot(np.asarray(cotangent), np.asarray(output_tangent))),
        np.real(np.vdot(np.asarray(gradient), np.asarray(tangent))),
        rtol=2e-9,
        atol=2e-10,
    )


@pytest.mark.filterwarnings("ignore:`axes` should not be `None`:DeprecationWarning")
def test_array_api_fftn_padding_vjp_discards_padded_tail() -> None:
    """The adjoint of zero-padding truncates back to the input length."""
    value = strict.asarray([[1.0 + 0.0j, -2.0 + 0.0j]], dtype=strict.complex128)
    cotangent = strict.asarray([[0.4 + 0.1j, -0.2 + 0.3j, 0.7 - 0.5j]], dtype=strict.complex128)

    def function(x: Any) -> Any:
        return x.__array_namespace__().fft.fftn(x, s=(3,), axes=None, norm="ortho")

    output, pullback = ad.vjp(function)(value)
    try:
        gradient = pullback(cotangent)
    finally:
        pullback.close()

    assert type(output) is type(value)
    assert type(gradient) is type(value)
    assert output.shape == (1, 3)
    assert gradient.shape == value.shape

    expected_full = np.fft.ifftn(np.asarray(cotangent), axes=(-1,), norm="ortho")
    assert_allclose(np.asarray(gradient), expected_full[..., :2], rtol=2e-9, atol=2e-10)


@pytest.mark.filterwarnings("ignore:.*axes.*:DeprecationWarning")
def test_array_api_fftn_crop_vjp_zero_extends_to_input_shape() -> None:
    value = strict.asarray([[1, -2, 3, 4, -5]], dtype=strict.complex128)
    cotangent = strict.asarray([[0.4, -0.2, 0.7]], dtype=strict.complex128)

    def function(x: Any) -> Any:
        return x.__array_namespace__().fft.fftn(x, s=(3,), axes=None, norm="ortho")

    _, pullback = ad.vjp(function)(value)
    try:
        gradient = pullback(cotangent)
    finally:
        pullback.close()

    expected = np.fft.ifftn(np.asarray(cotangent), axes=(-1,), norm="ortho")
    assert_allclose(np.asarray(gradient)[..., :3], expected, rtol=2e-9, atol=2e-10)
    assert_allclose(np.asarray(gradient)[..., 3:], 0.0)


def test_advanced_indexing_hessian_reports_first_order_boundary() -> None:
    """Higher-order advanced-index pullbacks fail with the public boundary."""
    value = np.arange(12.0).reshape(3, 4)
    tangent = np.full_like(value, 0.25)
    index = np.array([2, 0])

    def gradient_function(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        _output, pullback = ad.vjp(lambda y: y[index, 1:3])(x)
        try:
            return pullback(np.ones((2, 2)))
        finally:
            pullback.close()

    with pytest.raises(
        NotImplementedError,
        match="Higher-order pullbacks for advanced indexing",
    ):
        ad.jvp(gradient_function)(value, tangents=tangent)


def test_empty_mean_vjp_reports_runtime_boundary() -> None:
    """Empty means fail explicitly instead of dividing a gradient by zero."""
    value = np.empty((0,), dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        output, pullback = ad.vjp(np.mean)(value)
    assert np.isnan(output)
    try:
        with pytest.raises(RuntimeError, match="empty reduction axis"):
            pullback(np.array(1.0))
    finally:
        pullback.close()


def test_empty_second_difference_vjp_returns_zero_source_cotangent() -> None:
    value = np.array([0.2, 1.3])
    output, pullback = ad.vjp(lambda x: np.diff(x, n=2))(value)
    try:
        gradient = pullback(np.empty(0))
    finally:
        pullback.close()

    assert output.shape == (0,)
    assert_allclose(gradient, np.zeros_like(value))


@pytest.mark.parametrize(
    ("function", "value", "match"),
    [
        pytest.param(
            lambda x: np.linalg.qr(x, mode="complete"),
            np.arange(15.0).reshape(5, 3) + np.eye(5, 3),
            "provider-dependent null-space columns",
            id="tall-complete-qr",
        ),
        pytest.param(
            lambda x: np.linalg.svd(x, hermitian=True),
            np.array([[3.0, 1.0], [1.0, 2.0]]),
            "hermitian=False",
            id="hermitian-svd",
        ),
        pytest.param(
            lambda x: np.linalg.svd(x, full_matrices=True),
            np.arange(15.0).reshape(5, 3) + np.eye(5, 3),
            "full_matrices=False",
            id="rectangular-full-svd",
        ),
    ],
)
def test_nonunique_linalg_derivatives_report_public_boundaries(
    function: Any,
    value: np.ndarray[Any, Any],
    match: str,
) -> None:
    """Non-unique provider choices are outside the public derivative contract."""
    tangent = np.linspace(-0.2, 0.4, value.size).reshape(value.shape)
    with pytest.raises(NotImplementedError, match=match):
        ad.jvp(function)(value, tangents=tangent)

    output, pullback = ad.vjp(function)(value)
    try:
        with pytest.raises(NotImplementedError, match=match):
            pullback(tuple(np.ones_like(leaf) for leaf in output))
    finally:
        pullback.close()


@pytest.mark.parametrize("function", [np.var, np.std], ids=["var", "std"])
def test_degenerate_reduction_ddof_reports_runtime_boundary(function: Any) -> None:
    """Undefined reduction denominators fail explicitly in reverse mode."""
    value = np.array([2.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        output, pullback = ad.vjp(lambda x: function(x, ddof=1))(value)
    assert np.isnan(output)
    try:
        with pytest.raises(NotImplementedError, match="count > ddof"):
            pullback(np.array(1.0))
    finally:
        pullback.close()
