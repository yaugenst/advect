"""First- and higher-order qualification for the staged linalg closure."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import array_api_strict as strict
import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad

if TYPE_CHECKING:
    from collections.abc import Callable


_MATRIX = strict.asarray([[2.0, 0.3], [0.3, 1.4]], dtype=strict.float64)
_MATRIX_TANGENT = strict.asarray([[0.2, -0.1], [-0.1, 0.25]], dtype=strict.float64)
_RECTANGULAR = strict.asarray(
    [[1.4, 0.2], [0.3, 1.1], [0.5, -0.4]],
    dtype=strict.float64,
)
_RECTANGULAR_TANGENT = strict.asarray(
    [[0.1, -0.2], [0.3, 0.1], [-0.1, 0.25]],
    dtype=strict.float64,
)


def _cholesky_loss(value: Any) -> Any:
    namespace = value.__array_namespace__()
    return namespace.sum(namespace.linalg.cholesky(value))


def _det_loss(value: Any) -> Any:
    return value.__array_namespace__().linalg.det(value)


def _eigvalsh_loss(value: Any) -> Any:
    namespace = value.__array_namespace__()
    return namespace.sum(namespace.linalg.eigvalsh(value) ** 2)


def _inv_loss(value: Any) -> Any:
    namespace = value.__array_namespace__()
    inverse = namespace.linalg.inv(value)
    return namespace.sum(inverse * inverse)


def _pinv_loss(value: Any) -> Any:
    namespace = value.__array_namespace__()
    inverse = namespace.linalg.pinv(value)
    return namespace.sum(inverse * inverse)


def _svdvals_loss(value: Any) -> Any:
    namespace = value.__array_namespace__()
    values = namespace.linalg.svdvals(value)
    return namespace.sum(values * values)


def _matrix_norm_loss(value: Any) -> Any:
    return value.__array_namespace__().linalg.matrix_norm(value, ord="fro")


def _vector_norm_loss(value: Any) -> Any:
    namespace = value.__array_namespace__()
    return namespace.sum(namespace.linalg.vector_norm(value, axis=1))


def _qr_loss(value: Any) -> Any:
    namespace = value.__array_namespace__()
    q, r = namespace.linalg.qr(value, mode="reduced")
    q_weights = namespace.asarray(
        [[0.7, -0.2], [0.1, 0.4], [-0.3, 0.9]],
        dtype=value.dtype,
    )
    r_weights = namespace.asarray(
        [[0.5, -0.1], [0.2, 0.8]],
        dtype=value.dtype,
    )
    return namespace.sum(q * q_weights) + namespace.sum(r * r_weights)


@pytest.mark.parametrize(
    ("loss", "value", "tangent"),
    [
        (_cholesky_loss, _MATRIX, _MATRIX_TANGENT),
        (_det_loss, _MATRIX, _MATRIX_TANGENT),
        (_eigvalsh_loss, _MATRIX, _MATRIX_TANGENT),
        (_inv_loss, _MATRIX, _MATRIX_TANGENT),
        (_pinv_loss, _RECTANGULAR, _RECTANGULAR_TANGENT),
        (_svdvals_loss, _RECTANGULAR, _RECTANGULAR_TANGENT),
        (_matrix_norm_loss, _RECTANGULAR, _RECTANGULAR_TANGENT),
        (_vector_norm_loss, _RECTANGULAR, _RECTANGULAR_TANGENT),
        (_qr_loss, _RECTANGULAR, _RECTANGULAR_TANGENT),
    ],
    ids=[
        "cholesky",
        "det",
        "eigvalsh",
        "inv",
        "pinv",
        "svdvals",
        "matrix-norm",
        "vector-norm",
        "qr",
    ],
)
def test_linalg_gradients_match_directional_differences_and_staged_artifacts(
    loss: Callable[[Any], Any],
    value: Any,
    tangent: Any,
) -> None:
    dynamic = ad.grad(loss)(value)
    primal = ad.stage(
        loss,
        specs=(ad.ArraySpec(value.shape, value.dtype),),
    )
    staged_gradient = ad.grad(primal)
    restored_gradient = ad.StagedProgram.from_dict(staged_gradient.to_dict())

    epsilon = 1e-6
    finite_difference = (
        np.asarray(loss(value + epsilon * tangent)) - np.asarray(loss(value - epsilon * tangent))
    ) / (2 * epsilon)
    directional = np.real(np.vdot(np.asarray(dynamic), np.asarray(tangent)))

    assert_allclose(directional, finite_difference, rtol=2e-6, atol=2e-7)
    assert_allclose(np.asarray(staged_gradient(value)), np.asarray(dynamic), rtol=1e-10, atol=1e-10)
    assert_allclose(
        np.asarray(restored_gradient(value)),
        np.asarray(dynamic),
        rtol=1e-10,
        atol=1e-10,
    )


def test_complex_vecdot_gradient_uses_the_real_inner_product_convention() -> None:
    value = strict.asarray(
        [[1.0 + 0.5j, -0.2 + 0.7j], [0.3 - 0.4j, 1.2 + 0.1j]],
        dtype=strict.complex128,
    )
    weights = strict.asarray(
        [[0.5 - 0.25j, 0.8 + 0.2j], [-0.1 + 0.6j, 0.4 - 0.3j]],
        dtype=strict.complex128,
    )

    def loss(argument: Any) -> Any:
        namespace = argument.__array_namespace__()
        products = namespace.vecdot(argument, weights, axis=-1)
        return namespace.sum(namespace.real(products))

    dynamic = ad.grad(loss)(value)
    staged = ad.grad(ad.stage(loss, specs=(ad.ArraySpec(value.shape, value.dtype),)))(value)

    assert_allclose(np.asarray(dynamic), np.asarray(weights), rtol=1e-12, atol=1e-12)
    assert_allclose(np.asarray(staged), np.asarray(dynamic), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("loss", [_pinv_loss, _qr_loss], ids=["pinv", "qr"])
def test_traceable_linalg_pullbacks_support_higher_order_differentiation(
    loss: Callable[[Any], Any],
) -> None:
    _value, product = ad.hvp(loss)(_RECTANGULAR, vectors=_RECTANGULAR_TANGENT)
    epsilon = 1e-5
    expected = (
        ad.grad(loss)(_RECTANGULAR + epsilon * _RECTANGULAR_TANGENT)
        - ad.grad(loss)(_RECTANGULAR - epsilon * _RECTANGULAR_TANGENT)
    ) / (2 * epsilon)

    assert_allclose(np.asarray(product), np.asarray(expected), rtol=2e-6, atol=2e-7)
