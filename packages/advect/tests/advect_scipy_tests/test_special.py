"""Special-function primitive qualification."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING, NamedTuple

import array_api_strict as strict
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from numpy.testing import assert_allclose
from scipy import special as scipy_special

import advect as ad
from advect.core._array_api import providers as array_api_providers
from advect.scipy import special

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class _UnaryCase(NamedTuple):
    name: str
    actual: Callable[[object], object]
    expected: Callable[[object], object]
    derivative: Callable[[np.ndarray], np.ndarray]
    sample: np.ndarray


_UNARY_CASES = (
    _UnaryCase(
        "gammaln",
        special.gammaln,
        scipy_special.gammaln,
        scipy_special.digamma,
        np.array([0.7, 1.4, 2.8]),
    ),
    _UnaryCase(
        "digamma",
        special.digamma,
        scipy_special.digamma,
        lambda x: scipy_special.polygamma(1, x),
        np.array([0.7, 1.4, 2.8]),
    ),
    _UnaryCase(
        "erf",
        special.erf,
        scipy_special.erf,
        lambda x: 2 / np.sqrt(np.pi) * np.exp(-(x * x)),
        np.array([-1.2, 0.3, 1.7]),
    ),
    _UnaryCase(
        "erfc",
        special.erfc,
        scipy_special.erfc,
        lambda x: -2 / np.sqrt(np.pi) * np.exp(-(x * x)),
        np.array([-1.2, 0.3, 1.7]),
    ),
    _UnaryCase(
        "erfcx",
        special.erfcx,
        scipy_special.erfcx,
        lambda x: 2 * x * scipy_special.erfcx(x) - 2 / np.sqrt(np.pi),
        np.array([-1.2, 0.3, 1.7]),
    ),
    _UnaryCase(
        "erfinv",
        special.erfinv,
        scipy_special.erfinv,
        lambda x: np.sqrt(np.pi) / 2 * np.exp(scipy_special.erfinv(x) ** 2),
        np.array([-0.8, 0.3, 0.7]),
    ),
    _UnaryCase(
        "expit",
        special.expit,
        scipy_special.expit,
        lambda x: scipy_special.expit(x) * (1 - scipy_special.expit(x)),
        np.array([-1.2, 0.3, 1.7]),
    ),
    _UnaryCase(
        "log_expit",
        special.log_expit,
        scipy_special.log_expit,
        lambda x: scipy_special.expit(-x),
        np.array([-1.2, 0.3, 1.7]),
    ),
    _UnaryCase(
        "ndtr",
        special.ndtr,
        scipy_special.ndtr,
        lambda x: np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi),
        np.array([-1.2, 0.3, 1.7]),
    ),
    _UnaryCase(
        "log_ndtr",
        special.log_ndtr,
        scipy_special.log_ndtr,
        lambda x: np.exp(-0.5 * x * x - 0.5 * np.log(2 * np.pi) - scipy_special.log_ndtr(x)),
        np.array([-1.2, 0.3, 1.7]),
    ),
    _UnaryCase(
        "ndtri",
        special.ndtri,
        scipy_special.ndtri,
        lambda x: np.sqrt(2 * np.pi) * np.exp(0.5 * scipy_special.ndtri(x) ** 2),
        np.array([0.1, 0.3, 0.8]),
    ),
)


@pytest.mark.parametrize("case", _UNARY_CASES, ids=lambda case: case.name)
def test_unary_special_input_is_positional_only_like_the_scipy_ufunc(
    case: _UnaryCase,
) -> None:
    with pytest.raises(TypeError):
        case.expected(x=case.sample)
    with pytest.raises(TypeError):
        case.actual(x=case.sample)


def _summed(function: Callable[[object], object]) -> Callable[[object], object]:
    def loss(x: object) -> object:
        return np.sum(function(x))

    return loss


@pytest.mark.parametrize("case", _UNARY_CASES, ids=lambda case: case.name)
def test_unary_special_primitives_match_scipy_and_differentiate(case: _UnaryCase) -> None:
    assert_allclose(case.actual(case.sample), case.expected(case.sample), rtol=1e-13)

    gradient = ad.grad(lambda x: np.sum(case.actual(x)))(case.sample)

    assert_allclose(gradient, case.derivative(case.sample), rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize(
    ("dtype", "sample"),
    [
        (np.float32, np.array([1e3, 1e4])),
        (np.float64, np.array([1e8])),
    ],
)
def test_erfcx_gradient_remains_stable_in_the_positive_tail(
    dtype: object,
    sample: np.ndarray,
) -> None:
    value = np.asarray(sample, dtype=dtype)
    _primal, derivative = ad.jvp(special.erfcx)(value, tangents=np.ones_like(value))
    reference_value = np.asarray(sample, dtype=np.float64)
    step = 1e-4 * reference_value
    expected = (
        scipy_special.erfcx(reference_value + step) - scipy_special.erfcx(reference_value - step)
    ) / (2 * step)

    assert_allclose(derivative, expected, rtol=3e-6, atol=0)


def test_erfcx_gradient_is_accurate_across_the_tail_crossover() -> None:
    sample = np.array([7.9999, 8.0001])
    _primal, derivative = ad.jvp(special.erfcx)(sample, tangents=np.ones_like(sample))
    expected = 2 * sample * scipy_special.erfcx(sample) - 2 / np.sqrt(np.pi)

    assert_allclose(derivative, expected, rtol=2e-13, atol=2e-15)


@pytest.mark.parametrize(
    ("dtype", "sample"),
    [
        (np.float32, np.array([-1e4, -1e6])),
        (np.float64, np.array([-1e8])),
    ],
)
def test_log_ndtr_gradient_remains_stable_in_the_negative_tail(
    dtype: object,
    sample: np.ndarray,
) -> None:
    value = np.asarray(sample, dtype=dtype)
    _primal, derivative = ad.jvp(special.log_ndtr)(value, tangents=np.ones_like(value))
    reference_value = np.asarray(sample, dtype=np.float64)
    expected = np.sqrt(2 / np.pi) / scipy_special.erfcx(-reference_value / np.sqrt(2))

    assert_allclose(derivative, expected, rtol=3e-6, atol=0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_every_special_function_matches_value_and_gradient_at_supported_precision(
    dtype: object,
) -> None:
    common_sample = np.array([0.7, 1.4, 2.8], dtype=dtype)
    cases = [
        (
            case.name,
            case.actual,
            case.expected,
            case.derivative,
            np.asarray(case.sample, dtype=dtype),
        )
        for case in _UNARY_CASES
    ]
    cases.extend(
        (
            (
                "polygamma",
                lambda x: special.polygamma(2, x),
                lambda x: scipy_special.polygamma(2, x),
                lambda x: scipy_special.polygamma(3, x),
                common_sample,
            ),
            (
                "logsumexp",
                special.logsumexp,
                scipy_special.logsumexp,
                lambda x: np.exp(x - scipy_special.logsumexp(x)),
                common_sample,
            ),
        )
    )
    tolerance = 2e-5 if np.dtype(dtype) == np.dtype(np.float32) else 2e-12

    for name, actual, expected, derivative, sample in cases:
        value = actual(sample)
        expected_value = expected(sample)
        gradient = ad.grad(_summed(actual))(sample)

        assert np.asarray(value).dtype == np.asarray(expected_value).dtype, name
        assert_allclose(value, expected_value, rtol=tolerance, atol=tolerance, err_msg=name)
        assert_allclose(
            gradient,
            derivative(sample),
            rtol=tolerance,
            atol=tolerance,
            err_msg=name,
        )


def test_polygamma_broadcasts_array_orders_and_has_traceable_higher_derivatives() -> None:
    sample = np.array([0.7, 1.4, 2.8])
    orders = np.array([[0], [1], [2]])
    broadcast_sample = sample[None, :]

    value = special.polygamma(2, sample)
    gradient = ad.grad(lambda x: np.sum(special.polygamma(2, x)))(sample)
    broadcast_value = special.polygamma(orders, broadcast_sample)
    broadcast_gradient = ad.grad(lambda x: np.sum(special.polygamma(orders, x)))(broadcast_sample)
    second_gammaln = ad.grad(lambda x: np.sum(ad.grad(lambda y: np.sum(special.gammaln(y)))(x)))(
        sample
    )

    assert_allclose(value, scipy_special.polygamma(2, sample))
    assert_allclose(gradient, scipy_special.polygamma(3, sample))
    assert_allclose(
        broadcast_value,
        scipy_special.polygamma(orders, broadcast_sample),
    )
    assert_allclose(
        broadcast_gradient,
        np.sum(scipy_special.polygamma(orders + 1, broadcast_sample), axis=0, keepdims=True),
    )
    assert_allclose(second_gammaln, scipy_special.polygamma(1, sample))


def test_logsumexp_value_jvp_and_gradient_preserve_reduction_shape() -> None:
    sample = np.array([[-3.0, 0.4, 1.7], [2.0, -0.2, 0.8]])
    tangent = np.array([[0.2, -0.3, 0.1], [0.5, 0.7, -0.2]])
    expected_value = scipy_special.logsumexp(sample, axis=1, keepdims=True)
    weights = np.exp(sample - expected_value)
    expected_jvp = np.sum(weights * tangent, axis=1, keepdims=True)

    value, directional = ad.jvp(lambda x: special.logsumexp(x, axis=1, keepdims=True))(
        sample, tangents=tangent
    )
    gradient = ad.grad(lambda x: np.sum(special.logsumexp(x, axis=1, keepdims=True)))(sample)

    assert value.shape == (2, 1)
    assert directional.shape == (2, 1)
    assert_allclose(value, expected_value)
    assert_allclose(directional, expected_jvp)
    assert_allclose(gradient, weights)


@pytest.mark.parametrize(
    ("axis", "keepdims"),
    [(None, True), (0, True), (-1, True), ((), False), ((), True)],
)
def test_scalar_logsumexp_staging_matches_scipy_shape(
    axis: object,
    keepdims: object,
) -> None:
    assert isinstance(keepdims, bool)
    sample = np.array(2.0)
    program = ad.stage(
        lambda x: special.logsumexp(x, axis=axis, keepdims=keepdims),
        specs=(ad.ArraySpec((), sample.dtype),),
    )

    actual = program(sample)
    expected = scipy_special.logsumexp(sample, axis=axis, keepdims=keepdims)

    assert np.shape(actual) == np.shape(expected)
    assert_allclose(actual, expected)


def test_logsumexp_has_a_traceable_second_derivative() -> None:
    sample = np.array([-1.0, 0.4, 1.7])
    weights = np.exp(sample - scipy_special.logsumexp(sample))
    expected = np.diag(weights) - np.outer(weights, weights)

    actual = ad.hessian(
        lambda x: special.logsumexp(x)  # noqa: PLW0108 - Hessian boundary
    )(sample)

    assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("axis", [None, 0, 1, -1, (0, 1), ()])
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (special.softmax, scipy_special.softmax),
        (special.log_softmax, scipy_special.log_softmax),
    ],
)
def test_softmax_primitives_match_scipy_stage_serialize_and_differentiate(
    axis: object,
    actual: Callable[[object, object], object],
    expected: Callable[[object, object], object],
) -> None:
    sample = np.array([[-3.0, 0.4, 1.7], [2.0, -0.2, 0.8]])
    tangent = np.array([[0.3, -0.1, 0.5], [-0.2, 0.4, 0.1]])
    weights = np.array([[0.2, -0.3, 0.1], [0.5, 0.7, -0.2]])

    def function(x: object) -> object:
        return actual(x, axis)

    def loss(x: object) -> object:
        return np.sum(function(x) * weights)

    value, directional = ad.jvp(function)(sample, tangents=tangent)
    step = 1e-6
    expected_directional = (
        expected(sample + step * tangent, axis=axis) - expected(sample - step * tangent, axis=axis)
    ) / (2 * step)
    gradient = ad.grad(loss)(sample)
    probabilities = scipy_special.softmax(sample, axis=axis)
    if actual is special.softmax:
        expected_gradient = probabilities * (
            weights - np.sum(weights * probabilities, axis=axis, keepdims=True)
        )
    else:
        expected_gradient = weights - probabilities * np.sum(weights, axis=axis, keepdims=True)
    program = ad.stage(function, specs=(ad.ArraySpec(sample.shape, sample.dtype),))
    restored = ad.StagedProgram.from_dict(program.to_dict())
    gradient_program = ad.grad(ad.stage(loss, specs=(ad.ArraySpec(sample.shape, sample.dtype),)))
    restored_gradient = ad.StagedProgram.from_dict(gradient_program.to_dict())

    assert_allclose(value, expected(sample, axis=axis))
    assert_allclose(directional, expected_directional, rtol=2e-9, atol=2e-9)
    assert_allclose(
        np.vdot(directional, weights),
        np.vdot(tangent, gradient),
        rtol=2e-12,
        atol=2e-12,
    )
    assert_allclose(gradient, expected_gradient, rtol=2e-12, atol=2e-12)
    assert_allclose(program(sample), value)
    assert_allclose(restored(sample), value)
    assert_allclose(gradient_program(sample), expected_gradient, rtol=2e-12, atol=2e-12)
    assert_allclose(restored_gradient(sample), expected_gradient, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize(
    "sample",
    [
        np.array([np.inf, 0.0]),
        np.array([np.inf, np.inf]),
        np.array([-np.inf, -np.inf]),
        np.array([np.nan, 0.0]),
    ],
)
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (special.softmax, scipy_special.softmax),
        (special.log_softmax, scipy_special.log_softmax),
    ],
)
def test_softmax_primitives_preserve_scipy_nonfinite_values_when_staged(
    sample: np.ndarray,
    actual: Callable[[object, object], object],
    expected: Callable[[object, object], object],
) -> None:
    program = ad.stage(
        lambda x: actual(x, 0),
        specs=(ad.ArraySpec(sample.shape, sample.dtype),),
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        expected_value = expected(sample, axis=0)
        actual_value = program(sample)

    assert_allclose(actual_value, expected_value, equal_nan=True)


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int64, np.float16, np.complex64])
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (special.softmax, scipy_special.softmax),
        (special.log_softmax, scipy_special.log_softmax),
    ],
)
def test_softmax_primitives_preserve_scipy_dtype_families_when_staged(
    dtype: object,
    actual: Callable[[object, object], object],
    expected: Callable[[object, object], object],
) -> None:
    sample = np.asarray([[1, -2, 3], [0, 4, -1]], dtype=dtype)

    def function(value: object) -> object:
        return actual(value, axis=1)

    expected_value = expected(sample, axis=1)
    program = ad.stage(function, specs=(ad.ArraySpec(sample.shape, sample.dtype),))
    restored = ad.StagedProgram.from_dict(program.to_dict())

    assert np.asarray(program(sample)).dtype == np.asarray(expected_value).dtype
    assert_allclose(program(sample), expected_value, rtol=0, atol=0)
    assert_allclose(restored(sample), expected_value, rtol=0, atol=0)


@pytest.mark.parametrize("actual", [special.softmax, special.log_softmax])
def test_softmax_primitives_preserve_scipy_boolean_rejection(
    actual: Callable[[object, object], object],
) -> None:
    sample = np.array([[True, False, True]])

    with pytest.raises(TypeError, match="boolean subtract"):
        ad.stage(
            lambda value: actual(value, axis=1),
            specs=(ad.ArraySpec(sample.shape, sample.dtype),),
        )


def test_complex_erf_uses_advects_real_adjoint_convention() -> None:
    sample = np.array([0.4 + 0.3j, -0.2 + 0.7j])
    coefficient = 2 / np.sqrt(np.pi) * np.exp(-(sample * sample))

    gradient = ad.grad(lambda z: np.sum(np.real(special.erf(z))))(sample)

    assert_allclose(gradient, np.conj(coefficient), rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize(
    ("function", "coefficient"),
    [
        (
            special.erfc,
            lambda x: -2 / np.sqrt(np.pi) * np.exp(-(x * x)),
        ),
        (
            special.erfcx,
            lambda x: 2 * x * scipy_special.erfcx(x) - 2 / np.sqrt(np.pi),
        ),
        (
            special.log_ndtr,
            lambda x: np.exp(-0.5 * x * x - 0.5 * np.log(2 * np.pi) - scipy_special.log_ndtr(x)),
        ),
    ],
)
def test_new_complex_unary_functions_use_advects_real_adjoint_convention(
    function: Callable[[object], object],
    coefficient: Callable[[np.ndarray], np.ndarray],
) -> None:
    sample = np.array([0.4 + 0.3j, -0.2 + 0.7j])

    gradient = ad.grad(lambda z: np.sum(np.real(function(z))))(sample)

    assert_allclose(gradient, np.conj(coefficient(sample)), rtol=2e-12, atol=2e-12)


def test_complex_digamma_matches_scipy_and_uses_advects_real_adjoint_convention() -> None:
    sample = np.array([0.4 + 0.3j, 1.2 - 0.7j])
    tangent = np.array([0.2 - 0.4j, -0.3 + 0.7j])
    step = 1e-6
    expected_derivative = (
        scipy_special.digamma(sample + step) - scipy_special.digamma(sample - step)
    ) / (2 * step)

    value, directional = ad.jvp(special.digamma)(sample, tangents=tangent)
    gradient = ad.grad(lambda z: np.sum(np.real(special.digamma(z))))(sample)
    program = ad.stage(
        special.digamma,
        specs=(ad.ArraySpec(sample.shape, sample.dtype),),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())
    loss_program = ad.stage(
        lambda z: np.sum(np.real(special.digamma(z))),
        specs=(ad.ArraySpec(sample.shape, sample.dtype),),
    )
    gradient_program = ad.grad(loss_program)
    restored_gradient = ad.StagedProgram.from_dict(gradient_program.to_dict())

    assert_allclose(value, scipy_special.digamma(sample), rtol=2e-13, atol=2e-13)
    assert_allclose(directional, expected_derivative * tangent, rtol=2e-9, atol=2e-9)
    assert_allclose(gradient, np.conj(expected_derivative), rtol=2e-9, atol=2e-9)
    assert_allclose(program(sample), value)
    assert_allclose(restored(sample), value)
    assert_allclose(gradient_program(sample), gradient, rtol=2e-9, atol=2e-9)
    assert_allclose(restored_gradient(sample), gradient, rtol=2e-9, atol=2e-9)


@pytest.mark.parametrize("case", _UNARY_CASES, ids=lambda case: case.name)
def test_unary_ufunc_kwargs_out_and_where_match_scipy_and_differentiate(
    case: _UnaryCase,
) -> None:
    sample = case.sample
    mask = np.arange(sample.size) % 2 == 0

    def update(x: object) -> object:
        destination = (3 * x).copy()
        result = case.actual(
            x,
            out=(destination,),
            where=mask,
            casting="unsafe",
            order="C",
            dtype=np.float64,
            subok=False,
        )
        assert result is destination
        return result

    expected = 3 * sample
    case.expected(
        sample,
        out=expected,
        where=mask,
        casting="unsafe",
        order="C",
        dtype=np.float64,
        subok=False,
    )
    expected_derivative = np.where(mask, case.derivative(sample), 3)

    value, directional = ad.jvp(update)(sample, tangents=np.ones_like(sample))
    gradient = ad.grad(lambda x: np.sum(update(x)))(sample)
    program = ad.stage(
        update,
        specs=(ad.ArraySpec(sample.shape, sample.dtype),),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())
    loss_program = ad.stage(
        lambda x: np.sum(update(x)),
        specs=(ad.ArraySpec(sample.shape, sample.dtype),),
    )
    gradient_program = ad.grad(loss_program)
    restored_gradient = ad.StagedProgram.from_dict(gradient_program.to_dict())

    assert_allclose(value, expected)
    assert_allclose(directional, expected_derivative, rtol=2e-12, atol=2e-12)
    assert_allclose(gradient, expected_derivative, rtol=2e-12, atol=2e-12)
    assert_allclose(program(sample), expected)
    assert_allclose(restored(sample), expected)
    assert_allclose(
        gradient_program(sample),
        expected_derivative,
        rtol=2e-12,
        atol=2e-12,
    )
    assert_allclose(
        restored_gradient(sample),
        expected_derivative,
        rtol=2e-12,
        atol=2e-12,
    )


def test_unary_ufunc_out_can_expand_the_broadcast_shape_when_staged() -> None:
    sample = np.array([-0.7, 0.2, 1.3])
    tangent = np.array([0.3, -0.5, 0.1])

    def function(x: object) -> object:
        destination = np.stack((x, x), axis=0).copy()
        return special.erfc(x, out=destination)

    value, directional = ad.jvp(function)(sample, tangents=tangent)
    expected = np.broadcast_to(scipy_special.erfc(sample), (2, sample.size))
    derivative = -2 / np.sqrt(np.pi) * np.exp(-(sample * sample))
    expected_directional = np.broadcast_to(derivative * tangent, expected.shape)
    gradient = ad.grad(lambda x: np.sum(function(x)))(sample)
    program = ad.stage(function, specs=(ad.ArraySpec(sample.shape, sample.dtype),))
    restored = ad.StagedProgram.from_dict(program.to_dict())

    assert_allclose(value, expected, rtol=2e-13, atol=2e-13)
    assert_allclose(directional, expected_directional, rtol=2e-12, atol=2e-12)
    assert_allclose(gradient, 2 * derivative, rtol=2e-12, atol=2e-12)
    assert_allclose(restored(sample), expected, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("case", _UNARY_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("keyword", ["signature", "sig"])
def test_unary_ufunc_signature_aliases_stage_at_requested_precision(
    case: _UnaryCase,
    keyword: str,
) -> None:
    sample = np.asarray(case.sample, dtype=np.float32)
    signature: object = b"f->f" if keyword == "signature" else (np.float32, np.float32)

    def function(x: object) -> object:
        return case.actual(x, **{keyword: signature})

    expected = case.expected(sample, **{keyword: signature})
    value, tangent = ad.jvp(function)(sample, tangents=np.ones_like(sample))
    program = ad.stage(
        function,
        specs=(ad.ArraySpec(sample.shape, sample.dtype),),
    )

    assert np.asarray(value).dtype == np.float32
    assert_allclose(value, expected, rtol=2e-6, atol=2e-6)
    assert_allclose(tangent, case.derivative(sample), rtol=2e-5, atol=2e-5)
    assert_allclose(program(sample), expected, rtol=2e-6, atol=2e-6)


def test_unary_ufunc_partial_signature_tuple_preserves_unspecified_input() -> None:
    sample = np.array([0.2, 0.7], dtype=np.float32)
    signature = (None, np.float64)

    def function(x: object) -> object:
        return special.erf(
            x,
            casting=b"unsafe",
            order=None,
            signature=signature,
        )

    expected = scipy_special.erf(
        sample,
        casting=b"unsafe",
        order=None,
        signature=signature,
    )
    program = ad.stage(
        function,
        specs=(ad.ArraySpec(sample.shape, sample.dtype),),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())

    assert program(sample).dtype == np.float64
    assert_allclose(program(sample), expected)
    assert_allclose(restored(sample), expected)


@pytest.mark.filterwarnings("ignore:Casting complex values to real discards the imaginary part")
@pytest.mark.parametrize(
    ("keyword", "selection"),
    [("dtype", np.float64), ("signature", "d->d")],
)
def test_unary_ufunc_loop_casts_are_part_of_the_differentiated_program(
    keyword: str,
    selection: object,
) -> None:
    sample = np.array([0.2 + 0.3j, -0.7 + 0.4j])
    tangent = np.array([0.4 - 0.7j, -0.2 + 0.5j])

    def function(x: object) -> object:
        return special.erf(
            x,
            casting="unsafe",
            **{keyword: selection},
        )

    def loss(x: object) -> object:
        return np.sum(function(x))

    expected_value = scipy_special.erf(
        sample,
        casting="unsafe",
        **{keyword: selection},
    )
    coefficient = 2 / np.sqrt(np.pi) * np.exp(-(np.real(sample) ** 2))
    expected_tangent = coefficient * np.real(tangent)
    expected_gradient = np.astype(coefficient, np.complex128)

    value, directional = ad.jvp(function)(sample, tangents=tangent)
    gradient = ad.grad(loss)(sample)
    program = ad.stage(
        function,
        specs=(ad.ArraySpec(sample.shape, sample.dtype),),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())
    gradient_program = ad.grad(
        ad.stage(
            loss,
            specs=(ad.ArraySpec(sample.shape, sample.dtype),),
        )
    )
    restored_gradient = ad.StagedProgram.from_dict(gradient_program.to_dict())

    assert np.asarray(value).dtype == np.float64
    assert np.asarray(directional).dtype == np.float64
    assert_allclose(value, expected_value)
    assert_allclose(directional, expected_tangent)
    assert_allclose(gradient, expected_gradient)
    assert_allclose(program(sample), expected_value)
    assert_allclose(restored(sample), expected_value)
    assert_allclose(gradient_program(sample), expected_gradient)
    assert_allclose(restored_gradient(sample), expected_gradient)


def test_unary_ufunc_integer_out_has_zero_active_derivative() -> None:
    sample = np.array([0.2, 1.2, -0.5])

    def update(x: object) -> object:
        destination = np.zeros_like(x, dtype=np.int64)
        return special.erf(
            x,
            out=destination,
            casting="unsafe",
        )

    def loss(x: object) -> object:
        return np.sum(update(x) * 1.0)

    expected_value = np.zeros_like(sample, dtype=np.int64)
    scipy_special.erf(
        sample,
        out=expected_value,
        casting="unsafe",
    )
    expected_derivative = np.zeros_like(sample)

    value, directional = ad.jvp(update)(sample, tangents=np.ones_like(sample))
    gradient = ad.grad(loss)(sample)
    program = ad.stage(
        update,
        specs=(ad.ArraySpec(sample.shape, sample.dtype),),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())
    gradient_program = ad.grad(
        ad.stage(
            loss,
            specs=(ad.ArraySpec(sample.shape, sample.dtype),),
        )
    )
    restored_gradient = ad.StagedProgram.from_dict(gradient_program.to_dict())

    assert_allclose(value, expected_value)
    assert_allclose(directional, expected_derivative)
    assert_allclose(gradient, expected_derivative)
    assert_allclose(program(sample), expected_value)
    assert_allclose(restored(sample), expected_value)
    assert_allclose(gradient_program(sample), expected_derivative)
    assert_allclose(restored_gradient(sample), expected_derivative)


def test_where_without_out_preserves_defined_values_and_masks_the_jvp() -> None:
    sample = np.array([-0.7, 0.2, 1.1, 2.0])
    tangent = np.array([0.2, -0.4, 0.7, -0.1])
    mask = np.array([True, False, True, False])

    value, directional = ad.jvp(lambda x: special.erf(x, where=mask))(
        sample,
        tangents=tangent,
    )
    program = ad.stage(
        lambda x: special.erf(x, where=mask),
        specs=(ad.ArraySpec(sample.shape, sample.dtype),),
    )

    expected = scipy_special.erf(sample)
    expected_directional = 2 / np.sqrt(np.pi) * np.exp(-(sample * sample)) * tangent
    assert_allclose(value[mask], expected[mask])
    assert_allclose(directional[mask], expected_directional[mask])
    assert_allclose(directional[~mask], 0)
    assert_allclose(program(sample)[mask], expected[mask])


def test_live_where_mask_is_nondifferentiable_and_staged() -> None:
    sample = np.array([-0.7, 0.2, 1.1, 2.0])
    mask = np.array([True, False, False, True])

    def function(x: object, where: object) -> object:
        destination = (2 * x).copy()
        return special.erf(x, out=destination, where=where)

    expected = 2 * sample
    scipy_special.erf(sample, out=expected, where=mask)
    expected_gradient = np.where(
        mask,
        2 / np.sqrt(np.pi) * np.exp(-(sample * sample)),
        2,
    )

    value, directional = ad.jvp(function, argnums=0)(
        sample,
        mask,
        tangents=np.ones_like(sample),
    )
    loss_program = ad.stage(
        lambda x, where: np.sum(function(x, where)),
        specs=(
            ad.ArraySpec(sample.shape, sample.dtype),
            ad.ArraySpec(mask.shape, mask.dtype),
        ),
    )
    gradient_program = ad.grad(loss_program, argnums=0)
    restored_gradient = ad.StagedProgram.from_dict(gradient_program.to_dict())

    assert_allclose(value, expected)
    assert_allclose(directional, expected_gradient)
    assert_allclose(gradient_program(sample, mask), expected_gradient)
    assert_allclose(restored_gradient(sample, mask), expected_gradient)


def test_polygamma_array_orders_stage_serialize_and_differentiate_x() -> None:
    orders = np.array([[0], [1], [3]], dtype=np.int64)
    sample = np.array([[0.7, 1.4, 2.8, 4.1]])
    program = ad.stage(
        lambda n, x: special.polygamma(n, x),  # noqa: PLW0108 - trace boundary
        specs=(
            ad.ArraySpec(orders.shape, orders.dtype),
            ad.ArraySpec(sample.shape, sample.dtype),
        ),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())
    loss_program = ad.stage(
        lambda n, x: np.sum(special.polygamma(n, x)),
        specs=(
            ad.ArraySpec(orders.shape, orders.dtype),
            ad.ArraySpec(sample.shape, sample.dtype),
        ),
    )
    gradient_program = ad.grad(loss_program, argnums=1)
    restored_gradient = ad.StagedProgram.from_dict(gradient_program.to_dict())
    expected = scipy_special.polygamma(orders, sample)
    expected_gradient = np.sum(
        scipy_special.polygamma(orders + 1, sample),
        axis=0,
        keepdims=True,
    )

    assert_allclose(program(orders, sample), expected)
    assert_allclose(restored(orders, sample), expected)
    assert_allclose(
        ad.grad(lambda x: np.sum(special.polygamma(orders, x)))(sample),
        expected_gradient,
    )
    assert_allclose(gradient_program(orders, sample), expected_gradient)
    assert_allclose(restored_gradient(orders, sample), expected_gradient)


@settings(max_examples=30)
@given(
    values=st.lists(
        st.floats(
            min_value=-4,
            max_value=4,
            allow_nan=False,
            allow_infinity=False,
            width=64,
        ),
        min_size=1,
        max_size=8,
    ),
    weights=st.lists(
        st.floats(
            min_value=0.2,
            max_value=2,
            allow_nan=False,
            allow_infinity=False,
            width=64,
        ),
        min_size=1,
        max_size=8,
    ),
)
def test_weighted_logsumexp_jvp_matches_directional_finite_differences(
    values: list[float],
    weights: list[float],
) -> None:
    size = min(len(values), len(weights))
    a = np.asarray(values[:size])
    b = np.asarray(weights[:size])
    a_tangent = np.linspace(-0.4, 0.6, size)
    b_tangent = np.linspace(0.5, -0.3, size)
    step = 1e-5

    def function(aa: object, bb: object) -> object:
        return special.logsumexp(aa, b=bb)

    value, directional = ad.jvp(function, argnums=(0, 1))(
        a,
        b,
        tangents=(a_tangent, b_tangent),
    )
    expected_directional = (
        scipy_special.logsumexp(
            a + step * a_tangent,
            b=b + step * b_tangent,
        )
        - scipy_special.logsumexp(
            a - step * a_tangent,
            b=b - step * b_tangent,
        )
    ) / (2 * step)

    assert_allclose(value, scipy_special.logsumexp(a, b=b), rtol=2e-13, atol=2e-13)
    assert_allclose(directional, expected_directional, rtol=2e-8, atol=2e-8)


def test_logsumexp_broadcast_weights_have_unbroadcasted_gradients() -> None:
    a = np.array([[-2.0, 0.4, 1.7], [2.0, -0.2, 0.8]])
    b = np.array([0.5, 1.5, 2.0])

    def loss(aa: object, bb: object) -> object:
        return np.sum(special.logsumexp(aa, axis=1, b=bb))

    gradient_a, gradient_b = ad.grad(loss, argnums=(0, 1))(a, b)
    denominator = np.sum(b * np.exp(a), axis=1, keepdims=True)
    expected_a = b * np.exp(a) / denominator
    expected_b = np.sum(np.exp(a) / denominator, axis=0)
    program = ad.stage(
        loss,
        specs=(
            ad.ArraySpec(a.shape, a.dtype),
            ad.ArraySpec(b.shape, b.dtype),
        ),
    )
    gradient_program = ad.grad(program, argnums=(0, 1))

    assert_allclose(gradient_a, expected_a)
    assert_allclose(gradient_b, expected_b)
    staged_a, staged_b = gradient_program(a, b)
    assert_allclose(staged_a, expected_a)
    assert_allclose(staged_b, expected_b)


def test_signed_complex_logsumexp_differentiates_both_outputs_and_serializes() -> None:
    a = np.array([[-3.0 + 0.2j, 0.4 - 0.1j, 1.7 + 0.3j], [2.0 - 0.4j, -0.2 + 0.5j, 0.8 - 0.2j]])
    b = np.array([[1.0 + 0.1j, -2.0 + 0.2j, 0.5 - 0.3j], [0.4 + 0.2j, 2.0 - 0.1j, -1.0 + 0.4j]])
    a_tangent = np.array(
        [[0.2 - 0.1j, -0.4 + 0.3j, 0.1 + 0.2j], [0.3 + 0.2j, -0.2 - 0.4j, 0.5 - 0.1j]]
    )
    b_tangent = np.array(
        [[-0.1 + 0.2j, 0.4 - 0.3j, 0.2 + 0.1j], [0.3 - 0.1j, 0.2 + 0.4j, -0.5 + 0.2j]]
    )
    step = 1e-6

    def function(aa: object, bb: object) -> object:
        return special.logsumexp(
            aa,
            axis=1,
            b=bb,
            keepdims=True,
            return_sign=True,
        )

    value, directional = ad.jvp(function, argnums=(0, 1))(
        a,
        b,
        tangents=(a_tangent, b_tangent),
    )
    reduced_cotangent = np.array([[0.7], [-0.4]])
    sign_cotangent = np.array([[0.2 - 0.5j], [-0.3 + 0.6j]])
    _vjp_value, pullback = ad.vjp(function, argnums=(0, 1))(a, b)
    a_cotangent, b_cotangent = pullback((reduced_cotangent, sign_cotangent))
    positive = scipy_special.logsumexp(
        a + step * a_tangent,
        axis=1,
        b=b + step * b_tangent,
        keepdims=True,
        return_sign=True,
    )
    negative = scipy_special.logsumexp(
        a - step * a_tangent,
        axis=1,
        b=b - step * b_tangent,
        keepdims=True,
        return_sign=True,
    )
    expected_directional = tuple(
        (positive_part - negative_part) / (2 * step)
        for positive_part, negative_part in zip(positive, negative, strict=True)
    )
    program = ad.stage(
        function,
        specs=(
            ad.ArraySpec(a.shape, a.dtype),
            ad.ArraySpec(b.shape, b.dtype),
        ),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())

    def loss(aa: object, bb: object) -> object:
        return np.sum(
            special.logsumexp(
                aa,
                axis=1,
                b=bb,
                keepdims=True,
                return_sign=True,
            )[0]
        )

    expected_gradient = ad.grad(loss, argnums=(0, 1))(a, b)
    loss_program = ad.stage(
        loss,
        specs=(
            ad.ArraySpec(a.shape, a.dtype),
            ad.ArraySpec(b.shape, b.dtype),
        ),
    )
    gradient_program = ad.grad(loss_program, argnums=(0, 1))
    restored_gradient = ad.StagedProgram.from_dict(gradient_program.to_dict())

    for actual, expected in zip(
        value,
        scipy_special.logsumexp(
            a,
            axis=1,
            b=b,
            keepdims=True,
            return_sign=True,
        ),
        strict=True,
    ):
        assert_allclose(actual, expected)
    for actual, expected in zip(directional, expected_directional, strict=True):
        assert_allclose(actual, expected, rtol=2e-8, atol=2e-8)
    output_inner_product = np.real(
        np.vdot(reduced_cotangent, directional[0]) + np.vdot(sign_cotangent, directional[1])
    )
    input_inner_product = np.real(np.vdot(a_cotangent, a_tangent) + np.vdot(b_cotangent, b_tangent))
    assert_allclose(output_inner_product, input_inner_product, rtol=2e-12, atol=2e-12)
    for actual, expected in zip(program(a, b), value, strict=True):
        assert_allclose(actual, expected)
    for actual, expected in zip(restored(a, b), value, strict=True):
        assert_allclose(actual, expected)
    for actual, expected in zip(gradient_program(a, b), expected_gradient, strict=True):
        assert_allclose(actual, expected)
    for actual, expected in zip(restored_gradient(a, b), expected_gradient, strict=True):
        assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("sample", "axis", "b", "keepdims", "return_sign"),
    [
        (np.empty((0, 3)), 0, None, False, False),
        (np.array([1.0, 2.0]), None, np.zeros(2), True, True),
        (np.array([[1.0, -np.inf], [np.inf, 2.0]]), 1, None, False, True),
        (np.array([1.0, 2.0]), (), np.array([-1.0, 2.0]), False, True),
    ],
)
def test_logsumexp_edge_contract_stages_like_scipy(
    sample: np.ndarray,
    axis: object,
    b: object,
    keepdims: object,
    return_sign: object,
) -> None:
    assert isinstance(keepdims, bool)
    assert isinstance(return_sign, bool)

    def function(x: object) -> object:
        return special.logsumexp(
            x,
            axis=axis,
            b=b,
            keepdims=keepdims,
            return_sign=return_sign,
        )

    expected = scipy_special.logsumexp(
        sample,
        axis=axis,
        b=b,
        keepdims=keepdims,
        return_sign=return_sign,
    )
    program = ad.stage(
        function,
        specs=(ad.ArraySpec(sample.shape, sample.dtype),),
    )
    actual = program(sample)

    if return_sign:
        for actual_part, expected_part in zip(actual, expected, strict=True):
            assert_allclose(actual_part, expected_part, equal_nan=True)
    else:
        assert_allclose(actual, expected, equal_nan=True)


def test_special_primitives_stage_differentiate_and_serialize() -> None:
    sample = np.array([[0.7, 1.4, 2.8], [1.1, 2.2, 3.3]])

    def loss(x: object) -> object:
        probability = x / (1 + x)
        terms = (
            special.gammaln(x)
            + special.digamma(x)
            + special.polygamma(1, x)
            + special.erf(x)
            + special.erfc(x)
            + special.erfcx(x)
            + special.erfinv(probability)
            + special.expit(x)
            + special.log_expit(x)
            + special.ndtr(x)
            + special.log_ndtr(x)
            + special.ndtri(probability)
        )
        normalized = special.softmax(terms, axis=1) + special.log_softmax(terms, axis=1)
        return np.sum(special.logsumexp(normalized, axis=1, keepdims=True))

    program = ad.stage(loss, specs=(ad.ArraySpec(sample.shape, sample.dtype),))
    gradient_program = ad.grad(program)
    restored_program = ad.StagedProgram.from_dict(program.to_dict())
    restored_gradient = ad.StagedProgram.from_dict(gradient_program.to_dict())

    expected_value = loss(sample)
    expected_gradient = ad.grad(loss)(sample)
    assert_allclose(program(sample), expected_value)
    assert_allclose(restored_program(sample), expected_value)
    assert_allclose(gradient_program(sample), expected_gradient, rtol=1e-6, atol=1e-10)
    assert_allclose(restored_gradient(sample), expected_gradient, rtol=1e-6, atol=1e-10)


def test_serialized_special_program_requires_explicit_linking_in_fresh_process(
    tmp_path: Path,
) -> None:
    program = ad.stage(
        special.erf,
        specs=(ad.ArraySpec((2,), "float64"),),
    )
    artifact_path = tmp_path / "erf-program.json"
    artifact_path.write_text(json.dumps(program.to_dict()), encoding="utf-8")
    script = """
import json
import sys
from pathlib import Path

import numpy as np
from scipy import special as scipy_special

import advect as ad

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
try:
    ad.StagedProgram.from_dict(payload)
except ValueError as error:
    assert "unlinked primitive 'scipy.special.erf'" in str(error), str(error)
else:
    raise AssertionError("artifact loaded without linking advect.scipy")

import advect.scipy

restored = ad.StagedProgram.from_dict(payload)
np.testing.assert_allclose(restored(np.array([0.2, -0.7])), scipy_special.erf([0.2, -0.7]))
"""

    completed = subprocess.run(  # noqa: S603 - fixed interpreter and inline test program.
        [sys.executable, "-c", script, str(artifact_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_special_abstract_rules_preserve_scipy_float32_dtypes() -> None:
    sample = np.array([0.7, 1.4], dtype=np.float32)
    unary = ad.stage(
        lambda x: special.erf(x) + special.gammaln(x),
        specs=(ad.ArraySpec(sample.shape, sample.dtype),),
    )
    poly = ad.stage(
        lambda x: special.polygamma(1, x),
        specs=(ad.ArraySpec(sample.shape, sample.dtype),),
    )

    assert unary(sample).dtype == np.float32
    assert poly(sample).dtype == np.float64


def test_special_functions_reject_non_numpy_array_providers_clearly() -> None:
    sample = strict.asarray([0.7, 1.4], dtype=strict.float32)

    with pytest.raises(TypeError, match=r"supports NumPy arrays only.*array_api_strict"):
        special.erf(sample)
    with pytest.raises(TypeError, match="NumPy dtype specifications only"):
        ad.stage(
            special.erf,
            specs=(ad.ArraySpec((2,), strict.float32),),
        )

    program = ad.stage(
        special.erf,
        specs=(ad.ArraySpec((2,), "float32"),),
    )
    with pytest.raises(TypeError, match=r"supports NumPy arrays only.*array_api_strict"):
        program(sample)


@pytest.mark.parametrize(
    "function",
    [
        special.erf,
        lambda x: special.polygamma(1, x),
        special.logsumexp,
    ],
)
def test_provider_errors_use_advects_resolver_for_arrays_without_object_protocol(
    function: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ResolverOnlyArray:
        shape = (2,)
        dtype = "float32"
        ndim = 1
        size = 2

    class ResolverOnlyNamespace:
        __name__ = "resolver_only"

    def resolve(value: object, *, api_version: str | None) -> object | None:
        del api_version
        return ResolverOnlyNamespace() if isinstance(value, ResolverOnlyArray) else None

    monkeypatch.setattr(array_api_providers, "_ARRAY_NAMESPACE_FALLBACK", resolve)

    with pytest.raises(TypeError, match=r"supports NumPy arrays only.*resolver_only"):
        function(ResolverOnlyArray())
