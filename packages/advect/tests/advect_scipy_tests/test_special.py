"""Special-function primitive qualification."""

from __future__ import annotations

import json
import subprocess
import sys
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import array_api_strict as strict
import numpy as np
import pytest
from hypothesis import example, given, settings, strategies as st
from hypothesis.extra import numpy as hnp
from numpy.testing import assert_allclose, assert_array_equal
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
        lambda x: scipy_special.expit(x) * scipy_special.expit(-x),
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


def _and_restored(program: ad.StagedProgram) -> tuple[ad.StagedProgram, ad.StagedProgram]:
    """Return ``program`` and its serialized round trip, so both lifetimes run."""
    return program, ad.StagedProgram.from_dict(program.to_dict())


@pytest.mark.parametrize("case", _UNARY_CASES, ids=lambda case: case.name)
def test_unary_special_input_is_positional_only_like_the_scipy_ufunc(
    case: _UnaryCase,
) -> None:
    with pytest.raises(TypeError):
        case.expected(x=case.sample)
    with pytest.raises(TypeError):
        case.actual(x=case.sample)


def _erfcx_central_difference(x: np.ndarray) -> np.ndarray:
    step = 1e-4 * x
    return (scipy_special.erfcx(x + step) - scipy_special.erfcx(x - step)) / (2 * step)


def _log_ndtr_derivative(x: np.ndarray) -> np.ndarray:
    return np.sqrt(2 / np.pi) / scipy_special.erfcx(-x / np.sqrt(2))


def _expit_derivative(x: np.ndarray) -> np.ndarray:
    decay = np.exp(-np.abs(x))
    return decay / (1 + decay) ** 2


@pytest.mark.parametrize(
    ("name", "derivative", "dtype", "sample"),
    [
        ("erfcx", _erfcx_central_difference, np.float32, [1e3, 1e4]),
        ("erfcx", _erfcx_central_difference, np.float64, [1e8]),
        ("log_ndtr", _log_ndtr_derivative, np.float32, [-1e4, -1e6]),
        ("log_ndtr", _log_ndtr_derivative, np.float64, [-1e8]),
        ("expit", _expit_derivative, np.float32, [10.0, 15.0]),
        ("expit", _expit_derivative, np.float64, [30.0, 40.0]),
    ],
)
def test_special_gradient_keeps_relative_accuracy_in_the_tail(
    name: str,
    derivative: Callable[[np.ndarray], np.ndarray],
    dtype: object,
    sample: list[float],
) -> None:
    value = np.asarray(sample, dtype=dtype)
    _primal, actual = ad.jvp(getattr(special, name))(value, tangents=np.ones_like(value))

    assert_allclose(actual, derivative(np.asarray(sample)), rtol=3e-6, atol=0)


def test_erfcx_gradient_is_accurate_across_the_tail_crossover() -> None:
    sample = np.array([7.9999, 8.0001])
    _primal, derivative = ad.jvp(special.erfcx)(sample, tangents=np.ones_like(sample))
    expected = 2 * sample * scipy_special.erfcx(sample) - 2 / np.sqrt(np.pi)

    assert_allclose(derivative, expected, rtol=2e-13, atol=2e-15)


def _derivative(function: Callable[[object], object], x: np.ndarray) -> np.ndarray:
    return ad.jvp(function)(x, tangents=np.ones_like(x))[1]


def _away_from_poles(low: float, high: float, gap: float) -> st.SearchStrategy[float]:
    """Floats in ``[low, high]`` at least ``gap`` from every non-positive integer."""
    return st.floats(low, high).filter(lambda x: x > gap or abs(x - round(x)) >= gap)


#: ``(actual, expected, domain)``: each pairs two independent rules, a rule
#: with SciPy, or a rule with an independent closed form, so none restates the
#: formula it checks.
_DERIVATIVE_IDENTITIES = (
    pytest.param(
        lambda y: _derivative(lambda v: special.erf(special.erfinv(v)), y),
        np.ones_like,
        st.floats(-1 + 1e-12, 1 - 1e-12),
        id="erf-of-erfinv",
    ),
    pytest.param(
        lambda p: _derivative(lambda v: special.ndtr(special.ndtri(v)), p),
        np.ones_like,
        st.floats(1e-300, 1 - 1e-12),
        id="ndtr-of-ndtri",
    ),
    pytest.param(
        lambda x: _derivative(lambda v: special.log_expit(v) - special.log_expit(-v), x),
        np.ones_like,
        st.floats(-700, 700),
        id="log_expit-odd-part",
    ),
    pytest.param(
        partial(_derivative, special.expit),
        _expit_derivative,
        st.floats(-700, 700),
        id="expit",
    ),
    pytest.param(
        partial(_derivative, special.erfc),
        lambda x: -_derivative(special.erf, x),
        st.floats(-26, 26),
        id="erfc-complements-erf",
    ),
    pytest.param(
        partial(_derivative, special.ndtr),
        lambda x: np.exp(scipy_special.log_ndtr(x)) * _derivative(special.log_ndtr, x),
        st.floats(-20, 20),
        id="ndtr-exponentiates-log_ndtr",
    ),
    # Near a negative pole the reflection formula loses about |x| / gap ulps.
    pytest.param(
        lambda x: _derivative(lambda v: special.gammaln(v + 1) - special.gammaln(v), x),
        np.reciprocal,
        _away_from_poles(-10, 30, gap=0.05),
        id="gammaln-recurrence",
    ),
    pytest.param(
        partial(_derivative, special.digamma),
        partial(scipy_special.polygamma, 1),
        _away_from_poles(-10, 30, gap=0.05),
        id="digamma",
    ),
)


@pytest.mark.parametrize(("actual", "expected", "domain"), _DERIVATIVE_IDENTITIES)
@settings(max_examples=max(5, settings.default.max_examples // 20), deadline=None)
@given(data=st.data())
def test_special_derivatives_satisfy_formula_free_identities(
    actual: Callable[[np.ndarray], np.ndarray],
    expected: Callable[[np.ndarray], np.ndarray],
    domain: st.SearchStrategy[float],
    data: st.DataObject,
) -> None:
    x = data.draw(hnp.arrays(np.float64, st.integers(1, 16), elements=domain))

    assert_allclose(actual(x), expected(x), rtol=1e-12, atol=0)


@settings(max_examples=max(5, settings.default.max_examples // 20), deadline=None)
@given(
    z=hnp.arrays(
        np.complex128,
        st.integers(1, 16),
        elements=st.builds(complex, st.floats(-20, 20), st.floats(-20, 20)).filter(
            lambda z: abs(z - min(0, round(z.real))) >= 0.1
        ),
    )
)
def test_complex_trigamma_satisfies_the_recurrence(z: np.ndarray) -> None:
    # psi1(z) = psi1(z + 1) + 1 / z**2 across the reflection, recurrence, and
    # asymptotic branches of digamma's complex derivative. Reflection loses
    # about |pi z| ulps (the worst of a million draws is 1.3e-13), measured
    # against the largest term so that cancellation near a pole is absorbed.
    trigamma, shifted = _derivative(special.digamma, z), _derivative(special.digamma, z + 1)
    scale = np.max(np.abs([trigamma, shifted, 1 / z**2]), axis=0)

    assert np.all(np.abs(trigamma - shifted - 1 / z**2) <= 1e-12 * scale)


#: Functions outside the shared unary installer that also get the precision check.
_EXTRA_PRECISION_CASES = (
    _UnaryCase(
        "polygamma",
        lambda x: special.polygamma(2, x),
        lambda x: scipy_special.polygamma(2, x),
        lambda x: scipy_special.polygamma(3, x),
        np.array([0.7, 1.4, 2.8]),
    ),
    _UnaryCase(
        "logsumexp",
        special.logsumexp,
        scipy_special.logsumexp,
        lambda x: np.exp(x - scipy_special.logsumexp(x)),
        np.array([0.7, 1.4, 2.8]),
    ),
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", _UNARY_CASES + _EXTRA_PRECISION_CASES, ids=lambda case: case.name)
def test_every_special_function_matches_value_and_gradient_at_supported_precision(
    case: _UnaryCase,
    dtype: type[np.floating],
) -> None:
    sample = np.asarray(case.sample, dtype=dtype)
    tolerance = 2e-5 if dtype is np.float32 else 2e-12
    expected = case.expected(sample)

    value, _tangent = ad.jvp(case.actual)(sample, tangents=np.ones_like(sample))
    gradient = ad.grad(lambda x: np.sum(case.actual(x)))(sample)

    assert np.asarray(value).dtype == np.asarray(expected).dtype
    assert_array_equal(value, expected)
    assert gradient.dtype == sample.dtype
    assert_allclose(gradient, case.derivative(sample), rtol=tolerance, atol=tolerance)


@settings(max_examples=max(5, settings.default.max_examples // 20), deadline=None)
@given(x=hnp.arrays(np.float64, st.integers(1, 16), elements=st.floats(-60, 60)))
@example(x=np.array([37.7, 38.0, 40.0]))
def test_log_ndtr_second_derivative_satisfies_its_riccati_identity(x: np.ndarray) -> None:
    # (log ndtr)'' = -r (x + r) for r = (log ndtr)'. The lower tail cancels r x
    # against r**2, so measure against those terms. erfcx overflows from about
    # x = 37.6, and subnormal derivatives carry no relative precision.
    first = _derivative(special.log_ndtr, x)
    second = _derivative(partial(_derivative, special.log_ndtr), x)
    scale = np.abs(first) * np.maximum(np.abs(x), np.abs(first))

    error = np.abs(second + first * (x + first))
    assert np.all(error <= 1e-12 * scale + np.finfo(np.float64).tiny)


def test_float32_log_ndtr_second_derivative_is_finite_where_erfcx_overflows() -> None:
    sample = np.array([13.5, 14.0, 20.0], dtype=np.float32)

    second = ad.hessian_diag(lambda x: np.sum(special.log_ndtr(x)))(sample)

    assert np.all(np.isfinite(second))
    assert np.all(second <= 0)


def test_gammaln_has_a_traceable_second_derivative() -> None:
    sample = np.array([0.7, 1.4, 2.8])

    actual = ad.grad(lambda x: np.sum(ad.grad(lambda y: np.sum(special.gammaln(y)))(x)))(sample)

    assert_allclose(actual, scipy_special.polygamma(1, sample))


@st.composite
def _reductions(draw: st.DrawFn) -> tuple[np.ndarray, int | tuple[int, ...] | None]:
    """Draw an input and a None, integer, or unique (possibly empty) tuple axis."""
    shapes = hnp.array_shapes(min_dims=0, max_dims=3, max_side=4)
    sample = draw(hnp.arrays(np.float64, shapes, elements=st.floats(-50, 50)))
    ndim = sample.ndim
    axes = draw(st.lists(st.integers(0, max(ndim - 1, 0)), unique=True, max_size=ndim))
    signed = tuple(axis - ndim * draw(st.booleans()) for axis in axes)
    single = st.integers(-ndim, ndim - 1) if ndim else st.nothing()
    return sample, draw(st.none() | st.just(signed) | single)


@settings(max_examples=max(10, settings.default.max_examples // 8), deadline=None)
@given(case=_reductions(), keepdims=st.booleans())
def test_softmax_family_is_shift_invariant_along_reduced_axes(
    case: tuple[np.ndarray, int | tuple[int, ...] | None],
    keepdims: bool,  # noqa: FBT001 - Hypothesis argument.
) -> None:
    # Adding a constant along the reduced axes leaves softmax and log_softmax
    # unchanged and shifts logsumexp by that constant, so each derivative has
    # an exact value along ones (up to 1 - sum(p), a few ulps). Staging checks
    # the abstract reduced shape.
    sample, axis = case
    ones = np.ones_like(sample)
    cotangent = np.random.default_rng(0).normal(size=sample.shape)
    reduced = tuple(range(sample.ndim)) if axis is None else axis
    for actual, expected in (
        (special.softmax, scipy_special.softmax),
        (special.log_softmax, scipy_special.log_softmax),
    ):

        def function(x: object, actual: Callable[..., object] = actual) -> object:
            return actual(x, axis=axis)

        value, shift = ad.jvp(function)(sample, tangents=ones)
        pulled = ad.vjp(function)(sample)[1](cotangent)

        assert_array_equal(value, expected(sample, axis=axis))
        assert_allclose(shift, 0, atol=1e-12)
        assert_allclose(np.sum(pulled, axis=reduced), 0, atol=1e-12)

    def reduction(x: object) -> object:
        return special.logsumexp(x, axis=axis, keepdims=keepdims)

    value, shift = ad.jvp(reduction)(sample, tangents=ones)
    gradient = ad.vjp(reduction)(sample)[1](np.ones_like(value))
    expected = scipy_special.logsumexp(sample, axis=axis, keepdims=keepdims)

    for actual in (value, ad.stage(reduction, sample)(sample)):
        assert np.shape(actual) == np.shape(expected)
        assert_array_equal(actual, expected)
    assert np.shape(shift) == np.shape(expected)
    assert_allclose(shift, 1, rtol=1e-12)
    assert_allclose(gradient, scipy_special.softmax(sample, axis=axis), rtol=1e-12, atol=1e-15)


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

    assert_allclose(value, expected(sample, axis=axis))
    assert_allclose(directional, expected_directional, rtol=2e-9, atol=2e-9)
    assert_allclose(
        np.vdot(directional, weights),
        np.vdot(tangent, gradient),
        rtol=2e-12,
        atol=2e-12,
    )
    assert_allclose(gradient, expected_gradient, rtol=2e-12, atol=2e-12)
    for staged in _and_restored(ad.stage(function, sample)):
        assert_allclose(staged(sample), value)
    for staged in _and_restored(ad.grad(ad.stage(loss, sample))):
        assert_allclose(staged(sample), expected_gradient, rtol=2e-12, atol=2e-12)


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
    program = ad.stage(lambda x: actual(x, 0), sample)

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

    expected_value = expected(sample, axis=1)

    for staged in _and_restored(ad.stage(lambda value: actual(value, axis=1), sample)):
        assert np.asarray(staged(sample)).dtype == np.asarray(expected_value).dtype
        assert_array_equal(staged(sample), expected_value)


@pytest.mark.parametrize("actual", [special.softmax, special.log_softmax])
def test_softmax_primitives_preserve_scipy_boolean_rejection(
    actual: Callable[[object, object], object],
) -> None:
    sample = np.array([[True, False, True]])

    with pytest.raises(TypeError, match="boolean subtract"):
        ad.stage(lambda value: actual(value, axis=1), sample)


@pytest.mark.parametrize("axis", [True, [0], (True,), ("bad",)])
def test_staged_softmax_requires_static_integer_axes(axis: object) -> None:
    sample = np.arange(12.0).reshape(3, 4)

    with pytest.raises(TypeError, match="axis"):
        ad.stage(lambda x: special.softmax(x, axis=axis), sample)


@pytest.mark.parametrize(
    "case",
    [case for case in _UNARY_CASES if case.name in {"erf", "erfc", "erfcx", "ndtr", "log_ndtr"}],
    ids=lambda case: case.name,
)
def test_complex_unary_functions_use_advects_real_adjoint_convention(case: _UnaryCase) -> None:
    sample = np.array([0.4 + 0.3j, -0.2 + 0.7j])

    gradient = ad.grad(lambda z: np.sum(np.real(case.actual(z))))(sample)

    assert_allclose(gradient, np.conj(case.derivative(sample)), rtol=2e-12, atol=2e-12)


def test_complex_digamma_matches_scipy_and_uses_advects_real_adjoint_convention() -> None:
    sample = np.array([0.4 + 0.3j, 1.2 - 0.7j])
    tangent = np.array([0.2 - 0.4j, -0.3 + 0.7j])
    step = 1e-6
    expected_derivative = (
        scipy_special.digamma(sample + step) - scipy_special.digamma(sample - step)
    ) / (2 * step)

    value, directional = ad.jvp(special.digamma)(sample, tangents=tangent)

    def loss(z: object) -> object:
        return np.sum(np.real(special.digamma(z)))

    gradient = ad.grad(loss)(sample)

    assert_allclose(value, scipy_special.digamma(sample), rtol=2e-13, atol=2e-13)
    assert_allclose(directional, expected_derivative * tangent, rtol=2e-9, atol=2e-9)
    assert_allclose(gradient, np.conj(expected_derivative), rtol=2e-9, atol=2e-9)
    for staged in _and_restored(ad.stage(special.digamma, sample)):
        assert_allclose(staged(sample), value)
    for staged in _and_restored(ad.grad(ad.stage(loss, sample))):
        assert_allclose(staged(sample), gradient, rtol=2e-9, atol=2e-9)


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

    assert_allclose(value, expected)
    assert_allclose(directional, expected_derivative, rtol=2e-12, atol=2e-12)
    assert_allclose(gradient, expected_derivative, rtol=2e-12, atol=2e-12)
    for staged in _and_restored(ad.stage(update, sample)):
        assert_allclose(staged(sample), expected)
    for staged in _and_restored(ad.grad(ad.stage(lambda x: np.sum(update(x)), sample))):
        assert_allclose(staged(sample), expected_derivative, rtol=2e-12, atol=2e-12)


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

    assert_allclose(value, expected, rtol=2e-13, atol=2e-13)
    assert_allclose(directional, expected_directional, rtol=2e-12, atol=2e-12)
    assert_allclose(gradient, 2 * derivative, rtol=2e-12, atol=2e-12)
    for staged in _and_restored(ad.stage(function, sample)):
        assert_allclose(staged(sample), expected, rtol=2e-13, atol=2e-13)


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
    program = ad.stage(function, sample)

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

    for staged in _and_restored(ad.stage(function, sample)):
        assert staged(sample).dtype == np.float64
        assert_allclose(staged(sample), expected)


_OPTION_SAMPLE = np.arange(12.0).reshape(3, 4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"signature": np.str_("d->d")},
        {"signature": np.bytes_(b"d->d")},
        {"casting": np.str_("unsafe"), "order": np.bytes_(b"C")},
    ],
    ids=["str_-signature", "bytes_-signature", "text-options"],
)
def test_staged_unary_ufunc_accepts_numpy_scalar_text_options(kwargs: dict[str, object]) -> None:
    program = ad.stage(lambda x: special.erf(x, **kwargs), _OPTION_SAMPLE)

    assert_allclose(program(_OPTION_SAMPLE), scipy_special.erf(_OPTION_SAMPLE, **kwargs))


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"out": ()}, ValueError, "exactly one entry"),
        ({"out": (None, None)}, ValueError, "exactly one entry"),
        ({"unknown": True}, TypeError, "unexpected keyword"),
        ({"sig": "d->d", "signature": "d->d"}, TypeError, "both 'sig' and 'signature'"),
        ({"signature": None}, TypeError, "signature object"),
        ({"signature": 3}, TypeError, "signature object"),
        ({"subok": 1}, TypeError, "subok.*boolean"),
        ({"casting": 3}, TypeError, "casting must be str"),
        ({"order": 3}, TypeError, "order must be str"),
    ],
)
def test_staged_unary_ufunc_rejects_invalid_options(
    kwargs: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        ad.stage(lambda x: special.erf(x, **kwargs), _OPTION_SAMPLE)


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

    assert np.asarray(value).dtype == np.float64
    assert np.asarray(directional).dtype == np.float64
    assert_allclose(value, expected_value)
    assert_allclose(directional, expected_tangent)
    assert_allclose(gradient, expected_gradient)
    for staged in _and_restored(ad.stage(function, sample)):
        assert_allclose(staged(sample), expected_value)
    for staged in _and_restored(ad.grad(ad.stage(loss, sample))):
        assert_allclose(staged(sample), expected_gradient)


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

    assert_allclose(value, expected_value)
    assert_allclose(directional, expected_derivative)
    assert_allclose(gradient, expected_derivative)
    for staged in _and_restored(ad.stage(update, sample)):
        assert_allclose(staged(sample), expected_value)
    for staged in _and_restored(ad.grad(ad.stage(loss, sample))):
        assert_allclose(staged(sample), expected_derivative)


def test_where_without_out_preserves_defined_values_and_masks_the_jvp() -> None:
    sample = np.array([-0.7, 0.2, 1.1, 2.0])
    tangent = np.array([0.2, -0.4, 0.7, -0.1])
    mask = np.array([True, False, True, False])

    value, directional = ad.jvp(lambda x: special.erf(x, where=mask))(
        sample,
        tangents=tangent,
    )
    program = ad.stage(lambda x: special.erf(x, where=mask), sample)

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

    assert_allclose(value, expected)
    assert_allclose(directional, expected_gradient)
    for staged in _and_restored(
        ad.grad(ad.stage(lambda x, where: np.sum(function(x, where)), sample, mask))
    ):
        assert_allclose(staged(sample, mask), expected_gradient)


def test_polygamma_gradient_keeps_float_precision_for_integer_x() -> None:
    sample = np.array([1, 2, 3])

    def loss(x: object) -> object:
        return np.sum(special.polygamma(1, x))

    for gradient in (
        ad.grad(loss)(sample),
        *(staged(sample) for staged in _and_restored(ad.grad(ad.stage(loss, sample)))),
    ):
        assert gradient.dtype == np.float64
        assert_allclose(gradient, scipy_special.polygamma(2, sample))


def test_polygamma_array_orders_stage_serialize_and_differentiate_x() -> None:
    orders = np.array([[0], [1], [3]], dtype=np.int64)
    sample = np.array([[0.7, 1.4, 2.8, 4.1]])
    expected = scipy_special.polygamma(orders, sample)
    expected_gradient = np.sum(
        scipy_special.polygamma(orders + 1, sample),
        axis=0,
        keepdims=True,
    )

    assert_allclose(
        ad.grad(lambda x: np.sum(special.polygamma(orders, x)))(sample),
        expected_gradient,
    )
    for staged in _and_restored(ad.stage(special.polygamma, orders, sample)):
        assert_allclose(staged(orders, sample), expected)
    for staged in _and_restored(
        ad.grad(ad.stage(lambda n, x: np.sum(special.polygamma(n, x)), orders, sample), argnums=1)
    ):
        assert_allclose(staged(orders, sample), expected_gradient)


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


@pytest.mark.parametrize(
    ("a_dtype", "b_dtype"),
    [
        (np.float64, np.float64),
        (np.float32, np.float64),
        (np.float64, np.float32),
        (np.int64, np.float64),
        (np.float64, np.int64),
    ],
)
def test_logsumexp_broadcast_weights_have_unbroadcasted_primal_precision_gradients(
    a_dtype: type[np.number],
    b_dtype: type[np.number],
) -> None:
    a = np.array([[-2.0, 0.4, 1.7], [2.0, -0.2, 0.8]]).astype(a_dtype)
    b = np.array([0.5, 1.5, 2.0]).astype(b_dtype)
    reference_a = a.astype(np.float64)
    reference_b = b.astype(np.float64)
    rtol = 1e-6 if np.float32 in (a_dtype, b_dtype) else 1e-7

    def loss(aa: object, bb: object) -> object:
        return np.sum(special.logsumexp(aa, axis=1, b=bb))

    gradient_a, gradient_b = ad.grad(loss, argnums=(0, 1))(a, b)
    denominator = np.sum(reference_b * np.exp(reference_a), axis=1, keepdims=True)
    expected_a = reference_b * np.exp(reference_a) / denominator
    expected_b = np.sum(np.exp(reference_a) / denominator, axis=0)
    staged_a, staged_b = ad.grad(ad.stage(loss, a, b), argnums=(0, 1))(a, b)

    for actual, primal, expected in (
        (gradient_a, a, expected_a),
        (gradient_b, b, expected_b),
        (staged_a, a, expected_a),
        (staged_b, b, expected_b),
    ):
        # Integer operands keep SciPy's float64 result instead of truncating.
        inexact = np.issubdtype(primal.dtype, np.inexact)
        assert actual.dtype == (primal.dtype if inexact else np.float64)
        assert_allclose(actual, expected, rtol=rtol)


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
    for staged in _and_restored(ad.stage(function, a, b)):
        for actual, expected in zip(staged(a, b), value, strict=True):
            assert_allclose(actual, expected)
    for staged in _and_restored(ad.grad(ad.stage(loss, a, b), argnums=(0, 1))):
        for actual, expected in zip(staged(a, b), expected_gradient, strict=True):
            assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("sample", "axis", "b", "keepdims", "return_sign"),
    [
        (np.array(2.0), None, None, True, False),
        (np.array(2.0), 0, None, True, False),
        (np.array(2.0), -1, None, True, False),
        (np.array(2.0), (), None, False, False),
        (np.array(2.0), (), None, True, False),
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
    kwargs = {"axis": axis, "b": b, "keepdims": keepdims, "return_sign": return_sign}
    expected = scipy_special.logsumexp(sample, **kwargs)
    actual = ad.stage(lambda x: special.logsumexp(x, **kwargs), sample)(sample)

    for actual_part, expected_part in zip(
        actual if return_sign else (actual,),
        expected if return_sign else (expected,),
        strict=True,
    ):
        assert np.shape(actual_part) == np.shape(expected_part)
        assert_allclose(actual_part, expected_part, equal_nan=True)


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

    expected_value = loss(sample)
    expected_gradient = ad.grad(loss)(sample)
    program = ad.stage(loss, sample)
    for staged in _and_restored(program):
        assert_allclose(staged(sample), expected_value)
    for staged in _and_restored(ad.grad(program)):
        assert_allclose(staged(sample), expected_gradient, rtol=1e-12, atol=1e-12)


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

import advect as ad

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
try:
    ad.StagedProgram.from_dict(payload)
except ValueError as error:
    assert "unlinked primitive 'scipy.special.erf'" in str(error), str(error)
else:
    raise AssertionError("artifact loaded without linking advect.scipy")
# Neither the base import nor loading an artifact may import SciPy.
scipy_modules = [name for name in sys.modules if name.partition(".")[0] == "scipy"]
assert not scipy_modules and "advect.scipy" not in sys.modules, scipy_modules[:3]

import advect.scipy
from scipy import special as scipy_special

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
    unary = ad.stage(lambda x: special.erf(x) + special.gammaln(x), sample)
    poly = ad.stage(lambda x: special.polygamma(1, x), sample)

    assert unary(sample).dtype == np.float32
    assert poly(sample).dtype == np.float64


def test_special_functions_reject_non_numpy_array_providers_clearly() -> None:
    sample = strict.asarray([0.7, 1.4], dtype=strict.float32)

    with pytest.raises(TypeError, match=r"supports NumPy arrays only.*array_api_strict"):
        special.erf(sample)

    program = ad.stage(
        special.erf,
        specs=(ad.ArraySpec((2,), "float32"),),
    )
    with pytest.raises(TypeError, match=r"supports NumPy arrays only.*array_api_strict"):
        program(sample)


@pytest.mark.parametrize(
    "function",
    [special.erf, special.gammaln, special.logsumexp, special.softmax],
    ids=["erf", "gammaln", "logsumexp", "softmax"],
)
@pytest.mark.parametrize(
    "transform",
    [
        pytest.param(
            lambda function: ad.grad(lambda x: x.__array_namespace__().sum(function(x))),
            id="grad",
        ),
        pytest.param(
            lambda function: lambda x: ad.jvp(function)(x, tangents=x),
            id="jvp",
        ),
    ],
)
def test_special_derivatives_staged_from_other_providers_reject_them_at_call_time(
    function: Callable[[object], object],
    transform: Callable[[Callable[[object], object]], Callable[[object], object]],
) -> None:
    # Staged code sees strict's dtype objects; the derivative rules resolve
    # NumPy loops from the canonical dtype, so only the call rejects strict.
    sample = strict.asarray([0.7, 1.4], dtype=strict.float32)
    program = ad.stage(transform(function), sample)

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
