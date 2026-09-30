"""Elementwise derivative rules: discrete results, partial accuracy and domain edges."""

from __future__ import annotations

import math
import warnings
from decimal import Decimal, localcontext
from typing import Any

import array_api_strict as strict
import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, example, given
from numpy.testing import assert_allclose

import advect as ad


def _strict_namespace(value: Any) -> Any:
    return value.__array_namespace__()


def _integer_round_trip(v: Any) -> Any:
    xp = _strict_namespace(v)
    return xp.sum(xp.astype(xp.astype(v, xp.int64), xp.float64) * v)


_MASKED = np.array([-1.0, 0.2, 0.7])


@pytest.mark.parametrize("staged", [False, True], ids=["dynamic", "staged"])
@pytest.mark.parametrize(
    ("function", "value", "expected"),
    [
        pytest.param(
            _integer_round_trip,
            strict.asarray([1.5, 2.0]),
            np.array([1.0, 2.0]),
            id="array-api-round-trip",
        ),
        pytest.param(
            lambda v: np.sum(v.astype(np.int64).astype(np.float64)),
            np.array([1.5, 2.0], dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            id="numpy-float32",
        ),
        pytest.param(
            lambda v: np.sum(v * (v > 0)),
            _MASKED,
            np.array([0.0, 1.0, 1.0]),
            id="numpy-mask",
        ),
        pytest.param(
            lambda v: np.sum(v * np.logical_not(v > 0)),
            _MASKED,
            np.array([1.0, 0.0, 0.0]),
            id="numpy-logical-not",
        ),
        pytest.param(
            lambda v: _strict_namespace(v).sum(
                v * _strict_namespace(v).astype(v > 0.0, v.dtype),
            ),
            strict.asarray(_MASKED),
            np.array([0.0, 1.0, 1.0]),
            id="array-api-mask",
        ),
    ],
)
def test_reverse_mode_passes_zeros_through_discrete_results(
    function: Any,
    value: Any,
    expected: np.ndarray[Any, Any],
    *,
    staged: bool,
) -> None:
    """Integer and boolean results have no cotangent space in reverse mode.

    The zero cotangent of a cast into integers takes the source dtype, so it
    neither fails to promote against nor widens the source.
    """
    if staged:
        function = ad.stage(function, specs=(ad.ArraySpec(value.shape, value.dtype),))
    gradient = np.asarray(ad.grad(function)(value))

    assert gradient.dtype == expected.dtype
    assert_allclose(gradient, expected)


def _strict_cast(name: str) -> Any:
    return lambda v: _strict_namespace(v).astype(v, getattr(_strict_namespace(v), name))


def _strict_round_trip_loss(name: str) -> Any:
    def loss(v: Any) -> Any:
        xp = _strict_namespace(v)
        return xp.sum(xp.astype(_strict_cast(name)(v), v.dtype) * v * v)

    return loss


@pytest.mark.parametrize(
    ("cast", "loss", "value", "expected"),
    [
        pytest.param(
            _strict_cast("int64"),
            _strict_round_trip_loss("int64"),
            strict.asarray([1.5, 2.0]),
            np.array([2.0, 4.0]),
            id="array-api-int64",
        ),
        pytest.param(
            _strict_cast("bool"),
            _strict_round_trip_loss("bool"),
            strict.asarray([1.5, 0.0], dtype=strict.float32),
            np.array([2.0, 0.0], dtype=np.float32),
            id="array-api-bool",
        ),
        pytest.param(
            lambda v: v.astype(np.int64),
            lambda v: np.sum(v.astype(np.int64).astype(v.dtype) * v * v),
            np.array([1.5, 2.0], dtype=np.float32),
            np.array([2.0, 4.0], dtype=np.float32),
            id="numpy-float32",
        ),
    ],
)
def test_a_discrete_output_has_a_zero_tangent_of_the_tangent_dtype(
    cast: Any,
    loss: Any,
    value: Any,
    expected: np.ndarray[Any, Any],
) -> None:
    """An integer or boolean output does not promote its zero tangent.

    Array API providers need not promote a discrete dtype with a floating one,
    and NumPy would widen a float32 tangent to float64.
    """
    ones = value.__array_namespace__().ones_like(value)
    _output, tangent = ad.jvp(cast)(value, tangents=ones)
    assert np.asarray(tangent).dtype == expected.dtype
    assert not np.any(np.asarray(tangent))

    _value, product = ad.hvp(loss)(value, vectors=ones)
    assert np.asarray(product).dtype == expected.dtype
    assert_allclose(np.asarray(product), expected)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_a_discrete_output_leaf_does_not_widen_its_sibling_tangents(dtype: type) -> None:
    """An integer frexp exponent leaves both output tangents in the tangent dtype."""
    value = np.array([0.75, 3.0, -5.5], dtype=dtype)
    (mantissa, _exponent), (d_mantissa, d_exponent) = ad.jvp(np.frexp)(
        value,
        tangents=np.ones_like(value),
    )

    assert d_mantissa.dtype == d_exponent.dtype == dtype
    assert_allclose(d_mantissa, mantissa / value)
    assert not np.any(d_exponent)


_DOUBLE_ULP = float(np.finfo(np.float64).eps)
_NORMAL_DOUBLES = st.floats(allow_nan=False, allow_infinity=False, allow_subnormal=False)


@given(
    x=st.floats(-1.0, 1.0, exclude_min=True, exclude_max=True),
    operation=st.sampled_from(["arcsin", "arccos", "arctanh"]),
)
@example(x=1.0 - 2.0**-52, operation="arcsin")
@example(x=-(1.0 - 2.0**-52), operation="arccos")
@example(x=1.0 - 2.0**-30, operation="arctanh")
def test_inverse_unit_partials_stay_accurate_near_the_endpoints(x: float, operation: str) -> None:
    """``1 - x*x`` cancels near ``|x| = 1``; the factored partials do not."""
    with localcontext() as context:
        context.prec = 60
        exact = Decimal(x)
        denominator = (1 - exact) * (1 + exact)
        reference = 1 / denominator if operation == "arctanh" else 1 / denominator.sqrt()
    expected = -float(reference) if operation == "arccos" else float(reference)

    gradient = ad.grad(lambda v: np.sum(getattr(np, operation)(v)))(np.array([x]))

    assert_allclose(gradient, [expected], rtol=4 * _DOUBLE_ULP, atol=0)


def _sinc_derivative_reference(x: float, *, order: int) -> float:
    """Sum the Taylor series of sinc' or sinc'' in 60-digit arithmetic."""
    with localcontext() as context:
        context.prec = 60
        pi = Decimal("3.14159265358979323846264338327950288419716939937510582097494459")
        exact = Decimal(x)
        total = Decimal(0)
        for k in range(1, 40):
            power = 2 * k - order
            coefficient = (-1) ** k * 2 * k * (2 * k - 1 if order == 2 else 1)
            term = coefficient * pi ** (2 * k) / math.factorial(2 * k + 1)
            total += term * (exact**power if power else 1)
        return float(total)


@given(x=st.floats(-1.0, 1.0, allow_subnormal=False))
@example(x=0.0)
@example(x=1e-8)
@example(x=0.25)
def test_sinc_partials_stay_accurate_through_the_peak(x: float) -> None:
    """The closed form of sinc' cancels near zero; the rule's series does not."""
    value = np.array([x])

    gradient = ad.grad(lambda v: np.sum(np.sinc(v)))(value)
    curvature = ad.hessian(lambda v: np.sum(np.sinc(v)))(value)

    first = _sinc_derivative_reference(x, order=1)
    second = _sinc_derivative_reference(x, order=2)
    assert_allclose(gradient, [first], rtol=8 * _DOUBLE_ULP, atol=0)
    assert_allclose(curvature, [[second]], rtol=32 * _DOUBLE_ULP, atol=1e-14)


@given(x=_NORMAL_DOUBLES, y=_NORMAL_DOUBLES)
@example(x=1e-170, y=1e-170)
@example(x=1e-170, y=-2e-160)
def test_divide_denominator_partial_does_not_square_the_denominator(x: float, y: float) -> None:
    """``y * y`` over- or underflows where ``-x / y**2`` is a normal double."""
    with localcontext() as context:
        context.prec = 60
        reference = -Decimal(x) / (Decimal(y) * Decimal(y)) if y != 0 else Decimal(0)
    finfo = np.finfo(np.float64)
    assume(Decimal(float(finfo.tiny)) <= abs(reference) <= Decimal(float(finfo.max)))

    gradient = ad.grad(lambda d: np.sum(np.divide(np.array([x]), d)))(np.array([y]))

    assert_allclose(gradient, [float(reference)], rtol=4 * _DOUBLE_ULP, atol=0)


def test_cbrt_nested_derivatives_stay_finite_at_zero() -> None:
    """The cbrt partial is guarded to zero at zero in every derivative order.

    The ``hypot_origin`` case of ``test_array_api_nested_derivatives.py`` covers
    the other user of the guarded reciprocal.
    """
    value = np.array([0.0, 8.0])
    ones = np.ones_like(value)

    def directional(v: Any) -> Any:
        return np.sum(ad.jvp(np.cbrt)(v, tangents=ones)[1])

    _value, product = ad.hvp(lambda v: np.sum(np.cbrt(v)))(value, vectors=ones)
    assert_allclose(ad.grad(directional)(value), [0.0, -1.0 / 144.0], rtol=1e-15)
    assert_allclose(product, [0.0, -1.0 / 144.0], rtol=1e-15)


@st.composite
def _real_powers(draw: st.DrawFn) -> tuple[float, float]:
    """Draw a positive base with any exponent, or any base with an integer exponent."""
    if draw(st.booleans()):
        return draw(st.floats(0.25, 4.0)), draw(st.floats(-2.0, 2.0))
    return draw(st.floats(-2.0, 2.0, allow_subnormal=False)), float(draw(st.integers(0, 3)))


@given(point=_real_powers(), operation=st.sampled_from(["power", "float_power"]))
@example(point=(-2.0, 2.0), operation="power")
@example(point=(0.0, 0.0), operation="power")
@example(point=(-1.5, 3.0), operation="power")
@example(point=(1.25, 0.75), operation="power")
@example(point=(1.25, 0.75), operation="float_power")
@example(point=(2.0, 0.0), operation="power")
@example(point=(2.0, 0.0), operation="float_power")
@example(point=(6.90971716022898e-104, 3.0), operation="power")
@example(point=(0.0, 1.0), operation="power")
@example(point=(-2.0, -1.0), operation="power")
@example(point=(-0.5, 0.0), operation="float_power")
def test_power_exponent_partial_agrees_in_every_mode_and_exponent_form(
    point: tuple[float, float],
    operation: str,
) -> None:
    """The exponent partial is ``x**y * log(x)``, and zero for a base that is not positive.

    Forward and reverse mode agree for an array, 0-D or Python scalar exponent,
    and the Hessian is symmetric.
    """
    base, exponent = point
    function = getattr(np, operation)
    x = np.array([base])
    log_base = math.log(base) if base > 0 else 0.0
    expected = base**exponent * log_base if base > 0 else 0.0
    # Where x**y is subnormal, each side rounds it to within half its spacing.
    atol = 2.0**-1074 * abs(log_base)

    for y in (np.array([exponent]), np.array(exponent), exponent):
        _value, forward = ad.jvp(lambda e: function(x, e))(y, tangents=y * 0 + 1)
        reverse = ad.grad(lambda e: np.sum(function(x, e)))(y)

        assert_allclose(forward, [expected], rtol=1e-14, atol=atol)
        assert_allclose(reverse, np.full(np.shape(y), expected), rtol=1e-14, atol=atol)

    # d/dy (y * x**(y - 1)) == d/dx (x**y * log(x)), up to a sum that can
    # cancel. Off the positive axis both are zero, like the exponent partial.
    # Tiny positive bases are left out, where IEEE arithmetic limits power's
    # Hessian:
    # - Where x**y underflows, the product rule drops the x**(y - 1) term of
    #   d grad_y / dx, a relative error of 1 / |1 + y * log(x)| < 0.14%. JAX
    #   and PyTorch share it.
    # - At a subnormal base, 1 / x overflows to inf or NaN.
    # - At 0 < |y| <= 1e-10 and |x| < 1e-154, y * ((y - 1) * x**(y - 2))
    #   overflows before y scales it back.
    # - As for any function, a dense Hessian's zero seed turns another
    #   entry's legitimate infinity into NaN.
    # Off the positive axis, where x**(y - 1) overflows, forward mode also
    # multiplies it by the zero y-derivative, so d grad_x / dy is NaN there.
    if base >= 0.25 or base <= 0:
        mixed = base ** (exponent - 1) * (1 + exponent * log_base) if base > 0 else 0.0
        scale = base ** (exponent - 1) * (1 + abs(exponent * log_base)) if base > 0 else 0.0
        (_xx, xy), (yx, _yy) = ad.hessian(function, argnums=(0, 1))(
            np.array(base),
            np.array(exponent),
        )
        assert_allclose([xy, yx], [mixed, mixed], rtol=0, atol=1e-14 * scale)


@given(
    base=st.floats(0.25, 4.0).flatmap(lambda magnitude: st.sampled_from([magnitude, -magnitude])),
    exponent=st.complex_numbers(max_magnitude=2.0),
    operation=st.sampled_from(["power", "float_power"]),
)
@example(base=-2.0, exponent=2 + 0j, operation="power")
def test_power_exponent_partial_takes_a_real_bases_complex_log(
    base: float,
    exponent: complex,
    operation: str,
) -> None:
    """A complex exponent makes a real base's power complex, also off the positive axis.

    Its partial is ``x**y * log(x)`` with the principal complex logarithm.
    """
    function = getattr(np, operation)
    x = np.array([base])
    y = np.array([exponent])
    expected = function(x, y) * np.log(x.astype(np.complex128))

    _value, forward = ad.jvp(lambda e: function(x, e))(y, tangents=np.ones_like(y))
    _value, pullback = ad.vjp(lambda e: function(x, e))(y)

    assert_allclose(forward, expected, rtol=1e-15, atol=0)
    assert_allclose(pullback(np.ones_like(y)), np.conj(expected), rtol=1e-15, atol=0)


@given(
    base=st.sampled_from([0.0, -0.0, 0j]),
    exponent=st.builds(complex, st.floats(2.0**-10, 2.0), st.floats(-2.0, 2.0)),
    operation=st.sampled_from(["power", "float_power"]),
)
@example(base=0.0, exponent=2 + 0.5j, operation="power")
@example(base=-0.0, exponent=2 + 0j, operation="float_power")
@example(base=0j, exponent=2 + 0.5j, operation="power")
def test_complex_power_exponent_partial_is_zero_at_a_zero_base(
    base: complex,
    exponent: complex,
    operation: str,
) -> None:
    """``0**y`` is zero wherever ``Re(y) > 0``, so its partial is zero, not ``0 * log(0)``."""
    function = getattr(np, operation)
    x = np.array([base])
    y = np.array([exponent])

    _value, forward = ad.jvp(lambda e: function(x, e))(y, tangents=np.ones_like(y))
    _value, pullback = ad.vjp(lambda e: function(x, e))(y)

    np.testing.assert_array_equal(forward, [0j])
    np.testing.assert_array_equal(pullback(np.ones_like(y)), [0j])


@given(
    base=st.floats(-2.0, 2.0, allow_subnormal=False),
    exponent=st.integers(0, 3).map(float),
    operation=st.sampled_from(["power", "float_power"]),
)
@example(base=0.0, exponent=0.0, operation="float_power")
@example(base=0.0, exponent=0.0, operation="power")
@example(base=5e-324, exponent=0.0, operation="float_power")
@example(base=-1e-310, exponent=0.0, operation="power")
def test_power_base_partial_is_exactly_zero_for_a_zero_exponent(
    base: float,
    exponent: float,
    operation: str,
) -> None:
    """``d/dx x**0`` is zero, also where ``0 * x**-1`` would be NaN: a zero or subnormal ``x``."""
    function = getattr(np, operation)
    x = np.array([base])

    for y in (np.array([exponent]), exponent):
        _value, forward = ad.jvp(lambda b, y=y: function(b, y))(x, tangents=np.ones(1))
        reverse = ad.grad(lambda b, y=y: np.sum(function(b, y)))(x)

        expected = exponent * base ** (exponent - 1) if exponent else 0.0
        assert_allclose(forward, [expected], rtol=1e-14, atol=0)
        assert_allclose(reverse, [expected], rtol=1e-14, atol=0)


@pytest.mark.parametrize("function", [np.power, np.float_power], ids=["power", "float_power"])
def test_power_base_partial_promotes_a_boolean_exponent(function: Any) -> None:
    """``d/dx x**y`` reads a boolean ``y`` in the power's dtype, as NumPy promotes it.

    The partial subtracted one from the boolean exponent, which NumPy rejects:
    "numpy boolean subtract, the `-` operator, is not supported".
    """
    x = np.array([0.5, -1.0, 2.0], np.float32)
    expected = (x < 0).astype(function(x, x < 0).dtype)

    _value, tangent = ad.jvp(lambda b: function(b, b < 0))(x, tangents=np.ones_like(x))
    gradient = ad.grad(lambda b: np.sum(function(b, b < 0)))(x)
    scalar = ad.grad(lambda s, b: np.sum(function(s, s < b)))(0.5, x)

    assert tangent.dtype == expected.dtype
    assert_allclose(tangent, expected)
    assert_allclose(gradient, expected)
    assert scalar == pytest.approx(float(np.sum(x > 0.5)))


def test_staged_power_base_partial_promotes_a_boolean_exponent() -> None:
    x = np.array([0.5, -1.0, 2.0], np.float32)
    program = ad.stage(lambda b: np.sum(b ** (b < 0)), x)

    assert_allclose(ad.grad(program)(x), (x < 0).astype(np.float32))


@st.composite
def _normal_bases(draw: st.DrawFn) -> tuple[Any, float]:
    """Draw a dtype and a base of either sign whose reciprocal is normal."""
    dtype = draw(st.sampled_from([np.float32, np.float64]))
    info = np.finfo(dtype)
    smallest = float(info.smallest_normal)
    magnitude = draw(st.floats(smallest, 1.0 / smallest, width=info.bits))
    return dtype, draw(st.sampled_from([magnitude, -magnitude]))


@given(point=_normal_bases(), operation=st.sampled_from(["power", "float_power"]))
@example(point=(np.float64, 6.354477348262163e-161), operation="power")
@example(point=(np.float32, -1e-30), operation="float_power")
def test_power_hessian_at_a_zero_exponent_is_finite_at_every_normal_base(
    point: tuple[Any, float],
    operation: str,
) -> None:
    """At ``y == 0``, ``H_xx = 0``, the cross terms are ``1 / x`` and ``H_yy = log(x)**2``.

    Below about 1e-154 in float64, ``y * x**(y - 1)`` meets ``0 * x**-2``,
    whose NaN would fill the base's whole row. Off the positive axis the
    exponent partial is zero, and so are both cross terms.
    """
    dtype, base = point
    function = getattr(np, operation)
    x = np.array([base, 2.0], dtype=dtype)
    y = np.zeros(2, dtype=dtype)

    def total(a: Any, b: Any) -> Any:
        return np.sum(function(a, b))

    (xx, xy), (yx, yy) = ad.hessian(total, argnums=(0, 1))(x, y)
    _value, (hvp_x, hvp_y) = ad.hvp(total, argnums=(0, 1))(x, y, vectors=(np.ones_like(x), y))

    rtol = 4 * float(np.finfo(dtype).eps)
    inverse = 1.0 / x.astype(np.float64)
    positive = x > 0
    log_squared = np.where(positive, np.log(np.abs(x.astype(np.float64))) ** 2, 0.0)
    cross = np.diag(np.where(positive, inverse, 0.0))
    assert_allclose(xx, np.zeros((2, 2)), rtol=0, atol=0)
    assert_allclose(xy, cross, rtol=rtol, atol=0)
    assert_allclose(yx, cross, rtol=rtol, atol=0)
    assert_allclose(yy, np.diag(log_squared), rtol=rtol, atol=0)
    assert_allclose(hvp_x, np.zeros(2), rtol=0, atol=0)
    assert_allclose(hvp_y, np.diag(xy), rtol=rtol, atol=0)


@given(
    base=st.floats(2.0**-126, 2.0**126, width=32).flatmap(
        lambda magnitude: st.sampled_from([magnitude, -magnitude])
    ),
)
@example(base=3.7)
def test_float_power_cross_derivative_at_a_zero_exponent_keeps_double_precision(
    base: float,
) -> None:
    """``float_power`` of float32 operands computes in float64, also ``d2 f / dy dx = 1 / x``.

    Off the positive axis the cross derivative is zero.
    """
    x = np.array([base], dtype=np.float32)
    y = np.zeros(1, dtype=np.float32)

    def along_x(b: Any) -> Any:
        return ad.jvp(lambda a: np.float_power(a, b))(x, tangents=np.ones_like(x))[1]

    _value, cross = ad.jvp(along_x)(y, tangents=np.ones_like(y))

    assert cross.dtype == np.float64
    expected = 1.0 / x.astype(np.float64) if base > 0 else np.zeros(1)
    assert_allclose(cross, expected, rtol=float(np.finfo(np.float64).eps), atol=0)


@given(
    base=st.floats(1e-150, 1e150).flatmap(
        lambda magnitude: st.sampled_from([magnitude, -magnitude])
    ),
    operation=st.sampled_from(["power", "float_power"]),
)
@example(base=0.5, operation="power")
@example(base=-0.5, operation="float_power")
def test_power_third_derivative_at_a_zero_exponent_keeps_every_exponent_term(
    base: float,
    operation: str,
) -> None:
    """At ``y == 0``, ``d3 x**y / dx dy2 = 2 * log(x) / x``, and zero off the positive axis.

    The zero-exponent form of the base partial keeps each of its y-derivatives.
    Below about 1e-150, its intermediate ``-1 / x**2`` overflows.
    """
    function = getattr(np, operation)
    x = np.array([base, 2.0])
    y = np.zeros(2)

    def cross(a: Any, b: Any) -> Any:
        """Return ``d2 f / dx_i dy_i``: the x-part of the HVP along the exponent."""
        _value, (along_x, _along_y) = ad.hvp(
            lambda u, v: np.sum(function(u, v)),
            argnums=(0, 1),
        )(a, b, vectors=(np.zeros(2), np.ones(2)))
        return along_x

    third = ad.jacobian(cross, argnums=1)(x, y)
    expected = np.where(x > 0, 2 * np.log(np.abs(x)) / x, 0.0)
    assert_allclose(third, np.diag(expected), rtol=1e-14, atol=0)


def test_power_second_derivatives_of_a_staged_program_match_dynamic_mode() -> None:
    """A staged power's derivative reads its domain masks from traced values.

    Nested transforms evaluate the staged derivative with live tracers, which
    an as-array conversion of the base would reject. The zero exponent and the
    negative base select the masked branches.
    """
    x = np.array([0.5, 2.0, -1.5])
    y = np.array([0.0, 3.0, 2.0])
    vectors = (np.array([1.0, -0.5, 2.0]), np.array([0.5, 1.0, -1.0]))

    def total(a: Any, b: Any) -> Any:
        return np.sum(a**b)

    def second_derivatives(function: Any) -> tuple[Any, ...]:
        hessian = ad.hessian(function, argnums=(0, 1))(x, y)
        _value, product = ad.hvp(function, argnums=(0, 1))(x, y, vectors=vectors)
        return (*hessian, product)

    staged = second_derivatives(ad.stage(total, x, y))
    for actual, expected in zip(staged, second_derivatives(total), strict=True):
        assert_allclose(actual, expected, rtol=1e-15, atol=0)


def test_staged_power_base_gradient_evaluates_one_power() -> None:
    """A staged gradient with respect to the base does not keep the unused result ``x**y``.

    Its traced exponent needs the zero-exponent form, which reads ``x**y`` from
    the same power as ``x**(y - 1)``.
    """
    x = np.array([0.5, 2.0, -1.5])
    y = np.array([0.0, 3.0, 2.0])

    def total(a: Any, b: Any) -> Any:
        return np.sum(a**b)

    program = ad.stage(ad.grad(total), x, y)
    graph = program.graph
    operations = [graph.get_node(node_id).op for node_id in graph.node_ids()]

    assert operations.count("array.power") == 1
    assert_allclose(program(x, y), ad.grad(total)(x, y), rtol=1e-15, atol=0)


@pytest.mark.parametrize("operation", ["power", "float_power"])
def test_power_hessian_keeps_the_exponent_block_at_an_infinite_base(operation: str) -> None:
    """At ``x = inf, y = 0`` the base partial's ``0 * inf`` must not reach ``H_yy``."""
    function = getattr(np, operation)
    x = np.array(np.inf)
    y = np.array(0.0)

    with warnings.catch_warnings():
        # The x-block itself meets 0 * inf; only the exponent block is pinned.
        warnings.simplefilter("ignore", RuntimeWarning)
        (_xx, _xy), (_yx, yy) = ad.hessian(function, argnums=(0, 1))(x, y)
        exponent_only = ad.hessian(lambda e: function(x, e))(y)

    assert yy == np.inf
    assert exponent_only == np.inf


@given(
    shapes=hnp.mutually_broadcastable_shapes(num_shapes=3, max_dims=2, max_side=3),
    data=st.data(),
)
@example(shapes=hnp.BroadcastableShapes(((), (5,), ()), (5,)), data=None)
@example(shapes=hnp.BroadcastableShapes(((1,), (2,), (2,)), (2,)), data=None)
def test_clip_derivatives_match_its_minimum_of_maximum_form(
    shapes: hnp.BroadcastableShapes,
    data: st.DataObject | None,
) -> None:
    """Differentiate clip as ``minimum(maximum(x, min), max)``.

    That covers an operand smaller than the broadcast result and crossed
    bounds, where the result is ``max``. Small integers make ties common.
    """
    values = st.integers(-2, 2).map(float)
    if data is None:
        # x below crossed bounds: min = 2 > max = 1.
        fills = (0.0, 2.0, 1.0)
        operands = tuple(np.full(shape, fill) for shape, fill in zip(shapes[0], fills, strict=True))
    else:
        operands = tuple(
            data.draw(hnp.arrays(np.float64, shape, elements=values)) for shape in shapes[0]
        )
    tangents = tuple(np.ones_like(operand) for operand in operands)

    def reference(x: Any, lo: Any, hi: Any) -> Any:
        return np.minimum(np.maximum(x, lo), hi)

    _value, forward = ad.jvp(np.clip, argnums=(0, 1, 2))(*operands, tangents=tangents)
    _value, expected_forward = ad.jvp(reference, argnums=(0, 1, 2))(*operands, tangents=tangents)
    reverse = ad.grad(lambda *a: np.sum(np.clip(*a)), argnums=(0, 1, 2))(*operands)
    expected_reverse = ad.grad(lambda *a: np.sum(reference(*a)), argnums=(0, 1, 2))(*operands)

    assert forward.shape == shapes.result_shape
    np.testing.assert_array_equal(forward, expected_forward)
    for gradient, expected in zip(reverse, expected_reverse, strict=True):
        np.testing.assert_array_equal(gradient, expected)
