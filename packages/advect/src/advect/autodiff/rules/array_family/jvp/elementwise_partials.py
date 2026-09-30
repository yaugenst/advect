"""Local partial derivatives of the diagonal elementwise operations.

One table serves both modes. Forward mode sums each active tangent times its
operand's partial; reverse mode multiplies the cotangent by each conjugate
partial, which is the real adjoint of that diagonal map. An operand's partial
is a formula, a real constant, or ``None`` when it is identically zero.

The table an entry sits in states which primal values its formulas read, and
the reverse sweep retains exactly those. Operand formulas take the operands,
result formulas take the result, and mixed formulas take the result followed
by the operands. Each formula is evaluated only for an operand whose tangent
or cotangent is active. A ``_Shared`` entry computes work common to its
operands' formulas once, from the same values, and passes it to each formula
first.

An outer transform can trace one operand and not another, and an Array API
array rejects a traced right operand of a Python operator. A formula that
combines two operands, or an operand and the result, therefore calls ``xp``
functions, which dispatch to the traced operand's namespace.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

from advect.autodiff.rules.array_family._backend_runtime import _scalar_like, xp
from advect.autodiff.rules.array_family._transpose_utils import (
    _dtype_is_complex,
    _dtype_of,
    dtype_is_inexact,
)
from advect.autodiff.rules.array_family.jvp.common import (
    _asarray_preserving_trace,
    _asarray_unwrapped,
    _astype_preserving_trace,
    _iscomplex_unwrapped,
)
from advect.core._protocols import _innermost, _is_traced

type Partial = Callable[..., Any] | float | None
type _Of1 = Callable[[Any], Any] | float | None
type _Of2 = Callable[[Any, Any], Any] | float | None
type _Of3 = Callable[[Any, Any, Any], Any] | float | None
type _Of4 = Callable[[Any, Any, Any, Any], Any] | float | None


@dataclass(frozen=True, slots=True)
class ElementwisePartials:
    """One operation's partials and the primal values their formulas read."""

    partials: tuple[Partial, ...]
    reads_operands: bool
    reads_result: bool
    shared: Callable[..., Any] | None = None

    def bind(self, ans: Any, operands: tuple[Any, ...]) -> Callable[[int], Any]:
        """Return each operand's partial, computing the shared term at most once."""
        values = (
            *((ans,) if self.reads_result else ()),
            *(operands if self.reads_operands else ()),
        )
        shared: list[Any] = []

        def at(index: int) -> Any:
            formula = self.partials[index]
            if not callable(formula):
                return formula
            if self.shared is None:
                return formula(*values)
            if not shared:
                shared.append(self.shared(*values))
            return formula(shared[0], *values)

        return at


@dataclass(frozen=True, slots=True)
class _Shared:
    """Operand formulas that each take ``term(*values)`` before the values."""

    term: Callable[[Any, Any], Any] | Callable[[Any, Any, Any], Any]
    partials: tuple[_Of3, _Of3] | tuple[_Of4, _Of4]


def _inverse(value: Any) -> Any:
    return _scalar_like(1.0, value) / value


def _safe_inverse(value: Any) -> Any:
    """Return ``1 / value``, and zero where ``value`` is zero.

    The masked reciprocal reads one instead of zero, so it neither warns nor
    sends an infinite derivative into a nested one, where the mask's zero
    cotangent would turn it into NaN.
    """
    nonzero = value != _scalar_like(0, value)
    safe = xp.where(nonzero, value, _scalar_like(1, value))
    return xp.where(nonzero, _inverse(safe), xp.zeros_like(value))


def _inverse_unit_root(x: Any) -> Any:
    """Return ``1 / sqrt(1 - x**2)`` without cancellation near ``|x| = 1``."""
    one = _scalar_like(1, x)
    return _inverse(xp.sqrt(one - x) * xp.sqrt(one + x))


def _arcsin_partial(ans: Any, x: Any) -> Any:
    # On a complex branch cut the output records which side the primal chose.
    return _inverse(xp.cos(ans)) if _iscomplex_unwrapped(x) else _inverse_unit_root(x)


def _arccos_partial(ans: Any, x: Any) -> Any:
    return -_inverse(xp.sin(ans)) if _iscomplex_unwrapped(x) else -_inverse_unit_root(x)


def _arccosh_partial(x: Any) -> Any:
    one = _scalar_like(1, x)
    return _inverse(xp.sqrt(x - one) * xp.sqrt(x + one))


def _arctanh_partial(x: Any) -> Any:
    one = _scalar_like(1, x)
    return _inverse((one - x) * (one + x))


# Near zero the closed form of sinc' cancels, with relative error about
# 3 eps / (pi x)**2. Within the radius, eight terms of its Taylor series
# x * sum_k c_k x**(2k) are accurate to rounding instead.
_SINC_SERIES_RADIUS = 0.25
_SINC_SLOPE_SERIES = tuple(
    (-1) ** k * 2 * k * math.pi ** (2 * k) / math.factorial(2 * k + 1) for k in range(1, 9)
)


def _sinc_partial(x: Any) -> Any:
    # The partial stays traced, and each branch sees only the inputs it
    # selects, so nested derivatives are exact at and near the peak.
    x_arr = _asarray_preserving_trace(x)
    magnitude = xp.abs(x_arr)
    near_zero = magnitude < _scalar_like(_SINC_SERIES_RADIUS, magnitude)
    zero, one = _scalar_like(0.0, x_arr), _scalar_like(1.0, x_arr)
    series_x = xp.where(near_zero, x_arr, zero)
    square = series_x * series_x
    series = _scalar_like(_SINC_SLOPE_SERIES[-1], x_arr)
    for coefficient in reversed(_SINC_SLOPE_SERIES[:-1]):
        series = series * square + _scalar_like(coefficient, x_arr)
    safe_x = xp.where(near_zero, one, x_arr)
    pix = _scalar_like(math.pi, safe_x) * safe_x
    closed_form = (pix * xp.cos(pix) - xp.sin(pix)) / (pix * safe_x)
    return xp.where(near_zero, series_x * series, closed_form)


def _mask_operand(value: Any) -> Any:
    """Return the array a domain mask reads: ``value``'s innermost payload.

    A mask has no derivative. Under dynamic tracing it reads the concrete
    payload and records nothing. While staging it reads the abstract value
    itself, where ``_asarray_unwrapped`` would record an as-array conversion,
    which rejects the live tracers of an outer transform that evaluates the
    staged derivative.
    """
    return _asarray_preserving_trace(_innermost(value))


def _without_masked_exponent_derivative(x: Any, y: Any) -> Any:
    """Return ``y``, without a derivative where a real power's exponent partial is zero.

    That partial is zero at a base <= 0, where a real power needs an integer
    exponent, whose floor is the same value with a zero derivative.
    """
    dtype = _dtype_of(y)
    if not dtype_is_inexact(dtype) or _dtype_is_complex(dtype) or _iscomplex_unwrapped(x):
        return y
    base, whole = _mask_operand(x), xp.floor(y)
    integer = _mask_operand(whole) == _mask_operand(y)
    return xp.where(xp.logical_and(base <= _scalar_like(0, base), integer), whole, y)


def _power_base_partial(x: Any, y: Any, power: Callable[[Any, Any], Any]) -> Any:
    """Return ``y * power(x, y - 1)``, with a zero partial for a zero exponent."""
    if isinstance(y, (bool, int, float, complex)):
        if y == 0:
            return xp.zeros_like(x)
        return _scalar_like(y, x) * power(x, _scalar_like(y - 1, x))
    one, nil = _scalar_like(1.0, x), _scalar_like(0, x)
    exponent = _mask_operand(y)
    zero = exponent == _scalar_like(0, exponent)
    if not _is_traced(y):
        general = xp.multiply(y, power(xp.where(zero, one, x), y - _scalar_like(1, y)))
        return xp.where(zero, nil, general)
    # Where the exponent partial is zero, so is this partial's y-derivative,
    # which keeps the Hessian symmetric.
    y = _without_masked_exponent_derivative(x, y)
    # A traced exponent also needs the partial's derivatives at y == 0. There
    # it reads (y * x**y) / x, which has every y-derivative, and dividing the
    # zero product keeps its x-derivatives zero, where y * x**(y - 1) would
    # meet 0 * x**-2, which overflows below about 1e-154. One power, x**y at a
    # zero exponent and x**(y - 1) elsewhere, serves both forms, so a staged
    # derivative need not keep the result. Each branch sees only the bases it
    # selects, and the quotient takes the power's dtype, which keeps
    # float_power's double precision. A base whose reciprocal is not a finite
    # product keeps a constant zero: zero, subnormal, infinite, or NaN, for
    # which the comparisons are false.
    magnitude = xp.abs(_mask_operand(x))
    dtype = magnitude.dtype
    normal = xp.finfo(dtype).smallest_normal if dtype_is_inexact(dtype) else 1
    regular = xp.logical_and(magnitude >= _scalar_like(normal, magnitude), xp.isfinite(magnitude))
    reciprocal = xp.logical_and(zero, regular)
    product = xp.multiply(
        y,
        power(
            xp.where(xp.logical_xor(zero, reciprocal), one, x),
            xp.where(zero, y, y - _scalar_like(1, y)),
        ),
    )
    at_zero = xp.divide(product, xp.where(reciprocal, x, _scalar_like(1.0, product)))
    return xp.where(zero, xp.where(reciprocal, at_zero, nil), product)


def _power_exponent_partial(ans: Any, x: Any) -> Any:
    """Return ``x**y * log(x)``, zero where the base has no logarithm.

    A real power reads a positive base's. A complex power reads the principal
    logarithm, also of a real base, which every base but zero has; ``0**y`` is
    zero wherever it is differentiable.
    """
    base = _mask_operand(x)
    complex_power = _iscomplex_unwrapped(ans)
    zero = _scalar_like(0, base)
    logarithmic = xp.not_equal(base, zero) if complex_power else xp.greater(base, zero)
    safe_x = xp.where(logarithmic, x, _scalar_like(1.0, x))
    if complex_power and not _iscomplex_unwrapped(x):
        safe_x = xp.add(safe_x, _scalar_like(0j, ans))
    return xp.where(logarithmic, ans * xp.log(safe_x), _scalar_like(0, ans))


def _numeric_exponent(ans: Any, y: Any) -> Any:
    """Return a boolean exponent array in the power's dtype, as NumPy promotes it.

    A boolean array has no subtraction, which the base partial needs.
    """
    if isinstance(y, (bool, int, float, complex)) or "bool" not in str(_dtype_of(y)):
        return y
    return _astype_preserving_trace(y, dtype=_dtype_of(ans))


def _power(name: str) -> tuple[_Of3, _Of3]:
    return (
        lambda ans, x, y: _power_base_partial(x, _numeric_exponent(ans, y), getattr(xp, name)),
        lambda ans, x, _y: _power_exponent_partial(ans, x),
    )


def _chooses_x(ans: Any, x: Any, y: Any, *, maximum: bool, ignore_nan: bool) -> Any:
    """Return where a binary maximum or minimum selects ``x``.

    Ties select ``x``. With ``ignore_nan`` a NaN operand loses to a number, as
    in ``fmax`` and ``fmin``. A constant operand is compared in the result
    dtype, as the ufunc rounds a weak Python scalar.
    """
    dtype = _asarray_unwrapped(ans).dtype
    x_arr, y_arr = (
        value if _is_traced(value) else xp.asarray(value, dtype=dtype) for value in (x, y)
    )
    choose_x = xp.greater_equal(x_arr, y_arr) if maximum else xp.less_equal(x_arr, y_arr)
    if ignore_nan:
        choose_x = xp.logical_or(
            choose_x,
            xp.logical_and(xp.isnan(y_arr), xp.logical_not(xp.isnan(x_arr))),
        )
    return choose_x


def _selection(*, maximum: bool, ignore_nan: bool) -> _Shared:
    # The 0/1 partials cast the boolean choice, which carries no derivative, so
    # nested derivatives stay exact. A comparison of 0-d NumPy arrays returns a
    # bool scalar, which NumPy 2.0's astype rejects, so the choice is an array
    # before the cast.
    def indicator(choose: Any, ans: Any) -> Any:
        return xp.astype(_asarray_preserving_trace(choose), _asarray_unwrapped(ans).dtype)

    return _Shared(
        partial(_chooses_x, maximum=maximum, ignore_nan=ignore_nan),
        (
            lambda choose_x, ans, _x, _y: indicator(choose_x, ans),
            lambda choose_x, ans, _x, _y: indicator(xp.logical_not(choose_x), ans),
        ),
    )


# ``(x / y) / y`` neither overflows nor underflows where ``y * y`` would.
_DIVIDE: tuple[_Of2, _Of2] = (lambda _x, y: _inverse(y), lambda x, y: -xp.divide(x, y) / y)
_DEG2RAD: tuple[float] = (math.pi / 180.0,)
_RAD2DEG: tuple[float] = (180.0 / math.pi,)
_IDENTITY: tuple[float] = (1.0,)
_FLAT: tuple[None] = (None,)

_OPERAND_PARTIALS: dict[str, tuple[_Of1] | tuple[_Of2, _Of2] | _Shared] = {
    "array.sin": (lambda x: xp.cos(x),),  # noqa: PLW0108 - resolve xp per call
    "array.cos": (lambda x: -xp.sin(x),),
    "array.sinh": (lambda x: xp.cosh(x),),  # noqa: PLW0108 - resolve xp per call
    "array.cosh": (lambda x: xp.sinh(x),),  # noqa: PLW0108 - resolve xp per call
    "array.arctan": (lambda x: _inverse(_scalar_like(1, x) + x * x),),
    "array.arcsinh": (lambda x: _inverse(xp.sqrt(x * x + _scalar_like(1, x))),),
    "array.arccosh": (_arccosh_partial,),
    "array.arctanh": (_arctanh_partial,),
    "array.log": (_inverse,),
    "array.log1p": (lambda x: _inverse(_scalar_like(1, x) + x),),
    "array.log2": (lambda x: _inverse(x * _scalar_like(math.log(2.0), x)),),
    "array.log10": (lambda x: _inverse(x * _scalar_like(math.log(10.0), x)),),
    "array.square": (lambda x: _scalar_like(2.0, x) * x,),
    "array_ext.fabs": (lambda x: xp.sign(_asarray_unwrapped(x)),),
    "array_ext.sinc": (_sinc_partial,),
    "array.multiply": (lambda _x, y: y, lambda x, _y: x),
    "array.divide": _DIVIDE,
    "array_ext.true_divide": _DIVIDE,
    "array.remainder": (1.0, lambda x, y: -xp.floor(xp.divide(x, y))),
    "array_ext.fmod": (1.0, lambda x, y: -xp.trunc(xp.divide(x, y))),
    "array.arctan2": _Shared(
        lambda y, x: xp.add(y * y, x * x),
        (lambda square, _y, x: xp.divide(x, square), lambda square, y, _x: xp.divide(-y, square)),
    ),
}

_RESULT_PARTIALS: dict[str, tuple[_Of1]] = {
    "array.exp": (lambda ans: ans,),
    "array.expm1": (lambda ans: ans + _scalar_like(1.0, ans),),
    "array_ext.exp2": (lambda ans: ans * math.log(2.0),),
    "array.sqrt": (lambda ans: _inverse(_scalar_like(2.0, ans) * ans),),
    "array_ext.cbrt": (lambda ans: _safe_inverse(_scalar_like(3.0, ans) * ans * ans),),
    "array.reciprocal": (lambda ans: -(ans * ans),),
    "array.tan": (lambda ans: _scalar_like(1, ans) + ans * ans,),
    "array.tanh": (lambda ans: _scalar_like(1.0, ans) - ans * ans,),
}

_MIXED_PARTIALS: dict[str, tuple[_Of2] | tuple[_Of3, _Of3] | _Shared] = {
    "array.arcsin": (_arcsin_partial,),
    "array.arccos": (_arccos_partial,),
    "array.power": _power("pow"),
    "array_ext.float_power": _power("float_power"),
    "array.hypot": _Shared(
        lambda ans, _x, _y: _safe_inverse(ans),
        (
            lambda inverse, _ans, x, _y: xp.multiply(x, inverse),
            lambda inverse, _ans, _x, y: xp.multiply(y, inverse),
        ),
    ),
    "array.copysign": (lambda ans, x, _y: xp.multiply(xp.sign(x), xp.sign(ans)), None),
    "array.logaddexp": (
        lambda ans, x, _y: xp.exp(xp.subtract(x, ans)),
        lambda ans, _x, y: xp.exp(xp.subtract(y, ans)),
    ),
    "array_ext.logaddexp2": (
        lambda ans, x, _y: xp.exp2(xp.subtract(x, ans)),
        lambda ans, _x, y: xp.exp2(xp.subtract(y, ans)),
    ),
    "array_ext.heaviside": (
        None,
        lambda ans, x, _y: xp.where(xp.equal(x, 0), _scalar_like(1, ans), _scalar_like(0, ans)),
    ),
    "array.maximum": _selection(maximum=True, ignore_nan=False),
    "array.minimum": _selection(maximum=False, ignore_nan=False),
    "array_ext.fmax": _selection(maximum=True, ignore_nan=True),
    "array_ext.fmin": _selection(maximum=False, ignore_nan=True),
}

_CONSTANT_PARTIALS: dict[str, tuple[float | None, ...]] = {
    "array.negative": (-1.0,),
    "array.positive": _IDENTITY,
    "advect.copy": _IDENTITY,
    "array.add": (1.0, 1.0),
    "array.subtract": (1.0, -1.0),
    "array.nextafter": (1.0, None),
    "array_ext.deg2rad": _DEG2RAD,
    "array_ext.radians": _DEG2RAD,
    "array_ext.rad2deg": _RAD2DEG,
    "array_ext.degrees": _RAD2DEG,
    "array.floor": _FLAT,
    "array.ceil": _FLAT,
    "array.trunc": _FLAT,
    "array.rint": _FLAT,
    "array_ext.spacing": _FLAT,
    "array.floor_divide": (None, None),
}


def _join_partial_tables() -> dict[str, ElementwisePartials]:
    joined: dict[str, ElementwisePartials] = {}
    for table, reads_operands, reads_result in (
        (_OPERAND_PARTIALS, True, False),
        (_RESULT_PARTIALS, False, True),
        (_MIXED_PARTIALS, True, True),
        (_CONSTANT_PARTIALS, False, False),
    ):
        for op_name, partials in table.items():
            if op_name in joined:
                msg = f"Duplicate elementwise partials for {op_name!r}"
                raise RuntimeError(msg)
            joined[op_name] = (
                ElementwisePartials(partials.partials, reads_operands, reads_result, partials.term)
                if isinstance(partials, _Shared)
                else ElementwisePartials(partials, reads_operands, reads_result)
            )
    return joined


ELEMENTWISE_PARTIALS = _join_partial_tables()
