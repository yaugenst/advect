"""Elementwise JVP partial derivative formulas."""

from __future__ import annotations

import math
from typing import Any

from advect.autodiff.rules.array_family._backend_runtime import _scalar_like, xp
from advect.autodiff.rules.array_family.jvp.common import (
    _fmax_choice_mask,
    _fmin_choice_mask,
    _iscomplex_unwrapped,
    _maximum_choice_mask,
    _minimum_choice_mask,
    _positive_domain_mask,
)


def _partials_log10(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    _ = ans, rest, attrs
    return (_scalar_like(1.0, x) / (x * _scalar_like(math.log(10.0), x)),)


def _partials_logaddexp(
    ans: xp.ndarray,
    x: xp.ndarray,
    y: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray]:
    _ = rest, attrs
    return (xp.exp(x - ans), xp.exp(y - ans))


def _partials_logaddexp2(
    ans: xp.ndarray,
    x: xp.ndarray,
    y: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray]:
    _ = rest, attrs
    return (xp.exp2(x - ans), xp.exp2(y - ans))


def _partials_heaviside(
    ans: xp.ndarray,
    x: xp.ndarray,
    value_at_zero: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[float, xp.ndarray]:
    _ = ans, value_at_zero, rest, attrs
    return (0.0, xp.equal(x, 0))


def _partials_nextafter(
    ans: xp.ndarray,
    x: xp.ndarray,
    direction: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[float, float]:
    _ = ans, x, direction, rest, attrs
    return (1.0, 0.0)


def _partials_deg2rad(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[float]:
    _ = ans, x, rest, attrs
    return (xp.pi / 180.0,)


def _partials_rad2deg(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[float]:
    _ = ans, x, rest, attrs
    return (180.0 / xp.pi,)


def _partials_square(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    _ = ans, rest, attrs
    return (_scalar_like(2.0, x) * x,)


def _partials_power(
    ans: xp.ndarray,
    x: xp.ndarray,
    y: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray]:
    _ = rest, attrs
    complex_base = _iscomplex_unwrapped(x)
    if complex_base:
        safe_x = x
    else:
        positive_mask = _positive_domain_mask(x)
        safe_x = xp.where(positive_mask, x, _scalar_like(1.0, x))
    if isinstance(y, (bool, int, float, complex)):
        dx = xp.zeros_like(x) if y == 0 else y * xp.pow(x, y - 1)
    else:
        zero_exponent = y == _scalar_like(0, y)
        safe_derivative_base = xp.where(zero_exponent, _scalar_like(1.0, x), x)
        dx = xp.where(
            zero_exponent,
            xp.zeros_like(x),
            y * xp.pow(safe_derivative_base, y - _scalar_like(1, y)),
        )
    if complex_base:
        dy = ans * xp.log(safe_x)
    else:
        dy = xp.where(_positive_domain_mask(x), ans * xp.log(safe_x), xp.zeros_like(ans))
    return dx, dy


def _partials_reciprocal(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    _ = x, rest, attrs
    return (-(ans * ans),)


def _partials_zero(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[float]:
    _ = ans, x, rest, attrs
    return (0.0,)


def _partials_maximum(
    ans: xp.ndarray,
    x: xp.ndarray,
    y: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray]:
    _ = ans, rest, attrs
    choose_x = _maximum_choice_mask(x, y)
    return (
        xp.astype(choose_x, x.dtype),
        xp.astype(xp.logical_not(choose_x), y.dtype),
    )


def _partials_minimum(
    ans: xp.ndarray,
    x: xp.ndarray,
    y: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray]:
    _ = ans, rest, attrs
    choose_x = _minimum_choice_mask(x, y)
    return (
        xp.astype(choose_x, x.dtype),
        xp.astype(xp.logical_not(choose_x), y.dtype),
    )


def _partials_fmax(
    ans: xp.ndarray,
    x: xp.ndarray,
    y: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray]:
    _ = ans, rest, attrs
    choose_x = _fmax_choice_mask(x, y)
    return (choose_x, xp.logical_not(choose_x))


def _partials_fmin(
    ans: xp.ndarray,
    x: xp.ndarray,
    y: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray]:
    _ = ans, rest, attrs
    choose_x = _fmin_choice_mask(x, y)
    return (choose_x, xp.logical_not(choose_x))


def _partials_exp2(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    _ = x, rest, attrs
    return (ans * math.log(2.0),)


def _partials_expm1(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    _ = x, rest, attrs
    return (ans + _scalar_like(1.0, ans),)
