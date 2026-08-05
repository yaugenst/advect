"""Elementwise JVP partial derivative formulas."""

from __future__ import annotations

import math
from typing import Any

from advect.autodiff.rules.array_family._backend_runtime import _scalar_like, xp
from advect.autodiff.rules.array_family._impl.jvp.common import _positive_domain_mask


def _partials_divide(
    ans: xp.ndarray,
    x: xp.ndarray,
    y: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray]:
    _ = ans, rest, attrs
    return (_scalar_like(1.0, y) / y, -x / (y * y))


def _partials_true_divide(
    ans: xp.ndarray,
    x: xp.ndarray,
    y: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray]:
    return _partials_divide(ans, x, y, *rest, **attrs)


def _partials_floor_divide(
    ans: xp.ndarray,
    x: xp.ndarray,
    y: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[float, float]:
    _ = ans, x, y, rest, attrs
    return (0.0, 0.0)


def _partials_remainder(
    ans: xp.ndarray,
    x: xp.ndarray,
    y: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[float, xp.ndarray]:
    _ = ans, rest, attrs
    return (1.0, -xp.floor(x / y))


def _partials_fmod(
    ans: xp.ndarray,
    x: xp.ndarray,
    y: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[float, xp.ndarray]:
    _ = ans, rest, attrs
    return (1.0, -xp.trunc(x / y))


def _partials_float_power(
    ans: xp.ndarray,
    x: xp.ndarray,
    y: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray]:
    _ = rest, attrs
    positive_mask = _positive_domain_mask(x)
    safe_x = xp.where(positive_mask, x, _scalar_like(1.0, x))
    dx = y * xp.float_power(x, y - _scalar_like(1, y))
    dy = xp.where(positive_mask, ans * xp.log(safe_x), xp.zeros_like(ans))
    return (dx, dy)


def _partials_hypot(
    ans: xp.ndarray,
    x: xp.ndarray,
    y: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray]:
    _ = rest, attrs
    safe_inv = xp.where(
        ans != _scalar_like(0, ans),
        _scalar_like(1.0, ans) / ans,
        xp.zeros_like(ans),
    )
    return (x * safe_inv, y * safe_inv)


def _partials_copysign(
    ans: xp.ndarray,
    x: xp.ndarray,
    y: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray, float]:
    _ = y, rest, attrs
    return (xp.sign(x) * xp.sign(ans), 0.0)


def _partials_arctan2(
    ans: xp.ndarray,
    y: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray]:
    _ = ans, rest, attrs
    denom = y * y + x * x
    return (x / denom, -y / denom)


def _partials_tan(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    _ = x, rest, attrs
    return (_scalar_like(1, ans) + ans * ans,)


def _partials_arcsin(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    _ = x, rest, attrs
    return (_scalar_like(1.0, ans) / xp.cos(ans),)


def _partials_arccos(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    _ = x, rest, attrs
    return (-_scalar_like(1.0, ans) / xp.sin(ans),)


def _partials_arctan(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    _ = ans, rest, attrs
    return (_scalar_like(1.0, x) / (_scalar_like(1, x) + x * x),)


def _partials_arcsinh(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    _ = ans, rest, attrs
    return (_scalar_like(1.0, x) / xp.sqrt(x * x + _scalar_like(1, x)),)


def _partials_arccosh(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    _ = ans, rest, attrs
    one = _scalar_like(1, x)
    return (_scalar_like(1.0, x) / (xp.sqrt(x - one) * xp.sqrt(x + one)),)


def _partials_arctanh(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    _ = ans, rest, attrs
    return (_scalar_like(1.0, x) / (_scalar_like(1, x) - x * x),)


def _partials_sinh(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    _ = ans, rest, attrs
    return (xp.cosh(x),)


def _partials_cosh(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    _ = ans, rest, attrs
    return (xp.sinh(x),)


def _partials_cbrt(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    _ = x, rest, attrs
    denom = _scalar_like(3.0, ans) * ans * ans
    return (
        xp.where(
            denom != _scalar_like(0, denom),
            _scalar_like(1.0, denom) / denom,
            xp.zeros_like(denom),
        ),
    )


def _partials_log1p(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    _ = ans, rest, attrs
    return (_scalar_like(1.0, x) / (_scalar_like(1, x) + x),)


def _partials_log2(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    _ = ans, rest, attrs
    return (_scalar_like(1.0, x) / (x * _scalar_like(math.log(2.0), x)),)
