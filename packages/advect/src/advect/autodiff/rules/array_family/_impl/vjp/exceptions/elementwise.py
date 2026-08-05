# VJP signatures mirror NumPy op contracts
"""Explicit, traceable elementwise transpose rules."""

from __future__ import annotations

from typing import Any, cast

from advect.autodiff.rules.array_family._backend_runtime import _scalar_like, xp
from advect.autodiff.rules.array_family._impl.jvp.elementwise_partials_a import (
    _partials_divide,
)
from advect.autodiff.rules.array_family._transpose_utils import dtype_is_inexact


def _dtype_is_complex(dtype: object) -> bool:
    kind = getattr(dtype, "kind", None)
    return kind == "c" if kind is not None else "complex" in str(dtype).lower()


def _conjugate_if_complex(value: xp.ndarray) -> xp.ndarray:
    if isinstance(value, complex):
        return cast("xp.ndarray", value.conjugate())
    return xp.conj(value) if _dtype_is_complex(getattr(value, "dtype", None)) else value


def _astype(value: xp.ndarray, dtype: Any) -> xp.ndarray:
    """Normalize provider scalars before the standard array-only cast."""
    source = value if callable(getattr(value, "_advect_snapshot", None)) else xp.asarray(value)
    return xp.astype(source, dtype, copy=False)


def _cotangent_like(primal: xp.ndarray, cotangent: xp.ndarray) -> xp.ndarray:
    """Keep a local contribution in the primal's real tangent space."""
    primal_dtype = getattr(primal, "dtype", None)
    if primal_dtype is None:
        return cotangent
    cotangent_dtype = getattr(cotangent, "dtype", None)
    if not _dtype_is_complex(primal_dtype) and _dtype_is_complex(cotangent_dtype):
        cotangent = xp.real(cotangent)
        cotangent_dtype = getattr(cotangent, "dtype", None)
    if cotangent_dtype == primal_dtype:
        return cotangent
    return _astype(cotangent, primal_dtype)


def _same_real_dtype(*values: object) -> bool:
    dtype = getattr(values[0], "dtype", None)
    return (
        dtype is not None
        and not _dtype_is_complex(dtype)
        and all(getattr(value, "dtype", None) == dtype for value in values[1:])
    )


def _vjp_add(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray]:
    """Transpose addition directly."""
    _ = ans, inputs, attrs
    return g, g


def _vjp_subtract(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray]:
    """Transpose subtraction directly."""
    _ = ans, inputs, attrs
    return g, -g


def _vjp_multiply(
    ans: xp.ndarray,
    x: xp.ndarray,
    y: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray]:
    """Transpose multiplication under Advect's real inner product."""
    _ = ans, rest, attrs
    if _same_real_dtype(x, y, g):
        return g * y, g * x
    return (
        _cotangent_like(x, g * _conjugate_if_complex(y)),
        _cotangent_like(y, g * _conjugate_if_complex(x)),
    )


def _vjp_divide(
    ans: xp.ndarray,
    x: xp.ndarray,
    y: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray]:
    """Transpose division under Advect's real inner product."""
    dx, dy = _partials_divide(ans, x, y, *rest, **attrs)
    if _same_real_dtype(x, y, g):
        return g * dx, g * dy
    return (
        _cotangent_like(x, g * _conjugate_if_complex(dx)),
        _cotangent_like(y, g * _conjugate_if_complex(dy)),
    )


def _vjp_power(
    ans: xp.ndarray,
    x: xp.ndarray,
    y: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray | None]:
    """Transpose power using its analytic local derivatives."""
    dx, dy = _select_vjp_power(
        ans,
        x,
        y,
        *rest,
        g=g,
        active_input_indices=(0, 1),
        **attrs,
    )
    return cast("xp.ndarray", dx), dy


def _select_vjp_power(
    ans: xp.ndarray,
    x: xp.ndarray,
    y: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    active_input_indices: tuple[int, ...],
    **attrs: Any,
) -> tuple[xp.ndarray | None, xp.ndarray | None]:
    """Evaluate only power partials that can reach a selected input."""
    _ = rest, attrs
    active = frozenset(active_input_indices)
    dx = None
    if 0 in active:
        if isinstance(y, (bool, int, float, complex)):
            dx_scale = (
                xp.zeros_like(x)
                if y == 0
                else cast(
                    "xp.ndarray",
                    _scalar_like(y, x) * x ** _scalar_like(y - 1, x),
                )
            )
        else:
            zero_exponent = y == xp.zeros_like(y)
            safe_derivative_base = xp.where(zero_exponent, xp.ones_like(x), x)
            dx_scale = xp.where(
                zero_exponent,
                xp.zeros_like(ans),
                y * safe_derivative_base ** (y - xp.ones_like(y)),
            )
        dx = _cotangent_like(x, g * _conjugate_if_complex(dx_scale))

    dy = None
    if 1 in active and not isinstance(y, (bool, int, float, complex)):
        dy_scale = ans * xp.log(x)
        dy = _cotangent_like(y, g * _conjugate_if_complex(dy_scale))
    return dx, dy


cast("Any", _vjp_power).__advect_vjp_for_input_indices__ = _select_vjp_power


def _vjp_sin(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Transpose sine under Advect's real inner product."""
    _ = ans, rest, attrs
    if _same_real_dtype(x, g):
        return (g * xp.cos(x),)
    return (_cotangent_like(x, g * _conjugate_if_complex(xp.cos(x))),)


def _vjp_cos(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Transpose cosine under Advect's real inner product."""
    _ = ans, rest, attrs
    return (_cotangent_like(x, -g * _conjugate_if_complex(xp.sin(x))),)


def _vjp_exp(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Transpose exponential under Advect's real inner product."""
    _ = inputs, attrs
    return (g * _conjugate_if_complex(ans),)


def _vjp_ldexp(
    ans: xp.ndarray,
    value: xp.ndarray,
    exponent: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray, None]:
    """Transpose exact power-of-two scaling without an intermediate factor."""
    _ = ans, value, rest, attrs
    return xp.ldexp(g, exponent), None


def _vjp_zero(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray, ...]:
    """Transpose an explicitly zero local derivative."""
    _ = ans, g, attrs
    return tuple(xp.zeros_like(value) for value in inputs)


def _vjp_sign(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Transpose real sign or complex unit-phase sign away from zero."""
    _ = ans, rest, attrs
    if not _dtype_is_complex(getattr(x, "dtype", None)):
        return (xp.zeros_like(x),)
    magnitude = xp.abs(x)
    zero = xp.zeros_like(magnitude)
    safe = xp.where(magnitude == zero, xp.ones_like(magnitude), magnitude)
    magnitude_cotangent = xp.real(_conjugate_if_complex(x) * g) / safe
    result = g / safe - x * magnitude_cotangent / (safe * safe)
    return (_cotangent_like(x, xp.where(magnitude == zero, xp.zeros_like(result), result)),)


def _zero_like_input(
    x: xp.ndarray,
    g: xp.ndarray,
) -> xp.ndarray:
    return xp.zeros_like(x, dtype=xp.result_type(g, x))


def _vjp_identity(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Transpose the one-input identity map without tracing another op."""
    _ = ans, inputs, attrs
    return (g,)


def _vjp_negative(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Transpose negation, a self-adjoint real-linear map."""
    _ = ans, inputs, attrs
    return (-g,)


def _vjp_conjugate(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Transpose conjugation under Advect's real inner product."""
    _ = ans, inputs, attrs
    return (xp.conj(g),)


def _vjp_astype(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Cast the cotangent back into the source tangent space."""
    _ = rest
    target_dtype = attrs.get("dtype", getattr(ans, "dtype", None))
    if not dtype_is_inexact(target_dtype):
        return (_zero_like_input(x, g),)
    return (_cotangent_like(x, g),)


def _vjp_real(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Embed a real cotangent into the input's real tangent space."""
    _ = ans, rest, attrs
    return (_astype(g, x.dtype),)


def _vjp_where(
    ans: xp.ndarray,
    condition: xp.ndarray,
    x: xp.ndarray,
    y: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[None, xp.ndarray, xp.ndarray]:
    """Route the cotangent through the selected branch only."""
    _ = ans, x, y, rest, attrs
    zero = xp.zeros_like(g)
    return (None, xp.where(condition, g, zero), xp.where(condition, zero, g))


def _vjp_imag(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """VJP for numpy.imag."""
    _ = ans, rest, attrs
    if xp.iscomplexobj(x):
        return (_astype(_scalar_like(1j, x) * g, x.dtype),)
    return (_zero_like_input(x, g),)


def _vjp_absolute(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Real-adjoint VJP for numpy.absolute."""
    _ = ans, rest, attrs
    magnitude = xp.abs(x)
    zero = xp.zeros_like(magnitude)
    safe = xp.where(magnitude == zero, xp.ones_like(magnitude), magnitude)
    dx = g * x / safe
    dx = xp.where(magnitude == zero, xp.zeros_like(dx), dx)
    return (_astype(dx, x.dtype),)
