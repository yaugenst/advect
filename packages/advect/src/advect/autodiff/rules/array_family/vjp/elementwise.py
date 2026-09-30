# VJP signatures mirror NumPy op contracts
"""Explicit, traceable elementwise transpose rules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from advect.autodiff.rules.array_family._backend_runtime import _scalar_like, xp
from advect.autodiff.rules.array_family._transpose_utils import (
    _conjugate_if_complex,
    _dtype_is_complex,
    _dtype_of,
    dtype_is_inexact,
)
from advect.autodiff.rules.array_family.jvp.common import (
    _asarray_preserving_trace,
    scale_by_constant,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from advect.autodiff.rules.array_family.jvp.elementwise_partials import (
        ElementwisePartials,
    )


def make_diagonal_vjp_from_partials(
    entry: ElementwisePartials,
) -> Callable[..., tuple[Any | None, ...]]:
    """Transpose ``t -> sum_i p_i * t_i`` in closed form, ``g -> conj(p_i) * g``.

    The partials are those of the JVP, so the two rules share one formula and
    stay adjoint. Only the partials of active inputs are evaluated. The VJP
    binding casts each contribution into its input's dtype.
    """
    arity = len(entry.partials)

    def vjp_for_input_indices(
        ans: Any,
        *inputs: Any,
        g: Any,
        active_input_indices: Iterable[int],
        **attrs: Any,
    ) -> tuple[Any | None, ...]:
        del attrs
        cotangents: list[Any | None] = [None] * arity
        at = entry.bind(ans, inputs)
        for index in active_input_indices:
            local = at(index)
            if local is None:
                continue
            cotangents[index] = (
                scale_by_constant(g, local, g)
                if isinstance(local, (int, float))
                else xp.multiply(g, _conjugate_if_complex(local))
            )
        return tuple(cotangents)

    def vjp(ans: Any, *inputs: Any, g: Any, **attrs: Any) -> tuple[Any | None, ...]:
        return vjp_for_input_indices(
            ans,
            *inputs,
            g=g,
            active_input_indices=range(arity),
            **attrs,
        )

    cast("Any", vjp).__advect_vjp_for_input_indices__ = vjp_for_input_indices
    return vjp


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
        return (xp.zeros_like(_asarray_preserving_trace(x)),)
    magnitude = xp.abs(x)
    zero = xp.zeros_like(magnitude)
    safe = xp.where(magnitude == zero, xp.ones_like(magnitude), magnitude)
    magnitude_cotangent = xp.real(xp.multiply(_conjugate_if_complex(x), g)) / safe
    result = xp.divide(g, safe) - xp.multiply(x, magnitude_cotangent) / (safe * safe)
    return (xp.where(magnitude == zero, xp.zeros_like(result), result),)


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
    """Pass the cotangent back; the VJP binding casts it to the source dtype.

    The answer is not retained. An Array API ``asarray`` without a dtype
    records a cast that keeps the source dtype. A cast into integers or
    booleans passes no cotangent: the source receives zeros of its own dtype,
    which an integer cotangent must not promote.
    """
    _ = ans, rest
    target_dtype = attrs.get("dtype")
    if not dtype_is_inexact(_dtype_of(x) if target_dtype is None else target_dtype):
        return (xp.zeros_like(x),)
    return (g,)


def _vjp_real(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Pass the real cotangent back; the VJP binding embeds it in the input's dtype."""
    _ = ans, inputs, attrs
    return (g,)


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
    if _dtype_is_complex(_dtype_of(x)):
        return (_scalar_like(1j, g) * g,)
    return (xp.zeros_like(x),)


def _vjp_absolute(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Real-adjoint VJP for numpy.absolute."""
    _ = ans, rest, attrs
    if not _dtype_is_complex(_dtype_of(x)):
        # A real sign is locally constant; g * x / |x| would make nested
        # derivatives cancel two O(1/|x|) terms near zero.
        return (xp.multiply(g, xp.sign(_asarray_preserving_trace(x))),)
    magnitude = xp.abs(x)
    zero = xp.zeros_like(magnitude)
    safe = xp.where(magnitude == zero, xp.ones_like(magnitude), magnitude)
    dx = xp.multiply(g, x) / safe
    return (xp.where(magnitude == zero, xp.zeros_like(dx), dx),)
