"""Multi Input JVP rules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from advect.autodiff.rules.array_family._backend_runtime import xp
from advect.autodiff.rules.array_family._signal import native_signal_product
from advect.autodiff.rules.array_family._transpose_utils import _shape_of
from advect.autodiff.rules.array_family.jvp.common import (
    _coerce_tangent_or_zeros,
    _infer_tangent_dtype,
    _normalize_output_tangent,
    _validate_tangent_arity,
    _zeros_output_tangent,
    product_rule,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_MATRIX_RANK = 2


def _jvp_concatenate(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axis: int | None = 0,
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.concatenate."""
    _ = attrs
    _validate_tangent_arity(op_name="numpy.concatenate", inputs=inputs, tangents=tangents)
    dtype = _infer_tangent_dtype(ans, tangents)
    parts = [
        _coerce_tangent_or_zeros(tangent, primal=inp, dtype=dtype)
        for inp, tangent in zip(inputs, tangents, strict=True)
    ]
    return _normalize_output_tangent(ans, tangents, xp.concatenate(parts, axis=axis))


def _jvp_stack(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axis: int = 0,
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.stack."""
    _ = attrs
    _validate_tangent_arity(op_name="numpy.stack", inputs=inputs, tangents=tangents)
    dtype = _infer_tangent_dtype(ans, tangents)
    parts = [
        _coerce_tangent_or_zeros(tangent, primal=inp, dtype=dtype)
        for inp, tangent in zip(inputs, tangents, strict=True)
    ]
    return _normalize_output_tangent(ans, tangents, xp.stack(parts, axis=axis))


def _jvp_convolve(
    ans: xp.ndarray,
    left: xp.ndarray,
    right: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    mode: str = "full",
    **attrs: Any,
) -> xp.ndarray:
    _ = rest, attrs
    return product_rule(
        ans,
        tangents,
        lambda d: native_signal_product(d, right, mode=mode, correlate=False),
        lambda d: native_signal_product(left, d, mode=mode, correlate=False),
    )


def _jvp_correlate(
    ans: xp.ndarray,
    left: xp.ndarray,
    right: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    mode: str = "valid",
    **attrs: Any,
) -> xp.ndarray:
    _ = rest, attrs
    return product_rule(
        ans,
        tangents,
        lambda d: native_signal_product(d, right, mode=mode, correlate=True),
        lambda d: native_signal_product(left, d, mode=mode, correlate=True),
    )


def _jvp_matmul(
    ans: xp.ndarray,
    x: xp.ndarray,
    y: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    """Apply the matrix-product rule directly."""
    _ = rest, attrs
    return product_rule(ans, tangents, lambda d: xp.matmul(d, y), lambda d: xp.matmul(x, d))


def _jvp_matvec(
    ans: xp.ndarray,
    matrix: xp.ndarray,
    vector: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    """Differentiate ``matrix @ vector`` over the gufunc's broadcast loop dimensions."""
    _ = rest, attrs
    return product_rule(
        ans,
        tangents,
        lambda d: xp.matmul(d, vector[..., None])[..., 0],
        lambda d: xp.matmul(matrix, d[..., None])[..., 0],
    )


def _jvp_vecmat(
    ans: xp.ndarray,
    vector: xp.ndarray,
    matrix: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    """Differentiate ``conj(vector) @ matrix``, which conjugates its vector."""
    _ = rest, attrs
    return product_rule(
        ans,
        tangents,
        lambda d: xp.matmul(xp.conjugate(d)[..., None, :], matrix)[..., 0, :],
        lambda d: xp.matmul(xp.conjugate(vector)[..., None, :], d)[..., 0, :],
    )


def _jvp_ldexp(
    ans: xp.ndarray,
    value: xp.ndarray,
    exponent: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    _ = value, rest, attrs
    tangent = tangents[0] if tangents else None
    contribution = None if tangent is None else xp.ldexp(tangent, exponent)
    return cast(
        "xp.ndarray[Any, Any]",
        _zeros_output_tangent(ans, tangents)
        if contribution is None
        else _normalize_output_tangent(ans, tangents, contribution),
    )


def _jvp_dot(
    ans: xp.ndarray,
    x: xp.ndarray,
    y: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.dot."""
    _ = rest, attrs
    return product_rule(ans, tangents, lambda d: _dot(d, y), lambda d: _dot(x, d))


def _dot(a: Any, b: Any) -> Any:
    """Contract as ``numpy.dot`` does through Array API functions, which lack dot."""
    a_rank, b_rank = len(_shape_of(a)), len(_shape_of(b))
    if not a_rank or not b_rank:
        return xp.multiply(a, b)
    if b_rank <= _MATRIX_RANK:
        # A vector or matrix right operand contracts, and batches, as matmul.
        return xp.matmul(a, b)
    return xp.tensordot(a, b, axes=((a_rank - 1,), (b_rank - 2,)))


def _jvp_inner(
    ans: xp.ndarray,
    a: xp.ndarray,
    b: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.inner."""
    _ = rest, attrs
    return product_rule(ans, tangents, lambda d: xp.inner(d, b), lambda d: xp.inner(a, d))


def _jvp_outer(
    ans: xp.ndarray,
    a: xp.ndarray,
    b: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.outer."""
    _ = rest, attrs
    return product_rule(ans, tangents, lambda d: xp.outer(d, b), lambda d: xp.outer(a, d))


def _jvp_tensordot(
    ans: xp.ndarray,
    a: xp.ndarray,
    b: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axes: int | tuple[Any, Any] = 2,
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.tensordot."""
    _ = rest, attrs
    return product_rule(
        ans,
        tangents,
        lambda d: xp.tensordot(d, b, axes=axes),
        lambda d: xp.tensordot(a, d, axes=axes),
    )


def _jvp_cross(
    ans: xp.ndarray,
    a: xp.ndarray,
    b: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int | None = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = rest, attrs
    axes = {"axisa": axisa, "axisb": axisb, "axisc": axisc, "axis": axis}
    return product_rule(
        ans, tangents, lambda d: xp.cross(d, b, **axes), lambda d: xp.cross(a, d, **axes)
    )


def _jvp_kron(
    ans: xp.ndarray,
    a: xp.ndarray,
    b: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    _ = rest, attrs
    return product_rule(ans, tangents, lambda d: xp.kron(d, b), lambda d: xp.kron(a, d))


def _jvp_einsum(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    subscripts: str,
    optimize: bool | str | list[Any] | tuple[Any, ...] | None = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = attrs
    _validate_tangent_arity(op_name="numpy.einsum", inputs=inputs, tangents=tangents)
    optimize_arg = False if optimize is None else optimize

    def term(index: int) -> Callable[[Any], Any]:
        return lambda d: xp.einsum(
            subscripts, *inputs[:index], d, *inputs[index + 1 :], optimize=optimize_arg
        )

    return product_rule(ans, tangents, *(term(index) for index in range(len(inputs))))


def _jvp_linspace(
    ans: xp.ndarray,
    start: xp.ndarray,
    stop: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    num: int = 50,
    endpoint: bool = True,
    axis: int = 0,
    **attrs: Any,
) -> xp.ndarray:
    _ = start, stop, rest, attrs
    grid = {"num": num, "endpoint": endpoint, "axis": axis}
    return product_rule(
        ans,
        tangents,
        lambda d: xp.linspace(d, 0.0, **grid),
        lambda d: xp.linspace(0.0, d, **grid),
    )
