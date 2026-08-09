"""Multi Input JVP rules."""

from __future__ import annotations

from typing import Any, cast

from advect.autodiff.rules.array_family._backend_runtime import xp
from advect.autodiff.rules.array_family._signal import native_signal_product
from advect.autodiff.rules.array_family.jvp.common import (
    _asarray_preserving_trace,
    _coerce_tangent_or_zeros,
    _infer_tangent_dtype,
    _normalize_output_tangent,
    _validate_tangent_arity,
    _zeros_output_tangent,
)

_MATMUL_INPUT_ARITY = 2


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
    return cast(
        "xp.ndarray[Any, Any]",
        _normalize_output_tangent(ans, tangents, xp.concatenate(parts, axis=axis)),
    )


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
    return cast(
        "xp.ndarray[Any, Any]",
        _normalize_output_tangent(ans, tangents, xp.stack(parts, axis=axis)),
    )


def _signal_binary_jvp(
    ans: xp.ndarray,
    left: xp.ndarray,
    right: xp.ndarray,
    *,
    tangents: tuple[xp.ndarray | None, ...],
    mode: str,
    correlate: bool,
) -> xp.ndarray:
    """Differentiate a bilinear one-dimensional signal operation."""
    left_tangent = tangents[0] if tangents else None
    right_tangent = tangents[1] if len(tangents) > 1 else None
    if left_tangent is None and right_tangent is None:
        return _zeros_output_tangent(ans, tangents)

    result: Any | None = None
    if left_tangent is not None:
        result = native_signal_product(
            left_tangent,
            right,
            mode=mode,
            correlate=correlate,
        )
    if right_tangent is not None:
        contribution = native_signal_product(
            left,
            right_tangent,
            mode=mode,
            correlate=correlate,
        )
        result = contribution if result is None else result + contribution
    return cast(
        "xp.ndarray[Any, Any]",
        _normalize_output_tangent(ans, tangents, result),
    )


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
    return _signal_binary_jvp(
        ans,
        left,
        right,
        tangents=tangents,
        mode=mode,
        correlate=False,
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
    return _signal_binary_jvp(
        ans,
        left,
        right,
        tangents=tangents,
        mode=mode,
        correlate=True,
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
    if len(tangents) < _MATMUL_INPUT_ARITY:
        msg = (
            "numpy.matmul JVP tangent arity mismatch: "
            f"expected {_MATMUL_INPUT_ARITY}, got {len(tangents)}"
        )
        raise RuntimeError(msg)
    dx, dy = tangents[:_MATMUL_INPUT_ARITY]
    tangent: Any | None = None
    if dx is not None:
        tangent = xp.matmul(dx, y)
    if dy is not None:
        contribution = xp.matmul(x, dy)
        tangent = contribution if tangent is None else tangent + contribution
    if tangent is not None:
        tangent = _normalize_output_tangent(ans, tangents, tangent)
    return cast(
        "xp.ndarray[Any, Any]",
        _zeros_output_tangent(ans, tangents) if tangent is None else tangent,
    )


def _vector_matrix_product_jvp(
    ans: xp.ndarray,
    left: xp.ndarray,
    right: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    conjugate_left: bool,
    **attrs: Any,
) -> xp.ndarray:
    _ = rest, attrs
    if len(tangents) < _MATMUL_INPUT_ARITY:
        msg = (
            "vector-matrix JVP tangent arity mismatch: "
            f"expected {_MATMUL_INPUT_ARITY}, got {len(tangents)}"
        )
        raise RuntimeError(msg)
    d_left, d_right = tangents[:_MATMUL_INPUT_ARITY]
    primal_left = xp.conjugate(left) if conjugate_left else left
    tangent: Any | None = None
    if d_left is not None:
        tangent_left = xp.conjugate(d_left) if conjugate_left else d_left
        tangent = xp.matmul(tangent_left, right)
    if d_right is not None:
        contribution = xp.matmul(primal_left, d_right)
        tangent = contribution if tangent is None else tangent + contribution
    if tangent is not None:
        tangent = _normalize_output_tangent(ans, tangents, tangent)
    return cast(
        "xp.ndarray[Any, Any]",
        _zeros_output_tangent(ans, tangents) if tangent is None else tangent,
    )


def _jvp_matvec(
    ans: xp.ndarray,
    matrix: xp.ndarray,
    vector: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    return _vector_matrix_product_jvp(
        ans,
        matrix,
        vector,
        *rest,
        tangents=tangents,
        conjugate_left=False,
        **attrs,
    )


def _jvp_vecmat(
    ans: xp.ndarray,
    vector: xp.ndarray,
    matrix: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    return _vector_matrix_product_jvp(
        ans,
        vector,
        matrix,
        *rest,
        tangents=tangents,
        conjugate_left=True,
        **attrs,
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
    _ = ans, rest, attrs
    dx = tangents[0] if len(tangents) > 0 else None
    dy = tangents[1] if len(tangents) > 1 else None

    if dx is None and dy is None:
        return _zeros_output_tangent(ans, tangents)

    if dx is None:
        dy_arr = _asarray_preserving_trace(cast("xp.ndarray[Any, Any]", dy))
        return cast("xp.ndarray[Any, Any]", xp.dot(x, dy_arr))
    if dy is None:
        return cast("xp.ndarray[Any, Any]", xp.dot(_asarray_preserving_trace(dx), y))
    return cast(
        "xp.ndarray[Any, Any]",
        xp.dot(_asarray_preserving_trace(dx), y) + xp.dot(x, _asarray_preserving_trace(dy)),
    )


def _jvp_inner(
    ans: xp.ndarray,
    a: xp.ndarray,
    b: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.inner."""
    _ = ans, rest, attrs
    da = tangents[0] if len(tangents) > 0 else None
    db = tangents[1] if len(tangents) > 1 else None

    if da is None and db is None:
        return _zeros_output_tangent(ans, tangents)

    if da is None:
        db_arr = cast("xp.ndarray[Any, Any]", db)
        return cast("xp.ndarray[Any, Any]", xp.inner(a, db_arr))
    if db is None:
        return cast("xp.ndarray[Any, Any]", xp.inner(da, b))
    return cast("xp.ndarray[Any, Any]", xp.inner(da, b) + xp.inner(a, db))


def _jvp_outer(
    ans: xp.ndarray,
    a: xp.ndarray,
    b: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    """JVP for numpy.outer."""
    _ = ans, rest, attrs
    da = tangents[0] if len(tangents) > 0 else None
    db = tangents[1] if len(tangents) > 1 else None

    if da is None and db is None:
        return _zeros_output_tangent(ans, tangents)

    if da is None:
        db_arr = cast("xp.ndarray[Any, Any]", db)
        return xp.outer(a, db_arr)
    if db is None:
        return xp.outer(da, b)
    return cast("xp.ndarray[Any, Any]", xp.outer(da, b) + xp.outer(a, db))


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
    _ = ans, rest, attrs
    da = tangents[0] if len(tangents) > 0 else None
    db = tangents[1] if len(tangents) > 1 else None

    if da is None and db is None:
        return _zeros_output_tangent(ans, tangents)

    if da is None:
        db_arr = _asarray_preserving_trace(cast("xp.ndarray[Any, Any]", db))
        return xp.tensordot(a, db_arr, axes=axes)
    if db is None:
        return xp.tensordot(_asarray_preserving_trace(da), b, axes=axes)
    return cast(
        "xp.ndarray[Any, Any]",
        xp.tensordot(_asarray_preserving_trace(da), b, axes=axes)
        + xp.tensordot(a, _asarray_preserving_trace(db), axes=axes),
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
    _ = ans, rest, attrs
    da = tangents[0] if len(tangents) > 0 else None
    db = tangents[1] if len(tangents) > 1 else None
    if da is None and db is None:
        return _zeros_output_tangent(ans, tangents)
    out = _zeros_output_tangent(ans, tangents)
    if da is not None:
        out = out + xp.cross(da, b, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis)
    if db is not None:
        out = out + xp.cross(a, db, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis)
    return out


def _jvp_kron(
    ans: xp.ndarray,
    a: xp.ndarray,
    b: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, rest, attrs
    da = tangents[0] if len(tangents) > 0 else None
    db = tangents[1] if len(tangents) > 1 else None
    if da is None and db is None:
        return _zeros_output_tangent(ans, tangents)
    out = _zeros_output_tangent(ans, tangents)
    if da is not None:
        out = out + xp.kron(da, b)
    if db is not None:
        out = out + xp.kron(a, db)
    return out


def _jvp_einsum(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    subscripts: str,
    optimize: bool | str | list[Any] | tuple[Any, ...] | None = None,
    **attrs: Any,
) -> xp.ndarray:
    _ = ans, attrs
    _validate_tangent_arity(op_name="numpy.einsum", inputs=inputs, tangents=tangents)
    if all(tangent is None for tangent in tangents):
        return _zeros_output_tangent(ans, tangents)
    optimize_arg = False if optimize is None else optimize
    out = _zeros_output_tangent(ans, tangents)
    for index, tangent in enumerate(tangents):
        if tangent is None:
            continue
        args = list(inputs)
        args[index] = tangent
        out = out + xp.einsum(subscripts, *args, optimize=optimize_arg)
    return out


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
    _ = ans, rest, attrs
    d_start = tangents[0] if len(tangents) > 0 else None
    d_stop = tangents[1] if len(tangents) > 1 else None
    if d_start is None and d_stop is None:
        return _zeros_output_tangent(ans, tangents)
    out = _zeros_output_tangent(ans, tangents)
    if d_start is not None:
        out = out + xp.linspace(
            _asarray_preserving_trace(d_start),
            0.0,
            num=int(num),
            endpoint=bool(endpoint),
            axis=int(axis),
        )
    if d_stop is not None:
        out = out + xp.linspace(
            0.0,
            _asarray_preserving_trace(d_stop),
            num=int(num),
            endpoint=bool(endpoint),
            axis=int(axis),
        )
    return cast("xp.ndarray[Any, Any]", _asarray_preserving_trace(out))
