"""Explicit transposes for matrix decompositions."""

from __future__ import annotations

from typing import Any, cast

from advect.autodiff.rules.array_family._backend_runtime import (
    _scalar_like,
    xp,
)
from advect.autodiff.rules.array_family._transpose_utils import (
    _lower_triangular_halfdiag,
    _right_solve,
    _uses_standard_linalg_contract,
)
from advect.autodiff.rules.array_family.vjp.linalg.common import (
    _broadcast_eye,
    _dtype_of,
    _h,
    _hermitian_triangle_adjoint,
    _merge_multioutput_cotangent,
    _qr_skew_pullback,
    _shape_of,
)

_QR_OUTPUT_COUNT = 2
_SVD_OUTPUT_COUNT = 3


def _vjp_cholesky(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Transpose either triangular Cholesky differential."""
    _ = x, rest
    upper = attrs.get("upper", False)
    if type(upper) is not bool:
        msg = "linalg.cholesky VJP expects upper to be a bool"
        raise TypeError(msg)
    factor = _h(ans) if upper else ans
    factor_cotangent = _h(g) if upper else g
    projected = _lower_triangular_halfdiag(_h(factor) @ factor_cotangent)
    natural = xp.linalg.solve(_h(factor), _right_solve(factor, projected))
    natural = (natural + _h(natural)) / _scalar_like(2.0, natural)
    return (_hermitian_triangle_adjoint(natural, uplo="U" if upper else "L"),)


def _vjp_pinv(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray | None, ...]:
    """Transpose the constant-rank Moore-Penrose inverse differential."""
    _ = attrs
    rows, columns = _shape_of(x)[-2:]
    batch_ndim = len(_shape_of(x)) - 2
    result_dtype = _dtype_of(ans)
    projection_rows = x @ ans
    projection_columns = ans @ x
    identity_rows = xp.zeros_like(projection_rows) + _broadcast_eye(
        n=rows,
        batch_ndim=batch_ndim,
        dtype=xp.dtype(result_dtype),
    )
    identity_columns = xp.zeros_like(projection_columns) + _broadcast_eye(
        n=columns,
        batch_ndim=batch_ndim,
        dtype=xp.dtype(result_dtype),
    )

    term1 = -(ans @ _h(g) @ ans)
    term2 = (ans @ _h(ans) @ g) @ (identity_rows - projection_rows)
    term3 = (identity_columns - projection_columns) @ (g @ _h(ans) @ ans)
    input_cotangent = _h(term1 + term2 + term3)
    if not xp.iscomplexobj(x):
        input_cotangent = xp.real(input_cotangent)
    return (input_cotangent, *(None for _ in rest))


def _vjp_qr(
    ans: tuple[xp.ndarray, xp.ndarray],
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: tuple[xp.ndarray | None, ...],
    mode: str = "reduced",
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Transpose full-rank reduced QR factorizations."""
    _ = rest, attrs
    q, r = ans
    rows, columns = _shape_of(x)[-2:]
    if mode not in {"complete", "reduced"}:
        msg = f"linalg.qr derivatives do not support mode={mode!r}"
        raise NotImplementedError(msg)
    if mode == "complete" and rows > columns:
        msg = (
            "linalg.qr derivatives do not define the provider-dependent null-space "
            "columns returned by mode='complete' for tall matrices; use mode='reduced'"
        )
        raise NotImplementedError(msg)

    gq_value, gr_value = _merge_multioutput_cotangent(
        g,
        output_count=_QR_OUTPUT_COUNT,
        op_name="linalg.qr",
    )
    gq = xp.zeros_like(q) if gq_value is None else gq_value
    gr = xp.zeros_like(r) if gr_value is None else gr_value
    batch_shape = _shape_of(x)[:-2]

    if rows < columns:
        leading_r = r[..., :, :rows]
        bar_a = q @ gr
        bar_omega = _h(q) @ gq - gr @ _h(r)
        bar_local = _qr_skew_pullback(
            bar_omega,
            batch_dims=batch_shape,
            n=rows,
        )
        bar_tangent_r_inverse = q @ bar_local
        bar_leading = _right_solve(_h(leading_r), bar_tangent_r_inverse)
        trailing_zeros = xp.zeros_like(bar_a[..., :, rows:])
        leading_contribution = xp.concatenate(
            (bar_leading, trailing_zeros),
            axis=-1,
        )
        input_cotangent = bar_a + leading_contribution
        if not xp.iscomplexobj(x):
            input_cotangent = xp.real(input_cotangent)
        return (cast("xp.ndarray", input_cotangent),)

    bar_qt = xp.zeros_like(r)
    bar_do = xp.zeros_like(r)
    bar_dx_rinv = xp.zeros_like(q)

    bar_a = gr @ _h(r)
    bar_qt = bar_qt + bar_a
    bar_do = bar_do - bar_a
    bar_dx_rinv = bar_dx_rinv + gq
    bar_b = _h(q) @ gq
    bar_do = bar_do + bar_b
    bar_qt = bar_qt - bar_b
    bar_qt = bar_qt + _qr_skew_pullback(
        bar_do,
        batch_dims=batch_shape,
        n=columns,
    )
    bar_dx_rinv = bar_dx_rinv + q @ bar_qt
    input_cotangent = _right_solve(_h(r), bar_dx_rinv)
    if not xp.iscomplexobj(x):
        input_cotangent = xp.real(input_cotangent)
    return (input_cotangent,)


def _vjp_svdvals(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Transpose the singular-value differential."""
    _ = ans, rest, attrs
    if _uses_standard_linalg_contract():
        u, singular_values, vh = xp.linalg.svd(x, full_matrices=False)
    else:
        u, singular_values, vh = xp.linalg.svd(
            x,
            full_matrices=False,
            compute_uv=True,
            hermitian=False,
        )
    size = _shape_of(singular_values)[-1]
    diagonal = (
        _broadcast_eye(
            n=size,
            batch_ndim=len(_shape_of(x)) - 2,
            dtype=_dtype_of(u),
        )
        * g[..., :, None]
    )
    return (cast("xp.ndarray", u @ diagonal @ vh),)


def _vjp_svd(
    ans: tuple[xp.ndarray, xp.ndarray, xp.ndarray],
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: tuple[xp.ndarray | None, xp.ndarray | None, xp.ndarray | None],
    full_matrices: bool = True,
    compute_uv: bool = True,
    hermitian: bool = False,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Transpose a reduced SVD away from repeated or zero singular values."""
    _ = rest, attrs
    merged_g = _merge_multioutput_cotangent(
        g,
        output_count=_SVD_OUTPUT_COUNT,
        op_name="numpy.linalg.svd",
    )
    if not compute_uv:
        msg = "numpy.linalg.svd VJP expects compute_uv=True; use svdvals otherwise"
        raise NotImplementedError(msg)
    if hermitian:
        msg = "numpy.linalg.svd VJP currently requires hermitian=False"
        raise NotImplementedError(msg)

    u, singular_values, vh = ans
    gu_in, gs_in, gvh_in = merged_g
    m, n = _shape_of(x)[-2:]
    size = _shape_of(singular_values)[-1]

    if full_matrices and m != n and (gu_in is not None or gvh_in is not None):
        msg = (
            "numpy.linalg.svd VJP for rectangular matrices requires "
            "full_matrices=False when differentiating singular vectors because the "
            "completed null-space basis has no unique derivative"
        )
        raise NotImplementedError(msg)

    u = u[..., :, :size]
    vh = vh[..., :size, :]
    gu = xp.zeros_like(u) if gu_in is None else gu_in[..., :, :size]
    gs = xp.zeros_like(singular_values) if gs_in is None else gs_in
    gvh = xp.zeros_like(vh) if gvh_in is None else gvh_in[..., :size, :]

    u_h = _h(u)
    v = _h(vh)
    gv = _h(gvh)
    utgu = u_h @ gu
    vtgv = vh @ gv

    eye = _broadcast_eye(
        n=size,
        batch_ndim=len(_shape_of(x)) - 2,
        dtype=_dtype_of(u),
    )
    off_diagonal = xp.ones_like(eye) - eye
    squared = singular_values * singular_values
    denominator = squared[..., None, :] - squared[..., :, None]
    inverse_gaps = off_diagonal / (denominator + eye)

    core = (
        (inverse_gaps * (utgu - _h(utgu))) * singular_values[..., None, :]
        + eye * gs[..., :, None]
        + singular_values[..., :, None] * (inverse_gaps * (vtgv - _h(vtgv)))
    )

    if xp.issubdtype(_dtype_of(u), xp.complexfloating):
        gauge = xp.imag(xp.diagonal(utgu, axis1=-2, axis2=-1)) / singular_values
        core = core + _scalar_like(1j, core) * eye * gauge[..., :, None]

    grad = u @ core @ vh
    if m < n:
        eye_n = _broadcast_eye(
            n=n,
            batch_ndim=len(_shape_of(x)) - 2,
            dtype=_dtype_of(v),
        )
        grad = grad + (u / singular_values[..., None, :]) @ _h(gv) @ (eye_n - v @ vh)
    elif m > n:
        eye_m = _broadcast_eye(
            n=m,
            batch_ndim=len(_shape_of(x)) - 2,
            dtype=_dtype_of(u),
        )
        grad = grad + (eye_m - u @ u_h) @ gu @ _h(v / singular_values[..., None, :])

    return (cast("xp.ndarray", grad),)
