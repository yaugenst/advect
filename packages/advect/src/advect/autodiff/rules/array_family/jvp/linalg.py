"""Linalg JVP rules."""

from __future__ import annotations

from typing import Any, cast

from advect.autodiff.rules.array_family._backend_runtime import (
    _array_constructor_like,
    _moveaxis,
    _scalar_like,
    xp,
)
from advect.autodiff.rules.array_family._transpose_utils import (
    _conjugate_transpose as _h,
    _diagonal_matrix as _diag_matrix,
    _lower_triangular_halfdiag,
    _normalize_uplo,
    _right_solve,
    _uses_standard_linalg_contract,
    zeros_output_tangent_structure as _zeros_output_tangent_structure,
)
from advect.autodiff.rules.array_family.jvp.common import (
    _MATRIX_AXIS_COUNT,
    _asarray_unwrapped,
    _astype_preserving_trace,
    _flatten_reduction_axes,
    _infer_tangent_dtype,
    _is_traced_leaf,
    _iscomplex_unwrapped,
    _ndim_unwrapped,
    _normalize_axis_tuple,
    _reshape_reduction_result,
    _shape_unwrapped,
    _zeros_output_tangent,
)


def _hermitian_from_triangle(x: xp.ndarray, *, uplo: str) -> xp.ndarray:
    """Materialize the Hermitian matrix represented by one stored triangle."""
    diag = xp.real(xp.diagonal(x, axis1=-2, axis2=-1))
    diag_matrix = _diag_matrix(diag, dtype=xp.dtype(x.dtype))
    if uplo == "L":
        strict = xp.tril(x, k=-1)
    elif uplo == "U":
        strict = xp.triu(x, k=1)
    else:
        msg = f"expected UPLO='L' or 'U', got {uplo!r}"
        raise ValueError(msg)
    return cast("xp.ndarray", strict + _h(strict) + diag_matrix)


def _jvp_linalg_cholesky(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    """Differentiate either triangular Cholesky factor on the Hermitian domain."""
    _ = x, rest
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)

    upper = attrs.get("upper", False)
    if type(upper) is not bool:
        msg = "linalg.cholesky JVP expects upper to be a bool"
        raise TypeError(msg)
    factor = _h(ans) if upper else ans
    tangent_h = _hermitian_from_triangle(tangent, uplo="U" if upper else "L")
    left = xp.linalg.solve(factor, tangent_h)
    middle = _right_solve(_h(factor), left)
    lower_out = factor @ _lower_triangular_halfdiag(middle)
    out = _h(lower_out) if upper else lower_out
    return cast(
        "xp.ndarray",
        _astype_preserving_trace(out, dtype=_asarray_unwrapped(ans).dtype),
    )


def _jvp_linalg_eigh(
    ans: tuple[xp.ndarray, xp.ndarray],
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    UPLO: str = "L",  # noqa: N803 - NumPy spells this keyword in uppercase.
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray]:
    """Differentiate a Hermitian eigendecomposition with a horizontal gauge."""
    _ = x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        zeros = _zeros_output_tangent_structure(ans, tangents)
        return cast("tuple[xp.ndarray, xp.ndarray]", zeros)

    eigenvalues, eigenvectors = ans
    uplo = _normalize_uplo(UPLO)
    tangent_h = _hermitian_from_triangle(tangent, uplo=uplo)
    local = _h(eigenvectors) @ tangent_h @ eigenvectors

    size = _shape_unwrapped(eigenvalues)[-1]
    eye = _array_constructor_like(
        local,
        "eye",
        size,
        dtype=xp.dtype(eigenvectors.dtype),
    )
    off_diagonal = xp.ones_like(eye) - eye
    gaps = eigenvalues[..., None, :] - eigenvalues[..., :, None]
    inverse_gaps = off_diagonal / (gaps + eye)

    d_eigenvalues = xp.real(xp.diagonal(local, axis1=-2, axis2=-1))
    d_eigenvectors = eigenvectors @ (inverse_gaps * local)
    return (
        cast(
            "xp.ndarray",
            _astype_preserving_trace(
                d_eigenvalues,
                dtype=_asarray_unwrapped(eigenvalues).dtype,
            ),
        ),
        cast(
            "xp.ndarray",
            _astype_preserving_trace(
                d_eigenvectors,
                dtype=_asarray_unwrapped(eigenvectors).dtype,
            ),
        ),
    )


def _jvp_linalg_eigvalsh(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    UPLO: str = "L",  # noqa: N803 - NumPy spells this keyword in uppercase.
    **attrs: Any,
) -> xp.ndarray:
    """Differentiate Hermitian eigenvalues without differentiating eigenvectors."""
    _ = rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)

    uplo = _normalize_uplo(UPLO)
    if _uses_standard_linalg_contract():
        _eigenvalues, eigenvectors = xp.linalg.eigh(x)
    else:
        _eigenvalues, eigenvectors = xp.linalg.eigh(x, UPLO=uplo)
    tangent_h = _hermitian_from_triangle(tangent, uplo=uplo)
    local = _h(eigenvectors) @ tangent_h @ eigenvectors
    out = xp.real(xp.diagonal(local, axis1=-2, axis2=-1))
    return cast(
        "xp.ndarray",
        _astype_preserving_trace(out, dtype=_asarray_unwrapped(ans).dtype),
    )


def _jvp_linalg_eig(
    ans: tuple[xp.ndarray, xp.ndarray],
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray]:
    """Differentiate NumPy's normalized, largest-component-real eigenvectors."""
    _ = rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        zeros = _zeros_output_tangent_structure(ans, tangents)
        return cast("tuple[xp.ndarray, xp.ndarray]", zeros)

    eigenvalues, eigenvectors = ans
    local = xp.linalg.inv(eigenvectors) @ tangent @ eigenvectors
    size = _shape_unwrapped(eigenvalues)[-1]
    eye = _array_constructor_like(
        local,
        "eye",
        size,
        dtype=xp.dtype(eigenvectors.dtype),
    )
    off_diagonal = xp.ones_like(eye) - eye
    gaps = eigenvalues[..., None, :] - eigenvalues[..., :, None]
    inverse_gaps = off_diagonal / (gaps + eye)

    d_eigenvalues = xp.diagonal(local, axis1=-2, axis2=-1)
    raw_eigenvectors = eigenvectors @ (inverse_gaps * local)
    column_overlap = xp.sum(
        xp.conjugate(eigenvectors) * raw_eigenvectors,
        axis=-2,
        keepdims=True,
    )
    pivot_indices = xp.argmax(xp.abs(eigenvectors), axis=-2)
    row_indices = xp.reshape(
        _array_constructor_like(local, "arange", size, dtype=xp.int64),
        (1,) * (_ndim_unwrapped(eigenvectors) - 2) + (size, 1),
    )
    pivot_mask = row_indices == xp.expand_dims(pivot_indices, axis=-2)
    pivot_values = xp.sum(
        xp.where(pivot_mask, eigenvectors, xp.zeros_like(eigenvectors)),
        axis=-2,
        keepdims=True,
    )
    raw_pivots = xp.sum(
        xp.where(pivot_mask, raw_eigenvectors, xp.zeros_like(raw_eigenvectors)),
        axis=-2,
        keepdims=True,
    )
    # NumPy normalizes each eigenvector and rotates the largest-magnitude
    # component onto the positive real axis. The real part below preserves
    # unit norm; the imaginary part differentiates that phase convention.
    correction = -xp.real(column_overlap) - _scalar_like(1j, column_overlap) * (
        xp.imag(raw_pivots) / xp.real(pivot_values)
    )
    d_eigenvectors = raw_eigenvectors + eigenvectors * correction
    return (
        cast(
            "xp.ndarray",
            _astype_preserving_trace(
                d_eigenvalues,
                dtype=_asarray_unwrapped(eigenvalues).dtype,
            ),
        ),
        cast(
            "xp.ndarray",
            _astype_preserving_trace(
                d_eigenvectors,
                dtype=_asarray_unwrapped(eigenvectors).dtype,
            ),
        ),
    )


def _jvp_linalg_eigvals(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    """Differentiate simple eigenvalues of a general square matrix."""
    _ = rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    _eigenvalues, eigenvectors = xp.linalg.eig(x)
    local = xp.linalg.inv(eigenvectors) @ tangent @ eigenvectors
    out = xp.diagonal(local, axis1=-2, axis2=-1)
    return cast(
        "xp.ndarray",
        _astype_preserving_trace(out, dtype=_asarray_unwrapped(ans).dtype),
    )


def _jvp_linalg_svdvals(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    """Differentiate singular values using a reduced SVD."""
    _ = rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)

    if _uses_standard_linalg_contract():
        u, _singular_values, vh = xp.linalg.svd(x, full_matrices=False)
    else:
        u, _singular_values, vh = xp.linalg.svd(
            x,
            full_matrices=False,
            compute_uv=True,
            hermitian=False,
        )
    local = _h(u) @ tangent @ _h(vh)
    out = xp.real(xp.diagonal(local, axis1=-2, axis2=-1))
    return cast(
        "xp.ndarray",
        _astype_preserving_trace(out, dtype=_asarray_unwrapped(ans).dtype),
    )


def _jvp_linalg_svd(
    ans: tuple[xp.ndarray, xp.ndarray, xp.ndarray],
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    full_matrices: bool = True,
    compute_uv: bool = True,
    hermitian: bool = False,
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
    """Differentiate a reduced SVD away from repeated or zero singular values."""
    _ = rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        zeros = _zeros_output_tangent_structure(ans, tangents)
        return cast("tuple[xp.ndarray, xp.ndarray, xp.ndarray]", zeros)
    if not compute_uv:
        msg = "numpy.linalg.svd JVP expects compute_uv=True; use svdvals otherwise"
        raise NotImplementedError(msg)
    if hermitian:
        msg = "numpy.linalg.svd JVP currently requires hermitian=False"
        raise NotImplementedError(msg)

    u, singular_values, vh = ans
    m, n = _shape_unwrapped(x)[-2:]
    if full_matrices and m != n:
        msg = (
            "numpy.linalg.svd JVP for rectangular matrices requires "
            "full_matrices=False because the derivative of the completed null-space "
            "basis is not uniquely defined"
        )
        raise NotImplementedError(msg)

    v = _h(vh)
    local = _h(u) @ tangent @ v
    local_h = _h(local)

    size = _shape_unwrapped(singular_values)[-1]
    dtype = xp.dtype(u.dtype)
    eye = _array_constructor_like(local, "eye", size, dtype=dtype)
    off_diagonal = xp.ones_like(eye) - eye
    squared = singular_values * singular_values
    denominator = squared[..., None, :] - squared[..., :, None]
    inverse_gaps = off_diagonal / (denominator + eye)

    s_rows = singular_values[..., :, None]
    s_columns = singular_values[..., None, :]
    omega_u = inverse_gaps * (local * s_columns + s_rows * local_h)
    omega_v = inverse_gaps * (s_rows * local + local_h * s_columns)

    if _iscomplex_unwrapped(u):
        gauge = (
            _scalar_like(1j, local)
            * xp.imag(xp.diagonal(local, axis1=-2, axis2=-1))
            / singular_values
        )
        omega_u = omega_u + _diag_matrix(gauge, dtype=dtype)

    d_u = u @ omega_u + (tangent @ v - u @ local) / s_columns
    d_v = v @ omega_v + (_h(tangent) @ u - v @ local_h) / s_columns
    d_s = xp.real(xp.diagonal(local, axis1=-2, axis2=-1))
    d_vh = _h(d_v)

    return (
        cast("xp.ndarray", _astype_preserving_trace(d_u, dtype=_asarray_unwrapped(u).dtype)),
        cast(
            "xp.ndarray",
            _astype_preserving_trace(
                d_s,
                dtype=_asarray_unwrapped(singular_values).dtype,
            ),
        ),
        cast("xp.ndarray", _astype_preserving_trace(d_vh, dtype=_asarray_unwrapped(vh).dtype)),
    )


def _jvp_linalg_slogdet(
    ans: tuple[xp.ndarray, xp.ndarray],
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray]:
    _ = rest, attrs
    tangent = tangents[0] if tangents else None
    sign, logabsdet = ans
    zero_sign = xp.zeros_like(sign, dtype=_infer_tangent_dtype(sign, tangents))
    if tangent is None:
        zero_logabs = _zeros_output_tangent(logabsdet, tangents)
        return (zero_sign, zero_logabs)
    solved = xp.linalg.solve(x, tangent)
    trace = xp.trace(solved, axis1=-2, axis2=-1)
    d_logabs = xp.real(trace)
    d_sign = (
        _scalar_like(1j, sign) * sign * xp.imag(trace) if _iscomplex_unwrapped(sign) else zero_sign
    )
    return (
        _astype_preserving_trace(d_sign, dtype=_asarray_unwrapped(sign).dtype),
        _astype_preserving_trace(d_logabs, dtype=_asarray_unwrapped(logabsdet).dtype),
    )


def _vector_norm_jvp(
    ans: xp.ndarray,
    x: xp.ndarray,
    tangent: xp.ndarray,
    *,
    axes: tuple[int, ...],
    keepdims: bool,
    ord_value: float,
) -> xp.ndarray:
    """Differentiate a real p-norm over one or more flattened axes."""
    x_shape = _shape_unwrapped(x)
    x_flat, _ = _flatten_reduction_axes(x, axes=axes)
    tangent_flat, _ = _flatten_reduction_axes(tangent, axes=axes)
    reduced_shape = _shape_unwrapped(x_flat)[:-1]

    if ord_value == 0:
        reduced = xp.sum(
            xp.zeros_like(
                _astype_preserving_trace(
                    tangent_flat,
                    dtype=_asarray_unwrapped(ans).dtype,
                )
            ),
            axis=-1,
        )
    else:
        magnitude = xp.abs(x_flat)
        zero_magnitude = xp.zeros_like(magnitude)
        directional_abs = xp.where(
            magnitude == zero_magnitude,
            zero_magnitude,
            xp.real(xp.conjugate(x_flat) * tangent_flat)
            / xp.where(magnitude == zero_magnitude, xp.ones_like(magnitude), magnitude),
        )
        if ord_value in (float("inf"), float("-inf")):
            winner = (
                xp.argmax(magnitude, axis=-1)
                if ord_value == float("inf")
                else xp.argmin(magnitude, axis=-1)
            )
            winner_mask = xp.equal(
                _array_constructor_like(
                    tangent_flat,
                    "arange",
                    _shape_unwrapped(x_flat)[-1],
                    dtype=xp.int64,
                ),
                winner[..., None],
            )
            reduced = xp.sum(
                xp.where(winner_mask, directional_abs, xp.zeros_like(directional_abs)),
                axis=-1,
            )
        else:
            answer_reduced = xp.reshape(ans, reduced_shape)
            weighted = xp.where(
                magnitude == zero_magnitude,
                zero_magnitude,
                magnitude ** _scalar_like(ord_value - 1.0, magnitude) * directional_abs,
            )
            numerator = xp.sum(weighted, axis=-1)
            zero_answer = xp.zeros_like(answer_reduced)
            denominator = xp.where(
                answer_reduced == zero_answer,
                xp.ones_like(answer_reduced),
                answer_reduced ** _scalar_like(ord_value - 1.0, answer_reduced),
            )
            reduced = xp.where(
                answer_reduced == zero_answer,
                xp.zeros_like(numerator),
                numerator / denominator,
            )

    out = _reshape_reduction_result(
        reduced,
        input_shape=x_shape,
        axes=axes,
        keepdims=keepdims,
    )
    return cast(
        "xp.ndarray",
        _astype_preserving_trace(out, dtype=_asarray_unwrapped(ans).dtype),
    )


def _matrix_norm_jvp(
    ans: xp.ndarray,
    x: xp.ndarray,
    tangent: xp.ndarray,
    *,
    axes: tuple[int, int],
    keepdims: bool,
    ord_value: str | float,
) -> xp.ndarray:
    """Differentiate all NumPy matrix norm orders away from ties and zeros."""
    x_shape = _shape_unwrapped(x)
    moved = _moveaxis(x, axes, (-2, -1))
    tangent_moved = _moveaxis(tangent, axes, (-2, -1))

    if ord_value == "fro":
        reduced = xp.real(xp.sum(xp.conjugate(moved) * tangent_moved, axis=(-2, -1)))
        answer_reduced = xp.reshape(
            ans,
            tuple(size for index, size in enumerate(x_shape) if index not in set(axes)),
        )
        zero_answer = xp.zeros_like(answer_reduced)
        reduced = xp.where(
            answer_reduced == zero_answer,
            xp.zeros_like(reduced),
            reduced
            / xp.where(answer_reduced == zero_answer, xp.ones_like(answer_reduced), answer_reduced),
        )
    elif ord_value in {"nuc", 2, -2}:
        if _uses_standard_linalg_contract():
            u, _singular_values, vh = xp.linalg.svd(moved, full_matrices=False)
        else:
            u, _singular_values, vh = xp.linalg.svd(
                moved,
                full_matrices=False,
                compute_uv=True,
                hermitian=False,
            )
        singular_tangent = xp.real(xp.diagonal(_h(u) @ tangent_moved @ _h(vh), axis1=-2, axis2=-1))
        if ord_value == "nuc":
            reduced = xp.sum(singular_tangent, axis=-1)
        elif ord_value == _MATRIX_AXIS_COUNT:
            reduced = singular_tangent[..., 0]
        else:
            reduced = singular_tangent[..., -1]
    elif ord_value in {1, -1, xp.inf, -xp.inf}:
        magnitude = xp.abs(moved)
        zero_magnitude = xp.zeros_like(magnitude)
        directional_abs = xp.where(
            magnitude == zero_magnitude,
            zero_magnitude,
            xp.real(xp.conjugate(moved) * tangent_moved)
            / xp.where(magnitude == zero_magnitude, xp.ones_like(magnitude), magnitude),
        )
        sum_axis = -2 if ord_value in {1, -1} else -1
        scores = xp.sum(magnitude, axis=sum_axis)
        score_tangents = xp.sum(directional_abs, axis=sum_axis)
        winner = (
            xp.argmax(scores, axis=-1) if ord_value in {1, xp.inf} else xp.argmin(scores, axis=-1)
        )
        winner_mask = xp.equal(
            _array_constructor_like(
                score_tangents,
                "arange",
                _shape_unwrapped(scores)[-1],
                dtype=xp.int64,
            ),
            winner[..., None],
        )
        reduced = xp.sum(
            xp.where(winner_mask, score_tangents, xp.zeros_like(score_tangents)),
            axis=-1,
        )
    else:
        msg = f"unsupported matrix norm order {ord_value!r}"
        raise ValueError(msg)

    out = _reshape_reduction_result(
        reduced,
        input_shape=x_shape,
        axes=axes,
        keepdims=keepdims,
    )
    return cast(
        "xp.ndarray",
        _astype_preserving_trace(out, dtype=_asarray_unwrapped(ans).dtype),
    )


def _jvp_linalg_norm(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    ord_value: str | float | None = None,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    **attrs: Any,
) -> xp.ndarray:
    if ord_value is None and "ord" in attrs:
        ord_value = cast("str | float | None", attrs["ord"])
    _ = rest
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)

    ndim = _ndim_unwrapped(x)
    if axis is None:
        if ord_value is None and ndim != _MATRIX_AXIS_COUNT:
            return _vector_norm_jvp(
                ans,
                x,
                tangent,
                axes=tuple(range(ndim)),
                keepdims=keepdims,
                ord_value=2.0,
            )
        if ndim == 1:
            return _vector_norm_jvp(
                ans,
                x,
                tangent,
                axes=(0,),
                keepdims=keepdims,
                ord_value=2.0 if ord_value is None else float(ord_value),
            )
        if ndim == _MATRIX_AXIS_COUNT:
            return _matrix_norm_jvp(
                ans,
                x,
                tangent,
                axes=(0, 1),
                keepdims=keepdims,
                ord_value="fro" if ord_value is None else ord_value,
            )
        msg = "numpy.linalg.norm with ord= and axis=None requires a vector or matrix"
        raise ValueError(msg)

    axes = _normalize_axis_tuple(axis, ndim=ndim)
    if len(axes) == 1:
        if isinstance(ord_value, str):
            msg = f"vector norms do not accept ord={ord_value!r}"
            raise ValueError(msg)
        return _vector_norm_jvp(
            ans,
            x,
            tangent,
            axes=axes,
            keepdims=keepdims,
            ord_value=2.0 if ord_value is None else float(ord_value),
        )
    if len(axes) == _MATRIX_AXIS_COUNT:
        return _matrix_norm_jvp(
            ans,
            x,
            tangent,
            axes=cast("tuple[int, int]", axes),
            keepdims=keepdims,
            ord_value="fro" if ord_value is None else ord_value,
        )
    msg = "numpy.linalg.norm axis must select one vector axis or two matrix axes"
    raise ValueError(msg)


def _jvp_linalg_matrix_norm(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    keepdims: bool = False,
    ord: str | float | None = "fro",  # noqa: A002 - Array API keyword.
    **attrs: Any,
) -> xp.ndarray:
    """Differentiate matrix norms over the final two axes."""
    _ = rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    ndim = _ndim_unwrapped(x)
    if ndim < _MATRIX_AXIS_COUNT:
        msg = "linalg.matrix_norm requires an input with at least two dimensions"
        raise ValueError(msg)
    return _matrix_norm_jvp(
        ans,
        x,
        tangent,
        axes=(ndim - 2, ndim - 1),
        keepdims=keepdims,
        ord_value="fro" if ord is None else ord,
    )


def _jvp_linalg_vector_norm(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    ord: float = 2,  # noqa: A002 - Array API keyword.
    **attrs: Any,
) -> xp.ndarray:
    """Differentiate vector p-norms, including flattened inputs."""
    _ = rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    ndim = _ndim_unwrapped(x)
    axes = tuple(range(ndim)) if axis is None else _normalize_axis_tuple(axis, ndim=ndim)
    return _vector_norm_jvp(
        ans,
        x,
        tangent,
        axes=axes,
        keepdims=keepdims,
        ord_value=float(ord),
    )


def _jvp_linalg_det(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    _ = rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    inv_x = xp.linalg.inv(x)
    trace_term = xp.trace(inv_x @ tangent, axis1=-2, axis2=-1)
    out = ans * trace_term
    if not _iscomplex_unwrapped(x):
        out = xp.real(out)
    if _is_traced_leaf(out):
        return cast("xp.ndarray[Any, Any]", out)
    return xp.asarray(out, dtype=_asarray_unwrapped(ans).dtype)


def _jvp_linalg_inv(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    _ = x, rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    inv_x = ans
    dx = -(inv_x @ tangent @ inv_x)
    if not _iscomplex_unwrapped(inv_x):
        dx = xp.real(dx)
    if _is_traced_leaf(dx):
        return cast("xp.ndarray[Any, Any]", dx)
    return xp.asarray(dx, dtype=_asarray_unwrapped(inv_x).dtype)


def _jvp_linalg_pinv(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    """Differentiate the Moore-Penrose inverse on a constant-rank stratum."""
    _ = rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)

    rows, columns = _shape_unwrapped(x)[-2:]
    batch_ndim = _ndim_unwrapped(x) - 2
    result_dtype = xp.dtype(ans.dtype)
    identity_rows = xp.reshape(
        _array_constructor_like(tangent, "eye", rows, dtype=result_dtype),
        (1,) * batch_ndim + (rows, rows),
    )
    identity_columns = xp.reshape(
        _array_constructor_like(tangent, "eye", columns, dtype=result_dtype),
        (1,) * batch_ndim + (columns, columns),
    )
    tangent_h = _h(tangent)
    term1 = -(ans @ tangent @ ans)
    term2 = (ans @ _h(ans) @ tangent_h) @ (identity_rows - x @ ans)
    term3 = (identity_columns - ans @ x) @ (tangent_h @ _h(ans) @ ans)
    out = term1 + term2 + term3
    return cast(
        "xp.ndarray",
        _astype_preserving_trace(out, dtype=_asarray_unwrapped(ans).dtype),
    )


def _qr_skew(local: xp.ndarray) -> xp.ndarray:
    """Select the provider QR gauge from a square local differential."""
    lower = xp.tril(local, k=-1)
    skew = lower - _h(lower)
    if _iscomplex_unwrapped(local):
        diagonal = _scalar_like(1j, local) * xp.imag(xp.diagonal(local, axis1=-2, axis2=-1))
        skew = skew + _diag_matrix(diagonal, dtype=xp.dtype(local.dtype))
    return cast("xp.ndarray", skew)


def _jvp_linalg_qr(
    ans: tuple[xp.ndarray, xp.ndarray],
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    mode: str = "reduced",
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray]:
    """Differentiate full-rank reduced QR factorizations."""
    _ = rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        zeros = _zeros_output_tangent_structure(ans, tangents)
        return cast("tuple[xp.ndarray, xp.ndarray]", zeros)

    q, r = ans
    rows, columns = _shape_unwrapped(x)[-2:]
    if mode not in {"complete", "reduced"}:
        msg = f"linalg.qr derivatives do not support mode={mode!r}"
        raise NotImplementedError(msg)
    if mode == "complete" and rows > columns:
        msg = (
            "linalg.qr derivatives do not define the provider-dependent null-space "
            "columns returned by mode='complete' for tall matrices; use mode='reduced'"
        )
        raise NotImplementedError(msg)

    if rows < columns:
        # For a wide full-row-rank matrix, reduced QR returns square Q and
        # rectangular R.  The leading square block of R fixes Q's gauge; once
        # dQ is known, every column of dR follows from dA = dQ R + Q dR.
        leading_r = r[..., :, :rows]
        leading_tangent = tangent[..., :, :rows]
        tangent_r_inverse = _right_solve(leading_r, leading_tangent)
        local = _h(q) @ tangent_r_inverse
        omega = _qr_skew(local)
        d_q = q @ omega
        d_r = _h(q) @ tangent - omega @ r
        return (
            cast(
                "xp.ndarray",
                _astype_preserving_trace(d_q, dtype=_asarray_unwrapped(q).dtype),
            ),
            cast(
                "xp.ndarray",
                _astype_preserving_trace(d_r, dtype=_asarray_unwrapped(r).dtype),
            ),
        )

    tangent_r_inverse = _right_solve(r, tangent)
    local = _h(q) @ tangent_r_inverse
    omega = _qr_skew(local)
    d_q = tangent_r_inverse - q @ local + q @ omega
    d_r = (local - omega) @ r
    return (
        cast(
            "xp.ndarray",
            _astype_preserving_trace(d_q, dtype=_asarray_unwrapped(q).dtype),
        ),
        cast(
            "xp.ndarray",
            _astype_preserving_trace(d_r, dtype=_asarray_unwrapped(r).dtype),
        ),
    )


def _jvp_linalg_qr_r(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    """Differentiate the R-only QR result using the reduced factorization."""
    _ = rest, attrs
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(ans, tangents)
    q, r = xp.linalg.qr(x, mode="reduced")
    _d_q, d_r = _jvp_linalg_qr(
        (q, r),
        x,
        tangents=(tangent,),
        mode="reduced",
    )
    return cast(
        "xp.ndarray",
        _astype_preserving_trace(d_r, dtype=_asarray_unwrapped(ans).dtype),
    )


def _jvp_vecdot(
    ans: xp.ndarray,
    x1: xp.ndarray,
    x2: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    axis: int = -1,
    **attrs: Any,
) -> xp.ndarray:
    """Differentiate the Array API conjugating vector product."""
    _ = rest, attrs
    tangent1 = tangents[0] if tangents else None
    tangent2 = tangents[1] if len(tangents) > 1 else None
    if tangent1 is None and tangent2 is None:
        return _zeros_output_tangent(ans, tangents)
    vecdot = cast("Any", xp.linalg.vecdot)
    result: Any | None = None
    if tangent1 is not None:
        result = vecdot(tangent1, x2, axis=axis)
    if tangent2 is not None:
        contribution = vecdot(x1, tangent2, axis=axis)
        result = contribution if result is None else result + contribution
    return cast("xp.ndarray", result)


def _jvp_linalg_solve(
    ans: xp.ndarray,
    a: xp.ndarray,
    b: xp.ndarray,
    *rest: xp.ndarray,
    tangents: tuple[xp.ndarray | None, ...],
    **attrs: Any,
) -> xp.ndarray:
    _ = rest, attrs
    da = tangents[0] if len(tangents) > 0 else None
    db = tangents[1] if len(tangents) > 1 else None
    if da is None and db is None:
        return _zeros_output_tangent(ans, tangents)
    rhs_dtype = _infer_tangent_dtype(ans, tangents)
    rhs = xp.zeros_like(ans, dtype=rhs_dtype)
    if db is not None:
        rhs = rhs + db
    if da is not None:
        if len(_shape_unwrapped(ans)) == len(_shape_unwrapped(a)) - 1:
            matrix_product = xp.matmul(da, xp.expand_dims(ans, axis=-1))
            rhs = rhs - xp.squeeze(matrix_product, axis=-1)
        else:
            rhs = rhs - xp.matmul(da, ans)
    out = xp.linalg.solve(a, rhs)
    return cast("xp.ndarray[Any, Any]", _astype_preserving_trace(out, dtype=rhs_dtype))
