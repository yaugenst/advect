"""Explicit transposes for Hermitian eigendecompositions."""

from __future__ import annotations

from typing import Any

from advect.autodiff.rules.array_family._backend_runtime import _array_constructor_like, xp
from advect.autodiff.rules.array_family._transpose_utils import (
    _diagonal_matrix as _diag_matrix,
    _dtype_of,
    _normalize_uplo,
    _shape_of,
    _uses_standard_linalg_contract,
)
from advect.autodiff.rules.array_family.jvp.linalg import _hermitian_from_triangle
from advect.autodiff.rules.array_family.vjp.linalg.common import (
    _h,
    _hermitian_triangle_adjoint,
    _merge_multioutput_cotangent,
)

_EIGH_OUTPUT_COUNT = 2


def _vjp_eigvalsh(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    UPLO: str = "L",
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Transpose the Hermitian eigenvalue differential."""
    _ = ans, rest, attrs
    uplo = _normalize_uplo(UPLO)
    if _uses_standard_linalg_contract():
        # A standard eigh takes no UPLO, so it gets the selected triangle's matrix.
        _eigenvalues, eigenvectors = xp.linalg.eigh(_hermitian_from_triangle(x, uplo=uplo))
    else:
        _eigenvalues, eigenvectors = xp.linalg.eigh(x, UPLO=uplo)
    local = _diag_matrix(g, dtype=_dtype_of(eigenvectors))
    natural = xp.matmul(eigenvectors, local) @ _h(eigenvectors)
    return (_hermitian_triangle_adjoint(natural, uplo=uplo),)


def _vjp_eigh(
    ans: tuple[xp.ndarray, xp.ndarray],
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: tuple[xp.ndarray | None, xp.ndarray | None],
    UPLO: str = "L",
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Transpose a Hermitian eigendecomposition on distinct spectra."""
    _ = x, rest, attrs
    merged_g = _merge_multioutput_cotangent(
        g,
        output_count=_EIGH_OUTPUT_COUNT,
        op_name="numpy.linalg.eigh",
    )

    eigenvalues, eigenvectors = ans
    g_eigenvalues, g_eigenvectors = merged_g
    uplo = _normalize_uplo(UPLO)
    size = _shape_of(eigenvalues)[-1]
    dtype = _dtype_of(eigenvectors)

    values_cotangent = xp.zeros_like(eigenvalues) if g_eigenvalues is None else g_eigenvalues
    local = _diag_matrix(values_cotangent, dtype=dtype)

    if g_eigenvectors is not None:
        eye = _array_constructor_like((eigenvectors, g_eigenvectors), "eye", size, dtype=dtype)
        off_diagonal = xp.ones_like(eye) - eye
        gaps = eigenvalues[..., None, :] - eigenvalues[..., :, None]
        inverse_gaps = off_diagonal / xp.add(gaps, eye)
        local = xp.add(local, inverse_gaps * xp.matmul(_h(eigenvectors), g_eigenvectors))

    natural = xp.matmul(eigenvectors, local) @ _h(eigenvectors)
    return (_hermitian_triangle_adjoint(natural, uplo=uplo),)
