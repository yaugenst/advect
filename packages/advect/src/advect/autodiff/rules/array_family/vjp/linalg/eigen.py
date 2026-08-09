"""Explicit transposes for Hermitian eigendecompositions."""

from __future__ import annotations

from typing import Any, Literal, cast

from advect.autodiff.rules.array_family._backend_runtime import (
    current_array_backend_provider,
    xp,
)
from advect.autodiff.rules.array_family.vjp.linalg.common import (
    _dtype_of,
    _h,
    _hermitian_triangle_adjoint,
    _merge_multioutput_cotangent,
    _shape_of,
)

_EIGH_OUTPUT_COUNT = 2


def _uses_standard_linalg_contract() -> bool:
    provider = current_array_backend_provider()
    return provider is not None and provider.backend.split(".", 1)[0] != "numpy"


def _diag_matrix(values: xp.ndarray, *, dtype: xp.dtype[Any]) -> xp.ndarray:
    size = _shape_of(values)[-1]
    return cast("xp.ndarray", xp.eye(size, dtype=dtype) * values[..., None, :])


def _normalize_uplo(value: str) -> Literal["L", "U"]:
    normalized = str(value).upper()
    if normalized not in {"L", "U"}:
        msg = f"expected UPLO='L' or 'U', got {value!r}"
        raise ValueError(msg)
    return cast("Literal['L', 'U']", normalized)


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
        _eigenvalues, eigenvectors = xp.linalg.eigh(x)
    else:
        _eigenvalues, eigenvectors = xp.linalg.eigh(x, UPLO=uplo)
    local = _diag_matrix(g, dtype=_dtype_of(eigenvectors))
    natural = eigenvectors @ local @ _h(eigenvectors)
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
        eye = xp.eye(size, dtype=dtype)
        off_diagonal = xp.ones_like(eye) - eye
        gaps = eigenvalues[..., None, :] - eigenvalues[..., :, None]
        inverse_gaps = off_diagonal / (gaps + eye)
        local = local + inverse_gaps * (_h(eigenvectors) @ g_eigenvectors)

    natural = eigenvectors @ local @ _h(eigenvectors)
    return (_hermitian_triangle_adjoint(natural, uplo=uplo),)


__all__ = ["_vjp_eigh", "_vjp_eigvalsh"]
