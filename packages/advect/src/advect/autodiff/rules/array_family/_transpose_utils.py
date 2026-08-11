"""Shared transpose and tangent helpers for array-family JVP/VJP synthesis."""

from __future__ import annotations

from typing import Any, Literal, cast

from advect.autodiff.rules.array_family._backend_runtime import (
    _array_constructor_like,
    _scalar_like,
    current_array_backend_provider,
    xp,
)
from advect.core._protocols import _snapshot_traced

type FFTNorm = Literal["backward", "ortho", "forward"]


def _is_traced_leaf(value: Any) -> bool:
    return callable(getattr(value, "_advect_snapshot", None))


def _unwrap_traced_leaf(value: Any) -> Any:
    current = value
    while _is_traced_leaf(current):
        _node_id, next_value = _snapshot_traced(current)
        if next_value is current:
            break
        current = next_value
    return current


def _tangent_type_operand(value: Any) -> Any:
    """Preserve Python-scalar weakness while reducing array tangents to dtypes."""
    unwrapped = _unwrap_traced_leaf(value)
    if type(unwrapped) in {bool, complex, float, int}:
        return unwrapped
    return xp.asarray(unwrapped).dtype


def _adjoint_fft_norm(norm: FFTNorm | None) -> FFTNorm:
    if norm in {None, "backward"}:
        return "forward"
    if norm == "forward":
        return "backward"
    return "ortho"


def _conjugate_transpose(value: xp.ndarray) -> xp.ndarray:
    """Conjugate-transpose the final two axes."""
    return xp.conj(xp.swapaxes(value, -1, -2))


def _diagonal_matrix(values: xp.ndarray, *, dtype: xp.dtype[Any]) -> xp.ndarray:
    size = int(values.shape[-1])
    eye = _array_constructor_like(values, "eye", size, dtype=dtype)
    return cast("xp.ndarray", eye * values[..., None, :])


def _lower_triangular_halfdiag(value: xp.ndarray) -> xp.ndarray:
    """Project to the lower triangle and halve its diagonal."""
    lower = xp.tril(value)
    diagonal = xp.diagonal(lower, axis1=-2, axis2=-1)
    return cast(
        "xp.ndarray",
        lower - _scalar_like(0.5, lower) * _diagonal_matrix(diagonal, dtype=xp.dtype(lower.dtype)),
    )


def _normalize_uplo(value: str) -> Literal["L", "U"]:
    normalized = value.upper()
    if normalized not in {"L", "U"}:
        msg = f"expected UPLO='L' or 'U', got {value!r}"
        raise ValueError(msg)
    return cast("Literal['L', 'U']", normalized)


def _right_solve(a: xp.ndarray, b: xp.ndarray) -> xp.ndarray:
    """Solve ``result @ a = b`` over the final two axes."""
    return cast(
        "xp.ndarray",
        xp.swapaxes(
            xp.linalg.solve(xp.swapaxes(a, -1, -2), xp.swapaxes(b, -1, -2)),
            -1,
            -2,
        ),
    )


def _uses_standard_linalg_contract() -> bool:
    provider = current_array_backend_provider()
    return provider is not None and provider.backend.split(".", 1)[0] != "numpy"


def dtype_is_inexact(dtype: object) -> bool:
    """Return whether a provider dtype has a real or complex tangent space."""
    kind = getattr(dtype, "kind", None)
    if kind is not None:
        return kind in {"c", "f"}
    name = str(dtype).lower()
    return "float" in name or "complex" in name


def infer_tangent_dtype(ans: Any, tangents: tuple[Any | None, ...]) -> xp.dtype[Any]:
    """Infer a dtype for tangent computations from ans and non-None tangents."""
    ans_value = _unwrap_traced_leaf(ans)
    dtypes: list[Any] = [xp.asarray(ans_value).dtype]
    dtypes.extend(_tangent_type_operand(tangent) for tangent in tangents if tangent is not None)
    return xp.result_type(*dtypes)


def zeros_output_tangent(ans: Any, tangents: tuple[Any | None, ...]) -> xp.ndarray:
    """Create a zero tangent matching ``ans`` with inferred tangent dtype."""
    dtype = infer_tangent_dtype(ans, tangents)
    return xp.zeros_like(xp.asarray(_unwrap_traced_leaf(ans)), dtype=dtype)


def infer_output_tangent_dtype(ans: Any, tangents: tuple[Any | None, ...]) -> xp.dtype[Any]:
    """Infer tangent dtype for outputs, including tuple-output structures."""
    dtypes: list[Any] = []

    def collect(value: Any) -> None:
        if isinstance(value, tuple):
            for item in value:
                collect(item)
            return
        dtypes.append(xp.asarray(_unwrap_traced_leaf(value)).dtype)

    collect(ans)
    dtypes.extend(_tangent_type_operand(tangent) for tangent in tangents if tangent is not None)
    if not dtypes:
        return cast("xp.dtype[Any]", xp.float64)
    return xp.result_type(*dtypes)


def zeros_output_tangent_structure(ans: Any, tangents: tuple[Any | None, ...]) -> Any:
    """Create a zero output tangent matching ans shape/structure."""
    if isinstance(ans, tuple):
        dtype = infer_output_tangent_dtype(ans, tangents)
        return tuple(
            xp.zeros_like(xp.asarray(_unwrap_traced_leaf(item)), dtype=dtype) for item in ans
        )
    return zeros_output_tangent(ans, tangents)
