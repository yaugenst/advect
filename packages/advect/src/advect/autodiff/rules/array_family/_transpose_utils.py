"""Shared transpose and tangent helpers for array-family JVP/VJP synthesis."""

from __future__ import annotations

from typing import Any, cast

from advect.autodiff.rules.array_family._backend_runtime import xp
from advect.core._protocols import _snapshot_traced


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
    return cast("xp.dtype[Any]", xp.result_type(*dtypes))


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
    return cast("xp.dtype[Any]", xp.result_type(*dtypes))


def zeros_output_tangent_structure(ans: Any, tangents: tuple[Any | None, ...]) -> Any:
    """Create a zero output tangent matching ans shape/structure."""
    if isinstance(ans, tuple):
        dtype = infer_output_tangent_dtype(ans, tangents)
        return tuple(
            xp.zeros_like(xp.asarray(_unwrap_traced_leaf(item)), dtype=dtype) for item in ans
        )
    return zeros_output_tangent(ans, tangents)
