"""Stable result containers required by the Python Array API contract."""

from __future__ import annotations

from typing import Any, NamedTuple


class EighResult(NamedTuple):
    """Result of ``linalg.eigh``."""

    eigenvalues: Any
    eigenvectors: Any


class EigResult(NamedTuple):
    """Internal result of ``linalg.eig``."""

    eigenvalues: Any
    eigenvectors: Any


class QRResult(NamedTuple):
    """Result of ``linalg.qr``."""

    Q: Any
    R: Any


class SlogdetResult(NamedTuple):
    """Result of ``linalg.slogdet``."""

    sign: Any
    logabsdet: Any


class SVDResult(NamedTuple):
    """Result of ``linalg.svd``."""

    U: Any
    S: Any
    Vh: Any


_RESULT_TYPES: dict[str, type[tuple[Any, ...]]] = {
    "linalg.eig": EigResult,
    "linalg.eigh": EighResult,
    "linalg.qr": QRResult,
    "linalg.slogdet": SlogdetResult,
    "linalg.svd": SVDResult,
}

_SERIALIZED_RESULT_TYPES: dict[str, type[tuple[Any, ...]]] = {
    "array_api.eig_result": EigResult,
    "array_api.eigh_result": EighResult,
    "array_api.qr_result": QRResult,
    "array_api.slogdet_result": SlogdetResult,
    "array_api.svd_result": SVDResult,
}
_RESULT_TYPE_TAGS = {result_type: tag for tag, result_type in _SERIALIZED_RESULT_TYPES.items()}


def restore_array_api_result(path: str, values: tuple[Any, ...]) -> tuple[Any, ...]:
    """Restore the standard field-bearing result for one multi-output call."""
    result_type = _RESULT_TYPES.get(path)
    return values if result_type is None else result_type(*values)


__all__ = [
    "EigResult",
    "EighResult",
    "QRResult",
    "SVDResult",
    "SlogdetResult",
    "restore_array_api_result",
]
