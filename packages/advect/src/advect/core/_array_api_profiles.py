"""Frozen Array API revision profiles used by every Advect frontend."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

SUPPORTED_ARRAY_API_VERSIONS = ("2022.12", "2023.12", "2024.12")
LATEST_ARRAY_API_VERSION = SUPPORTED_ARRAY_API_VERSIONS[-1]


@dataclass(frozen=True, slots=True)
class ArrayAPIProfile:
    """One immutable official callable/signature surface."""

    version: str
    signatures: Mapping[str, str]

    def admits(self, path: str) -> bool:
        """Return whether this revision includes one callable path."""
        return path in self.signatures


@dataclass(frozen=True, slots=True)
class _ProfileDelta:
    additions: Mapping[str, str]
    overrides: Mapping[str, str]


_SIGNATURES_2022_12: dict[str, str] = {
    "abs": "(x, /)",
    "acos": "(x, /)",
    "acosh": "(x, /)",
    "add": "(x1, x2, /)",
    "all": "(x, /, *, axis=None, keepdims=False)",
    "any": "(x, /, *, axis=None, keepdims=False)",
    "arange": "(start, /, stop=None, step=1, *, dtype=None, device=None)",
    "argmax": "(x, /, *, axis=None, keepdims=False)",
    "argmin": "(x, /, *, axis=None, keepdims=False)",
    "argsort": "(x, /, *, axis=-1, descending=False, stable=True)",
    "asarray": "(obj, /, *, dtype=None, device=None, copy=None)",
    "asin": "(x, /)",
    "asinh": "(x, /)",
    "astype": "(x, dtype, /, *, copy=True)",
    "atan": "(x, /)",
    "atan2": "(x1, x2, /)",
    "atanh": "(x, /)",
    "bitwise_and": "(x1, x2, /)",
    "bitwise_invert": "(x, /)",
    "bitwise_left_shift": "(x1, x2, /)",
    "bitwise_or": "(x1, x2, /)",
    "bitwise_right_shift": "(x1, x2, /)",
    "bitwise_xor": "(x1, x2, /)",
    "broadcast_arrays": "(*arrays)",
    "broadcast_to": "(x, /, shape)",
    "can_cast": "(from_, to, /)",
    "ceil": "(x, /)",
    "concat": "(arrays, /, *, axis=0)",
    "conj": "(x, /)",
    "cos": "(x, /)",
    "cosh": "(x, /)",
    "divide": "(x1, x2, /)",
    "empty": "(shape, *, dtype=None, device=None)",
    "empty_like": "(x, /, *, dtype=None, device=None)",
    "equal": "(x1, x2, /)",
    "exp": "(x, /)",
    "expand_dims": "(x, /, axis)",
    "expm1": "(x, /)",
    "eye": "(n_rows, n_cols=None, /, *, k=0, dtype=None, device=None)",
    "fft.fft": "(x, /, *, n=None, axis=-1, norm='backward')",
    "fft.fftfreq": "(n, /, *, d=1.0, device=None)",
    "fft.fftn": "(x, /, *, s=None, axes=None, norm='backward')",
    "fft.fftshift": "(x, /, *, axes=None)",
    "fft.hfft": "(x, /, *, n=None, axis=-1, norm='backward')",
    "fft.ifft": "(x, /, *, n=None, axis=-1, norm='backward')",
    "fft.ifftn": "(x, /, *, s=None, axes=None, norm='backward')",
    "fft.ifftshift": "(x, /, *, axes=None)",
    "fft.ihfft": "(x, /, *, n=None, axis=-1, norm='backward')",
    "fft.irfft": "(x, /, *, n=None, axis=-1, norm='backward')",
    "fft.irfftn": "(x, /, *, s=None, axes=None, norm='backward')",
    "fft.rfft": "(x, /, *, n=None, axis=-1, norm='backward')",
    "fft.rfftfreq": "(n, /, *, d=1.0, device=None)",
    "fft.rfftn": "(x, /, *, s=None, axes=None, norm='backward')",
    "finfo": "(type, /)",
    "flip": "(x, /, *, axis=None)",
    "floor": "(x, /)",
    "floor_divide": "(x1, x2, /)",
    "from_dlpack": "(x, /)",
    "full": "(shape, fill_value, *, dtype=None, device=None)",
    "full_like": "(x, /, fill_value, *, dtype=None, device=None)",
    "greater": "(x1, x2, /)",
    "greater_equal": "(x1, x2, /)",
    "iinfo": "(type, /)",
    "imag": "(x, /)",
    "isdtype": "(dtype, kind)",
    "isfinite": "(x, /)",
    "isinf": "(x, /)",
    "isnan": "(x, /)",
    "less": "(x1, x2, /)",
    "less_equal": "(x1, x2, /)",
    "linalg.cholesky": "(x, /, *, upper=False)",
    "linalg.cross": "(x1, x2, /, *, axis=-1)",
    "linalg.det": "(x, /)",
    "linalg.diagonal": "(x, /, *, offset=0)",
    "linalg.eigh": "(x, /)",
    "linalg.eigvalsh": "(x, /)",
    "linalg.inv": "(x, /)",
    "linalg.matmul": "(x1, x2, /)",
    "linalg.matrix_norm": "(x, /, *, keepdims=False, ord='fro')",
    "linalg.matrix_power": "(x, n, /)",
    "linalg.matrix_rank": "(x, /, *, rtol=None)",
    "linalg.matrix_transpose": "(x, /)",
    "linalg.outer": "(x1, x2, /)",
    "linalg.pinv": "(x, /, *, rtol=None)",
    "linalg.qr": "(x, /, *, mode='reduced')",
    "linalg.slogdet": "(x, /)",
    "linalg.solve": "(x1, x2, /)",
    "linalg.svd": "(x, /, *, full_matrices=True)",
    "linalg.svdvals": "(x, /)",
    "linalg.tensordot": "(x1, x2, /, *, axes=2)",
    "linalg.trace": "(x, /, *, offset=0, dtype=None)",
    "linalg.vecdot": "(x1, x2, /, *, axis=-1)",
    "linalg.vector_norm": "(x, /, *, axis=None, keepdims=False, ord=2)",
    "linspace": "(start, stop, /, num, *, dtype=None, device=None, endpoint=True)",
    "log": "(x, /)",
    "log10": "(x, /)",
    "log1p": "(x, /)",
    "log2": "(x, /)",
    "logaddexp": "(x1, x2, /)",
    "logical_and": "(x1, x2, /)",
    "logical_not": "(x, /)",
    "logical_or": "(x1, x2, /)",
    "logical_xor": "(x1, x2, /)",
    "matmul": "(x1, x2, /)",
    "matrix_transpose": "(x, /)",
    "max": "(x, /, *, axis=None, keepdims=False)",
    "mean": "(x, /, *, axis=None, keepdims=False)",
    "meshgrid": "(*arrays, indexing='xy')",
    "min": "(x, /, *, axis=None, keepdims=False)",
    "multiply": "(x1, x2, /)",
    "negative": "(x, /)",
    "nonzero": "(x, /)",
    "not_equal": "(x1, x2, /)",
    "ones": "(shape, *, dtype=None, device=None)",
    "ones_like": "(x, /, *, dtype=None, device=None)",
    "permute_dims": "(x, /, axes)",
    "positive": "(x, /)",
    "pow": "(x1, x2, /)",
    "prod": "(x, /, *, axis=None, dtype=None, keepdims=False)",
    "real": "(x, /)",
    "remainder": "(x1, x2, /)",
    "reshape": "(x, /, shape, *, copy=None)",
    "result_type": "(*arrays_and_dtypes)",
    "roll": "(x, /, shift, *, axis=None)",
    "round": "(x, /)",
    "sign": "(x, /)",
    "sin": "(x, /)",
    "sinh": "(x, /)",
    "sort": "(x, /, *, axis=-1, descending=False, stable=True)",
    "sqrt": "(x, /)",
    "square": "(x, /)",
    "squeeze": "(x, /, axis)",
    "stack": "(arrays, /, *, axis=0)",
    "std": "(x, /, *, axis=None, correction=0.0, keepdims=False)",
    "subtract": "(x1, x2, /)",
    "sum": "(x, /, *, axis=None, dtype=None, keepdims=False)",
    "take": "(x, indices, /, *, axis=None)",
    "tan": "(x, /)",
    "tanh": "(x, /)",
    "tensordot": "(x1, x2, /, *, axes=2)",
    "tril": "(x, /, *, k=0)",
    "triu": "(x, /, *, k=0)",
    "trunc": "(x, /)",
    "unique_all": "(x, /)",
    "unique_counts": "(x, /)",
    "unique_inverse": "(x, /)",
    "unique_values": "(x, /)",
    "var": "(x, /, *, axis=None, correction=0.0, keepdims=False)",
    "vecdot": "(x1, x2, /, *, axis=-1)",
    "where": "(condition, x1, x2, /)",
    "zeros": "(shape, *, dtype=None, device=None)",
    "zeros_like": "(x, /, *, dtype=None, device=None)",
}

_DELTAS: dict[str, _ProfileDelta] = {
    "2023.12": _ProfileDelta(
        additions=MappingProxyType(
            {
                "clip": "(x, /, min=None, max=None)",
                "copysign": "(x1, x2, /)",
                "cumulative_sum": "(x, /, *, axis=None, dtype=None, include_initial=False)",
                "hypot": "(x1, x2, /)",
                "maximum": "(x1, x2, /)",
                "minimum": "(x1, x2, /)",
                "moveaxis": "(x, source, destination, /)",
                "repeat": "(x, repeats, /, *, axis=None)",
                "searchsorted": "(x1, x2, /, *, side='left', sorter=None)",
                "signbit": "(x, /)",
                "tile": "(x, repetitions, /)",
                "unstack": "(x, /, *, axis=0)",
            }
        ),
        overrides=MappingProxyType(
            {
                "astype": "(x, dtype, /, *, copy=True, device=None)",
                "from_dlpack": "(x, /, *, device=None, copy=None)",
            }
        ),
    ),
    "2024.12": _ProfileDelta(
        additions=MappingProxyType(
            {
                "count_nonzero": "(x, /, *, axis=None, keepdims=False)",
                "cumulative_prod": "(x, /, *, axis=None, dtype=None, include_initial=False)",
                "diff": "(x, /, *, axis=-1, n=1, prepend=None, append=None)",
                "nextafter": "(x1, x2, /)",
                "reciprocal": "(x, /)",
                "take_along_axis": "(x, indices, /, *, axis=-1)",
            }
        ),
        overrides=MappingProxyType(
            {
                "fft.fftfreq": "(n, /, *, d=1.0, dtype=None, device=None)",
                "fft.rfftfreq": "(n, /, *, d=1.0, dtype=None, device=None)",
            }
        ),
    ),
}


@cache
def materialize_array_api_profile(version: str) -> ArrayAPIProfile:
    """Fold the frozen base and thin deltas into one immutable profile."""
    if version not in SUPPORTED_ARRAY_API_VERSIONS:
        supported = ", ".join(SUPPORTED_ARRAY_API_VERSIONS)
        message = f"Unsupported Array API revision {version!r}; choose one of {supported}"
        raise ValueError(message)

    signatures = dict(_SIGNATURES_2022_12)
    target_index = SUPPORTED_ARRAY_API_VERSIONS.index(version)
    for candidate in SUPPORTED_ARRAY_API_VERSIONS[1 : target_index + 1]:
        delta = _DELTAS[candidate]
        signatures.update(delta.additions)
        signatures.update(delta.overrides)
    return ArrayAPIProfile(version=version, signatures=MappingProxyType(signatures))


@cache
def minimum_array_api_version(path: str) -> str:
    """Return the first supported revision containing one callable."""
    for version in SUPPORTED_ARRAY_API_VERSIONS:
        if materialize_array_api_profile(version).admits(path):
            return version
    raise KeyError(path)


__all__ = [
    "LATEST_ARRAY_API_VERSION",
    "SUPPORTED_ARRAY_API_VERSIONS",
    "ArrayAPIProfile",
    "materialize_array_api_profile",
    "minimum_array_api_version",
]
