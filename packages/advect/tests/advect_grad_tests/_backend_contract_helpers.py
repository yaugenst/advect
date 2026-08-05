"""Shared helpers for backend contract tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from advect.autodiff.rules.array_family.providers import ArrayFamilyBackendProvider


@dataclass(frozen=True, slots=True)
class BackendProviderStub(ArrayFamilyBackendProvider):
    backend: str
    namespace: Any
    ext: Any | None = None


class StrictHigherOrderNamespace:
    __slots__ = ("_xp", "float64")

    def __init__(self, namespace: Any) -> None:
        self._xp = namespace
        self.float64 = namespace.float64

    @staticmethod
    def _unwrap(value: Any) -> Any:
        if isinstance(value, StrictArrayWithMethods):
            return value.value
        return value

    def __getattr__(self, name: str) -> Any:
        return getattr(self._xp, name)

    def asarray(self, value: Any, dtype: Any | None = None) -> Any:
        unwrapped = self._unwrap(value)
        if dtype is None:
            return StrictArrayWithMethods(self._xp.asarray(unwrapped), self._xp)
        return StrictArrayWithMethods(self._xp.asarray(unwrapped, dtype=dtype), self._xp)

    def zeros(self, shape: tuple[int, ...], dtype: Any) -> Any:
        return StrictArrayWithMethods(self._xp.zeros(shape, dtype=dtype), self._xp)

    def result_type(self, *dtypes: Any) -> Any:
        return self._xp.result_type(*dtypes)

    def zeros_like(self, value: Any) -> Any:
        unwrapped = self._unwrap(value)
        return StrictArrayWithMethods(
            self._xp.zeros(unwrapped.shape, dtype=unwrapped.dtype),
            self._xp,
        )

    def diag(self, value: Any) -> Any:
        unwrapped = self._unwrap(value)
        return StrictArrayWithMethods(self._xp.linalg.diagonal(unwrapped), self._xp)


class StrictArrayWithMethods:
    __slots__ = ("_xp", "value")

    def __init__(self, value: Any, namespace: Any) -> None:
        self.value = value
        self._xp = namespace

    @property
    def shape(self) -> tuple[int, ...]:
        return cast("tuple[int, ...]", self.value.shape)

    @property
    def size(self) -> int:
        return int(self.value.size)

    @property
    def dtype(self) -> Any:
        return self.value.dtype

    def reshape(self, *shape: int) -> StrictArrayWithMethods:
        target_shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        reshaped = self._xp.reshape(self.value, target_shape)
        return StrictArrayWithMethods(reshaped, self._xp)

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(value, StrictArrayWithMethods):
            value = value.value
        self.value[key] = value

    def __getitem__(self, key: Any) -> Any:
        value = self.value[key]
        if hasattr(value, "shape"):
            return StrictArrayWithMethods(value, self._xp)
        return value

    def __array__(self, dtype: Any | None = None) -> np.ndarray[Any, Any]:
        return np.asarray(self.value, dtype=dtype)
