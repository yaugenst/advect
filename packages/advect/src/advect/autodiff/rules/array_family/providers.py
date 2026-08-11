"""Runtime providers for canonical array-family derivative execution."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any, Protocol, cast, runtime_checkable

from advect.core._array_api.providers import (
    _get_backend_key_from_namespace,
    _negotiate_array_namespace_for_call,
)

__all__ = [
    "ArrayFamilyBackendProvider",
    "resolve_array_family_backend_provider",
    "try_resolve_array_family_backend_provider",
]


@runtime_checkable
class ArrayFamilyBackendProvider(Protocol):
    """Runtime provider for array-family derivative kernels."""

    @property
    def backend(self) -> str:
        """Return the canonical backend key."""

    @property
    def namespace(self) -> object:
        """Return the backend array namespace."""

    @property
    def ext(self) -> object | None:
        """Return the optional backend extension namespace."""


_RUNTIME_ARRAY_API_PROVIDERS: dict[tuple[str, int, str], ArrayFamilyBackendProvider] = {}


@dataclass(frozen=True, slots=True)
class _RuntimeArrayAPIProvider:
    """Provider for a standards-compliant runtime namespace without a plugin."""

    backend: str
    namespace: object
    ext: object | None = None


class _ComplexFloatingDTypeCategory:
    """Compatibility marker for existing real-linear derivative helpers."""


class _StandardArrayAPIExtensions:
    """Tiny compatibility layer for derivative helpers, never a NumPy fallback."""

    __slots__ = ("_namespace",)

    complexfloating = _ComplexFloatingDTypeCategory

    def __init__(self, namespace: object) -> None:
        self._namespace = namespace

    def _namespace_for(self, *values: object) -> object:
        for value in values:
            if not callable(getattr(value, "_advect_snapshot", None)):
                continue
            namespace = getattr(value, "__array_namespace__", None)
            if callable(namespace):
                return namespace()
        return self._namespace

    def dtype(self, value: object) -> object:
        dtype = getattr(value, "dtype", None)
        if dtype is not None:
            return dtype
        if not isinstance(value, str):
            return value
        dtype_name = value.rsplit(".", 1)[-1]
        resolved = getattr(self._namespace, dtype_name, None)
        return value if resolved is None else resolved

    def iscomplexobj(self, value: object) -> bool:
        dtype = getattr(value, "dtype", value)
        try:
            isdtype = cast("Any", self._namespace).isdtype
        except (AttributeError, NotImplementedError):
            return "complex" in str(dtype).lower()
        return bool(isdtype(dtype, "complex floating"))

    def conjugate(self, value: object) -> object:
        """Expose NumPy's spelling for the standard ``conj`` operation."""
        namespace = self._namespace_for(value)
        return cast("Any", namespace).conj(value)

    def concatenate(
        self,
        values: tuple[object, ...] | list[object],
        *,
        axis: int = 0,
    ) -> object:
        """Expose NumPy's spelling for the standard ``concat`` operation."""
        namespace = self._namespace_for(*values)
        return cast("Any", namespace).concat(values, axis=axis)

    def cumprod(
        self,
        value: object,
        *,
        axis: int | None = None,
        dtype: object | None = None,
    ) -> object:
        """Expose NumPy's spelling for standard cumulative products."""
        namespace = self._namespace_for(value)
        return cast("Any", namespace).cumulative_prod(value, axis=axis, dtype=dtype)

    def cumsum(
        self,
        value: object,
        *,
        axis: int | None = None,
        dtype: object | None = None,
    ) -> object:
        """Expose NumPy's spelling for standard cumulative sums."""
        namespace = self._namespace_for(value)
        return cast("Any", namespace).cumulative_sum(value, axis=axis, dtype=dtype)

    def cross(
        self,
        left: object,
        right: object,
        *,
        axisa: int = -1,
        axisb: int = -1,
        axisc: int = -1,
        axis: int | None = None,
    ) -> object:
        """Normalize NumPy cross-product axes to the standard linalg operation."""
        selected_axis = axisa if axis is None else axis
        if axis is None and (axisb != selected_axis or axisc != selected_axis):
            msg = "Array API cross-product derivatives require one shared vector axis"
            raise NotImplementedError(msg)
        if selected_axis >= 0:
            selected_axis -= len(cast("Any", left).shape)
        namespace = self._namespace_for(left, right)
        return cast("Any", namespace).linalg.cross(
            left,
            right,
            axis=selected_axis,
        )

    def diagonal(
        self,
        value: object,
        *,
        offset: int = 0,
        axis1: int = 0,
        axis2: int = 1,
    ) -> object:
        """Expose NumPy's spelling for the standard linalg operation."""
        namespace = self._namespace_for(value)
        matrix = self._matrix_axes_last(
            value,
            axis1=axis1,
            axis2=axis2,
            namespace=namespace,
        )
        return cast("Any", namespace).linalg.diagonal(matrix, offset=offset)

    def outer(self, left: object, right: object) -> object:
        """Expose NumPy's spelling for the standard linalg operation."""
        namespace = self._namespace_for(left, right)
        return cast("Any", namespace).linalg.outer(left, right)

    def tensordot(
        self,
        left: object,
        right: object,
        *,
        axes: int | tuple[tuple[int, ...], tuple[int, ...]] = 2,
    ) -> object:
        """Expose NumPy's spelling for the standard linalg operation."""
        namespace = self._namespace_for(left, right)
        return cast("Any", namespace).linalg.tensordot(left, right, axes=axes)

    def trace(
        self,
        value: object,
        *,
        offset: int = 0,
        axis1: int = 0,
        axis2: int = 1,
    ) -> object:
        """Expose NumPy's spelling for the standard linalg operation."""
        namespace = self._namespace_for(value)
        matrix = self._matrix_axes_last(
            value,
            axis1=axis1,
            axis2=axis2,
            namespace=namespace,
        )
        return cast("Any", namespace).linalg.trace(matrix, offset=offset)

    def _matrix_axes_last(
        self,
        value: object,
        *,
        axis1: int,
        axis2: int,
        namespace: object,
    ) -> object:
        rank = len(cast("Any", value).shape)
        first = axis1 + rank if axis1 < 0 else axis1
        second = axis2 + rank if axis2 < 0 else axis2
        if first == second or not (0 <= first < rank and 0 <= second < rank):
            msg = "matrix axes must be distinct valid array axes"
            raise ValueError(msg)
        axes = (
            *(index for index in range(rank) if index not in {first, second}),
            first,
            second,
        )
        if axes == tuple(range(rank)):
            return value
        return cast("Any", namespace).permute_dims(value, axes)

    def transpose(
        self,
        value: object,
        axes: tuple[int, ...] | None = None,
    ) -> object:
        """Expose NumPy's spelling for the standard axis permutation."""
        if axes is None:
            rank = len(cast("Any", value).shape)
            axes = tuple(reversed(range(rank)))
        namespace = self._namespace_for(value)
        return cast("Any", namespace).permute_dims(value, axes)

    def swapaxes(
        self,
        value: object,
        axis1: int,
        axis2: int,
    ) -> object:
        """Expose NumPy's spelling through the standard axis permutation."""
        rank = len(cast("Any", value).shape)
        first = axis1 + rank if axis1 < 0 else axis1
        second = axis2 + rank if axis2 < 0 else axis2
        if not (0 <= first < rank and 0 <= second < rank):
            msg = f"swapaxes axes {(axis1, axis2)!r} are invalid for rank {rank}"
            raise ValueError(msg)
        axes = list(range(rank))
        axes[first], axes[second] = axes[second], axes[first]
        namespace = self._namespace_for(value)
        return cast("Any", namespace).permute_dims(value, tuple(axes))

    def issubdtype(self, dtype: object, category: object) -> bool:
        if category is not self.complexfloating:
            return False
        isdtype = cast("Any", self._namespace).isdtype
        return bool(isdtype(dtype, "complex floating"))


def _missing_backend_error(backend: str) -> RuntimeError:
    msg = f"Array namespace '{backend}' does not implement the required Python Array API protocol."
    return RuntimeError(msg)


def _is_standard_array_api_namespace(
    namespace: object,
    *,
    array_api_version: str,
) -> bool:
    """Recognize the standard protocol, not merely a similarly named module."""
    return (
        isinstance(getattr(namespace, "__array_api_version__", None), str)
        and (
            array_api_version == "2022.12"
            or callable(getattr(namespace, "__array_namespace_info__", None))
        )
        and callable(getattr(namespace, "asarray", None))
        and callable(getattr(namespace, "zeros_like", None))
        and callable(getattr(namespace, "ones_like", None))
    )


def _runtime_array_api_provider(
    backend: str,
    namespace: object,
    *,
    array_api_version: str,
) -> ArrayFamilyBackendProvider | None:
    if not _is_standard_array_api_namespace(
        namespace,
        array_api_version=array_api_version,
    ):
        return None
    if not isinstance(namespace, ModuleType):
        # Trace-aware namespace proxies are invocation-local. Retaining them in
        # a process-global provider cache would leak completed traces.
        return _RuntimeArrayAPIProvider(
            backend=backend,
            namespace=namespace,
            ext=_StandardArrayAPIExtensions(namespace),
        )
    key = (backend, id(namespace), array_api_version)
    provider = _RUNTIME_ARRAY_API_PROVIDERS.get(key)
    if provider is None or provider.namespace is not namespace:
        provider = _RuntimeArrayAPIProvider(
            backend=backend,
            namespace=namespace,
            ext=_StandardArrayAPIExtensions(namespace),
        )
        _RUNTIME_ARRAY_API_PROVIDERS[key] = provider
    return provider


def resolve_array_family_backend_provider(
    *values: object,
    array_api_version: str | None = None,
) -> ArrayFamilyBackendProvider:
    """Resolve a provider from runtime values.

    Standards-compliant Python Array API namespaces execute canonical rules
    directly. No backend plugin or NumPy fallback is used.
    """
    resolution = _negotiate_array_namespace_for_call(
        args=values,
        kwargs={},
        required_version=array_api_version,
    )
    if resolution is None:
        msg = (
            "Could not resolve an array-family backend from runtime values. "
            "Pass arrays implementing __array_namespace__."
        )
        raise RuntimeError(msg)

    runtime_namespace = resolution.raw_namespace
    backend_key = _get_backend_key_from_namespace(runtime_namespace)
    if backend_key is None:
        msg = "Array API provider namespace does not expose a backend name"
        raise RuntimeError(msg)
    runtime_provider = _runtime_array_api_provider(
        backend_key,
        runtime_namespace,
        array_api_version=resolution.requested_version,
    )
    if runtime_provider is not None:
        return runtime_provider
    raise _missing_backend_error(backend_key)


def try_resolve_array_family_backend_provider(
    *values: object,
    array_api_version: str | None = None,
) -> ArrayFamilyBackendProvider | None:
    """Resolve one provider for a whole derivative call, or return ``None``.

    The caller may then retain dynamic per-rule resolution as the precise-error
    fallback. Successful calls let a complete derivative plan run under one
    provider scope and use the unwrapped rule functions directly.
    """
    resolution = _negotiate_array_namespace_for_call(
        args=values,
        kwargs={},
        required_version=array_api_version,
    )
    if resolution is None:
        return None
    runtime_namespace = resolution.raw_namespace
    backend_key = _get_backend_key_from_namespace(runtime_namespace)
    if backend_key is None:
        return None
    return _runtime_array_api_provider(
        backend_key,
        runtime_namespace,
        array_api_version=resolution.requested_version,
    )
