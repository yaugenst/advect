"""Backend-provider registry for canonical array-family derivative execution."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any, Protocol, cast, runtime_checkable

from advect.core._array_api.providers import (
    _get_backend_key_from_namespace,
    _negotiate_array_namespace_for_call,
)
from advect.core._protocols import _snapshot_traced

__all__ = [
    "ArrayFamilyBackendProvider",
    "get_array_family_backend_provider",
    "register_array_family_backend_provider",
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


_ARRAY_FAMILY_BACKEND_PROVIDERS: dict[str, ArrayFamilyBackendProvider] = {}
_RUNTIME_ARRAY_API_PROVIDERS: dict[tuple[str, int, str], ArrayFamilyBackendProvider] = {}
_SCALAR_VALUE_MISSING = object()


class _ExtensionNamespaceChain:
    """Resolve attributes across extension packs for one array backend."""

    __slots__ = ("namespaces",)

    def __init__(self, namespaces: tuple[object, ...]) -> None:
        self.namespaces = namespaces

    def __getattr__(self, name: str) -> object:
        for namespace in self.namespaces:
            try:
                return getattr(namespace, name)
            except AttributeError:
                continue
        msg = f"No registered array extension exposes {name!r}"
        raise AttributeError(msg)

    def __dir__(self) -> list[str]:
        names = set(super().__dir__())
        for namespace in self.namespaces:
            names.update(dir(namespace))
        return sorted(names)


@dataclass(frozen=True, slots=True)
class _ComposedArrayFamilyBackendProvider:
    backend: str
    namespace: object
    ext: object | None


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


def _extension_namespaces(provider: ArrayFamilyBackendProvider) -> tuple[object, ...]:
    extension = provider.ext
    if extension is None or extension is provider.namespace:
        return ()
    if isinstance(extension, _ExtensionNamespaceChain):
        return extension.namespaces
    return (extension,)


def _deduplicate_namespaces(namespaces: tuple[object, ...]) -> tuple[object, ...]:
    result: list[object] = []
    seen_ids: set[int] = set()
    for namespace in namespaces:
        namespace_id = id(namespace)
        if namespace_id in seen_ids:
            continue
        seen_ids.add(namespace_id)
        result.append(namespace)
    return tuple(result)


def _public_namespace_attributes(namespace: object) -> frozenset[str]:
    return frozenset(name for name in dir(namespace) if not name.startswith("_"))


def _reject_extension_collisions(
    backend: str,
    namespaces: tuple[object, ...],
) -> None:
    public_attributes = tuple(_public_namespace_attributes(namespace) for namespace in namespaces)
    for left_index, left_attributes in enumerate(public_attributes):
        for right_attributes in public_attributes[left_index + 1 :]:
            overlap = sorted(left_attributes & right_attributes)
            if overlap:
                names = ", ".join(repr(name) for name in overlap)
                msg = (
                    f"Array-family extension namespaces for backend {backend!r} "
                    f"export overlapping public attributes: {names}"
                )
                raise ValueError(msg)


def _normalize_backend_key(backend: str) -> str:
    normalized = backend.strip()
    if normalized == "":
        msg = "backend must be non-empty"
        raise ValueError(msg)
    return normalized


def _missing_backend_error(backend: str) -> RuntimeError:
    msg = (
        f"Array namespace '{backend}' does not implement the required Python Array API "
        "protocol and has no explicitly registered array-family provider."
    )
    return RuntimeError(msg)


def register_array_family_backend_provider(provider: ArrayFamilyBackendProvider) -> None:
    """Register a provider, composing extension packs for the same namespace.

    A concrete array namespace has one provider. Optional extension namespaces
    may add kernels not present in the standard Array API. Re-registering the
    base provider must not erase those extensions, and extension ordering must
    not affect availability.
    """
    backend = _normalize_backend_key(provider.backend)
    existing = _ARRAY_FAMILY_BACKEND_PROVIDERS.get(backend)
    if existing is None or existing.namespace is not provider.namespace:
        _ARRAY_FAMILY_BACKEND_PROVIDERS[backend] = provider
        return

    existing_extensions = _extension_namespaces(existing)
    incoming_extensions = _extension_namespaces(provider)
    if not incoming_extensions or all(
        any(extension is candidate for candidate in existing_extensions)
        for extension in incoming_extensions
    ):
        return
    extensions = _deduplicate_namespaces((*existing_extensions, *incoming_extensions))
    _reject_extension_collisions(backend, extensions)
    if len(extensions) == 1 and provider.ext is extensions[0]:
        _ARRAY_FAMILY_BACKEND_PROVIDERS[backend] = provider
        return
    extension: object = (
        extensions[0] if len(extensions) == 1 else _ExtensionNamespaceChain(extensions)
    )
    _ARRAY_FAMILY_BACKEND_PROVIDERS[backend] = _ComposedArrayFamilyBackendProvider(
        backend=backend,
        namespace=provider.namespace,
        ext=extension,
    )


def get_array_family_backend_provider(backend: str) -> ArrayFamilyBackendProvider:
    """Return a previously registered array-family backend provider."""
    normalized = _normalize_backend_key(backend)
    provider = _ARRAY_FAMILY_BACKEND_PROVIDERS.get(normalized)
    if provider is None:
        raise _missing_backend_error(normalized)
    return provider


def _candidate_backend_keys(backend_key: str) -> tuple[str, ...]:
    if "." not in backend_key:
        return (backend_key,)
    head = backend_key.split(".", 1)[0]
    if head == backend_key:
        return (backend_key,)
    return backend_key, head


def _is_standard_array_api_namespace(
    namespace: object,
    *,
    array_api_version: str,
) -> bool:
    """Recognize the standard protocol, not merely a similarly named module."""
    return bool(
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
        # a process-global provider registry would leak completed traces.
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


def _sole_registered_provider() -> ArrayFamilyBackendProvider | None:
    if len(_ARRAY_FAMILY_BACKEND_PROVIDERS) != 1:
        return None
    return next(iter(_ARRAY_FAMILY_BACKEND_PROVIDERS.values()))


def _is_runtime_scalar(value: object) -> bool:
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes, bytearray)):
        return True
    if isinstance(value, (tuple, list)):
        return all(_is_runtime_scalar(item) for item in value)
    if isinstance(value, dict):
        return all(_is_runtime_scalar(item) for item in value.values())
    snapshot = getattr(value, "_advect_snapshot", None)
    if callable(snapshot):
        _node_id, wrapped = _snapshot_traced(value)
    else:
        wrapped = _SCALAR_VALUE_MISSING
    if wrapped is not _SCALAR_VALUE_MISSING and wrapped is not value:
        return _is_runtime_scalar(wrapped)
    shape = getattr(value, "shape", _SCALAR_VALUE_MISSING)
    return isinstance(shape, tuple) and shape == ()


def resolve_array_family_backend_provider(
    *values: object,
    scalar_backend_hint: str | None = None,
    array_api_version: str | None = None,
) -> ArrayFamilyBackendProvider:
    """Resolve a provider from runtime values.

    Detection uses runtime array namespaces and first maps namespace keys to
    registered providers. Standards-compliant Python Array API namespaces can
    execute canonical rules directly without a backend plugin; no NumPy fallback
    is used. For namespace names with dotted suffixes (for example,
    ``numpy.array_api``), both the full key and its leading segment are checked.
    A caller-scoped ``scalar_backend_hint`` is used only when every runtime value
    is scalar and no array namespace can be inferred. Array values always retain
    ordinary dynamic provider resolution.
    """
    resolution = _negotiate_array_namespace_for_call(
        args=values,
        kwargs={},
        required_version=array_api_version,
    )
    runtime_namespace = None if resolution is None else resolution.raw_namespace
    backend_key = (
        None if runtime_namespace is None else _get_backend_key_from_namespace(runtime_namespace)
    )
    if (
        scalar_backend_hint is not None
        and backend_key is None
        and values
        and all(_is_runtime_scalar(value) for value in values)
    ):
        return get_array_family_backend_provider(scalar_backend_hint)

    if backend_key is not None:
        for candidate in _candidate_backend_keys(backend_key):
            provider = _ARRAY_FAMILY_BACKEND_PROVIDERS.get(candidate)
            if provider is not None:
                return provider
        if runtime_namespace is not None:
            selected = resolution.requested_version if resolution is not None else array_api_version
            if selected is None:  # pragma: no cover - runtime namespace implies a resolution
                msg = "Array API provider resolution did not retain its selected revision"
                raise RuntimeError(msg)
            runtime_provider = _runtime_array_api_provider(
                backend_key,
                runtime_namespace,
                array_api_version=selected,
            )
            if runtime_provider is not None:
                return runtime_provider

    sole_provider = _sole_registered_provider()
    if sole_provider is not None and backend_key is None:
        return sole_provider

    if backend_key is None:
        if len(_ARRAY_FAMILY_BACKEND_PROVIDERS) == 1:
            return next(iter(_ARRAY_FAMILY_BACKEND_PROVIDERS.values()))
        msg = (
            "Could not resolve an array-family backend from runtime values. "
            "Pass arrays implementing __array_namespace__ or register an explicit provider."
        )
        raise RuntimeError(msg)

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
    for candidate in _candidate_backend_keys(backend_key):
        provider = _ARRAY_FAMILY_BACKEND_PROVIDERS.get(candidate)
        if provider is not None:
            return provider
    return _runtime_array_api_provider(
        backend_key,
        runtime_namespace,
        array_api_version=resolution.requested_version,
    )
