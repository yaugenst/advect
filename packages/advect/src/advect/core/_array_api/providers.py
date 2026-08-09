# ruff: noqa: ANN401
"""Array-namespace discovery and invocation-local revision negotiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from advect.core._array_api.profiles import (
    LATEST_ARRAY_API_VERSION,
    SUPPORTED_ARRAY_API_VERSIONS,
    materialize_array_api_profile,
)
from advect.core._protocols import _snapshot_traced
from advect.core._pytree import _get_node_impl

__all__ = [
    "_ARRAY_API_VERSION",
    "ResolvedArrayNamespace",
    "_array_namespace_can_donate",
    "_clear_array_namespace_caches",
    "_configure_array_namespace_fallback",
    "_get_array_namespace",
    "_get_backend_key_from_namespace",
    "_get_provider_array_api_version",
    "_infer_array_namespace_for_call",
    "_negotiate_array_namespace_for_call",
]

# Kept as the newest supported revision for internal callers which need a
# concrete default. Trace and stage paths negotiate or carry an explicit target.
_ARRAY_API_VERSION = LATEST_ARRAY_API_VERSION
_DEFAULT_API_VERSION = object()
_NAMESPACE_CACHE_MISS = object()
_NAMESPACE_BY_TYPE: dict[tuple[type[Any], str | None], Any | None] = {}
_WRAPPED_NAMESPACE_BY_TYPES: dict[tuple[type[Any], type[Any], str | None], Any | None] = {}
_ARRAY_NAMESPACE_FALLBACK: Any | None = None
_ARRAY_NAMESPACE_DONATION_CHECKER: Any | None = None
_PRIMITIVE_TYPES = (bool, int, float, complex, str, bytes, bytearray)


@dataclass(frozen=True, slots=True)
class ResolvedArrayNamespace:
    """One provider namespace resolved for an explicit Advect revision."""

    raw_namespace: Any
    requested_version: str

    @property
    def _advect_requested_array_api_version(self) -> str:
        return self.requested_version

    def __getattr__(self, name: str) -> Any:
        return getattr(self.raw_namespace, name)


def _configure_array_namespace_fallback(
    resolver: Any,
    *,
    can_donate: Any,
) -> None:
    """Install Advect's deterministic base-provider fallback."""
    global _ARRAY_NAMESPACE_DONATION_CHECKER, _ARRAY_NAMESPACE_FALLBACK  # noqa: PLW0603
    _ARRAY_NAMESPACE_FALLBACK = resolver
    _ARRAY_NAMESPACE_DONATION_CHECKER = can_donate


def _array_namespace_can_donate(value: Any) -> bool:
    """Return whether the built-in provider bridge admits buffer donation."""
    checker = _ARRAY_NAMESPACE_DONATION_CHECKER
    return bool(checker is not None and checker(value))


def _clear_array_namespace_caches() -> None:
    """Forget type-level protocol results after a provider changes profiles."""
    _NAMESPACE_BY_TYPE.clear()
    _WRAPPED_NAMESPACE_BY_TYPES.clear()


def _has_instance_namespace_override(value: Any) -> bool:
    attrs = getattr(value, "__dict__", None)
    return isinstance(attrs, dict) and "__array_namespace__" in attrs


def _call_namespace_fn(ns_fn: Any, *, api_version: str | None) -> Any | None:
    try:
        return ns_fn() if api_version is None else ns_fn(api_version=api_version)
    except Exception:  # noqa: BLE001 - best-effort backend detection
        return None


def _resolve_direct_namespace(value: Any, *, api_version: str | None) -> Any:
    value_type = type(value)
    type_ns = getattr(value_type, "__array_namespace__", None)
    if type_ns is None:
        ns_fn = getattr(value, "__array_namespace__", None)
        if ns_fn is None:
            return _NAMESPACE_CACHE_MISS
        return _call_namespace_fn(ns_fn, api_version=api_version)

    if _has_instance_namespace_override(value) or bool(
        getattr(value_type, "__advect_namespace_is_instance_specific__", False)
    ):
        ns_fn = getattr(value, "__array_namespace__", None)
        if ns_fn is None:
            return _NAMESPACE_CACHE_MISS
        return _call_namespace_fn(ns_fn, api_version=api_version)

    cache_key = (value_type, api_version)
    cached = _NAMESPACE_BY_TYPE.get(cache_key, _NAMESPACE_CACHE_MISS)
    if cached is not _NAMESPACE_CACHE_MISS:
        return cached

    ns_fn = getattr(value, "__array_namespace__", None)
    if ns_fn is None:
        return _NAMESPACE_CACHE_MISS

    namespace = _call_namespace_fn(ns_fn, api_version=api_version)
    _NAMESPACE_BY_TYPE[cache_key] = namespace
    return namespace


def _resolve_wrapped_namespace(value: Any, *, api_version: str | None) -> Any | None:
    snapshot = getattr(value, "_advect_snapshot", None)
    if not callable(snapshot):
        return None
    _node_id, underlying = _snapshot_traced(value)
    if underlying is None or underlying is value:
        return None

    underlying_type = type(underlying)
    if getattr(underlying_type, "__array_namespace__", None) is None:
        return _resolve_wrapped_namespace(underlying, api_version=api_version)

    if _has_instance_namespace_override(underlying) or bool(
        getattr(underlying_type, "__advect_namespace_is_instance_specific__", False)
    ):
        ns_fn = getattr(underlying, "__array_namespace__", None)
        if ns_fn is None:
            return None
        return _call_namespace_fn(ns_fn, api_version=api_version)

    wrapped_key = (type(value), underlying_type, api_version)
    cached = _WRAPPED_NAMESPACE_BY_TYPES.get(wrapped_key, _NAMESPACE_CACHE_MISS)
    if cached is not _NAMESPACE_CACHE_MISS:
        return cached

    ns_fn = getattr(underlying, "__array_namespace__", None)
    if ns_fn is None:
        return None

    namespace = _call_namespace_fn(ns_fn, api_version=api_version)
    _WRAPPED_NAMESPACE_BY_TYPES[wrapped_key] = namespace
    return namespace


def _default_api_version() -> str | None:
    from advect.core._context import _get_active_array_api_version  # noqa: PLC0415

    return _get_active_array_api_version() or LATEST_ARRAY_API_VERSION


def _get_array_namespace(
    value: Any,
    *,
    api_version: str | None | object = _DEFAULT_API_VERSION,
) -> Any | None:
    if value is None or type(value) in _PRIMITIVE_TYPES:
        return None
    if api_version is _DEFAULT_API_VERSION:
        requested: str | None = _default_api_version()
    elif isinstance(api_version, str) or api_version is None:
        requested = api_version
    else:
        message = f"Invalid Array API version request {api_version!r}"
        raise TypeError(message)

    direct_namespace = _resolve_direct_namespace(value, api_version=requested)
    if direct_namespace is not _NAMESPACE_CACHE_MISS:
        return direct_namespace

    wrapped_namespace = _resolve_wrapped_namespace(value, api_version=requested)
    if wrapped_namespace is not None:
        return wrapped_namespace

    fallback = _ARRAY_NAMESPACE_FALLBACK
    return None if fallback is None else fallback(value, api_version=requested)


def _get_backend_key_from_namespace(xp: Any) -> str | None:
    name = getattr(xp, "__name__", None)
    return name if isinstance(name, str) and name else None


def _version_key(version: str) -> tuple[int, int] | None:
    year, separator, month = version.partition(".")
    if separator != "." or not year.isdigit() or not month.isdigit():
        return None
    return int(year), int(month)


def _get_provider_array_api_version(namespace: Any) -> str | None:
    """Return the Array API revision declared by a provider namespace."""
    version = getattr(namespace, "__array_api_version__", None)
    return version if isinstance(version, str) else None


def _provider_can_report(
    namespace: Any,
    *,
    requested_version: str,
) -> str | None:
    backend = _get_backend_key_from_namespace(namespace)
    if backend is None:
        return None
    if not callable(getattr(namespace, "asarray", None)):
        return None

    provider_version = _get_provider_array_api_version(namespace)
    if provider_version is not None:
        reported_key = _version_key(provider_version)
        requested_key = _version_key(requested_version)
        if reported_key is None or requested_key is None or reported_key < requested_key:
            return None

    if requested_version != "2022.12" and not callable(
        getattr(namespace, "__array_namespace_info__", None)
    ):
        return None
    return backend


def _array_candidates(value: Any) -> tuple[Any, ...]:
    """Collect every provider-backed leaf without resolving a revision."""
    if value is None or type(value) in _PRIMITIVE_TYPES:
        return ()
    if callable(getattr(value, "_advect_snapshot", None)):
        return (value,)

    node_impl = _get_node_impl(type(value))
    if node_impl is not None:
        flatten_fn, _unflatten_fn = node_impl
        children, _aux_data = flatten_fn(value)
        return tuple(candidate for child in children for candidate in _array_candidates(child))

    if callable(getattr(value, "__array_namespace__", None)) or (
        hasattr(value, "shape") and hasattr(value, "dtype")
    ):
        return (value,)
    return ()


def _call_array_candidates(
    *,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, ...]:
    return tuple(
        candidate for value in (*args, *kwargs.values()) for candidate in _array_candidates(value)
    )


def _requested_versions(required_version: str | None) -> tuple[str, ...]:
    if required_version is not None:
        materialize_array_api_profile(required_version)
        return (required_version,)

    from advect.core._context import _get_active_array_api_version  # noqa: PLC0415

    enclosing = _get_active_array_api_version()
    if enclosing is not None:
        materialize_array_api_profile(enclosing)
        return (enclosing,)
    return tuple(reversed(SUPPORTED_ARRAY_API_VERSIONS))


def _negotiate_array_namespace_for_call(
    *,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    required_version: str | None = None,
) -> ResolvedArrayNamespace | None:
    """Select the newest revision served by every array leaf in one call."""
    candidates = _call_array_candidates(args=args, kwargs=kwargs)
    if not candidates:
        return None

    attempted = _requested_versions(required_version)
    for version in attempted:
        resolutions: list[tuple[Any, str]] = []
        for value in candidates:
            namespace = _get_array_namespace(value, api_version=version)
            if namespace is None:
                break
            backend = _provider_can_report(namespace, requested_version=version)
            if backend is None:
                break
            resolutions.append((namespace, backend))
        else:
            backends = {backend for _namespace, backend in resolutions}
            if len(backends) != 1:
                names = ", ".join(sorted(backends))
                message = f"Cannot combine different array providers in one transform: {names}"
                raise TypeError(message)
            namespace, _backend = resolutions[0]
            return ResolvedArrayNamespace(
                raw_namespace=namespace,
                requested_version=version,
            )

    versions = ", ".join(attempted)
    provider_types = ", ".join(sorted({type(value).__name__ for value in candidates}))
    if required_version is None:
        message = (
            f"Array inputs {provider_types} cannot serve a common Advect Array API revision; "
            f"attempted {versions}"
        )
    else:
        message = (
            f"Array inputs {provider_types} cannot serve required Array API "
            f"{required_version}; attempted {versions}"
        )
    raise TypeError(message)


def _infer_array_namespace_for_call(
    *,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    api_version: str | None | object = _DEFAULT_API_VERSION,
) -> Any | None:
    """Resolve one validated namespace for a whole call."""
    if api_version is _DEFAULT_API_VERSION:
        required: str | None = None
    elif isinstance(api_version, str) or api_version is None:
        required = api_version
    else:
        message = f"Invalid Array API version request {api_version!r}"
        raise TypeError(message)
    if required is None and api_version is None:
        for value in _call_array_candidates(args=args, kwargs=kwargs):
            xp = _get_array_namespace(value, api_version=None)
            if xp is not None:
                return xp
        return None
    resolution = _negotiate_array_namespace_for_call(
        args=args,
        kwargs=kwargs,
        required_version=required,
    )
    return None if resolution is None else resolution.raw_namespace
