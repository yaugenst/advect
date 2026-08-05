"""Runtime backend context for canonical array-family derivative rules."""

from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
from typing import TYPE_CHECKING, Any, cast

from advect.autodiff.rules.array_family.providers import (
    ArrayFamilyBackendProvider,
    resolve_array_family_backend_provider,
)
from advect.core._abstract_helpers import dtype_name
from advect.core._abstract_model import ArraySpec
from advect.core._basic_index import decode_index
from advect.core._context import _get_active_array_api_version

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    xp = np

__all__ = [
    "decode_array_index",
    "wrap_array_family_jvp_rule",
    "wrap_array_family_vjp_rule",
    "xp",
]

_CurrentProvider = ContextVar[ArrayFamilyBackendProvider | None]
type _JVPFn = Callable[..., object]
type _VJPResult = tuple[object | None, ...]
type _VJPFn = Callable[..., _VJPResult]


class _ArrayTypeFallback:
    """Fallback type used for module-load-time `xp.ndarray` aliases."""


_TYPE_ONLY_NAMESPACE_FALLBACKS: dict[str, object] = {
    "ndarray": _ArrayTypeFallback,
}

_CURRENT_ARRAY_FAMILY_PROVIDER: _CurrentProvider = ContextVar(
    "advect_array_family_backend_provider",
    default=None,
)
_ARRAY_NAMESPACE_ATTR_CACHE: dict[tuple[str, int, int, str], object] = {}
_ARRAY_FAMILY_JVP_RULE_ATTR = "__advect_array_family_jvp_rule__"
_ARRAY_FAMILY_VJP_RULE_ATTR = "__advect_array_family_vjp_rule__"
_SELECT_INPUTS_VJP_ATTR = "__advect_vjp_for_input_indices__"
_PREBIND_VJP_ATTR = "__advect_prebind_for_call__"
_EXECUTE_PREBOUND_ATTR = "__advect_execute_prebound__"
_TRACED_NAMESPACE_ALIASES = {
    "cumprod": "cumulative_prod",
    "cumsum": "cumulative_sum",
}


def decode_array_index(payload: object) -> object:
    """Normalize canonical or already materialized provider index metadata."""

    def materialize(values: object, dtype: str, shape: tuple[int, ...]) -> object:
        return xp.asarray(values, dtype=xp.dtype(dtype)).reshape(shape)

    if isinstance(payload, (list, tuple)):
        return tuple(decode_array_index(item) for item in payload)
    if isinstance(payload, (int, slice)) or payload is None or payload is Ellipsis:
        return payload
    if not isinstance(payload, dict):
        return payload
    return decode_index(payload, array_decoder=materialize)


def current_array_backend_provider() -> ArrayFamilyBackendProvider | None:
    """Return the active backend provider for the current derivative call."""
    return _CURRENT_ARRAY_FAMILY_PROVIDER.get()


def _instance_specific_tracer(value: object) -> object | None:
    if bool(getattr(value, "__advect_namespace_is_instance_specific__", False)) and callable(
        getattr(getattr(value, "recorder", None), "runtime_trace_identity", None)
    ):
        return value
    fallback = (
        value if bool(getattr(value, "__advect_namespace_is_instance_specific__", False)) else None
    )
    if isinstance(value, (tuple, list)):
        for item in value:
            tracer = _instance_specific_tracer(item)
            if tracer is None:
                continue
            if callable(getattr(getattr(tracer, "recorder", None), "runtime_trace_identity", None)):
                return tracer
            if fallback is None:
                fallback = tracer
    if isinstance(value, dict):
        for item in value.values():
            tracer = _instance_specific_tracer(item)
            if tracer is None:
                continue
            if callable(getattr(getattr(tracer, "recorder", None), "runtime_trace_identity", None)):
                return tracer
            if fallback is None:
                fallback = tracer
    return fallback


def _instance_namespace_callable(
    name: str,
    values: object,
) -> Callable[..., object] | None:
    tracer = _instance_specific_tracer(values)
    if tracer is None:
        return None
    namespace = cast("Any", tracer).__array_namespace__()
    paths = [name.split(".")]
    alias = _TRACED_NAMESPACE_ALIASES.get(paths[0][-1])
    if alias is not None:
        paths.append([*paths[0][:-1], alias])
    for path in paths:
        target = namespace
        try:
            for part in path:
                target = getattr(target, part)
        except (AttributeError, NotImplementedError):
            continue
        if callable(target):
            return cast("Callable[..., object]", target)
    return None


def _array_constructor_like(
    like: object,
    name: str,
    /,
    *args: object,
    **kwargs: object,
) -> xp.ndarray:
    """Construct an array in the dynamic tangent trace selected by ``like``."""
    tracer = _instance_specific_tracer(like)
    target = _instance_namespace_callable(name, like)
    if target is None:
        target = cast("Callable[..., object]", getattr(xp, name))
    result = target(*args, **kwargs)
    if tracer is None:
        return cast("xp.ndarray", result)
    if getattr(result, "recorder", None) is getattr(tracer, "recorder", None):
        return cast("xp.ndarray", result)

    materialize = getattr(
        cast("Any", tracer).__array_namespace__(),
        "_advect_materialize_constant",
        None,
    )
    spec = getattr(result, "spec", None)
    if spec is None and hasattr(result, "shape") and hasattr(result, "dtype"):
        spec = ArraySpec(
            tuple(int(dimension) for dimension in cast("Any", result).shape),
            cast("Any", result).dtype,
        )
    if callable(materialize) and spec is not None:
        traced_result = materialize(result, spec)
        if traced_result is not NotImplemented:
            return cast("xp.ndarray", traced_result)
    return cast("xp.ndarray", result)


def _scalar_like(value: object, like: object) -> xp.ndarray:
    """Materialize a derivative constant with an array operand's dtype.

    Array API revisions before 2024.12 require both operands of an
    elementwise operation to be arrays. Derivative formulas therefore lift
    their numeric constants explicitly instead of relying on later weak-scalar
    semantics. A rank-zero ``asarray`` result preserves scalar broadcasting and
    remains independent of ``like`` in higher-order traces.
    """
    dtype = getattr(like, "dtype", None)
    if isinstance(value, complex) and dtype is not None:
        # Weak complex scalars promote a real single-precision operand to
        # complex64 and every wider real operand to complex128. Preserve that
        # behavior while still constructing an explicit rank-zero array for
        # revisions whose elementwise operations require array operands.
        source_name = dtype_name(dtype)
        if not source_name.startswith("complex"):
            target_name = "complex64" if source_name in {"float16", "float32"} else "complex128"
            dtype = getattr(xp, target_name)
    kwargs = {} if dtype is None else {"dtype": dtype}
    return _array_constructor_like(like, "asarray", value, **kwargs)


def _supports_cumulative_prod() -> bool:
    """Return whether the active provider has an efficient cumulative product."""
    provider = current_array_backend_provider()
    if provider is None:
        return False
    namespace = provider.namespace
    if (
        getattr(namespace, "__array_api_version__", None) is not None
        and _get_active_array_api_version() != "2024.12"
    ):
        return False
    return callable(getattr(namespace, "cumulative_prod", None)) or callable(
        getattr(namespace, "cumprod", None)
    )


def _moveaxis(
    value: object,
    source: int | tuple[int, ...],
    destination: int | tuple[int, ...],
) -> xp.ndarray:
    """Lower ``moveaxis`` through the 2022.12 ``permute_dims`` primitive."""
    rank = len(cast("Any", value).shape)
    sources = (source,) if isinstance(source, int) else tuple(source)
    destinations = (destination,) if isinstance(destination, int) else tuple(destination)
    if len(sources) != len(destinations):
        msg = "moveaxis source and destination must have equal length"
        raise ValueError(msg)

    normalized_sources = tuple(axis % rank for axis in sources)
    normalized_destinations = tuple(axis % rank for axis in destinations)
    if len(set(normalized_sources)) != len(normalized_sources):
        msg = f"repeated moveaxis source axes: {sources!r}"
        raise ValueError(msg)
    if len(set(normalized_destinations)) != len(normalized_destinations):
        msg = f"repeated moveaxis destination axes: {destinations!r}"
        raise ValueError(msg)

    order = [axis for axis in range(rank) if axis not in normalized_sources]
    for destination_axis, source_axis in sorted(
        zip(normalized_destinations, normalized_sources, strict=True)
    ):
        order.insert(destination_axis, source_axis)
    return cast("xp.ndarray", xp.permute_dims(cast("Any", value), tuple(order)))


def _take_along_axis(value: object, indices: object, *, axis: int) -> xp.ndarray:
    """Lower ``take_along_axis`` through 2022.12 broadcasting and reduction."""
    rank = len(cast("Any", value).shape)
    normalized_axis = axis % rank
    values_last = _moveaxis(value, normalized_axis, -1)
    indices_last = _moveaxis(indices, normalized_axis, -1)
    axis_size = int(cast("Any", values_last).shape[-1])
    positions = _array_constructor_like(
        indices_last,
        "arange",
        axis_size,
        dtype=xp.int64,
    )
    mask = xp.equal(indices_last[..., None], positions)
    selected = xp.sum(
        xp.where(mask, values_last[..., None, :], xp.zeros_like(values_last[..., None, :])),
        axis=-1,
    )
    return _moveaxis(selected, -1, normalized_axis)


class _InstanceAwareNamespaceCall:
    """Route derivative helpers through an instance-specific traced namespace."""

    __slots__ = ("_name", "_resolved")

    def __init__(self, name: str, resolved: object) -> None:
        self._name = name
        self._resolved = resolved

    def __call__(self, *args: object, **kwargs: object) -> object:
        target = _instance_namespace_callable(self._name, (args, kwargs))
        if target is not None:
            return target(*args, **kwargs)
        return cast("Callable[..., object]", self._resolved)(*args, **kwargs)


class _InstanceAwareSubnamespace:
    """Defer nested namespace calls to an operand's trace-aware namespace."""

    __slots__ = ("_name", "_resolved")

    def __init__(self, name: str, resolved: object) -> None:
        self._name = name
        self._resolved = resolved

    def __getattr__(self, name: str) -> object:
        resolved = getattr(self._resolved, name)
        if not callable(resolved):
            return resolved
        return _InstanceAwareNamespaceCall(f"{self._name}.{name}", resolved)


class _ArrayNamespaceProxy:
    """Proxy object exposing the active backend namespace via ``xp``."""

    @staticmethod
    def _provider_attr_cache_key(
        provider: ArrayFamilyBackendProvider,
        *,
        attr_name: str,
    ) -> tuple[str, int, int, str]:
        ext_namespace = provider.namespace if provider.ext is None else provider.ext
        return (provider.backend, id(provider.namespace), id(ext_namespace), attr_name)

    @staticmethod
    def _resolve_provider_attribute(
        provider: ArrayFamilyBackendProvider,
        name: str,
    ) -> object:
        namespace = provider.namespace
        try:
            return getattr(namespace, name)
        except (AttributeError, NotImplementedError):
            ext_namespace = namespace if provider.ext is None else provider.ext
            try:
                return getattr(ext_namespace, name)
            except AttributeError:
                msg = (
                    f"Backend provider '{provider.backend}' does not expose "
                    f"array attribute '{name}'."
                )
                raise AttributeError(msg) from None

    def __getattr__(self, name: str) -> object:
        if name == "__wrapped__":
            raise AttributeError(name)

        provider = _CURRENT_ARRAY_FAMILY_PROVIDER.get()
        if provider is not None:
            # A derivative sweep establishes one provider scope around all
            # rules. Resolve directly there: constructing the global-cache key
            # for every ``xp`` access costs more than backend module lookup.
            namespace = provider.namespace
            resolved = self._resolve_provider_attribute(provider, name)
            if (
                provider.backend.split(".", 1)[0] != "numpy"
                and getattr(namespace, "__array_api_version__", None) is not None
            ):
                cache_key = self._provider_attr_cache_key(provider, attr_name=name)
                cached = _ARRAY_NAMESPACE_ATTR_CACHE.get(cache_key)
                if cached is None:
                    if callable(resolved):
                        cached = _InstanceAwareNamespaceCall(name, resolved)
                    elif name in {"fft", "linalg"}:
                        cached = _InstanceAwareSubnamespace(name, resolved)
                    else:
                        return resolved
                    _ARRAY_NAMESPACE_ATTR_CACHE[cache_key] = cached
                return cached
            return resolved

        try:
            provider = resolve_array_family_backend_provider()
        except RuntimeError:
            fallback = _TYPE_ONLY_NAMESPACE_FALLBACKS.get(name)
            if fallback is not None:
                return fallback
            raise

        cache_key = self._provider_attr_cache_key(provider, attr_name=name)
        cached = _ARRAY_NAMESPACE_ATTR_CACHE.get(cache_key)
        if cached is not None:
            return cached

        resolved = self._resolve_provider_attribute(provider, name)
        _ARRAY_NAMESPACE_ATTR_CACHE[cache_key] = resolved
        return resolved


if not TYPE_CHECKING:
    xp = _ArrayNamespaceProxy()


def run_with_array_family_backend_provider(
    provider: ArrayFamilyBackendProvider,
    fn: _JVPFn | _VJPFn,
    /,
    *args: object,
    **kwargs: object,
) -> object:
    """Run ``fn`` under an explicit backend-provider context."""
    token = _CURRENT_ARRAY_FAMILY_PROVIDER.set(provider)
    try:
        return fn(*args, **kwargs)
    finally:
        _CURRENT_ARRAY_FAMILY_PROVIDER.reset(token)


def _resolve_provider_for_call(
    *values: object,
    scalar_backend_hint: str | None = None,
) -> ArrayFamilyBackendProvider:
    active_provider = _CURRENT_ARRAY_FAMILY_PROVIDER.get()
    if active_provider is not None:
        return active_provider
    if scalar_backend_hint is None:
        return resolve_array_family_backend_provider(*values)
    return resolve_array_family_backend_provider(
        *values,
        scalar_backend_hint=scalar_backend_hint,
    )


def _maybe_unwrap_array_family_jvp_rule(rule: _JVPFn) -> _JVPFn | None:
    raw = getattr(rule, _ARRAY_FAMILY_JVP_RULE_ATTR, None)
    if raw is None:
        return None
    return cast("_JVPFn", raw)


def _maybe_unwrap_array_family_vjp_rule(rule: _VJPFn) -> _VJPFn | None:
    raw = getattr(rule, _ARRAY_FAMILY_VJP_RULE_ATTR, None)
    if raw is None:
        return None
    return cast("_VJPFn", raw)


def wrap_array_family_jvp_rule(
    rule: _JVPFn,
    *,
    scalar_backend_hint: str | None = None,
) -> _JVPFn:
    """Wrap a JVP rule so it executes inside a resolved backend context."""

    @wraps(rule)
    def wrapped_jvp(
        ans: object,
        *inputs: object,
        tangents: tuple[object | None, ...],
        **attrs: object,
    ) -> object:
        # Hot-path: an enclosing derivative scope already resolved the provider.
        if _CURRENT_ARRAY_FAMILY_PROVIDER.get() is not None:
            return rule(ans, *inputs, tangents=tangents, **attrs)
        provider = _resolve_provider_for_call(
            ans,
            *inputs,
            *tangents,
            scalar_backend_hint=scalar_backend_hint,
        )
        token = _CURRENT_ARRAY_FAMILY_PROVIDER.set(provider)
        try:
            return rule(ans, *inputs, tangents=tangents, **attrs)
        finally:
            _CURRENT_ARRAY_FAMILY_PROVIDER.reset(token)

    setattr(wrapped_jvp, _ARRAY_FAMILY_JVP_RULE_ATTR, rule)
    return wrapped_jvp


def wrap_array_family_vjp_rule(
    rule: _VJPFn,
    *,
    scalar_backend_hint: str | None = None,
) -> _VJPFn:
    """Wrap a VJP rule so it executes inside a resolved backend context."""

    @wraps(rule)
    def wrapped_vjp(
        ans: object,
        *inputs: object,
        g: object,
        **attrs: object,
    ) -> _VJPResult:
        # Hot-path: an enclosing derivative scope already resolved the provider.
        if _CURRENT_ARRAY_FAMILY_PROVIDER.get() is not None:
            return rule(ans, *inputs, g=g, **attrs)
        provider = _resolve_provider_for_call(
            ans,
            *inputs,
            g,
            scalar_backend_hint=scalar_backend_hint,
        )
        token = _CURRENT_ARRAY_FAMILY_PROVIDER.set(provider)
        try:
            return rule(ans, *inputs, g=g, **attrs)
        finally:
            _CURRENT_ARRAY_FAMILY_PROVIDER.reset(token)

    setattr(wrapped_vjp, _ARRAY_FAMILY_VJP_RULE_ATTR, rule)
    selective_rule = getattr(rule, _SELECT_INPUTS_VJP_ATTR, None)
    if callable(selective_rule):
        selective_vjp = cast("_VJPFn", selective_rule)

        def _wrapped_selective_vjp(
            ans: object,
            *inputs: object,
            g: object,
            active_input_indices: tuple[int, ...],
            **attrs: object,
        ) -> _VJPResult:
            if _CURRENT_ARRAY_FAMILY_PROVIDER.get() is not None:
                return selective_vjp(
                    ans,
                    *inputs,
                    g=g,
                    active_input_indices=active_input_indices,
                    **attrs,
                )
            provider = _resolve_provider_for_call(
                ans,
                *inputs,
                g,
                scalar_backend_hint=scalar_backend_hint,
            )
            token = _CURRENT_ARRAY_FAMILY_PROVIDER.set(provider)
            try:
                return selective_vjp(
                    ans,
                    *inputs,
                    g=g,
                    active_input_indices=active_input_indices,
                    **attrs,
                )
            finally:
                _CURRENT_ARRAY_FAMILY_PROVIDER.reset(token)

        setattr(wrapped_vjp, _SELECT_INPUTS_VJP_ATTR, _wrapped_selective_vjp)

    prebind_rule = getattr(rule, _PREBIND_VJP_ATTR, None)
    if callable(prebind_rule):
        prebind_vjp = cast("_JVPFn", prebind_rule)

        def _wrapped_prebind_vjp(
            *,
            ans: object,
            inputs: tuple[object, ...],
            g: object,
            attrs: dict[str, object],
            active_input_indices: tuple[int, ...] | None = None,
        ) -> object:
            provider = _resolve_provider_for_call(
                ans,
                *inputs,
                g,
                scalar_backend_hint=scalar_backend_hint,
            )
            bound = run_with_array_family_backend_provider(
                provider,
                prebind_vjp,
                ans=ans,
                inputs=inputs,
                g=g,
                attrs=attrs,
                active_input_indices=active_input_indices,
            )
            if not callable(bound):
                return bound
            bound_vjp = cast("_VJPFn", bound)

            @wraps(bound_vjp)
            def _wrapped_bound_vjp(*args: object, **kwargs: object) -> _VJPResult:
                return cast(
                    "_VJPResult",
                    run_with_array_family_backend_provider(
                        provider,
                        bound_vjp,
                        *args,
                        **kwargs,
                    ),
                )

            execute_prebound = getattr(bound_vjp, _EXECUTE_PREBOUND_ATTR, None)
            if callable(execute_prebound):
                execute = cast("_JVPFn", execute_prebound)

                def _wrapped_execute_prebound(runtime_g: object) -> _VJPResult:
                    return cast(
                        "_VJPResult",
                        run_with_array_family_backend_provider(
                            provider,
                            execute,
                            runtime_g,
                        ),
                    )

                setattr(_wrapped_bound_vjp, _EXECUTE_PREBOUND_ATTR, _wrapped_execute_prebound)
            return _wrapped_bound_vjp

        setattr(wrapped_vjp, _PREBIND_VJP_ATTR, _wrapped_prebind_vjp)
    return wrapped_vjp
