"""Runtime backend context for canonical array-family derivative rules.

Rules reach the active provider through ``xp``. An outer transform can trace
any of a rule's primals, tangents or cotangents while the others stay provider
arrays, and an Array API array rejects a traced right operand of a Python
operator. A rule therefore combines values from different sources through
``xp`` functions, which dispatch to the traced operand's namespace.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from math import prod
from typing import TYPE_CHECKING, Any, cast

from advect.autodiff.rules.array_family.providers import resolve_array_family_backend_provider
from advect.core._abstract_helpers import PYTHON_SCALAR_TYPES, dtype_name
from advect.core._abstract_model import ArraySpec
from advect.core._array_api.profiles import (
    LATEST_ARRAY_API_VERSION,
    SUPPORTED_ARRAY_API_VERSIONS,
    minimum_array_api_version,
)
from advect.core._array_family_ops import (
    ARRAY_API_TO_CANONICAL,
    _canonical_array_family_op_name,
)
from advect.core._array_protocol_helpers import materialize_weak_scalar_operands
from advect.core._basic_index import decode_index
from advect.core._context import _get_active_array_api_version
from advect.core._protocols import _is_traced

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from advect.autodiff.rules.array_family.providers import ArrayFamilyBackendProvider

    xp = np

__all__ = [
    "decode_array_index",
    "wrap_array_family_jvp_rule",
    "xp",
]

type _JVPFn = Callable[..., object]


@dataclass(frozen=True, slots=True)
class _ProviderScope:
    """One active provider and the ``xp`` attributes resolved under it."""

    provider: ArrayFamilyBackendProvider
    attributes: dict[str, object] = field(default_factory=dict)


_CURRENT_ARRAY_FAMILY_PROVIDER: ContextVar[_ProviderScope | None] = ContextVar(
    "advect_array_family_backend_provider",
    default=None,
)
_ARRAY_FAMILY_JVP_RULE_ATTR = "__advect_array_family_jvp_rule__"
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
    scope = _CURRENT_ARRAY_FAMILY_PROVIDER.get()
    return None if scope is None else scope.provider


def _has_trace_identity(value: object) -> bool:
    return callable(getattr(getattr(value, "recorder", None), "runtime_trace_identity", None))


def _instance_specific_tracer(value: object) -> object | None:
    """Find a traced leaf whose namespace records into its own trace.

    A leaf that identifies its runtime trace wins over one that does not.
    """
    instance_specific = bool(getattr(value, "__advect_namespace_is_instance_specific__", False))
    if instance_specific and _has_trace_identity(value):
        return value
    fallback = value if instance_specific else None
    items = tuple(value.values()) if isinstance(value, dict) else value
    for item in items if isinstance(items, (tuple, list)) else ():
        tracer = _instance_specific_tracer(item)
        if tracer is not None and _has_trace_identity(tracer):
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
            return target
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


_INEXACT_PYTHON_SCALARS = frozenset({float, complex})


def _scalar_like(value: object, like: object) -> xp.ndarray:
    """Materialize a derivative constant with an array operand's dtype.

    Array API revisions before 2024.12 require both operands of an
    elementwise operation to be arrays. Derivative formulas therefore lift
    their numeric constants explicitly instead of relying on later weak-scalar
    semantics. A rank-zero ``asarray`` result preserves scalar broadcasting and
    remains independent of ``like`` in higher-order traces.
    """
    dtype = getattr(like, "dtype", None)
    if dtype is None and type(like) in _INEXACT_PYTHON_SCALARS:
        # A weak Python-scalar operand promotes as its default dtype would.
        dtype = getattr(xp, dtype_name(type(like)))
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


def _count_like(value: float, like: object) -> xp.ndarray:
    """Materialize a count or divisor in at least single precision.

    A half-precision count overflows past 65504. Promoting ``like``'s dtype
    with float32 keeps the count finite and leaves wider dtypes unchanged;
    rules cast their result back to the answer dtype.
    """
    dtype = xp.result_type(cast("Any", like).dtype, xp.float32)
    return _array_constructor_like(like, "asarray", value, dtype=dtype)


def _target_admits(name: str) -> bool:
    """Return whether the active Array API target includes one standard function.

    A NumPy program may use operations newer than its target, but its
    derivative rules emit only functions of that target, which staging and
    every provider serving it replay.
    """
    active = _get_active_array_api_version() or LATEST_ARRAY_API_VERSION
    return SUPPORTED_ARRAY_API_VERSIONS.index(active) >= SUPPORTED_ARRAY_API_VERSIONS.index(
        minimum_array_api_version(name)
    )


def _supports_cumulative_prod() -> bool:
    """Return whether the active provider has an efficient cumulative product."""
    provider = current_array_backend_provider()
    if provider is None:
        return False
    namespace = provider.namespace
    if getattr(namespace, "__array_api_version__", None) is not None and not _target_admits(
        "cumulative_prod"
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
    if order == sorted(order):
        return cast("xp.ndarray", value)
    return cast("xp.ndarray", xp.permute_dims(cast("Any", value), tuple(order)))


def _zeros_in_trace(like: object, shape: tuple[int, ...], dtype: xp.dtype[Any]) -> xp.ndarray:
    """Return writable zeros of ``shape`` in the dynamic trace that ``like`` is in.

    A constructed constant enters an Array API trace read-only, so its
    ``zeros_like`` supplies the writable value. A NumPy trace dispatches
    ``zeros_like`` through its traced prototype.
    """
    if _instance_specific_tracer(like) is not None:
        constant = _array_constructor_like(like, "zeros", shape, dtype=dtype)
        return xp.zeros_like(constant)
    if callable(getattr(like, "_advect_snapshot", None)):
        return xp.zeros_like(cast("Any", like), shape=shape, dtype=dtype)
    return xp.zeros(shape, dtype=dtype)


def _zero_pad_axis(value: object, *, axis: int, before: int, after: int) -> xp.ndarray:
    """Concatenate exact zeros around ``value`` along one axis.

    Unlike zeros computed arithmetically from ``value``, exact zeros keep its
    non-finite entries local; they are built in the trace of ``value``.
    """
    array = cast("Any", value)
    if before == 0 and after == 0:
        return cast("xp.ndarray", array)
    normalized_axis = axis % array.ndim
    shape = [int(dimension) for dimension in array.shape]

    def zeros(length: int) -> xp.ndarray:
        shape[normalized_axis] = length
        return _array_constructor_like(array, "zeros", tuple(shape), dtype=array.dtype)

    parts = (
        *((zeros(before),) if before else ()),
        array,
        *((zeros(after),) if after else ()),
    )
    return xp.concatenate(parts, axis=normalized_axis)


def _int64_indices(indices: object) -> xp.ndarray:
    """Read gather indices as ``int64``, as NumPy reads any integer index as ``intp``.

    Index arithmetic needs room for the axis length, which a narrow index
    dtype lacks, and NumPy promotes ``uint64`` with ``int64`` to ``float64``.
    A weak Python int already combines as ``int64``.
    """
    dtype = getattr(indices, "dtype", None)
    if dtype is None or dtype == xp.dtype(xp.int64):
        return cast("xp.ndarray", indices)
    if _is_traced(indices):
        return xp.astype(cast("xp.ndarray", indices), xp.int64)
    return cast("xp.ndarray", xp.asarray(indices, dtype=xp.int64))


def _clip_indices(indices: object, axis_size: int) -> xp.ndarray:
    """Clamp gather positions into ``[0, axis_size)``.

    ``clip`` entered the Array API in 2023.12; an older target clamps through
    two ``where`` selections, which cost more than one ``clip``.
    """
    positions = cast("Any", indices)
    last = axis_size - 1
    if _target_admits("clip"):
        return xp.clip(positions, 0, last)
    lower, upper = _scalar_like(0, positions), _scalar_like(last, positions)
    return xp.where(positions < 0, lower, xp.where(positions > last, upper, positions))


def _along_axis_positions(
    indices: object,
    source_shape: tuple[int, ...],
    axis: int,
) -> tuple[xp.ndarray, tuple[int, ...]]:
    """Return the flat source positions a ``take_along_axis`` gathers.

    As in NumPy, a negative index counts from the end of the axis and the
    other axes broadcast. The positions address the source with ``axis``
    moved last and broadcast to the returned leading shape, then flattened.
    """
    indices_last = cast("Any", _int64_indices(_moveaxis(indices, axis, -1)))
    axis_size = source_shape[axis]
    source_leading = (*source_shape[:axis], *source_shape[axis + 1 :])
    *index_leading, count = (int(size) for size in indices_last.shape)
    leading = tuple(
        index_size if source_size == 1 else source_size
        for source_size, index_size in zip(source_leading, index_leading, strict=True)
    )
    offsets = _array_constructor_like(indices_last, "arange", prod(leading), dtype=xp.int64)
    positions = xp.broadcast_to(indices_last, (*leading, count)) % axis_size + xp.reshape(
        offsets * axis_size, (*leading, 1)
    )
    return xp.reshape(positions, (-1,)), leading


def _take_along_axis(value: object, indices: object, *, axis: int) -> xp.ndarray:
    """Lower ``take_along_axis`` to a 2022.12 ``take`` of flat positions.

    A one-hot selection would cost the square of the axis length; ``take``
    gathers directly, and its transpose scatters.
    """
    source_shape = tuple(int(size) for size in cast("Any", value).shape)
    normalized_axis = axis % len(source_shape)
    positions, leading = _along_axis_positions(indices, source_shape, normalized_axis)
    values_last = _moveaxis(value, normalized_axis, -1)
    flat_values = xp.reshape(
        xp.broadcast_to(values_last, (*leading, source_shape[normalized_axis])), (-1,)
    )
    selected = xp.take(flat_values, positions, axis=0)
    count = int(cast("Any", indices).shape[normalized_axis])
    return _moveaxis(xp.reshape(selected, (*leading, count)), -1, normalized_axis)


class _InstanceAwareNamespaceCall:
    """Route derivative helpers through an instance-specific traced namespace."""

    __slots__ = ("_name", "_resolved")

    def __init__(self, name: str, resolved: object) -> None:
        self._name = name
        self._resolved = resolved

    def __call__(self, *args: object, **kwargs: object) -> object:
        # Weak primals, partials and tangents are Python scalars, which an Array
        # API function may reject without an array operand. They compute on
        # arrays of their promoted dtype, as a NumPy function converts them, so
        # a rule keeps IEEE results such as 1.0 / 0.0 == inf.
        if (
            not kwargs
            and args
            and all(type(arg) in PYTHON_SCALAR_TYPES for arg in args)
            and (provider := current_array_backend_provider()) is not None
        ):
            op = _canonical_array_family_op_name(ARRAY_API_TO_CANONICAL.get(self._name, self._name))
            args = materialize_weak_scalar_operands(op, args, namespace=provider.namespace)
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


def _resolve_provider_attribute(provider: ArrayFamilyBackendProvider, name: str) -> object:
    namespace = provider.namespace
    try:
        resolved = getattr(namespace, name)
    except (AttributeError, NotImplementedError):
        ext_namespace = namespace if provider.ext is None else provider.ext
        try:
            resolved = getattr(ext_namespace, name)
        except AttributeError:
            msg = f"Backend provider '{provider.backend}' does not expose array attribute '{name}'."
            raise AttributeError(msg) from None
    if (
        provider.backend.split(".", 1)[0] != "numpy"
        and getattr(namespace, "__array_api_version__", None) is not None
    ):
        # Classes such as the ``complexfloating`` marker are compared by
        # identity, so only functions are routed through an operand's trace.
        if callable(resolved) and not isinstance(resolved, type):
            return _InstanceAwareNamespaceCall(name, resolved)
        if name in {"fft", "linalg"}:
            return _InstanceAwareSubnamespace(name, resolved)
    return resolved


class _ArrayNamespaceProxy:
    """Proxy object exposing the active backend namespace via ``xp``."""

    def __getattr__(self, name: str) -> object:
        if name == "__wrapped__":
            raise AttributeError(name)
        scope = _CURRENT_ARRAY_FAMILY_PROVIDER.get()
        if scope is None:
            msg = (
                "Array-family derivative rules need an active backend provider. "
                "Pass arrays implementing __array_namespace__."
            )
            raise RuntimeError(msg)
        # A derivative sweep runs every rule in one scope, so each attribute
        # is resolved once per sweep rather than on every ``xp`` access.
        attributes = scope.attributes
        if name not in attributes:
            attributes[name] = _resolve_provider_attribute(scope.provider, name)
        return attributes[name]


if not TYPE_CHECKING:
    xp = _ArrayNamespaceProxy()


def run_with_array_family_backend_provider(
    provider: ArrayFamilyBackendProvider,
    fn: Callable[..., object],
    /,
    *args: object,
    **kwargs: object,
) -> object:
    """Run ``fn`` under an explicit backend-provider context."""
    token = _CURRENT_ARRAY_FAMILY_PROVIDER.set(_ProviderScope(provider))
    try:
        return fn(*args, **kwargs)
    finally:
        _CURRENT_ARRAY_FAMILY_PROVIDER.reset(token)


def _maybe_unwrap_array_family_jvp_rule(rule: _JVPFn) -> _JVPFn | None:
    return cast("_JVPFn | None", getattr(rule, _ARRAY_FAMILY_JVP_RULE_ATTR, None))


def wrap_array_family_jvp_rule(rule: _JVPFn) -> _JVPFn:
    """Wrap a JVP rule so it executes inside a resolved backend context.

    A derivative sweep calls the unwrapped rule inside its own provider scope;
    the wrapper serves direct calls such as structural transposition.
    """

    @wraps(rule)
    def wrapped_jvp(
        ans: object,
        *inputs: object,
        tangents: tuple[object | None, ...],
        **attrs: object,
    ) -> object:
        if _CURRENT_ARRAY_FAMILY_PROVIDER.get() is not None:
            return rule(ans, *inputs, tangents=tangents, **attrs)
        return run_with_array_family_backend_provider(
            resolve_array_family_backend_provider(ans, *inputs, *tangents),
            rule,
            ans,
            *inputs,
            tangents=tangents,
            **attrs,
        )

    setattr(wrapped_jvp, _ARRAY_FAMILY_JVP_RULE_ATTR, rule)
    return wrapped_jvp
