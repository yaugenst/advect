# ruff: noqa: ANN401
"""Explicit array and trace-integration helpers."""

from __future__ import annotations

from typing import Any, cast

from advect.core._array_api.providers import _get_array_namespace
from advect.core._errors import TracingError
from advect.core._pytree import tree_flatten, tree_unflatten


def is_traced(value: object) -> bool:
    """Return whether ``value`` is an Advect tracer.

    This check does not read the trace-time payload and remains safe for an
    escaped tracer. It tests the value itself rather than recursively searching
    an arbitrary object graph.

    Examples
    --------
    >>> import advect as ad
    >>> import numpy as np
    >>> ad.is_traced(np.array([1.0]))
    False
    >>> def contains_tracer(value):
    ...     assert ad.is_traced(value)
    ...     return np.sum(value**2)
    >>> ad.grad(contains_tracer)(np.array([2.0])).tolist()
    [4.0]
    """
    return callable(getattr(value, "_advect_snapshot", None))


def _first_sequence_tracer(
    value: object,
    *,
    seen: set[int] | None = None,
) -> object | None:
    if is_traced(value):
        return value
    if not isinstance(value, (tuple, list)):
        return None

    visited = set() if seen is None else seen
    identity = id(value)
    if identity in visited:
        msg = "Advect array construction does not accept cyclic sequences"
        raise ValueError(msg)
    visited.add(identity)
    try:
        for item in value:
            tracer = _first_sequence_tracer(item, seen=visited)
            if tracer is not None:
                return tracer
    finally:
        visited.remove(identity)
    return None


def _tracer_namespace(value: object) -> Any:
    if getattr(value, "__advect_frontend__", None) == "numpy":
        import numpy as np  # noqa: PLC0415 - NumPy is Advect's required frontend

        return np
    namespace = getattr(value, "__array_namespace__", None)
    if not callable(namespace):
        msg = f"Advect tracer {type(value).__name__} does not expose an array namespace"
        raise TypeError(msg)
    return namespace()


def _same_dtype(value: object, dtype: object) -> bool:
    current = getattr(value, "dtype", None)
    return current == dtype or str(current) == str(dtype)


def _cast_traced(value: Any, dtype: object | None, *, copy: bool | None) -> Any:
    if dtype is None or _same_dtype(value, dtype):
        return value.copy() if copy is True else value
    if copy is False:
        msg = "Unable to avoid a copy while changing dtype in advect.asarray"
        raise ValueError(msg)
    astype = getattr(value, "astype", None)
    if not callable(astype):
        msg = f"Advect tracer {type(value).__name__} does not support dtype conversion"
        raise TypeError(msg)
    return astype(dtype, copy=True)


def _namespace_asarray(
    namespace: Any,
    value: object,
    *,
    dtype: object | None,
    copy: bool | None,
) -> Any:
    asarray_fn = getattr(namespace, "asarray", None)
    if not callable(asarray_fn):
        msg = f"Array namespace {namespace!r} does not provide asarray"
        raise TypeError(msg)
    kwargs: dict[str, object] = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if copy is not None:
        kwargs["copy"] = copy
    return asarray_fn(value, **kwargs)


def _stack_traced_sequence(value: object, *, namespace: Any) -> Any:
    if is_traced(value):
        return value
    if not isinstance(value, (tuple, list)) or not value:
        return _namespace_asarray(namespace, value, dtype=None, copy=None)

    children = tuple(_stack_traced_sequence(item, namespace=namespace) for item in value)
    stack = getattr(namespace, "stack", None)
    if not callable(stack):
        msg = f"Array namespace {namespace!r} does not provide stack"
        raise TypeError(msg)
    return stack(children, axis=0)


def asarray(
    obj: object,
    dtype: object | None = None,
    *,
    copy: bool | None = None,
) -> Any:
    """Construct an array without detaching Advect tracers.

    Direct tracers and rectangular nested tracer sequences remain
    differentiable. This is the provider-neutral explicit alternative to
    NumPy's standard ``numpy.asarray(..., like=tracer)`` dispatch. Ordinary
    non-traced values retain their provider when they expose the pinned Array
    API namespace and otherwise use NumPy.

    Examples
    --------
    >>> import advect as ad
    >>> import numpy as np
    >>> def total(value):
    ...     return np.sum(ad.asarray([value[0], value[1]]))
    >>> ad.grad(total)(np.array([2.0, 3.0])).tolist()
    [1.0, 1.0]
    """
    tracer = _first_sequence_tracer(obj)
    if tracer is not None:
        if is_traced(obj):
            return _cast_traced(cast("Any", obj), dtype, copy=copy)
        if copy is False:
            msg = "Unable to avoid a copy when constructing an array from a traced sequence"
            raise ValueError(msg)
        namespace = _tracer_namespace(tracer)
        result = _stack_traced_sequence(obj, namespace=namespace)
        return _cast_traced(result, dtype, copy=None)

    namespace = _get_array_namespace(obj)
    if namespace is not None:
        return _namespace_asarray(namespace, obj, dtype=dtype, copy=copy)

    import numpy as np  # noqa: PLC0415 - NumPy is the default constructor provider

    return np.asarray(obj, dtype=cast("Any", dtype), copy=copy)


def array(
    obj: object,
    dtype: object | None = None,
    *,
    copy: bool = True,
) -> Any:
    """Construct an owned array while preserving traced dependencies.

    This is the explicit traced counterpart of the common
    ``numpy.array(obj, dtype=..., copy=...)`` forms. It intentionally does not
    mirror NumPy's complete constructor signature.

    Examples
    --------
    >>> import advect as ad
    >>> import numpy as np
    >>> def total(value):
    ...     return np.sum(ad.array([value[0], value[1]]))
    >>> ad.grad(total)(np.array([2.0, 3.0])).tolist()
    [1.0, 1.0]
    """
    return asarray(obj, dtype=dtype, copy=copy)


def _concrete_tracer_copy(value: object) -> object:
    current = value
    seen: set[int] = set()
    while is_traced(current):
        if bool(getattr(type(current), "__advect_abstract_array__", False)):
            msg = (
                "stop_gradient requires a concrete dynamic trace; abstract staged "
                "values have no primal payload"
            )
            raise TracingError(msg)
        identity = id(current)
        if identity in seen:
            msg = "Advect tracer payload chain is cyclic"
            raise RuntimeError(msg)
        seen.add(identity)
        snapshot = cast("Any", current)._advect_snapshot()  # noqa: SLF001
        next_value = snapshot[1]
        if next_value is current:
            msg = "Advect tracer did not expose a concrete primal payload"
            raise TracingError(msg)
        current = next_value

    copy_fn = getattr(current, "copy", None)
    if callable(copy_fn):
        return copy_fn()
    namespace = _get_array_namespace(current)
    if namespace is not None:
        try:
            return _namespace_asarray(namespace, current, dtype=None, copy=True)
        except (TypeError, ValueError):
            pass
    return current


def stop_gradient[T](value: T) -> T:
    """Return a concrete copy of traced leaves, explicitly stopping gradients.

    Registered pytree structure is preserved. The operation is available only
    during concrete dynamic tracing; staging rejects it because an abstract
    value has no concrete primal to validate or serialize.

    Examples
    --------
    >>> import advect as ad
    >>> import numpy as np
    >>> def loss(value):
    ...     return np.sum(value * ad.stop_gradient(value))
    >>> ad.grad(loss)(np.array([2.0, 3.0])).tolist()
    [2.0, 3.0]
    """
    leaves, treedef = tree_flatten(value)
    stopped = [_concrete_tracer_copy(leaf) if is_traced(leaf) else leaf for leaf in leaves]
    return cast("T", tree_unflatten(treedef, stopped))


__all__ = ["array", "asarray", "is_traced", "stop_gradient"]
