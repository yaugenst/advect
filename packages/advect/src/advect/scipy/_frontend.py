"""Shared frontend helpers for traceable SciPy adapters."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from advect.core._array_api.providers import _get_array_namespace
from advect.core._array_protocol_helpers import _staged_value
from advect.core._context import is_tracing

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from numpy.typing import DTypeLike


def _is_traced_value(value: object) -> bool:
    return callable(getattr(value, "_advect_snapshot", None)) or bool(
        getattr(value, "__advect_abstract_array__", False)
    )


def _numpy_dtype(dtype: object) -> np.dtype[Any]:
    try:
        return np.dtype(cast("DTypeLike", dtype))
    except (TypeError, ValueError) as error:
        msg = f"advect.scipy supports NumPy dtype specifications only; got {dtype!r}"
        raise TypeError(msg) from error


def _operand_dtype(value: object) -> np.dtype[Any]:
    staged = _staged_value(value)
    if staged is not None:
        # Staged code sees its provider's dtype objects; the rules resolve
        # NumPy loops from the canonical dtype, and the call rejects the
        # provider when it runs.
        return _numpy_dtype(staged.spec.dtype)
    dtype = getattr(value, "dtype", None)
    return np.asarray(value).dtype if dtype is None else _numpy_dtype(dtype)


def _traceable_astype(value: Any, dtype: object) -> Any:  # noqa: ANN401 - any provider array.
    """Cast ``value`` to ``dtype``, keeping only the real part for real targets."""
    normalized = _numpy_dtype(dtype)
    source = _operand_dtype(value)
    if source == normalized:
        return value
    if np.issubdtype(source, np.complexfloating) and not np.issubdtype(
        normalized,
        np.complexfloating,
    ):
        value = np.real(value)
    astype = getattr(value, "astype", None)
    return astype(normalized) if callable(astype) else np.asarray(value, dtype=normalized)


def _array_operand(value: object) -> object:
    if _is_traced_value(value) or type(value) in (bool, int, float, complex):
        return value
    if _get_array_namespace(value) is not None:
        return value
    return np.asarray(value)


def _require_numpy_values(module: str, name: str, *values: object) -> None:
    for value in values:
        if value is None:
            continue
        if isinstance(value, tuple):
            _require_numpy_values(module, name, *value)
            continue
        if _is_traced_value(value):
            continue
        namespace = _get_array_namespace(value)
        if namespace is None:
            continue
        provider = getattr(namespace, "__name__", None) or type(namespace).__name__
        if provider != "numpy":
            msg = (
                f"advect.scipy.{module}.{name} supports NumPy arrays only; got Array API "
                f"provider {provider!r}. Convert to a NumPy array before calling "
                "this function."
            )
            raise TypeError(msg)


def _concrete_scipy[**P, R](module: ModuleType) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Forward concrete calls verbatim to ``module``'s same-named function.

    Only traced calls run the decorated body, so concrete values, dtypes, and
    ``output=`` handling are SciPy's own. Every argument must be NumPy-backed.
    """

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        name = function.__name__
        scipy_function = getattr(module, name)
        module_name = module.__name__.rpartition(".")[2]

        @functools.wraps(function)
        def call(*args: P.args, **kwargs: P.kwargs) -> R:
            if is_tracing():
                return function(*args, **kwargs)
            _require_numpy_values(module_name, name, *args, *kwargs.values())
            return scipy_function(*args, **kwargs)

        return call

    return decorate


def _replace_out(
    destination: object,
    replacement: object,
    *,
    argument: str,
    operation: str,
) -> object:
    require_mutable = getattr(destination, "advect_require_mutable", None)
    replace = getattr(destination, "advect_replace", None)
    snapshot = getattr(replacement, "_advect_snapshot", None)
    if not callable(require_mutable) or not callable(replace) or not callable(snapshot):
        msg = f"{argument}= must be an owned traced array from the active trace"
        raise TypeError(msg)
    require_mutable(operation)
    node_id, value = cast("Callable[[], tuple[int, object]]", snapshot)()
    replace(value=value, node_id=int(node_id), operation=operation)
    return destination
