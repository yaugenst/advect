"""Shared frontend helpers for traceable SciPy adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from advect.core._array_api.providers import _get_array_namespace

if TYPE_CHECKING:
    from collections.abc import Callable


def _is_traced_value(value: object) -> bool:
    return callable(getattr(value, "_advect_snapshot", None)) or bool(
        getattr(value, "__advect_abstract_array__", False)
    )


def _array_operand(value: object) -> object:
    if _is_traced_value(value) or type(value) in (bool, int, float, complex):
        return value
    if _get_array_namespace(value) is not None:
        return value
    return np.asarray(value)


def _provider_name(value: object) -> str:
    namespace = _get_array_namespace(value)
    if namespace is None:
        module = type(value).__module__.partition(".")[0]
        return module or type(value).__name__
    name = getattr(namespace, "__name__", None)
    return name if isinstance(name, str) and name else type(namespace).__name__


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
        provider = None if namespace is None else _provider_name(value)
        if provider is not None and provider != "numpy":
            msg = (
                f"advect.scipy.{module}.{name} supports NumPy arrays only; got Array API "
                f"provider {provider!r}. Convert to a NumPy array before calling "
                "this function."
            )
            raise TypeError(msg)


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
