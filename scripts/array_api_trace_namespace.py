"""Array API test-suite bridge that executes selected calls through Advect.

The official suite imports this module as its array namespace. Array creation
and unselected helpers pass directly to array-api-strict. Selected operations
with array operands are reconstructed inside either a concrete Advect trace or an
abstractly staged program before their concrete results return to the suite.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING

import array_api_strict as _provider

import advect as _ad
from advect.autodiff._ephemeral import trace_call as _trace_call
from advect.core._array_api.frontend import _ARRAY_API_COMPOSITES, _FUNCTION_SPECS
from advect.core._array_api.profiles import (
    LATEST_ARRAY_API_VERSION as _LATEST_ARRAY_API_VERSION,
    materialize_array_api_profile as _materialize_array_api_profile,
)
from advect.core._context import _use_array_api_version

if TYPE_CHECKING:
    from collections.abc import Callable


__array_api_version__ = os.environ.get("ARRAY_API_TESTS_VERSION", _LATEST_ARRAY_API_VERSION)
_materialize_array_api_profile(__array_api_version__)
_provider.set_array_api_strict_flags(api_version=__array_api_version__)
__version__ = f"advect-bridge+{getattr(_provider, '__version__', 'unknown')}"


@dataclass(frozen=True, slots=True)
class _ArraySlot:
    index: int


@dataclass(frozen=True, slots=True)
class _EncodedCall:
    args: tuple[object, ...]
    kwargs: dict[str, object]


def _selected_operations() -> frozenset[str]:
    raw = os.environ.get("ADVECT_ARRAY_API_QUALIFICATION_OPS", "")
    return frozenset(item.strip() for item in raw.split(",") if item.strip())


def _is_array(value: object) -> bool:
    return (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and callable(getattr(value, "__array_namespace__", None))
    )


def _encode(value: object, leaves: list[object]) -> object:
    if _is_array(value):
        slot = _ArraySlot(len(leaves))
        leaves.append(value)
        return slot
    if isinstance(value, tuple):
        return tuple(_encode(item, leaves) for item in value)
    if isinstance(value, list):
        return [_encode(item, leaves) for item in value]
    if isinstance(value, dict):
        return {key: _encode(item, leaves) for key, item in value.items()}
    return value


def _decode(value: object, leaves: tuple[object, ...]) -> object:
    if isinstance(value, _ArraySlot):
        return leaves[value.index]
    if isinstance(value, tuple):
        return tuple(_decode(item, leaves) for item in value)
    if isinstance(value, list):
        return [_decode(item, leaves) for item in value]
    if isinstance(value, dict):
        return {key: _decode(item, leaves) for key, item in value.items()}
    return value


def _resolve(namespace: object, path: str) -> Callable[..., object]:
    resolved = namespace
    for component in path.split("."):
        resolved = getattr(resolved, component)
    if not callable(resolved):
        msg = f"Array API attribute {path!r} is not callable"
        raise TypeError(msg)
    return resolved


def _encode_call(
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> tuple[_EncodedCall, tuple[object, ...]]:
    leaves: list[object] = []
    encoded = _EncodedCall(
        args=tuple(_encode(arg, leaves) for arg in args),
        kwargs={name: _encode(value, leaves) for name, value in kwargs.items()},
    )
    return encoded, tuple(leaves)


def _record(
    path: str,
    mode: str,
    inputs: tuple[object, ...],
    output: object,
    *,
    selected_array_api_version: str,
) -> None:
    destination = os.environ.get("ADVECT_ARRAY_API_TRACE_LOG")
    if not destination:
        return
    payload = {
        "input_count": len(inputs),
        "mode": mode,
        "operation": path,
        "output_dtype": str(getattr(output, "dtype", None)),
        "output_shape": list(getattr(output, "shape", ())),
        "selected_array_api_version": selected_array_api_version,
    }
    with Path(destination).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True))
        stream.write("\n")


def _execute_selected(
    path: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    encoded, inputs = _encode_call(args, kwargs)
    if not inputs:
        return _resolve(_provider, path)(*args, **kwargs)

    def transformed(*runtime_inputs: object) -> object:
        namespace = runtime_inputs[0].__array_namespace__()  # type: ignore[attr-defined]
        function = _resolve(namespace, path)
        decoded_args = _decode(encoded.args, runtime_inputs)
        decoded_kwargs = _decode(encoded.kwargs, runtime_inputs)
        return function(
            *decoded_args,  # type: ignore[arg-type]
            **decoded_kwargs,
        )

    mode = os.environ.get("ADVECT_ARRAY_API_QUALIFICATION_MODE", "dynamic")
    if mode == "dynamic":
        # A provider may implement several revisions. Qualification constrains
        # negotiation to the revision under test rather than selecting the
        # provider's newest revision.
        with _use_array_api_version(__array_api_version__):
            trace = _trace_call(
                transformed,
                args=inputs,
                kwargs={},
                argnums=tuple(range(len(inputs))),
                argnames=None,
            )
        try:
            output = trace.output
            selected_array_api_version = trace.array_api_version
        finally:
            trace.tape.release_payloads()
    elif mode in {"stage", "serialized"}:
        specs = tuple(
            _ad.ArraySpec(tuple(value.shape), value.dtype)  # type: ignore[attr-defined]
            for value in inputs
        )
        program = _ad.stage(
            transformed,
            specs=specs,
            array_api_version=__array_api_version__,
        )
        if mode == "serialized":
            program = _ad.StagedProgram.from_dict(program.to_dict())
        output = program(*inputs)
        selected_array_api_version = program.array_api_version
    else:
        msg = f"Unknown ADVECT_ARRAY_API_QUALIFICATION_MODE {mode!r}"
        raise RuntimeError(msg)

    _record(
        path,
        mode,
        inputs,
        output,
        selected_array_api_version=selected_array_api_version,
    )
    return output


def _wrap(path: str, function: Callable[..., object]) -> Callable[..., object]:
    @wraps(function)
    def wrapped(*args: object, **kwargs: object) -> object:
        return _execute_selected(path, args, kwargs)

    return wrapped


class _ExtensionNamespace:
    __slots__ = ("_name", "_namespace")

    def __init__(self, name: str, namespace: object) -> None:
        self._name = name
        self._namespace = namespace

    @property
    def __name__(self) -> str:
        return f"{__name__}.{self._name}"

    def __getattr__(self, name: str) -> object:
        value = getattr(self._namespace, name)
        path = f"{self._name}.{name}"
        if (
            path in _selected_operations()
            and path in set(_FUNCTION_SPECS) | _ARRAY_API_COMPOSITES
            and callable(value)
        ):
            return _wrap(path, value)
        return value

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()).union(dir(self._namespace)))


fft = _ExtensionNamespace("fft", _provider.fft)
linalg = _ExtensionNamespace("linalg", _provider.linalg)


def __getattr__(name: str) -> object:
    value = getattr(_provider, name)
    if (
        name in _selected_operations()
        and name in set(_FUNCTION_SPECS) | _ARRAY_API_COMPOSITES
        and callable(value)
    ):
        return _wrap(name, value)
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(dir(_provider)))
