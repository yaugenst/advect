"""Focused tests for NumPy protocol normalization and result structure."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import advect.numpy._array_function.runtime as array_function_runtime_module
from advect.core._errors import TracingError
from advect.numpy._protocol_runtime import NUMPY_PROTOCOL_RUNTIME


class _Traced:
    def __init__(self, value: object, node_id: int, recorder: object) -> None:
        self.value = value
        self.node_id = node_id
        self.recorder = recorder


class _BinaryUfunc:
    __name__ = "binary"
    nin = 2
    nout = 1


def test_array_function_signature_is_resolved_once(monkeypatch: Any) -> None:
    def target(first: object, second: object, *, optional: object = None) -> object:
        del second, optional
        return first

    array_function_runtime_module._cached_positional_parameters.cache_clear()
    original_signature = array_function_runtime_module.inspect.signature
    calls = 0

    def counted_signature(func: object) -> object:
        nonlocal calls
        if func is target:
            calls += 1
        return original_signature(func)

    monkeypatch.setattr(array_function_runtime_module.inspect, "signature", counted_signature)

    first = NUMPY_PROTOCOL_RUNTIME._normalize_array_function_args_and_kwargs(
        func=target,
        args=(1,),
        kwargs={"second": 2},
    )
    second = NUMPY_PROTOCOL_RUNTIME._normalize_array_function_args_and_kwargs(
        func=target,
        args=(3,),
        kwargs={"second": 4},
    )

    assert first == ((1, 2), {})
    assert second == ((3, 4), {})
    assert calls == 1


def test_unhashable_callable_signature_bypasses_cache() -> None:
    class UnhashableCallable:
        def __call__(
            self,
            first: object,
            second: object,
            *,
            optional: object = None,
        ) -> object:
            del second, optional
            return first

    type.__setattr__(UnhashableCallable, "__hash__", None)

    parameters = array_function_runtime_module._positional_parameters(UnhashableCallable())

    assert tuple(parameter.name for parameter in parameters) == ("first", "second")


def test_array_function_normalizes_einsum_keyword_operands() -> None:
    def einsum(*values: object, **kwargs: object) -> object:
        return values, kwargs

    args, kwargs = NUMPY_PROTOCOL_RUNTIME._normalize_array_function_args_and_kwargs(
        func=einsum,
        args=(),
        kwargs={
            "subscripts": "ij,jk->ik",
            "x1": "left",
            "x2": "right",
            "optimize": True,
        },
    )

    assert args == ("ij,jk->ik", "left", "right")
    assert kwargs == {"optimize": True}


def test_array_function_normalizes_known_positional_aliases() -> None:
    def clip(value: object, minimum: object, maximum: object) -> object:
        return value, minimum, maximum

    args, kwargs = NUMPY_PROTOCOL_RUNTIME._normalize_array_function_args_and_kwargs(
        func=clip,
        args=("value",),
        kwargs={"a_min": 0, "a_max": 1},
    )

    assert args == ("value", 0, 1)
    assert kwargs == {}


def test_array_function_normalization_preserves_uninspectable_callables(
    monkeypatch: Any,
) -> None:
    def target(value: object) -> object:
        return value

    def unavailable(_func: object) -> None:
        message = "signature unavailable"
        raise ValueError(message)

    monkeypatch.setattr(array_function_runtime_module, "_positional_parameters", unavailable)

    args, kwargs = NUMPY_PROTOCOL_RUNTIME._normalize_array_function_args_and_kwargs(
        func=target,
        args=("value",),
        kwargs={"flag": True},
    )

    assert args == ("value",)
    assert kwargs == {"flag": True}


def test_array_function_normalization_stops_at_optional_or_missing_parameters() -> None:
    def target(first: object, second: object, third: object = 3) -> object:
        return first, second, third

    optional = NUMPY_PROTOCOL_RUNTIME._normalize_array_function_args_and_kwargs(
        func=target,
        args=(1, 2),
        kwargs={"third": 4},
    )
    missing = NUMPY_PROTOCOL_RUNTIME._normalize_array_function_args_and_kwargs(
        func=target,
        args=(1,),
        kwargs={},
    )

    assert optional == ((1, 2), {"third": 4})
    assert missing == ((1,), {})


def test_array_function_out_argument_contract() -> None:
    traced = _Traced("value", 1, object())
    resolve = NUMPY_PROTOCOL_RUNTIME._resolve_array_function_out_arg
    round_function = SimpleNamespace(__name__="round")
    clip_function = SimpleNamespace(__name__="clip")

    assert resolve(round_function, _Traced, None) is None
    assert resolve(round_function, _Traced, traced) is traced
    assert resolve(clip_function, _Traced, (traced,)) is traced
    with pytest.raises(TracingError, match="tuple destination"):
        resolve(round_function, _Traced, (traced,))
    with pytest.raises(TracingError, match="tuple destination"):
        resolve(clip_function, _Traced, (traced, traced))


def test_array_function_result_tree_is_wrapped_without_backend_adapter() -> None:
    recorder = object()
    result = NUMPY_PROTOCOL_RUNTIME._wrap_array_function_result(
        result_value=("left", ["middle", "right"]),
        node_ids=(3, [4, 5]),
        traced_type=_Traced,
        recorder=recorder,
    )

    left, nested = result
    assert (left.value, left.node_id, left.recorder) == ("left", 3, recorder)
    assert [(item.value, item.node_id, item.recorder) for item in nested] == [
        ("middle", 4, recorder),
        ("right", 5, recorder),
    ]


def test_ufunc_keyword_operands_are_normalized_once() -> None:
    inputs, kwargs = NUMPY_PROTOCOL_RUNTIME._normalize_ufunc_inputs_and_kwargs(
        ufunc=_BinaryUfunc(),
        inputs=("left",),
        kwargs={"x2": "right", "where": True},
    )

    assert inputs == ("left", "right")
    assert kwargs == {"where": True}
    with pytest.raises(TracingError, match="duplicate operand"):
        NUMPY_PROTOCOL_RUNTIME._normalize_ufunc_inputs_and_kwargs(
            ufunc=_BinaryUfunc(),
            inputs=("left", "right"),
            kwargs={"x2": "duplicate"},
        )
