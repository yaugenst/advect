"""Focused tests for NumPy protocol normalization and result structure."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import advect.numpy._signature as signature_module
from advect.core._errors import TracingError
from advect.numpy._protocol_runtime import NUMPY_PROTOCOL_RUNTIME


class _Traced:
    def __init__(self, value: object, node_id: int, recorder: object) -> None:
        self.value = value
        self.node_id = node_id
        self.recorder = recorder


def test_array_function_signature_is_resolved_once(monkeypatch: Any) -> None:
    def target(first: object, second: object, *, optional: object = None) -> object:
        del second, optional
        return first

    signature_module.positional_parameters.cache_clear()
    original_signature = signature_module.inspect.signature
    calls = 0

    def counted_signature(func: object) -> object:
        nonlocal calls
        if func is target:
            calls += 1
        return original_signature(func)

    monkeypatch.setattr(signature_module.inspect, "signature", counted_signature)

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


def test_array_function_normalization_preserves_uninspectable_callables(
    monkeypatch: Any,
) -> None:
    def target(value: object) -> object:
        return value

    original_signature = signature_module.inspect.signature

    def unavailable(func: object) -> object:
        if func is target:
            message = "signature unavailable"
            raise ValueError(message)
        return original_signature(func)

    signature_module.positional_parameters.cache_clear()
    monkeypatch.setattr(signature_module.inspect, "signature", unavailable)

    args, kwargs = NUMPY_PROTOCOL_RUNTIME._normalize_array_function_args_and_kwargs(
        func=target,
        args=("value",),
        kwargs={"flag": True},
    )

    assert args == ("value",)
    assert kwargs == {"flag": True}


class _UninspectableFunction:
    """Stand-in for a NumPy 2.0 C function whose signature ``inspect`` cannot read."""

    def __init__(self, name: str) -> None:
        self.__name__ = name


@pytest.mark.parametrize(
    ("name", "args", "kwargs", "expected"),
    [
        ("bincount", (), {"x": 1, "minlength": 2}, "'x'"),
        ("dot", (), {"a": 1, "b": 2}, ((1, 2), {})),
        ("lexsort", (), {"keys": 1, "axis": 0}, ((1,), {"axis": 0})),
        ("concatenate", (), {"arrays": 1}, "'arrays'"),
        ("inner", (1,), {"b": 2}, "'b'"),
        ("vdot", (), {"a": 1, "b": 2}, "'a, b'"),
        ("where", (1,), {"x": 2, "y": 3}, "'x, y'"),
        # NumPy's C empty_like accepts the prototype its signature marks positional-only.
        ("empty_like", (), {"prototype": 1, "dtype": 2}, ((1,), {"dtype": 2})),
    ],
)
def test_uninspectable_c_functions_bind_their_published_signatures(
    name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    expected: object,
) -> None:
    signature_module.positional_parameters.cache_clear()
    function = _UninspectableFunction(name)
    normalize = NUMPY_PROTOCOL_RUNTIME._normalize_array_function_args_and_kwargs
    if isinstance(expected, str):
        with pytest.raises(TypeError, match=f"passed as keyword arguments: {expected}"):
            normalize(func=function, args=args, kwargs=kwargs)
    else:
        assert normalize(func=function, args=args, kwargs=kwargs) == expected


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
