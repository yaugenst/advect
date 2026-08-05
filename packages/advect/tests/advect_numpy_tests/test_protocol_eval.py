"""Tests for concrete NumPy evaluator routing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

import advect.numpy._protocol_eval as eval_module
from advect.core import _array_namespace
from advect.core._eval_dispatch import _can_donate_array, bind_native_node_evaluator
from advect.numpy._protocol_eval import ArrayProtocolEvalRuntime

if TYPE_CHECKING:
    import pytest


class _OwnedArrayWithoutWritableFlag:
    flags = type("_Flags", (), {"owndata": True})()
    base = None


def test_donation_uses_provider_capability_when_writability_flag_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _array_namespace,
        "_ARRAY_NAMESPACE_DONATION_CHECKER",
        lambda value: isinstance(value, _OwnedArrayWithoutWritableFlag),
    )

    assert _can_donate_array(_OwnedArrayWithoutWritableFlag())


def test_donation_rejects_views_before_consulting_provider_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_call(_value: object) -> bool:
        message = "provider capability should not run for a view"
        raise AssertionError(message)

    monkeypatch.setattr(
        _array_namespace,
        "_ARRAY_NAMESPACE_DONATION_CHECKER",
        unexpected_call,
    )
    value = _OwnedArrayWithoutWritableFlag()
    value.base = object()

    assert not _can_donate_array(value)


def test_array_output_ownership_is_classified_by_operation_semantics() -> None:
    addition = bind_native_node_evaluator("array.add", {})
    reshape = bind_native_node_evaluator("array.reshape", {"shape": (2, 2)})
    real = bind_native_node_evaluator("array.real", {})

    assert addition.__advect_owned_output__
    assert reshape.__advect_alias_positions__ == (0,)
    assert not hasattr(real, "__advect_owned_output__")
    assert not hasattr(real, "__advect_alias_positions__")


def test_evaluate_op_filters_unknown_kwargs_for_dynamic_callables() -> None:
    runtime = ArrayProtocolEvalRuntime()
    x = np.arange(6, dtype=np.float64)

    result = runtime.evaluate_op(
        "array.reshape",
        (x,),
        {
            "shape": (2, 3),
            "order": "C",
            "ignored": "drop-me",
        },
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 3)


def test_evaluate_op_allows_var_keyword_kwargs_for_backend_function() -> None:
    runtime = ArrayProtocolEvalRuntime()
    x = np.array([1.0, 2.0], dtype=np.float64)

    result = runtime.evaluate_op(
        "array.pad",
        (x,),
        {
            "pad_width": ((1, 1),),
            "mode": "constant",
            "constant_values": 1.5,
        },
    )

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, np.array([1.5, 1.0, 2.0, 1.5], dtype=np.float64))


def test_evaluate_op_var_keyword_kwargs_drop_internal_attrs() -> None:
    runtime = ArrayProtocolEvalRuntime()
    x = np.array([1.0, 2.0], dtype=np.float64)

    result = runtime.evaluate_op(
        "array.pad",
        (x,),
        {
            "pad_width": ((1, 1),),
            "mode": "constant",
            "constant_values": np.nan,
            "_advect_internal_flag": True,
        },
    )

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(
        result,
        np.array([np.nan, 1.0, 2.0, np.nan], dtype=np.float64),
        equal_nan=True,
    )


def test_evaluate_op_signature_fallback_allows_kwargs(monkeypatch: Any) -> None:
    runtime = ArrayProtocolEvalRuntime()
    x = np.array([1.0, 2.0], dtype=np.float64)

    original_signature = eval_module.inspect.signature

    def _patched_signature(func: object) -> object:
        if func is np.pad:
            msg = "signature unavailable"
            raise ValueError(msg)
        return original_signature(func)

    monkeypatch.setattr(eval_module.inspect, "signature", _patched_signature)

    result = runtime.evaluate_op(
        "array.pad",
        (x,),
        {
            "pad_width": ((1, 1),),
            "mode": "constant",
            "constant_values": np.inf,
            "_advect_internal_flag": True,
        },
    )

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(
        result,
        np.array([np.inf, 1.0, 2.0, np.inf], dtype=np.float64),
    )


def test_evaluate_op_checks_special_registry_with_decanonicalized_key() -> None:
    runtime = ArrayProtocolEvalRuntime()

    seen: dict[str, object] = {}

    def _special(inputs: tuple[object, ...], attrs: dict[str, object]) -> object:
        seen["inputs"] = inputs
        seen["attrs"] = attrs
        return "ok"

    runtime.register_evaluator("numpy.custom_op", _special)

    result = runtime.evaluate_op("array.custom_op", (1, 2), {"k": "v"})
    assert result == "ok"
    assert seen["inputs"] == (1, 2)
    assert seen["attrs"] == {"k": "v"}


def test_bind_evaluator_returns_none_for_unknown_operation() -> None:
    runtime = ArrayProtocolEvalRuntime()

    assert runtime.bind_evaluator("array.not_a_numpy_operation", {}) is None


def test_bound_plain_ufunc_preserves_direct_call_semantics() -> None:
    runtime = ArrayProtocolEvalRuntime()
    evaluator = runtime.bind_evaluator("array.add", {})
    assert evaluator is not None

    x = np.array([1.0, 2.0, 3.0])
    y = np.array([10.0, 20.0, 30.0])

    result = evaluator((x, y))

    np.testing.assert_allclose(np.asarray(result), np.array([11.0, 22.0, 33.0]))


def test_bound_ufunc_preserves_static_keyword_arguments_without_out() -> None:
    runtime = ArrayProtocolEvalRuntime()
    evaluator = runtime.bind_evaluator("array.add", {"dtype": "float32"})
    assert evaluator is not None

    result = evaluator(
        (
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([3.0, 4.0], dtype=np.float64),
        )
    )

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(result, np.array([4.0, 6.0], dtype=np.float32))
