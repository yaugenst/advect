"""Tests for concrete NumPy evaluator routing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from hypothesis import example, given, settings, strategies as st
from hypothesis.extra import numpy as hnp

import advect as ad
import advect.numpy._protocol_eval as eval_module
from advect.core._array_api import providers as array_api_providers
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
        array_api_providers,
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
        array_api_providers,
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


def test_attributeless_nodes_share_one_bound_evaluator() -> None:
    multiply = bind_native_node_evaluator("array.multiply", {})
    reshape = bind_native_node_evaluator("array.reshape", {"shape": (2,)})

    assert bind_native_node_evaluator("array.multiply", {}) is multiply
    assert bind_native_node_evaluator("array.add", {}) is not multiply
    assert bind_native_node_evaluator("array.reshape", {"shape": (2,)}) is not reshape
    assert multiply.__advect_owned_output__
    left, right = np.array([2.0, 3.0]), np.array([5.0, 7.0])
    np.testing.assert_array_equal(multiply((left, right), np, None), [10.0, 21.0])


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


@st.composite
def _vecdot_case(draw: st.DrawFn) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], int]:
    shape = draw(hnp.array_shapes(max_dims=3, max_side=3))
    # Small integers keep every sum exact.
    left, right = (
        draw(hnp.arrays(np.float64, shape, elements=st.integers(-8, 8))) for _ in range(2)
    )
    return left, right, draw(st.integers(-len(shape), len(shape) - 1))


@settings(deadline=None)
@given(case=_vecdot_case(), function=st.sampled_from((np.vecdot, np.linalg.vecdot)))
# Square operands give the last-axis contraction the requested result shape.
@example(case=(np.arange(9.0).reshape(3, 3), np.eye(3)[::-1], 0), function=np.linalg.vecdot)
def test_vecdot_contracts_the_requested_axis_in_every_lifetime(
    case: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], int],
    function: Any,
) -> None:
    left, right, axis = case
    tangent = np.ones_like(left)

    def contract(x: Any, y: Any) -> Any:
        return function(x, y, axis=axis)

    expected = contract(left, right)
    primal, derivative = ad.jvp(contract, argnums=(0, 1))(left, right, tangents=(tangent, tangent))
    program = ad.stage(contract, left, right)

    np.testing.assert_array_equal(primal, expected)
    np.testing.assert_array_equal(derivative, contract(tangent, right) + contract(left, tangent))
    np.testing.assert_array_equal(program(left, right), expected)
    np.testing.assert_array_equal(
        ad.StagedProgram.from_dict(program.to_dict())(left, right), expected
    )
