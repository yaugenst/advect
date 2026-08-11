"""Core and staged ownership contracts for primitive residuals."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import advect as ad
from advect.core._backends import dispatch_input
from advect.core._context import _set_active_recorder
from advect.core._native import DynamicTape, dynamic_vjp

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def _active_tape() -> Iterator[DynamicTape]:
    tape = DynamicTape()
    _set_active_recorder(tape, trace_kind="autodiff_dynamic")
    try:
        yield tape
    finally:
        _set_active_recorder(None)


def test_primitive_result_is_public_and_direct_calls_release_immediately() -> None:
    released: list[object] = []
    token = object()

    @ad.primitive(name="tests.residual.direct", residual=True)
    def primitive(x: float) -> ad.PrimitiveResult[float]:
        return ad.PrimitiveResult(x * 2, token, release=released.append)

    assert primitive(3.0) == 6.0
    assert released == [token]


def test_implementation_result_must_match_declared_residual_contract() -> None:
    @ad.primitive(name="tests.residual.declared_mismatch", residual=True)
    def declared(x: float) -> float:
        return x

    with pytest.raises(TypeError, match=r"declares residual=True.*PrimitiveResult"):
        declared(1.0)

    released: list[object] = []
    token = object()

    @ad.primitive(name="tests.residual.undeclared_mismatch")
    def ordinary(x: float) -> ad.PrimitiveResult[float]:
        return ad.PrimitiveResult(x, token, release=released.append)

    with pytest.raises(TypeError, match=r"does not declare residual=True"):
        ordinary(1.0)
    assert released == [token]


def test_dynamic_tape_pairs_each_residual_with_its_primitive_node() -> None:
    released: list[object] = []
    tokens = [object(), object()]
    calls = iter(tokens)

    @ad.primitive(name="tests.residual.exact_pairing", residual=True)
    def primitive(
        x: np.ndarray[Any, Any],
    ) -> ad.PrimitiveResult[np.ndarray[Any, Any]]:
        return ad.PrimitiveResult(x + 1, next(calls), release=released.append)

    with _active_tape() as tape:
        traced = dispatch_input(np.array(2.0))
        first = primitive(traced)
        second = primitive(traced)
        input_id = traced.node_id
        output_ids = [first.node_id, second.node_id]

    observed: list[object] = []

    def transpose(
        _output: object,
        _operands: object,
        cotangent: object,
        _attrs: object,
        _active: object,
        residual: object,
        _parent_specs: object,
        _source: object,
    ) -> list[object]:
        observed.append(residual)
        return [cotangent]

    for output_id in output_ids:
        tape.mark_output(output_id)
    tape.freeze(
        [None] * len(tape.op_names),
        [transpose if op == primitive.op_name else None for op in tape.op_names],
        [(True, True, True) if op == primitive.op_name else None for op in tape.op_names],
    )
    assert tape.stats()["residual_count"] == 2
    assert dynamic_vjp(
        tape,
        [(output_ids[0], np.array(1.0)), (output_ids[1], np.array(1.0))],
        [input_id],
    ) == [np.array(2.0)]
    assert observed == list(reversed(tokens))

    tape.release_payloads()
    assert released == tokens
    assert tape.stats()["residual_count"] == 0
    tape.release_payloads()
    assert released == tokens


def test_multi_output_residual_belongs_to_the_primitive_parent() -> None:
    released: list[object] = []
    token = object()

    @ad.primitive(name="tests.residual.multi_output", residual=True)
    def primitive(
        x: np.ndarray[Any, Any],
    ) -> ad.PrimitiveResult[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]:
        return ad.PrimitiveResult((x, x + 1), token, release=released.append)

    with _active_tape() as tape:
        traced = dispatch_input(np.array(2.0))
        first, second = primitive(traced)
        first_id = first.node_id
        second_id = second.node_id

    assert first_id != second_id
    assert tape.op_names == ["advect.input", primitive.op_name, "advect.getoutput"]
    assert tape.node_count == 4
    assert tape.stats()["residual_count"] == 1
    tape.release_payloads()
    assert released == [token]


def test_invalid_primitive_output_releases_unattached_residual() -> None:
    released: list[object] = []
    token = object()

    @ad.primitive(name="tests.residual.invalid_output", residual=True)
    def primitive(x: np.ndarray[Any, Any]) -> ad.PrimitiveResult[object]:
        del x
        return ad.PrimitiveResult(object(), token, release=released.append)

    with _active_tape():
        traced = dispatch_input(np.array(2.0))
        with pytest.raises(TypeError, match="invalid output"):
            primitive(traced)

    assert released == [token]


def test_tape_attempts_every_residual_release_when_one_fails() -> None:
    released: list[str] = []

    def fail(payload: object) -> None:
        released.append(str(payload))
        message = "release failed"
        raise RuntimeError(message)

    def succeed(payload: object) -> None:
        released.append(str(payload))

    @ad.primitive(name="tests.residual.release_failure.first", residual=True)
    def first(
        x: np.ndarray[Any, Any],
    ) -> ad.PrimitiveResult[np.ndarray[Any, Any]]:
        return ad.PrimitiveResult(x, "first", release=fail)

    @ad.primitive(name="tests.residual.release_failure.second", residual=True)
    def second(
        x: np.ndarray[Any, Any],
    ) -> ad.PrimitiveResult[np.ndarray[Any, Any]]:
        return ad.PrimitiveResult(x, "second", release=succeed)

    with _active_tape() as tape:
        traced = dispatch_input(np.array(2.0))
        first(traced)
        second(traced)

    with pytest.raises(RuntimeError, match="release failed"):
        tape.release_payloads()
    assert released == ["first", "second"]
    assert tape.stats()["residual_count"] == 0


def test_staged_artifact_never_serializes_or_returns_a_residual() -> None:
    released: list[object] = []
    token = object()
    implementation_calls = 0

    @ad.primitive(name="tests.residual.staged", residual=True)
    def primitive(x: np.ndarray[Any, Any]) -> ad.PrimitiveResult[np.ndarray[Any, Any]]:
        nonlocal implementation_calls
        implementation_calls += 1
        return ad.PrimitiveResult(x * 2, token, release=released.append)

    @primitive.def_abstract
    def abstract(x: ad.AbstractValue) -> ad.ArraySpec:
        return x.spec

    program = ad.stage(
        lambda x: primitive(x),  # noqa: PLW0108 - explicit trace boundary
        specs=(ad.ArraySpec((2,), "float64"),),
    )
    assert implementation_calls == 0
    payload = program.to_dict()
    assert "PrimitiveResult" not in repr(payload)
    assert repr(token) not in repr(payload)

    value = np.array([2.0, 3.0])
    np.testing.assert_array_equal(program(value), 2 * value)
    assert implementation_calls == 1
    assert released == [token]


def test_staged_custom_replay_under_dynamic_trace_stays_atomic() -> None:
    seen_input_types: list[type[object]] = []

    @ad.primitive(name="tests.residual.staged_atomic")
    def primitive(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        seen_input_types.append(type(x))
        return x * x

    @primitive.def_abstract
    def abstract(x: ad.AbstractValue) -> ad.ArraySpec:
        return x.spec

    program = ad.stage(
        lambda x: primitive(x),  # noqa: PLW0108 - explicit trace boundary
        specs=(ad.ArraySpec((2,), "float64"),),
    )
    value = np.array([2.0, 3.0])

    with _active_tape() as tape:
        traced = dispatch_input(value)
        program(traced)

    assert seen_input_types == [np.ndarray]
    assert tape.op_names == ["advect.input", primitive.op_name]
    assert tape.node_count == 2


def test_nested_trace_rejects_residual_before_calling_implementation() -> None:
    calls = 0

    @ad.primitive(name="tests.residual.nested", residual=True)
    def primitive(
        x: np.ndarray[Any, Any],
    ) -> ad.PrimitiveResult[np.ndarray[Any, Any]]:
        nonlocal calls
        calls += 1
        return ad.PrimitiveResult(x, object())

    with _active_tape():
        outer = dispatch_input(np.array(2.0))
        with _active_tape():
            inner = dispatch_input(outer)
            with pytest.raises(
                ad.TracingError,
                match=r"opaque residual.*first-order differentiation only",
            ):
                primitive(inner)

    assert calls == 0
