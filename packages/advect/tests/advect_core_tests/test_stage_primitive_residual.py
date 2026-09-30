"""Core and staged ownership contracts for primitive residuals."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import advect as ad
from advect.core._backends import dispatch_input
from advect.core._context import _set_active_recorder
from advect.core._native import DynamicTape
from advect.core._primitive_call import _attach_residual
from advect.core._residual import _PrimitiveExecution, _ResidualSlot

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


def test_residual_is_released_when_tape_attachment_fails() -> None:
    released: list[object] = []
    token = object()
    execution = _PrimitiveExecution(token, _ResidualSlot(token, released.append))

    class FailingRecorder:
        @staticmethod
        def record_residual(_node_id: int, _residual: object) -> None:
            raise RuntimeError("recording failed")

    with pytest.raises(RuntimeError, match="recording failed"):
        _attach_residual(FailingRecorder(), 1, execution)  # type: ignore[arg-type]

    assert released == [token]


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
