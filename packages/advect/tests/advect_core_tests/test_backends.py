"""Tests for the narrow backend dispatch boundary."""

from __future__ import annotations

from contextlib import contextmanager
from importlib import import_module
from typing import TYPE_CHECKING, Any

import array_api_strict as xp
import numpy as np
import pytest

from advect.core._array_api.frontend import ArrayAPITracer, _accepts_array_api
from advect.core._backend_hooks import resolve_backend_hooks
from advect.core._backends import (
    dispatch_input,
    get_hook,
    register_hook,
    register_input_handler,
)
from advect.core._context import _set_active_recorder
from advect.core._native import DynamicTape
from advect_core_tests._backend_state import isolated_backend_state

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _restore_backend_state() -> Iterator[None]:
    with isolated_backend_state():
        yield


@contextmanager
def _active_tape() -> Iterator[DynamicTape]:
    tape = DynamicTape()
    _set_active_recorder(tape, trace_kind="autodiff_dynamic")
    try:
        yield tape
    finally:
        _set_active_recorder(None)


def test_custom_input_handler_dispatches_the_matching_value() -> None:
    handled: list[object] = []

    def accepts(value: Any) -> bool:
        return type(value) is object

    def handle(value: object, name: str | None = None) -> tuple[object, str | None]:
        handled.append(value)
        return value, name

    register_input_handler(accepts, handle)
    value = object()

    assert dispatch_input(value, name="custom") == (value, "custom")
    assert handled == [value]


def test_exact_input_handler_bypasses_its_predicate() -> None:
    predicate_calls = 0

    def should_not_run(value: Any) -> bool:
        nonlocal predicate_calls
        predicate_calls += 1
        return isinstance(value, str)

    def handle(value: str, name: str | None = None) -> tuple[str, str | None]:
        return value, name

    register_input_handler(should_not_run, handle, exact_types=(str,))

    assert dispatch_input("value", name="input") == ("value", "input")
    assert predicate_calls == 0


def test_conflicting_exact_handlers_are_rejected() -> None:
    def first(value: Any, name: str | None = None) -> tuple[Any, str | None]:
        return value, name

    def second(value: Any, name: str | None = None) -> tuple[str | None, Any]:
        return name, value

    register_input_handler(lambda value: isinstance(value, str), first, exact_types=(str,))

    with pytest.raises(ValueError, match="already registered for str"):
        register_input_handler(
            lambda value: isinstance(value, str),
            second,
            exact_types=(str,),
        )


def test_unsupported_input_reports_the_missing_backend() -> None:
    with pytest.raises(TypeError, match="No backend can handle"):
        dispatch_input(object())


def test_core_array_api_semantics_precede_an_eager_provider_frontend() -> None:
    import_module("advect.numpy")

    value = xp.asarray([1.0, 2.0], dtype=xp.float32)
    with _active_tape():
        traced = dispatch_input(value)

    assert isinstance(traced, ArrayAPITracer)


def test_builtin_numpy_provider_is_registered_at_import() -> None:
    value = np.array([1.0, 2.0])

    with _active_tape():
        traced = dispatch_input(value)

    assert type(traced).__name__ == "TracedArray"


def test_hook_registration_is_idempotent_for_the_identical_callable() -> None:
    def evaluate(op: str, inputs: tuple[Any, ...], attrs: dict[str, Any]) -> object:
        return op, inputs, attrs

    register_hook("idempotent.evaluate_op", evaluate)
    resolved, _decode = resolve_backend_hooks("idempotent.add", ())
    assert resolved is evaluate

    register_hook("idempotent.evaluate_op", evaluate)

    resolved_again, _decode = resolve_backend_hooks("idempotent.add", ())
    assert resolved_again is evaluate
    assert get_hook("idempotent.evaluate_op") is evaluate


def test_hook_registration_rejects_rebinding() -> None:
    def first(op: str, inputs: tuple[Any, ...], attrs: dict[str, Any]) -> object:
        return op, inputs, attrs, "first"

    def second(op: str, inputs: tuple[Any, ...], attrs: dict[str, Any]) -> object:
        return op, inputs, attrs, "second"

    register_hook("single_assignment.evaluate_op", first)

    with pytest.raises(ValueError, match="already registered"):
        register_hook("single_assignment.evaluate_op", second)

    resolved, _decode = resolve_backend_hooks("single_assignment.add", ())
    assert resolved is first
    assert get_hook("single_assignment.evaluate_op") is first


def test_generic_array_api_version_guard_never_claims_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(np, "__array_api_version__", "2099.12")

    assert not _accepts_array_api(np.arange(2.0))
