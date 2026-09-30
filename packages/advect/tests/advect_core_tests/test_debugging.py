"""Tests for scoped debugging and user-facing trace diagnostics."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest

import advect as ad
from advect.core._context import _is_numerics_debug, is_debug, is_tracing


def test_debug_is_scoped_and_expands_live_tracer_repr() -> None:
    representations: list[str] = []

    def model(x: object) -> object:
        representations.append(repr(x))
        return np.sin(x)

    value = np.array([1.0, 2.0])
    tangent = np.ones_like(value)
    ad.jvp(model)(value, tangents=tangent)

    assert not is_debug()
    with ad.debug():
        assert is_debug()
        assert not _is_numerics_debug()
        with ad.debug(numerics=True):
            assert _is_numerics_debug()
        assert not _is_numerics_debug()
        ad.jvp(model)(value, tangents=tangent)
    assert not is_debug()

    normal, debug = representations
    assert "finite=" not in normal
    assert "values=" not in normal
    assert "finite=2/2" in debug


def _trace_state() -> tuple[bool, bool, bool]:
    return is_debug(), _is_numerics_debug(), is_tracing()


def test_debug_and_trace_state_are_isolated_per_thread() -> None:
    observed: list[tuple[bool, bool, bool]] = []

    def model(x: np.ndarray[Any, Any]) -> object:
        observed.append(_trace_state())
        with ThreadPoolExecutor(max_workers=1) as executor:
            observed.append(executor.submit(_trace_state).result(timeout=10))
        return np.sum(x * x)

    with ad.debug(numerics=True):
        ad.grad(model)(np.ones(2))
    assert observed == [(True, True, True), (False, False, False)]

    entered = threading.Event()
    release = threading.Event()

    def blocking_model(x: np.ndarray[Any, Any]) -> object:
        entered.set()
        assert release.wait(timeout=10)
        return np.sum(x)

    def worker() -> object:
        with ad.debug(numerics=True):
            return ad.grad(blocking_model)(np.ones(2))

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(worker)
        assert entered.wait(timeout=10)
        outside = _trace_state()
        release.set()
        future.result(timeout=10)
    assert outside == (False, False, False)


def test_debug_records_the_user_callsite_for_derivative_errors() -> None:
    @ad.primitive(name="tests.debugging.no_derivative")
    def opaque(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return 2 * x

    def model(x: np.ndarray[Any, Any]) -> object:
        return np.sum(opaque(x))

    with pytest.raises(ad.NoVJPError) as ordinary_error:
        ad.grad(model)(np.array([1.0, 2.0]))
    assert "with advect.debug()" in str(ordinary_error.value)

    with ad.debug(), pytest.raises(ad.NoVJPError) as caught:
        ad.grad(model)(np.array([1.0, 2.0]))

    message = str(caught.value)
    assert "tests.debugging.no_derivative" in message
    assert __file__ in message
    assert "in model()" in message
    assert "/src/advect/" not in caught.value.source_location


def test_numerics_debug_finds_primal_jvp_and_vjp_failures() -> None:
    value = np.array([0.0])

    # A JVP-only primitive is transposed structurally, so the VJP failure stays
    # in its JVP body whatever reverse rules the built-in operations use.
    @ad.primitive(name="tests.debugging.jvp_only_root")
    def root(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.sqrt(x)

    @root.def_jvp
    def root_jvp(output: Any, primals: object, tangents: tuple[Any, ...]) -> object:
        del primals
        return tangents[0] / (2.0 * output)

    with np.errstate(divide="ignore", invalid="ignore"):
        with ad.debug(numerics=True), pytest.raises(ad.NumericsError) as primal_error:
            ad.grad(lambda x: np.sum(np.log(x)))(np.array([-1.0]))
        with ad.debug(numerics=True), pytest.raises(ad.NumericsError) as jvp_error:
            ad.jvp(np.sqrt)(value, tangents=np.ones_like(value))
        with ad.debug(numerics=True), pytest.raises(ad.NumericsError) as vjp_error:
            ad.grad(lambda x: np.sum(root(x)))(value)
        with ad.debug(numerics=True), pytest.raises(ad.NumericsError) as builtin_vjp_error:
            ad.grad(lambda x: np.sum(np.sqrt(x)))(value)

    assert primal_error.value.phase == "primal evaluation"
    assert primal_error.value.op == "array.log"
    assert "finite=0/1" in primal_error.value.summary
    assert jvp_error.value.phase == "JVP propagation"
    assert jvp_error.value.op == "array.sqrt"
    assert vjp_error.value.phase == "VJP propagation"
    assert vjp_error.value.op == "array.divide"
    # Built-in elementwise rules transpose in closed form, so the user's op is named.
    assert builtin_vjp_error.value.phase == "VJP propagation"
    assert builtin_vjp_error.value.op == "array.sqrt"


def test_staged_programs_render_and_fail_with_local_graph_context() -> None:
    state = {"fail": False}

    @ad.primitive(name="tests.debugging.staged_failure")
    def fragile(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        if state["fail"]:
            msg = "provider exploded"
            raise RuntimeError(msg)
        return x + 1

    @fragile.def_abstract
    def fragile_abstract(x: ad.AbstractValue) -> ad.ArraySpec:
        return x.spec

    with ad.debug():
        program = ad.stage(fragile, specs=(ad.ArraySpec((2,), "float64"),))

    assert repr(program).startswith("StagedProgram(GraphStore(nodes=")
    assert "custom.tests.debugging.staged_failure" in str(program)
    assert str(program.graph).startswith("GraphStore(nodes=")
    state["fail"] = True

    with pytest.raises(RuntimeError, match="provider exploded") as caught:
        program(np.ones(2))

    notes = "\n".join(caught.value.__notes__)
    assert "Advect graph context:" in notes
    assert "->" in notes
    assert "custom.tests.debugging.staged_failure" in notes
    assert __file__ in notes


def test_large_staged_program_rendering_is_bounded() -> None:
    def many_operations(x: object) -> object:
        for _index in range(50):
            x = np.sin(x)
        return x

    program = ad.stage(many_operations, specs=(ad.ArraySpec((2,), "float64"),))

    assert program.graph.node_count == 51
    assert "11 nodes omitted" in str(program)
