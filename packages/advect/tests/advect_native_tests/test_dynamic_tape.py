"""Tests for native dynamic-tape ownership and derivative execution."""

from __future__ import annotations

import gc
import weakref
from typing import Any, cast

import pytest

from advect import _native_core as native


def _input(
    tape: native.DynamicTape,
    value: object,
    *,
    active: bool = True,
) -> int:
    shape = tuple(getattr(value, "shape", ()))
    dtype = getattr(value, "dtype", "float64")
    return tape.record_input(value, shape, dtype, active=active)


def _operation(
    tape: native.DynamicTape,
    op: str,
    parents: list[int],
    value: object,
    *,
    parent_positions: list[int] | None = None,
    literals: list[object] | None = None,
    attrs: dict[str, object] | None = None,
    residual: object | None = None,
) -> int:
    shape = tuple(getattr(value, "shape", ()))
    dtype = getattr(value, "dtype", "float64")
    positions = list(range(len(parents))) if parent_positions is None else parent_positions
    literal_values = [] if literals is None else literals
    if literal_values:
        node_id = tape.record_operation_with_literals(
            op,
            parents,
            positions,
            literal_values,
            value,
            {} if attrs is None else attrs,
            shape,
            dtype,
        )
    else:
        node_id = tape.record_operation(
            op,
            parents,
            value,
            {} if attrs is None else attrs,
            shape,
            dtype,
        )
    if residual is not None:
        tape.record_residual(node_id, residual)
    return node_id


def _freeze(
    tape: native.DynamicTape,
    *,
    jvps: dict[str, object] | None = None,
    vjps: dict[str, object] | None = None,
    reverse_needs: dict[str, tuple[bool, bool, bool]] | None = None,
) -> None:
    jvp_rules = {} if jvps is None else jvps
    vjp_rules = {} if vjps is None else vjps
    needs = {} if reverse_needs is None else reverse_needs
    tape.freeze(
        [jvp_rules.get(op) for op in tape.op_names],
        [vjp_rules.get(op) for op in tape.op_names],
        [
            needs.get(op, (True, True, False)) if vjp_rules.get(op) is not None else None
            for op in tape.op_names
        ],
    )


def test_dynamic_tape_records_compact_shared_arena_nodes() -> None:
    tape = native.DynamicTape()
    left = _input(tape, 2.0)
    right = _input(tape, 3.0)
    third = _input(tape, 4.0)
    _operation(tape, "multiply", [left, right], 6.0)
    _operation(tape, "sum3", [left, right, third], 9.0)
    _operation(
        tape,
        "scale",
        [left],
        10.0,
        parent_positions=[1],
        literals=[5.0],
    )

    stats = tape.stats()
    assert stats["node_count"] == 6
    assert stats["edge_count"] == 3
    assert stats["operand_position_count"] == 1
    assert stats["literal_count"] == 1
    assert stats["node_core_bytes"] <= 24
    assert stats["input_ref_bytes"] <= 16
    structural = stats["native_structural"]
    assert structural["nodes"]["len"] == stats["node_count"]
    assert structural["edges"]["len"] == stats["edge_count"]
    assert all(entry["capacity"] >= entry["len"] for entry in structural.values())
    assert stats["native_structural_bytes"] == sum(entry["bytes"] for entry in structural.values())
    assert tape.op_names == ["advect.input", "multiply", "sum3", "scale"]
    tape.release_payloads()


def test_passive_inputs_remain_primal_operands_without_receiving_cotangents() -> None:
    seen_active_positions: list[tuple[int, ...]] = []

    def multiply_vjp(
        _output: float,
        operands: tuple[float, float],
        cotangent: float,
        _attrs: object,
        active_positions: tuple[int, ...],
        _residual: object,
        _parent_specs: object,
        _source: object,
    ) -> list[float | None]:
        seen_active_positions.append(active_positions)
        return [cotangent * operands[1], cotangent * operands[0]]

    tape = native.DynamicTape()
    passive = _input(tape, 3.0, active=False)
    active = _input(tape, 2.0)
    product = _operation(tape, "multiply", [passive, active], 6.0)
    tape.mark_output(product)
    _freeze(tape, vjps={"multiply": multiply_vjp})

    assert native.dynamic_vjp(tape, [(product, 1.0)], [active]) == [3.0]
    assert seen_active_positions == [(1,)]
    tape.release_payloads()


def test_dynamic_tape_preserves_custom_operation_schema_versions() -> None:
    tape = native.DynamicTape()
    input_id = _input(tape, 2.0)
    first = tape.record_operation(
        "custom.versioned",
        [input_id],
        2.0,
        {},
        (),
        "float64",
        schema_version=7,
    )
    assert first == 1

    with pytest.raises(ValueError, match=r"already schema version 7, not 8"):
        tape.record_operation(
            "custom.versioned",
            [input_id],
            2.0,
            {},
            (),
            "float64",
            schema_version=8,
        )
    tape.release_payloads()


def test_native_real_linearity_analysis_returns_dependency_set_and_specific_errors() -> None:
    linear = native.DynamicTape()
    coefficient = _input(linear, 3.0)
    tangent = _input(linear, 0.0)
    output = _operation(linear, "array.multiply", [coefficient, tangent], 0.0)
    zeroed = _operation(
        linear,
        "array.multiply",
        [tangent],
        0.0,
        parent_positions=[0],
        literals=[0.0],
    )
    linear.mark_output(output)
    linear.mark_output(zeroed)
    _freeze(linear)

    assert linear.analyze_real_linearity([tangent], "tests.linear") == [tangent, output]
    linear.release_payloads()

    nonlinear = native.DynamicTape()
    tangent = _input(nonlinear, 0.0)
    output = _operation(nonlinear, "array.multiply", [tangent, tangent], 0.0)
    nonlinear.mark_output(output)
    _freeze(nonlinear)

    with pytest.raises(
        ValueError,
        match=r"JVP rule for 'tests\.nonlinear'.*multiplies tangent-dependent operands"
        r".*'array\.multiply'.*tape value %1",
    ):
        nonlinear.analyze_real_linearity([tangent], "tests.nonlinear")
    nonlinear.release_payloads()


def test_native_forward_and_reverse_handle_repeated_parents() -> None:
    def multiply_jvp(
        _output: float,
        operands: tuple[float, float],
        tangents: tuple[float | None, float | None],
        _attrs: object,
        _source: object,
    ) -> float:
        left, right = operands
        left_tangent = 0.0 if tangents[0] is None else tangents[0]
        right_tangent = 0.0 if tangents[1] is None else tangents[1]
        return left_tangent * right + left * right_tangent

    def multiply_vjp(
        _output: float,
        operands: tuple[float, float],
        cotangent: float,
        _attrs: object,
        _active: tuple[int, ...],
        _residual: object,
        _parent_specs: object,
        _source: object,
    ) -> list[float]:
        left, right = operands
        return [cotangent * right, cotangent * left]

    tape = native.DynamicTape()
    left = _input(tape, 2.0)
    right = _input(tape, 3.0)
    product = _operation(tape, "multiply", [left, right], 6.0)
    square = _operation(tape, "multiply", [product, product], 36.0)
    tape.mark_output(square)
    _freeze(
        tape,
        jvps={"multiply": multiply_jvp},
        vjps={"multiply": multiply_vjp},
    )

    assert native.dynamic_jvp(tape, [(left, 1.0), (right, 1.0)], [square]) == [60.0]
    assert native.dynamic_vjp(tape, [(square, 1.0)], [left, right]) == [36.0, 24.0]
    assert native.dynamic_jvp_many(
        tape,
        [
            [(left, 1.0), (right, 0.0)],
            [(left, 0.0), (right, 1.0)],
        ],
        [square],
    ) == [[36.0], [24.0]]
    assert native.dynamic_vjp_many(
        tape,
        [
            [(square, 1.0)],
            [(square, 2.0)],
        ],
        [left, right],
    ) == [[36.0, 24.0], [72.0, 48.0]]
    tape.release_payloads()


def test_native_multi_seed_traversals_are_bounded() -> None:
    tape = native.DynamicTape()
    value = _input(tape, 2.0)
    tape.mark_output(value)
    _freeze(tape)

    with pytest.raises(ValueError, match="at most 16 seeds"):
        native.dynamic_jvp_many(tape, [[(value, 1.0)]] * 17, [value])
    with pytest.raises(ValueError, match="at most 16 seeds"):
        native.dynamic_vjp_many(tape, [[(value, 1.0)]] * 17, [value])
    tape.release_payloads()


def test_native_multi_seed_reverse_uses_one_optional_batched_callback() -> None:
    scalar_calls: list[float] = []
    batched_calls: list[tuple[float, ...]] = []

    def transpose(
        _output: float,
        _operands: tuple[float],
        cotangent: float,
        *_args: object,
    ) -> list[float]:
        scalar_calls.append(cotangent)
        return [3.0 * cotangent]

    def transpose_many(
        _output: float,
        _operands: tuple[float],
        cotangents: tuple[float, ...],
        *_args: object,
    ) -> list[list[float]]:
        batched_calls.append(cotangents)
        return [[3.0 * cotangent] for cotangent in cotangents]

    cast("Any", transpose).__advect_vjp_many__ = transpose_many

    tape = native.DynamicTape()
    value = _input(tape, 2.0)
    unrelated = _input(tape, 5.0)
    scaled = _operation(tape, "scale", [value], 6.0)
    tape.mark_output(scaled)
    tape.mark_output(unrelated)
    _freeze(tape, vjps={"scale": transpose})

    assert native.dynamic_vjp(tape, [(scaled, 1.0)], [value]) == [3.0]
    assert native.dynamic_vjp_many(
        tape,
        [
            [(scaled, 2.0)],
            [(unrelated, 4.0)],
        ],
        [value, unrelated],
    ) == [[6.0, None], [None, 4.0]]
    assert scalar_calls == [1.0]
    assert batched_calls == [(2.0,)]
    tape.release_payloads()


def test_native_reverse_preserves_mixed_operand_order_and_residual_payload() -> None:
    seen: list[tuple[Any, ...]] = []

    class Slot:
        payload = "forward-residual"

        def close(self) -> None:
            return

    def transpose(
        output: float,
        operands: tuple[float, float],
        cotangent: float,
        attrs: object,
        active: tuple[int, ...],
        residual: object,
        parent_specs: object,
        _source: object,
    ) -> list[float | None]:
        seen.append((output, operands, attrs, active, residual, parent_specs))
        return [None, cotangent * operands[0]]

    tape = native.DynamicTape()
    value = _input(tape, 2.0)
    output = _operation(
        tape,
        "scale",
        [value],
        6.0,
        parent_positions=[1],
        literals=[3.0],
        attrs={"kind": "literal-left"},
        residual=Slot(),
    )
    tape.mark_output(output)
    _freeze(
        tape,
        vjps={"scale": transpose},
        reverse_needs={"scale": (True, True, True)},
    )

    assert native.dynamic_vjp(tape, [(output, 2.0)], [value]) == [6.0]
    assert seen[0][:5] == (
        6.0,
        (3.0, 2.0),
        {"kind": "literal-left"},
        (1,),
        "forward-residual",
    )
    tape.release_payloads()


def test_reverse_only_pruning_drops_values_unused_by_the_vjp() -> None:
    observed: list[tuple[object, tuple[object, ...]]] = []

    def transpose(
        output: object,
        operands: tuple[object, ...],
        cotangent: float,
        *_args: object,
    ) -> list[float]:
        observed.append((output, operands))
        return [cotangent, cotangent]

    tape = native.DynamicTape()
    left_value = _Payload()
    right_value = _Payload()
    output_value = _Payload()
    refs = [weakref.ref(value) for value in (left_value, right_value, output_value)]
    left = _input(tape, left_value)
    right = _input(tape, right_value)
    output = _operation(tape, "add", [left, right], output_value)
    tape.mark_output(output)
    _freeze(
        tape,
        vjps={"add": transpose},
        reverse_needs={"add": (False, False, False)},
    )
    del left_value, right_value, output_value

    assert tape.stats()["retained_value_count"] == 3
    tape.prune_reverse_payloads()
    gc.collect()

    assert tape.stats()["retained_value_count"] == 0
    assert all(ref() is None for ref in refs)
    assert native.dynamic_vjp(tape, [(output, 2.0)], [left, right]) == [2.0, 2.0]
    assert observed == [(None, (None, None))]
    tape.release_payloads()


def test_consuming_reverse_releases_a_primal_at_its_last_callback() -> None:
    middle_ref: weakref.ReferenceType[object]
    seen: list[str] = []

    def first_transpose(
        _output: object,
        _operands: object,
        cotangent: float,
        *_args: object,
    ) -> list[float]:
        assert middle_ref() is None
        seen.append("first")
        return [cotangent]

    def second_transpose(
        _output: object,
        operands: tuple[object],
        cotangent: float,
        *_args: object,
    ) -> list[float]:
        assert operands[0] is middle_ref()
        seen.append("second")
        return [cotangent]

    tape = native.DynamicTape()
    source_value = _Payload()
    source = _input(tape, source_value)
    middle_value = _Payload()
    middle_ref = weakref.ref(middle_value)
    middle = _operation(tape, "first", [source], middle_value)
    output = _operation(tape, "second", [middle], _Payload())
    tape.mark_output(output)
    _freeze(
        tape,
        vjps={"first": first_transpose, "second": second_transpose},
        reverse_needs={
            "first": (False, False, False),
            "second": (False, True, False),
        },
    )
    del source_value, middle_value

    assert native.dynamic_vjp(tape, [(output, 1.0)], [source], consume=True) == [1.0]
    gc.collect()
    assert seen == ["second", "first"]
    assert middle_ref() is None
    assert tape.stats()["retained_value_count"] == 0


def test_reusable_reverse_retains_and_closes_residual_once() -> None:
    events: list[object] = []
    residual = _Residual("token", events)

    def transpose(
        _output: object,
        _operands: object,
        cotangent: float,
        _attrs: object,
        _active: object,
        payload: object,
        _parent_specs: object,
        _source: object,
    ) -> list[float]:
        assert payload == "token"
        return [cotangent]

    tape = native.DynamicTape()
    value = _input(tape, 1.0)
    output = _operation(tape, "identity", [value], 1.0, residual=residual)
    tape.mark_output(output)
    _freeze(
        tape,
        vjps={"identity": transpose},
        reverse_needs={"identity": (False, False, True)},
    )

    assert native.dynamic_vjp(tape, [(output, 2.0)], [value]) == [2.0]
    assert native.dynamic_vjp(tape, [(output, 3.0)], [value]) == [3.0]
    assert residual.close_count == 0
    tape.release_payloads()
    tape.release_payloads()
    assert residual.close_count == 1
    assert events == ["token"]


def test_consuming_reverse_releases_literals_after_their_callback() -> None:
    literal = _Payload()
    literal_ref = weakref.ref(literal)

    def transpose(
        _output: object,
        operands: tuple[object, object],
        cotangent: float,
        *_args: object,
    ) -> list[float | None]:
        assert operands[0] is literal_ref()
        return [None, cotangent]

    tape = native.DynamicTape()
    value = _input(tape, 2.0)
    output = _operation(
        tape,
        "scale",
        [value],
        6.0,
        parent_positions=[1],
        literals=[literal],
    )
    tape.mark_output(output)
    _freeze(
        tape,
        vjps={"scale": transpose},
        reverse_needs={"scale": (False, True, False)},
    )
    del literal
    tape.prune_reverse_payloads()

    assert tape.stats()["literal_count"] == 1
    assert native.dynamic_vjp(tape, [(output, 3.0)], [value], consume=True) == [3.0]
    gc.collect()
    assert literal_ref() is None
    assert tape.stats()["literal_count"] == 0


def test_sparse_tuple_cotangents_accumulate_slotwise() -> None:
    def first_vjp(
        _output: object,
        _operands: object,
        cotangent: object,
        _attrs: object,
        _active: object,
        _residual: object,
        _parent_specs: object,
        _source: object,
    ) -> list[tuple[object | None, object | None]]:
        return [(cotangent, None)]

    def second_vjp(
        _output: object,
        _operands: object,
        cotangent: object,
        _attrs: object,
        _active: object,
        _residual: object,
        _parent_specs: object,
        _source: object,
    ) -> list[tuple[object | None, object | None]]:
        return [(None, cotangent)]

    tape = native.DynamicTape()
    pair = _input(tape, (1.0, 2.0))
    first = _operation(tape, "first", [pair], 1.0)
    second = _operation(tape, "second", [pair], 2.0)
    tape.mark_output(first)
    tape.mark_output(second)
    _freeze(tape, vjps={"first": first_vjp, "second": second_vjp})

    assert native.dynamic_vjp(tape, [(first, 3.0), (second, 5.0)], [pair]) == [(3.0, 5.0)]
    tape.release_payloads()


def test_reverse_allows_outer_tape_reentry_and_rejects_same_tape_recursion() -> None:
    outer = native.DynamicTape()
    outer_input = _input(outer, 4.0)
    recursive_errors: list[str] = []

    inner = native.DynamicTape()
    inner_input = _input(inner, 2.0)
    inner_output = _operation(inner, "identity", [inner_input], 2.0)

    def transpose(
        _output: object,
        _operands: object,
        cotangent: object,
        _attrs: object,
        _active: object,
        _residual: object,
        _parent_specs: object,
        _source: object,
    ) -> list[object]:
        _operation(outer, "nested", [outer_input], 4.0)
        with pytest.raises(RuntimeError, match="already executing") as error:
            native.dynamic_vjp(inner, [(inner_output, cotangent)], [inner_input])
        recursive_errors.append(str(error.value))
        return [cotangent]

    inner.mark_output(inner_output)
    _freeze(inner, vjps={"identity": transpose})

    assert native.dynamic_vjp(
        inner,
        [(inner_output, 1.0)],
        [inner_input],
        consume=True,
    ) == [1.0]
    assert outer.node_count == 2
    assert recursive_errors
    inner.release_payloads()
    outer.release_payloads()


class _Payload:
    pass


class _Residual:
    def __init__(
        self, payload: object, events: list[object], error: Exception | None = None
    ) -> None:
        self.payload = payload
        self.events = events
        self.error = error
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1
        self.events.append(self.payload)
        if self.error is not None:
            raise self.error


def test_release_closes_every_residual_once_and_propagates_first_error() -> None:
    events: list[object] = []
    first = _Residual("first", events, ValueError("first release failed"))
    second = _Residual("second", events)
    tape = native.DynamicTape()
    value = _input(tape, 1.0)
    first_node = _operation(tape, "identity", [value], 1.0, residual=first)
    output = _operation(tape, "identity", [first_node], 1.0, residual=second)
    tape.mark_output(output)
    _freeze(tape)

    with pytest.raises(ValueError, match="first release failed"):
        tape.release_payloads()
    tape.release_payloads()

    assert events == ["first", "second"]
    assert first.close_count == 1
    assert second.close_count == 1
    assert tape.is_consumed


def test_consume_releases_payloads_even_when_reverse_callback_fails() -> None:
    events: list[object] = []
    residual = _Residual(_Payload(), events)
    payload = _Payload()
    payload_ref = weakref.ref(payload)
    residual_ref = weakref.ref(residual.payload)

    def fail(*_args: object) -> list[object]:
        message = "expected callback failure"
        raise ValueError(message)

    tape = native.DynamicTape()
    value = _input(tape, payload)
    output = _operation(tape, "fail", [value], _Payload(), residual=residual)
    tape.mark_output(output)
    _freeze(tape, vjps={"fail": fail})
    del payload

    with pytest.raises(ValueError, match="expected callback failure"):
        native.dynamic_vjp(tape, [(output, 1.0)], [value], consume=True)
    gc.collect()

    assert residual.close_count == 1
    assert payload_ref() is None
    assert residual_ref() is not None  # retained by the test's event log
    assert tape.is_consumed
    assert tape.stats()["retained_value_count"] == 0


def test_consume_closes_residual_when_payload_access_fails() -> None:
    events: list[object] = []

    class BrokenPayload(_Residual):
        @property
        def payload(self) -> object:
            message = "expected payload failure"
            raise ValueError(message)

        @payload.setter
        def payload(self, value: object) -> None:
            self._payload = value

        def close(self) -> None:
            self.close_count += 1
            self.events.append(self._payload)

    residual = BrokenPayload("token", events)
    tape = native.DynamicTape()
    value = _input(tape, 1.0)
    output = _operation(tape, "identity", [value], 1.0, residual=residual)
    tape.mark_output(output)
    _freeze(
        tape,
        vjps={"identity": lambda *_args: [1.0]},
        reverse_needs={"identity": (False, False, True)},
    )

    with pytest.raises(ValueError, match="expected payload failure"):
        native.dynamic_vjp(tape, [(output, 1.0)], [value], consume=True)

    assert residual.close_count == 1
    assert events == ["token"]
    assert tape.stats()["residual_count"] == 0
