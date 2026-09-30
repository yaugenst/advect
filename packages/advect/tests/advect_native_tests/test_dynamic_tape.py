"""Tests for native dynamic-tape ownership and derivative execution."""

from __future__ import annotations

import gc
import re
import weakref
from typing import TYPE_CHECKING, Any, cast

import pytest

from advect import _native_core as native

if TYPE_CHECKING:
    from collections.abc import Callable


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
    node_id = tape.record_operation(
        op,
        parents,
        value,
        {} if attrs is None else attrs,
        tuple(getattr(value, "shape", ())),
        getattr(value, "dtype", "float64"),
        input_positions=parent_positions,
        literals=() if literals is None else literals,
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


def test_dynamic_tape_reports_node_activity() -> None:
    tape = native.DynamicTape()
    active = _input(tape, 2.0)
    passive = _input(tape, 3.0, active=False)

    assert tape.node_is_active(active)
    assert not tape.node_is_active(passive)

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
    # A constant that is zero at this point is still tangent-independent, so
    # dividing by it scales linearly to IEEE infinities, as forward mode does.
    zero = _operation(linear, "advect.const", [], 0.0)
    divided = _operation(linear, "array.divide", [tangent, zero], 0.0)
    linear.mark_output(output)
    linear.mark_output(zeroed)
    linear.mark_output(divided)
    _freeze(linear)

    assert linear.analyze_real_linearity([tangent], "tests.linear") == [tangent, output, divided]
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


@pytest.mark.parametrize(
    "op",
    ["custom.mylib.negative", "custom.tests.nonlinear.sum", "custom.array.transpose"],
)
def test_native_real_linearity_does_not_trust_custom_names_of_linear_builtins(op: str) -> None:
    tape = native.DynamicTape()
    tangent = _input(tape, 0.0)
    output = _operation(tape, op, [tangent], 0.0)
    tape.mark_output(output)
    _freeze(tape)

    with pytest.raises(
        ValueError,
        match=rf"uses unsupported tangent-dependent operation '{re.escape(op)}'",
    ):
        tape.analyze_real_linearity([tangent], "tests.custom_name")
    tape.release_payloads()

    builtin = native.DynamicTape()
    tangent = _input(builtin, 0.0)
    output = _operation(builtin, "array.negative", [tangent], 0.0)
    builtin.mark_output(output)
    _freeze(builtin)
    assert builtin.analyze_real_linearity([tangent], "tests.builtin") == [tangent, output]
    builtin.release_payloads()


def _identity_tape(
    *, frozen: bool = True, vjp: Callable[..., object] | None = None
) -> native.DynamicTape:
    """Record %0 = input 2.0 and %1 = identity(%0), a marked output."""
    tape = native.DynamicTape()
    _operation(tape, "identity", [_input(tape, 2.0)], 2.0)
    tape.mark_output(1)
    if frozen:
        _freeze(tape, vjps={} if vjp is None else {"identity": vjp})
    return tape


# message fragment -> (frozen, error, misuse of an identity tape without rules)
_MISUSE: dict[str, tuple[bool, type[Exception], Callable[[native.DynamicTape], object]]] = {
    "dynamic tape operation name must not be empty": (
        False,
        ValueError,
        lambda tape: tape.record_operation("", [0], 1.0, {}, (), "float64"),
    ),
    "dynamic operand layout has 1 literals but no parent positions": (
        False,
        ValueError,
        lambda tape: tape.record_operation("add", [0], 2.0, {}, (), "float64", literals=[1.0]),
    ),
    "operand layout repeats parent position 0": (
        False,
        ValueError,
        lambda tape: tape.record_operation(
            "add", [0, 0], 2.0, {}, (), "float64", input_positions=[0, 0]
        ),
    ),
    "dynamic tape output %1 is already marked": (
        False,
        ValueError,
        lambda tape: tape.mark_output(1),
    ),
    "DynamicTape is already bound to a trace frame": (
        False,
        RuntimeError,
        lambda tape: [tape.bind_trace_frame(0, 1) for _ in range(2)],
    ),
    "DynamicTape node %1 already owns a primitive residual": (
        False,
        RuntimeError,
        lambda tape: [tape.record_residual(1, _Residual(None, [])) for _ in range(2)],
    ),
    "only rank-zero dynamic tape values can be weak scalars": (
        False,
        ValueError,
        lambda tape: tape.mark_weak(tape.record_input(1.0, (2,), "float64")),
    ),
    "dynamic tape node %9 does not exist": (False, ValueError, lambda tape: tape.value(9)),
    "DynamicTape JVP binding count 0 does not match operation count 2": (
        False,
        ValueError,
        lambda tape: tape.freeze([], [], []),
    ),
    "DynamicTape JVP binding 1 must be callable or None": (
        False,
        ValueError,
        lambda tape: tape.freeze([None, 1], [None, None], [None, None]),
    ),
    "DynamicTape reverse-needs count 1 does not match operation count 2": (
        False,
        ValueError,
        lambda tape: tape.freeze([None, None], [None, None], [None]),
    ),
    "DynamicTape VJP binding 1 is missing reverse-needs metadata": (
        False,
        ValueError,
        lambda tape: tape.freeze([None, None], [None, print], [None, None]),
    ),
    "DynamicTape reverse-needs metadata 1 has no VJP binding": (
        False,
        ValueError,
        lambda tape: tape.freeze([None, None], [None, None], [None, (True, True, False)]),
    ),
    "DynamicTape must be frozen before differentiation": (
        False,
        RuntimeError,
        lambda tape: native.dynamic_vjp(tape, [(1, 1.0)], [0]),
    ),
    "DynamicTape is frozen": (
        True,
        RuntimeError,
        lambda tape: tape.record_input(1.0, (), "float64"),
    ),
    "cannot replace DynamicTape activity after reverse payload pruning": (
        True,
        RuntimeError,
        lambda tape: (tape.prune_reverse_payloads(), tape.set_active_nodes([0])),
    ),
    "DynamicTape has released its invocation payloads": (
        True,
        RuntimeError,
        lambda tape: (tape.release_payloads(), native.dynamic_vjp(tape, [(1, 1.0)], [0])),
    ),
    "dynamic JVP supports at most 16 seeds per traversal": (
        True,
        ValueError,
        lambda tape: native.dynamic_jvp_many(tape, [[(0, 1.0)]] * 17, [1]),
    ),
    "dynamic VJP supports at most 16 seeds per traversal": (
        True,
        ValueError,
        lambda tape: native.dynamic_vjp_many(tape, [[(1, 1.0)]] * 17, [0]),
    ),
    "dynamic JVP seed node %1 is not a tape input": (
        True,
        ValueError,
        lambda tape: native.dynamic_jvp(tape, [(1, 1.0)], [1]),
    ),
    "dynamic JVP repeats input seed %0": (
        True,
        ValueError,
        lambda tape: native.dynamic_jvp(tape, [(0, 1.0), (0, 2.0)], [1]),
    ),
    "dynamic JVP requested node %0, which is not a marked output": (
        True,
        ValueError,
        lambda tape: native.dynamic_jvp(tape, [(0, 1.0)], [0]),
    ),
    "dynamic VJP seed node %0 is not a marked output": (
        True,
        ValueError,
        lambda tape: native.dynamic_vjp(tape, [(0, 1.0)], [0]),
    ),
    "dynamic VJP repeats output seed %1": (
        True,
        ValueError,
        lambda tape: native.dynamic_vjp(tape, [(1, 1.0), (1, 2.0)], [0]),
    ),
    "dynamic VJP requested node %1, which is not a tape input": (
        True,
        ValueError,
        lambda tape: native.dynamic_vjp(tape, [(1, 1.0)], [1]),
    ),
    "dynamic operation 'identity' has no JVP binding": (
        True,
        RuntimeError,
        lambda tape: native.dynamic_jvp(tape, [(0, 1.0)], [1]),
    ),
    "dynamic operation 'identity' has no VJP binding": (
        True,
        RuntimeError,
        lambda tape: native.dynamic_vjp(tape, [(1, 1.0)], [0]),
    ),
}


@pytest.mark.parametrize(("message", "case"), _MISUSE.items(), ids=list(_MISUSE))
def test_dynamic_tape_rejects_misuse_with_a_specific_error(
    message: str,
    case: tuple[bool, type[Exception], Callable[[native.DynamicTape], object]],
) -> None:
    frozen, error, misuse = case
    tape = _identity_tape(frozen=frozen)

    with pytest.raises(error, match=re.escape(message)):
        misuse(tape)
    tape.release_payloads()


@pytest.mark.parametrize(
    "query",
    [
        lambda tape: tape.is_weak(1),
        lambda tape: tape.weak_mask([0, 1]),
        lambda tape: tape.record_residual(1, object()),
    ],
    ids=["is_weak", "weak_mask", "record_residual"],
)
def test_node_queries_after_release_name_the_released_payloads(
    query: Callable[[native.DynamicTape], object],
) -> None:
    tape = _identity_tape()
    tape.release_payloads()

    with pytest.raises(RuntimeError, match="released its invocation payloads"):
        query(tape)


def _batched_vjp(many: Callable[..., object]) -> Callable[..., object]:
    def vjp(*_args: object) -> list[float]:
        return [1.0]

    cast("Any", vjp).__advect_vjp_many__ = many
    return vjp


@pytest.mark.parametrize(
    ("vjp", "message"),
    [
        (lambda *_args: 1.0, "VJP for 'identity' at dynamic node %1 must return a sequence"),
        (
            lambda *_args: [1.0, 2.0],
            "VJP for 'identity' at dynamic node %1 returned 2 contributions for 1 operands",
        ),
        (
            _batched_vjp(lambda *_args: 1.0),
            "Batched VJP for 'identity' at dynamic node %1 must return a sequence of sequences",
        ),
        (
            _batched_vjp(lambda *_args: [[1.0]]),
            "Batched VJP for 'identity' at dynamic node %1 returned 1 contribution sets",
        ),
        (
            _batched_vjp(lambda *_args: [[1.0], []]),
            "Batched VJP for 'identity' at dynamic node %1 returned 0 contributions for 1 operands",
        ),
    ],
)
def test_reverse_validates_vjp_results(vjp: Callable[..., object], message: str) -> None:
    tape = _identity_tape(vjp=vjp)

    with pytest.raises(ValueError, match=re.escape(message)):
        native.dynamic_vjp_many(tape, [[(1, 1.0)], [(1, 2.0)]], [0])
    tape.release_payloads()


def _raise_rule_error(*_args: object) -> object:
    message = "rule failed"
    raise RuntimeError(message)


@pytest.mark.parametrize(
    ("rules", "traverse", "seeds", "requested"),
    [
        ("vjps", native.dynamic_vjp, [(1, 1.0)], [0]),
        ("jvps", native.dynamic_jvp, [(0, 1.0)], [1]),
    ],
)
def test_rule_errors_propagate_with_the_rule_and_node_noted(
    rules: str,
    traverse: Callable[..., object],
    seeds: list[tuple[int, float]],
    requested: list[int],
) -> None:
    tape = _identity_tape(frozen=False)
    _freeze(tape, **{rules: {"identity": _raise_rule_error}})

    with pytest.raises(RuntimeError, match="rule failed") as caught:
        traverse(tape, seeds, requested)
    label = rules[:3].upper()
    assert caught.value.__notes__ == [f"while executing {label} for 'identity' at dynamic node %1"]
    tape.release_payloads()


@pytest.mark.parametrize(
    ("later", "message"),
    [
        ((1.0,), "cannot add cotangent tuples with lengths 1 and 2"),
        (1.0, "cannot add tuple and non-tuple cotangents"),
    ],
)
def test_reverse_rejects_mismatched_cotangent_structures(later: object, message: str) -> None:
    tape = native.DynamicTape()
    pair = _input(tape, (1.0, 2.0))
    outputs = [_operation(tape, op, [pair], 1.0) for op in ("first", "second")]
    for output in outputs:
        tape.mark_output(output)
    # The reverse sweep reaches "second" first, then adds "first".
    _freeze(tape, vjps={"first": lambda *_args: [(1.0, 1.0)], "second": lambda *_args: [later]})

    with pytest.raises(ValueError, match=re.escape(message)):
        native.dynamic_vjp(tape, [(output, 1.0) for output in outputs], [pair])
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


def test_consuming_reverse_skips_an_unreached_operation_without_a_transpose() -> None:
    tape = native.DynamicTape()
    value = _input(tape, 1.0)
    output = _operation(tape, "identity", [value], 1.0)
    _operation(tape, "unreached", [value], 2.0)
    tape.mark_output(output)
    _freeze(
        tape,
        vjps={"identity": lambda _output, _operands, cotangent, *_args: [cotangent]},
        reverse_needs={"identity": (False, False, False)},
    )

    assert native.dynamic_vjp(tape, [(output, 1.0)], [value]) == [1.0]
    assert native.dynamic_vjp(tape, [(output, 1.0)], [value], consume=True) == [1.0]
    assert tape.is_consumed


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
