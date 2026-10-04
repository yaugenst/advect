"""Property tests for native dynamic-tape derivative traversals.

Random integer programs make every comparison exact. Forward and reverse
sweeps must match a pure-Python reference, multi-seed sweeps must match
their single-seed lanes with or without batched VJPs, pruning reverse
payloads or consuming the tape must never change a result, and every
primitive residual is released exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from hypothesis import example, given, strategies as st

from advect import _native_core as native

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    Seeds = tuple[tuple[int, int | None], ...]


@dataclass(frozen=True)
class _Rule:
    """An integer primitive with at most one literal operand."""

    parent_count: int
    literal_slot: int | None
    evaluate: Callable[[Sequence[Any]], int]
    # One partial derivative per operand; `None` marks the literal.
    partials: Callable[[Sequence[Any]], tuple[int | None, ...]]
    # The rule's minimal reverse need: its operands, or its partials saved
    # as a primitive residual when it runs forward.
    needs_primals: bool = False
    needs_residual: bool = False


_RULES = {
    "add": _Rule(2, None, lambda x: x[0] + x[1], lambda _x: (1, 1)),
    "sub": _Rule(2, None, lambda x: x[0] - x[1], lambda _x: (1, -1)),
    "mul": _Rule(2, None, lambda x: x[0] * x[1], lambda x: (x[1], x[0]), needs_primals=True),
    "square": _Rule(1, None, lambda x: x[0] * x[0], lambda x: (2 * x[0],), needs_residual=True),
    # literal - x and x * literal place the literal in either operand slot.
    "lsub": _Rule(1, 0, lambda x: x[0] - x[1], lambda _x: (None, -1)),
    "rmul": _Rule(1, 1, lambda x: x[0] * x[1], lambda x: (x[1], None), needs_primals=True),
}


class _Residual:
    """Saved partials that count their releases and vanish once released."""

    def __init__(self, payload: tuple[int | None, ...]) -> None:
        self.payload: tuple[int | None, ...] | None = payload
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1
        self.payload = None


@dataclass(frozen=True)
class _Step:
    op: str
    parents: tuple[int, ...]
    literal: int | None = None

    def operands(self, nodes: Sequence[Any], literal: object) -> list[Any]:
        operands = [nodes[parent] for parent in self.parents]
        slot = _RULES[self.op].literal_slot
        if slot is not None:
            operands.insert(slot, literal)
        return operands


@dataclass(frozen=True)
class _Program:
    inputs: tuple[tuple[int, bool], ...]  # (value, active)
    steps: tuple[_Step, ...]
    outputs: tuple[int, ...]

    def values(self) -> list[int]:
        values = [value for value, _active in self.inputs]
        for step in self.steps:
            values.append(_RULES[step.op].evaluate(step.operands(values, step.literal)))
        return values

    def activity(self) -> list[bool]:
        active = [flag for _value, flag in self.inputs]
        active.extend(any(active[parent] for parent in step.parents) for step in self.steps)
        return active

    def jvp(self, seeds: Seeds) -> list[int | None]:
        """Dual-number reference; `None` means no seeded input reaches a node."""
        values = self.values()
        tangents: list[int | None] = [None] * len(self.inputs)
        for node, tangent in seeds:
            tangents[node] = tangent
        for step in self.steps:
            operand_tangents = step.operands(tangents, None)
            if all(tangent is None for tangent in operand_tangents):
                tangents.append(None)
                continue
            partials = _RULES[step.op].partials(step.operands(values, step.literal))
            tangents.append(_dot(partials, operand_tangents))
        return [tangents[output] for output in self.outputs]

    def vjp(self, seeds: Seeds) -> list[int | None]:
        """Reverse-accumulation reference over active nodes only."""
        values = self.values()
        active = self.activity()
        cotangents: list[int | None] = [None] * len(values)
        for node, cotangent in seeds:
            cotangents[node] = cotangent
        first_step = len(self.inputs)
        for index in reversed(range(first_step, len(values))):
            cotangent = cotangents[index]
            if cotangent is None or not active[index]:
                continue
            step = self.steps[index - first_step]
            partials = _RULES[step.op].partials(step.operands(values, step.literal))
            parent_partials = [partial for partial in partials if partial is not None]
            for parent, partial in zip(step.parents, parent_partials, strict=True):
                if active[parent]:
                    previous = cotangents[parent]
                    contribution = partial * cotangent
                    cotangents[parent] = (
                        contribution if previous is None else previous + contribution
                    )
        return cotangents[:first_step]


@dataclass(frozen=True)
class _Case:
    program: _Program
    tangent_lanes: tuple[Seeds, ...]
    cotangent_lanes: tuple[Seeds, ...]
    batched: bool


def _dot(partials: Sequence[int | None], tangents: Sequence[int | None]) -> int:
    return sum(
        partial * tangent
        for partial, tangent in zip(partials, tangents, strict=True)
        if partial is not None and tangent is not None
    )


_SMALL = st.integers(-3, 3)


@st.composite
def _cases(draw: st.DrawFn) -> _Case:
    inputs = tuple(draw(st.lists(st.tuples(_SMALL, st.booleans()), min_size=1, max_size=3)))
    steps = []
    for _ in range(draw(st.integers(1, 10))):
        node_count = len(inputs) + len(steps)
        op = draw(st.sampled_from(sorted(_RULES)))
        rule = _RULES[op]
        parents = tuple(draw(st.integers(0, node_count - 1)) for _ in range(rule.parent_count))
        literal = None if rule.literal_slot is None else draw(_SMALL)
        steps.append(_Step(op, parents, literal))
    node_count = len(inputs) + len(steps)
    outputs = tuple(
        draw(st.lists(st.integers(0, node_count - 1), min_size=1, max_size=3, unique=True))
    )
    active_inputs = [node for node, (_value, active) in enumerate(inputs) if active]
    tangent_lane = (
        st.lists(
            st.tuples(st.sampled_from(active_inputs), _SMALL), unique_by=lambda seed: seed[0]
        ).map(tuple)
        if active_inputs
        else st.just(())
    )
    cotangent_lane = st.tuples(*(st.tuples(st.just(node), st.none() | _SMALL) for node in outputs))
    return _Case(
        _Program(inputs, tuple(steps), outputs),
        tuple(draw(st.lists(tangent_lane, min_size=1, max_size=3))),
        tuple(draw(st.lists(cotangent_lane, min_size=1, max_size=3))),
        draw(st.booleans()),
    )


def _jvp_rule(rule: _Rule, residuals: list[_Residual]) -> Callable[..., int]:
    def jvp(
        _output: object,
        operands: tuple[Any, ...],
        tangents: tuple[int | None, ...],
        _attrs: object,
        _source: object,
        residual: tuple[int | None, ...] | None,
    ) -> int:
        assert (residual is not None) is rule.needs_residual
        if residual is not None:
            assert any(saved.payload is residual and saved.close_count == 0 for saved in residuals)
        partials = rule.partials(operands) if residual is None else residual
        return _dot(partials, tangents)

    return jvp


def _vjp_rule(rule: _Rule, *, batched: bool) -> Callable[..., list[int | None]]:
    def vjp(
        output: object,
        operands: tuple[Any, ...],
        cotangent: int,
        _attrs: object,
        _active: object,
        residual: tuple[int | None, ...] | None,
        *_rest: object,
    ) -> list[int | None]:
        # Only the declared reverse needs reach the rule.
        assert output is None
        assert all((operand is not None) is rule.needs_primals for operand in operands)
        assert (residual is not None) is rule.needs_residual
        partials = rule.partials(operands) if residual is None else residual
        return [None if partial is None else partial * cotangent for partial in partials]

    def vjp_many(
        output: object,
        operands: tuple[Any, ...],
        cotangents: tuple[int, ...],
        *rest: object,
    ) -> list[list[int | None]]:
        return [vjp(output, operands, cotangent, *rest) for cotangent in cotangents]

    if batched:
        cast("Any", vjp).__advect_vjp_many__ = vjp_many
    return vjp


def _record(case: _Case) -> tuple[native.DynamicTape, list[_Residual]]:
    program = case.program
    values = program.values()
    tape = native.DynamicTape()
    for value, active in program.inputs:
        tape.record_input(value, (), "int64", active=active)
    residuals = []
    for node, step in enumerate(program.steps, start=len(program.inputs)):
        rule = _RULES[step.op]
        slot = rule.literal_slot
        tape.record_operation(
            step.op,
            list(step.parents),
            values[node],
            {},
            (),
            "int64",
            input_positions=(
                None
                if slot is None
                else [position for position in range(len(step.parents) + 1) if position != slot]
            ),
            literals=() if slot is None else [step.literal],
        )
        if rule.needs_residual:
            residuals.append(_Residual(rule.partials(step.operands(values, step.literal))))
            tape.record_residual(node, residuals[-1])
    for output in program.outputs:
        tape.mark_output(output)
    rules = [_RULES.get(op) for op in tape.op_names]
    tape.freeze(
        [None if rule is None else _jvp_rule(rule, residuals) for rule in rules],
        [None if rule is None else _vjp_rule(rule, batched=case.batched) for rule in rules],
        [
            None if rule is None else (False, rule.needs_primals, rule.needs_residual)
            for rule in rules
        ],
    )
    return tape, residuals


_REPEATED_PARENTS = _Program(
    inputs=((2, True), (3, True)),
    steps=(_Step("mul", (0, 1)), _Step("mul", (2, 2))),
    outputs=(3,),
)


@given(case=_cases())
@example(
    case=_Case(
        _REPEATED_PARENTS,
        tangent_lanes=(((0, 1), (1, 1)), ((0, 1),), ((1, 1),)),
        cotangent_lanes=(((3, 1),), ((3, 2),)),
        batched=False,
    )
)
@example(
    case=_Case(
        _Program(
            inputs=((2, True),),
            steps=(_Step("square", (0,)), _Step("square", (1,))),
            outputs=(1, 2),
        ),
        tangent_lanes=(((0, 1),), ((0, 2),), ()),
        cotangent_lanes=(((1, 1), (2, 1)),),
        batched=False,
    )
)
def test_dynamic_tape_traversals_match_the_integer_reference(case: _Case) -> None:
    program = case.program
    inputs = list(range(len(program.inputs)))
    outputs = list(program.outputs)
    tangents = [program.jvp(lane) for lane in case.tangent_lanes]
    cotangents = [program.vjp(lane) for lane in case.cotangent_lanes]

    tape, residuals = _record(case)
    try:
        assert [tape.node_is_active(node) for node in range(tape.node_count)] == program.activity()
        # One tape serves repeated single and multi-seed traversals.
        assert [native.dynamic_jvp(tape, lane, outputs) for lane in case.tangent_lanes] == tangents
        assert native.dynamic_jvp_many(tape, case.tangent_lanes, outputs) == tangents
        for _ in range(2):
            assert [
                native.dynamic_vjp(tape, lane, inputs) for lane in case.cotangent_lanes
            ] == cotangents
        assert native.dynamic_vjp_many(tape, case.cotangent_lanes, inputs) == cotangents
    finally:
        tape.release_payloads()
    assert [residual.close_count for residual in residuals] == [1] * len(residuals)

    # Retiring payloads early must keep everything a rule declared.
    reverse = (native.dynamic_vjp, case.cotangent_lanes[0], inputs, cotangents[0])
    forward = (native.dynamic_jvp, case.tangent_lanes[0], outputs, tangents[0])
    for (traverse, lane, requested, expected), prune, consume in (
        (reverse, True, False),
        (reverse, False, True),
        (reverse, True, True),
        (forward, False, True),
    ):
        tape, residuals = _record(case)
        try:
            if prune:
                tape.prune_reverse_payloads()
            assert traverse(tape, lane, requested, consume=consume) == expected
            if consume:
                stats = tape.stats()
                assert tape.is_consumed
                assert stats["retained_value_count"] == stats["literal_count"] == 0
                assert stats["residual_count"] == 0
        finally:
            tape.release_payloads()
        assert [residual.close_count for residual in residuals] == [1] * len(residuals)
