"""Direct registered-rule checks, fed by real frontend invocations."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any

import numpy as np

import advect as ad
from advect.core._eval_dispatch import _evaluate_array_op, evaluate_node_value
from advect.core._pytree import tree_flatten, tree_map
from advect.core._registry import get_registry
from advect.testing import _real_inner_product
from advect_conformance_tests._harness._cases import Law, NumericalReference
from advect_conformance_tests._harness._frontends import to_numpy
from advect_conformance_tests._harness._laws import (
    ConformanceError,
    _align_eigen_output,
    _assert_close,
    _finite_difference_step,
    _numerical_tolerances,
    _numpy_leaves,
    _probe_like,
    _promote_numerical_reference,
    _seed_for,
    _traced_arguments,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from advect_conformance_tests._harness._cases import InvocationCase

type _JvpRule = Callable[..., Any]
type _VjpRule = Callable[..., tuple[Any | None, ...]]

__all__ = [
    "RawRuleCase",
    "check_raw_jvp",
    "check_raw_vjp",
    "check_registered_jvp",
    "check_registered_vjp",
]


@dataclass(frozen=True, slots=True)
class _CapturedRuleCall:
    answer: Any
    operands: tuple[Any, ...]
    tangents: tuple[Any | None, ...]
    attrs: Mapping[str, Any]
    jvp: _JvpRule
    vjp: _VjpRule | None
    vjp_needs_inputs: bool


@dataclass(frozen=True, slots=True)
class RawRuleCase:
    """A rule whose operation currently has no frontend invocation."""

    op: str
    operands: tuple[Any, ...]
    tangents: tuple[Any | None, ...]
    attrs: Mapping[str, Any]
    tolerance: float = 1e-5
    numerical: bool = True


# Replacing a registry rule is process-global. Pytest runs tests serially inside
# one worker, and this lock makes that assumption explicit for any future
# threaded runner. xdist workers have separate processes and registries.
_CAPTURE_LOCK = Lock()


def _directions(case: InvocationCase, values: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(
        _probe_like(values[index], position)
        for position, index in enumerate(case.differentiable_indices)
    )


def _capture_calls(
    case: InvocationCase,
    values: tuple[Any, ...],
) -> tuple[_CapturedRuleCall, ...]:
    directions = _directions(case, values)
    arguments = _traced_arguments(case, values)
    tangent_arguments = _traced_arguments(case, directions)

    registry = get_registry()
    definition = registry.get(case.op)
    original_jvp = definition.jvp
    if original_jvp is None:
        msg = f"{case.op}: invocation declares differentiation but has no JVP"
        raise ConformanceError(msg)

    seen: list[_CapturedRuleCall] = []

    def capture(
        answer: Any,
        *operands: Any,
        tangents: tuple[Any | None, ...],
        **attrs: Any,
    ) -> Any:
        seen.append(
            _CapturedRuleCall(
                answer=answer,
                operands=tuple(operands),
                tangents=tangents,
                attrs=dict(attrs),
                jvp=original_jvp,
                vjp=definition.vjp,
                vjp_needs_inputs=definition.vjp_needs_inputs,
            ),
        )
        return original_jvp(answer, *operands, tangents=tangents, **attrs)

    with _CAPTURE_LOCK:
        registry.update(case.op, jvp=capture)
        try:
            ad.jvp(case.call, case.differentiable_indices)(
                *arguments,
                tangents=tangent_arguments,
                **dict(case.static),
            )
        finally:
            registry.update(case.op, jvp=original_jvp)

    if not seen:
        msg = f"{case.op}: declared {case.frontend.value} invocation did not emit the target op"
        raise ConformanceError(msg)
    return tuple(seen)


def _array_namespace(operands: tuple[Any, ...]) -> Any:
    for operand in operands:
        namespace = getattr(operand, "__array_namespace__", None)
        if callable(namespace):
            return namespace()
        if isinstance(operand, np.ndarray | np.generic):
            return np
    return np


def _evaluate_rule_op(
    op: str,
    operands: tuple[Any, ...],
    attrs: Mapping[str, Any],
) -> Any:
    result: Any
    if op.startswith(("array.", "array_ext.")):
        namespace = _array_namespace(operands)
        if namespace is np:
            from advect.numpy._eval import evaluate_op  # noqa: PLC0415
            from advect.numpy._op_bindings import (  # noqa: PLC0415
                decanonicalize_array_op,
            )

            result = evaluate_op(
                decanonicalize_array_op(op),
                operands,
                dict(attrs),
            )
        else:
            result = _evaluate_array_op(op, operands, attrs, namespace)
    elif op == "advect.getitem":
        result = operands[0][attrs["index"]]
    elif op == "advect.index_update":
        result = operands[0].copy()
        if attrs.get("mode", "set") == "add":
            result[attrs["index"]] += operands[1]
        else:
            result[attrs["index"]] = operands[1]
    elif op == "advect.copy":
        result = operands[0].copy()
    elif op == "advect.getoutput":
        result = operands[0][int(attrs["index"])]
    else:
        result = evaluate_node_value(op, operands, attrs)
    # NumPy 2 returns namedtuple subclasses for decompositions while Advect's
    # atomic multi-output node deliberately owns a plain tuple.
    if isinstance(result, tuple) and type(result) is not tuple:
        return tuple(result)
    return result


def _shift_operands(
    operands: tuple[Any, ...],
    tangents: tuple[Any | None, ...],
    step: complex,
) -> tuple[Any, ...]:
    def shift(operand: Any, tangent: Any) -> Any:
        if tangent is None:
            return operand
        if isinstance(operand, tuple) and isinstance(tangent, tuple):
            return tuple(
                shift(item, item_tangent)
                for item, item_tangent in zip(operand, tangent, strict=True)
            )
        return operand + step * tangent

    return tuple(
        shift(operand, tangent) for operand, tangent in zip(operands, tangents, strict=True)
    )


def _pairing_leaves(value: Any) -> list[Any]:
    return [
        None if leaf is None else _promote_numerical_reference(to_numpy(leaf))
        for leaf in tree_flatten(value)[0]
    ]


def _raw_numerical_derivative(
    case: InvocationCase,
    captured: _CapturedRuleCall,
) -> list[Any]:
    if case.numerical_reference is NumericalReference.COMPLEX_STEP:
        operands = tuple(_promote_numerical_reference(value) for value in captured.operands)
        tangents = tuple(
            None if value is None else _promote_numerical_reference(value)
            for value in captured.tangents
        )
        step = case.tolerance.complex_step
        shifted = _evaluate_rule_op(
            case.op,
            _shift_operands(operands, tangents, 1j * step),
            captured.attrs,
        )
        return [np.imag(to_numpy(leaf)) / step for leaf in tree_flatten(shifted)[0]]

    operands = tuple(_promote_numerical_reference(value) for value in captured.operands)
    tangents = tuple(
        None if value is None else _promote_numerical_reference(value)
        for value in captured.tangents
    )
    reference = _evaluate_rule_op(case.op, operands, captured.attrs)
    step = _finite_difference_step(case, operands)
    positive = _evaluate_rule_op(
        case.op,
        _shift_operands(operands, tangents, step),
        captured.attrs,
    )
    negative = _evaluate_rule_op(
        case.op,
        _shift_operands(operands, tangents, -step),
        captured.attrs,
    )
    input_is_real = not np.iscomplexobj(operands[0])
    positive = _align_eigen_output(case.op, reference, positive, input_is_real=input_is_real)
    negative = _align_eigen_output(case.op, reference, negative, input_is_real=input_is_real)
    return [
        (to_numpy(upper) - to_numpy(lower)) / (2.0 * step)
        for upper, lower in zip(
            tree_flatten(positive)[0],
            tree_flatten(negative)[0],
            strict=True,
        )
    ]


def check_registered_jvp(
    case: InvocationCase,
    values: tuple[Any, ...],
    *,
    variant: int = 0,
) -> None:
    """Check the exact registered JVP reached by one frontend invocation."""
    case = case.resolve_variant(variant)
    if Law.FINITE_DIFFERENCE not in case.laws:
        msg = f"{case.op}: direct JVP check requires a numerical-reference law"
        raise ValueError(msg)
    for captured in _capture_calls(case, values):
        evaluated = _evaluate_rule_op(case.op, captured.operands, captured.attrs)
        _assert_close(
            evaluated,
            captured.answer,
            rtol=case.tolerance.primal_rtol,
            atol=case.tolerance.primal_atol,
            context=f"\n  op: {case.op}\n  boundary: direct rule primal",
        )
        tangent = captured.jvp(
            captured.answer,
            *captured.operands,
            tangents=captured.tangents,
            **captured.attrs,
        )
        rtol, atol = _numerical_tolerances(case)
        _assert_close(
            tangent,
            _raw_numerical_derivative(case, captured),
            rtol=rtol,
            atol=atol,
            context=f"\n  op: {case.op}\n  boundary: registered JVP",
        )


def check_registered_vjp(
    case: InvocationCase,
    values: tuple[Any, ...],
    *,
    variant: int = 0,
) -> None:
    """Check an explicit VJP directly against the registered JVP."""
    case = case.resolve_variant(variant)
    for captured in _capture_calls(case, values):
        if captured.vjp is None:
            msg = f"{case.op}: no explicit VJP is registered"
            raise ValueError(msg)
        cotangent = _seed_for(case, captured.answer)
        tangent = captured.jvp(
            captured.answer,
            *captured.operands,
            tangents=captured.tangents,
            **captured.attrs,
        )
        contributions = captured.vjp(
            captured.answer,
            *(captured.operands if captured.vjp_needs_inputs else ()),
            g=cotangent,
            **captured.attrs,
        )
        left = _real_inner_product(_numpy_leaves(cotangent), _numpy_leaves(tangent))
        right = _real_inner_product(
            _pairing_leaves(contributions),
            _pairing_leaves(captured.tangents),
        )
        scale = max(abs(left), abs(right), 1.0)
        tolerance = case.tolerance.adjoint_atol + case.tolerance.adjoint_rtol * scale
        if abs(left - right) > tolerance:
            msg = (
                f"{case.op}: registered VJP violates the direct-rule adjoint identity: "
                f"left={left!r}, right={right!r}, tolerance={tolerance:.3e}"
            )
            raise ConformanceError(msg)


def check_raw_jvp(case: RawRuleCase) -> None:
    """Check an intentionally unbound registered JVP from raw operands."""
    definition = get_registry().get(case.op)
    if definition.jvp is None:
        msg = f"{case.op}: raw rule case has no registered JVP"
        raise ConformanceError(msg)
    answer = _evaluate_rule_op(case.op, case.operands, case.attrs)
    actual = definition.jvp(
        answer,
        *case.operands,
        tangents=case.tangents,
        **case.attrs,
    )
    if case.numerical:
        step = 1e-6
        positive = _evaluate_rule_op(
            case.op,
            _shift_operands(case.operands, case.tangents, step),
            case.attrs,
        )
        negative = _evaluate_rule_op(
            case.op,
            _shift_operands(case.operands, case.tangents, -step),
            case.attrs,
        )
        reference = [
            (to_numpy(upper) - to_numpy(lower)) / (2.0 * step)
            for upper, lower in zip(
                tree_flatten(positive)[0],
                tree_flatten(negative)[0],
                strict=True,
            )
        ]
    else:
        reference = tree_map(lambda value: np.zeros_like(to_numpy(value)), answer)
    _assert_close(
        actual,
        reference,
        rtol=case.tolerance,
        atol=case.tolerance,
        context=f"\n  op: {case.op}\n  boundary: unbound registered JVP",
    )


def check_raw_vjp(case: RawRuleCase) -> None:
    """Check an explicit raw VJP against the raw JVP."""
    definition = get_registry().get(case.op)
    if definition.jvp is None or definition.vjp is None:
        msg = f"{case.op}: raw VJP check requires both registered rules"
        raise ConformanceError(msg)
    answer = _evaluate_rule_op(case.op, case.operands, case.attrs)
    tangent = definition.jvp(
        answer,
        *case.operands,
        tangents=case.tangents,
        **case.attrs,
    )
    cotangent = tree_map(_probe_like, answer)
    contributions = definition.vjp(
        answer,
        *(case.operands if definition.vjp_needs_inputs else ()),
        g=cotangent,
        **case.attrs,
    )
    left = _real_inner_product(_pairing_leaves(cotangent), _pairing_leaves(tangent))
    right = _real_inner_product(
        _pairing_leaves(contributions),
        _pairing_leaves(case.tangents),
    )
    scale = max(abs(left), abs(right), 1.0)
    if abs(left - right) > case.tolerance * scale:
        msg = (
            f"{case.op}: raw registered VJP violates the adjoint identity: "
            f"left={left!r}, right={right!r}"
        )
        raise ConformanceError(msg)
