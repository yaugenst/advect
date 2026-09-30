"""Direct registered-rule checks, fed by real frontend invocations."""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any, cast

import numpy as np

import advect as ad
from advect.autodiff.rules.array_family._backend_runtime import (
    run_with_array_family_backend_provider,
)
from advect.autodiff.rules.array_family.providers import resolve_array_family_backend_provider
from advect.core._eval_dispatch import _bind_array_op, bind_node_evaluator
from advect.core._pytree import tree_map
from advect.core._registry import get_registry
from advect_conformance_tests._harness._cases import Law
from advect_conformance_tests._harness._frontends import to_numpy
from advect_conformance_tests._harness._laws import (
    ConformanceError,
    Probes,
    _assert_adjoint,
    _assert_close,
    _directional_oracle,
    _directions,
    _numerical_tolerances,
    _oracle_step,
    _probe_like,
    _promote_numerical_reference,
    _seed_for,
    _shifted,
    _traced_arguments,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from hypothesis.strategies import DataObject

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


# Raw operands are small float64 values, so a fixed central step suffices.
_RAW_STEP = 1e-6

# Replacing a registry rule is process-global. Pytest runs tests serially inside
# one worker, and this lock makes that assumption explicit for any future
# threaded runner. xdist workers have separate processes and registries.
_CAPTURE_LOCK = Lock()


def _capture_calls(
    case: InvocationCase,
    values: tuple[Any, ...],
    probes: Probes,
) -> tuple[_CapturedRuleCall, ...]:
    directions = _directions(case, values, probes)
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
            result = _bind_array_op(op, attrs)(operands, namespace, None)
    elif op == "advect.getitem":
        result = operands[0][attrs["index"]]
    elif op == "advect.index_update":
        result = operands[0].copy()
        # Like NumPy's setitem, a complex replacement of a real base drops its
        # imaginary part; the invocation silences the same warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
            if attrs.get("mode", "set") == "add":
                result[attrs["index"]] += operands[1]
            else:
                result[attrs["index"]] = operands[1]
    elif op == "advect.copy":
        result = operands[0].copy()
    elif op == "advect.getoutput":
        result = operands[0][int(attrs["index"])]
    else:
        result = bind_node_evaluator(op, attrs)(operands, None, None)
    # NumPy 2 returns namedtuple subclasses for decompositions while Advect's
    # atomic multi-output node deliberately owns a plain tuple.
    if isinstance(result, tuple) and type(result) is not tuple:
        return tuple(result)
    return result


def _rule_oracle(
    case: InvocationCase,
    operands: tuple[Any, ...],
    tangents: tuple[Any | None, ...],
    attrs: Mapping[str, Any],
) -> list[Any]:
    operands = tuple(map(_promote_numerical_reference, operands))
    tangents = tuple(
        None if value is None else _promote_numerical_reference(value) for value in tangents
    )
    return _directional_oracle(
        lambda step: _evaluate_rule_op(case.op, _shifted(operands, tangents, step), attrs),
        _oracle_step(case, operands),
        op=case.op,
        input_is_real=not np.iscomplexobj(operands[0]),
    )


def _call_vjp(
    vjp: _VjpRule,
    answer: Any,
    operands: tuple[Any, ...],
    cotangent: Any,
    attrs: Mapping[str, Any],
) -> tuple[Any | None, ...]:
    """Call a VJP in the provider scope that a reverse sweep establishes."""
    provider = resolve_array_family_backend_provider(answer, *operands, cotangent)
    return cast(
        "tuple[Any | None, ...]",
        run_with_array_family_backend_provider(
            provider, vjp, answer, *operands, g=cotangent, **attrs
        ),
    )


def check_registered_jvp(
    case: InvocationCase,
    values: tuple[Any, ...],
    *,
    variant: int = 0,
    data: DataObject | None = None,
) -> None:
    """Check the exact registered JVP reached by one frontend invocation."""
    case = case.resolve_variant(variant)
    if Law.FINITE_DIFFERENCE not in case.laws:
        msg = f"{case.op}: direct JVP check requires a numerical-reference law"
        raise ValueError(msg)
    for captured in _capture_calls(case, values, Probes(data)):
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
            _rule_oracle(case, captured.operands, captured.tangents, captured.attrs),
            rtol=rtol,
            atol=atol,
            context=f"\n  op: {case.op}\n  boundary: registered JVP",
        )


def check_registered_vjp(
    case: InvocationCase,
    values: tuple[Any, ...],
    *,
    variant: int = 0,
    data: DataObject | None = None,
) -> None:
    """Check an explicit VJP directly against the registered JVP."""
    case = case.resolve_variant(variant)
    probes = Probes(data)
    for captured in _capture_calls(case, values, probes):
        if captured.vjp is None:
            msg = f"{case.op}: no explicit VJP is registered"
            raise ValueError(msg)
        cotangent = _seed_for(case, captured.answer, probes)
        tangent = captured.jvp(
            captured.answer,
            *captured.operands,
            tangents=captured.tangents,
            **captured.attrs,
        )
        contributions = _call_vjp(
            captured.vjp,
            captured.answer,
            captured.operands if captured.vjp_needs_inputs else (),
            cotangent,
            captured.attrs,
        )
        _assert_adjoint(
            cotangent,
            tangent,
            contributions,
            captured.tangents,
            rtol=case.tolerance.adjoint_rtol,
            atol=case.tolerance.adjoint_atol,
            context=f"\n  op: {case.op}\n  boundary: registered VJP",
        )
        _check_selective_vjp(case, captured, cotangent, contributions)


def _check_selective_vjp(
    case: InvocationCase,
    captured: _CapturedRuleCall,
    cotangent: Any,
    contributions: tuple[Any | None, ...],
) -> None:
    """Require the input-selective rule reverse sweeps prefer to slice the full one."""
    selective = getattr(captured.vjp, "__advect_vjp_for_input_indices__", None)
    if not callable(selective):
        return
    active = [index for index, tangent in enumerate(captured.tangents) if tangent is not None]
    for size in range(1, len(active) + 1):
        for subset in itertools.combinations(active, size):
            selected = _call_vjp(
                selective,
                captured.answer,
                captured.operands if captured.vjp_needs_inputs else (),
                cotangent,
                {**captured.attrs, "active_input_indices": subset},
            )
            context = f"\n  op: {case.op}\n  boundary: selective VJP for inputs {subset}"
            for index, (partial, full) in enumerate(zip(selected, contributions, strict=True)):
                if index not in subset:
                    if partial is not None:
                        msg = f"inactive input {index} received a cotangent{context}"
                        raise ConformanceError(msg)
                elif (partial is None) != (full is None):
                    msg = f"input {index} cotangent presence differs from the full rule{context}"
                    raise ConformanceError(msg)
                elif full is not None:
                    _assert_close(
                        partial,
                        full,
                        rtol=case.tolerance.adjoint_rtol,
                        atol=case.tolerance.adjoint_atol,
                        context=context,
                    )


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
        reference = _directional_oracle(
            lambda step: _evaluate_rule_op(
                case.op, _shifted(case.operands, case.tangents, step), case.attrs
            ),
            _RAW_STEP,
        )
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
    contributions = _call_vjp(
        definition.vjp,
        answer,
        case.operands if definition.vjp_needs_inputs else (),
        cotangent,
        case.attrs,
    )
    _assert_adjoint(
        cotangent,
        tangent,
        contributions,
        case.tangents,
        rtol=case.tolerance,
        atol=0.0,
        context=f"\n  op: {case.op}\n  boundary: raw registered VJP",
    )
