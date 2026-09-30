"""Public-transform laws for primitive conformance."""

from __future__ import annotations

import functools
import itertools
import json
import math
from typing import TYPE_CHECKING, Any

import hypothesis.strategies as st
import numpy as np

import advect as ad
from advect.core._pytree import tree_flatten, tree_map
from advect.core._registry import get_registry
from advect.testing import _real_inner_product, _real_inner_product_magnitude
from advect_conformance_tests._harness._cases import Law, NumericalReference
from advect_conformance_tests._harness._frontends import (
    Frontend,
    is_python_number,
    to_numpy,
    wrap_for,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from hypothesis.strategies import DataObject

    from advect_conformance_tests._harness._cases import InvocationCase

__all__ = ["ConformanceError", "Probes", "check_law"]


class ConformanceError(AssertionError):
    """A primitive violated one of its declared contracts."""


def _describe(case: InvocationCase, law: Law, variant: int) -> str:
    domains = ", ".join(
        f"{argument.name}: {argument.domain.condition_note}" for argument in case.arguments
    )
    return (
        f"\n  op       : {case.op}"
        f"\n  frontend : {case.frontend.value}"
        f"\n  law      : {law.value}"
        f"\n  variant  : {variant}"
        f"\n  domains  : {domains}"
    )


def _leaves(value: Any) -> list[Any]:
    leaves, _treedef = tree_flatten(value)
    return leaves


def _numpy_leaves(value: Any) -> list[Any]:
    return [
        None if leaf is None else _promote_numerical_reference(to_numpy(leaf))
        for leaf in _leaves(value)
    ]


def _invoke(case: InvocationCase, values: Sequence[Any]) -> Any:
    wrapped = tuple(wrap_for(case.frontend, value) for value in values)
    return case.call(*wrapped, **dict(case.static))


def _traced_arguments(case: InvocationCase, values: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(wrap_for(case.frontend, value) for value in values)


def _transformed_output(case: InvocationCase, values: Sequence[Any]) -> Any:
    """Return a transformed call's output, which cotangent seeds mirror.

    Its result containers may differ from the concrete provider's (for example
    linalg results).
    """
    value, pullback = ad.vjp(case.call, case.differentiable_indices)(
        *_traced_arguments(case, values),
        **dict(case.static),
    )
    with pullback:
        return value


def _stage_specs(values: Sequence[Any]) -> tuple[ad.ArraySpec, ...]:
    return tuple(
        ad.ArraySpec(
            tuple(np.shape(value)),
            np.asarray(value).dtype,
            weak=is_python_number(value),
        )
        for value in values
    )


def _probe_like(value: Any, ordinal: int = 0) -> Any:
    """Return a deterministic, dense direction with the primal's metadata."""
    array = np.asarray(to_numpy(value))
    direction = _probe_pattern(array.shape, array.dtype, ordinal)
    if is_python_number(value):
        return type(value)(direction.reshape(()).item())
    return direction.copy()


@functools.cache
def _probe_pattern(
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    ordinal: int,
) -> np.ndarray[Any, Any]:
    size = max(int(np.prod(shape, dtype=np.int64)), 1)
    positions = np.arange(size, dtype=np.float64)
    real = 1.0 + ((positions + ordinal) % 5.0)
    real[1::2] *= -1.0
    direction: np.ndarray[Any, Any]
    if np.issubdtype(dtype, np.complexfloating):
        imaginary = 1.0 + ((positions[::-1] + 2 * ordinal) % 7.0)
        direction = real + 1j * imaginary
    else:
        direction = real
    return (direction / np.max(np.abs(direction))).reshape(shape).astype(dtype)


class Probes:
    """Independent dense tangent and cotangent probes for one law check.

    Every request takes the next ``_probe_like`` ordinal, so a cotangent seed
    never equals a tangent direction and an untransposed pullback cannot
    satisfy the adjoint identity by symmetry. Under Hypothesis, a drawn offset
    of at most one half per component searches around that anchor and shrinks
    back to it; renormalising keeps every probe's largest magnitude at one.
    """

    __slots__ = ("_data", "_ordinals")

    def __init__(self, data: DataObject | None = None) -> None:
        self._data = data
        self._ordinals = itertools.count()

    def __call__(self, like: Any) -> Any:
        anchor = _probe_like(like, next(self._ordinals))
        if self._data is None:
            return anchor
        array = np.asarray(anchor)
        complex_parts = np.iscomplexobj(array)
        raw = self._data.draw(
            _offset_bytes(array.size * (2 if complex_parts else 1)), label="probe"
        )
        shift = np.frombuffer(raw, dtype=np.int8) / 256.0
        if complex_parts:
            shift = shift[0::2] + 1j * shift[1::2]
        moved = array + shift.reshape(array.shape)
        probe = np.asarray(moved / np.max(np.abs(moved)), dtype=array.dtype)
        return type(anchor)(probe.item()) if is_python_number(anchor) else probe


@functools.cache
def _offset_bytes(size: int) -> st.SearchStrategy[bytes]:
    return st.binary(min_size=size, max_size=size)


def _shifted(values: Sequence[Any], tangents: Sequence[Any], step: complex) -> tuple[Any, ...]:
    """Move each value ``step`` along its tangent; a ``None`` tangent holds it fixed.

    Values keep their kind: a Python number stays a number, a NumPy value
    stays an array, and another provider's array uses its own arithmetic.
    """

    def shift(value: Any, tangent: Any) -> Any:
        if tangent is None:
            return value
        if isinstance(value, tuple):
            return _shifted(value, tangent, step)
        moved = value + step * tangent
        return np.asarray(moved) if isinstance(value, np.ndarray | np.generic) else moved

    return tuple(shift(value, tangent) for value, tangent in zip(values, tangents, strict=True))


def _promote_numerical_reference(value: Any) -> Any:
    """Evaluate numerical oracles above low-precision roundoff."""
    if is_python_number(value):
        return value
    array = np.asarray(value)
    if array.dtype == np.dtype("float32"):
        return array.astype(np.float64)
    if array.dtype == np.dtype("complex64"):
        return array.astype(np.complex128)
    return value


def _finite_difference_step(case: InvocationCase, values: Sequence[Any]) -> float:
    magnitude = max(
        (float(np.max(np.abs(np.asarray(value)))) for value in values),
        default=1.0,
    )
    return case.tolerance.finite_difference_step * (1.0 + magnitude)


def _oracle_step(case: InvocationCase, values: Sequence[Any]) -> complex:
    """Return the central step for ``values``, or the imaginary complex step."""
    if case.numerical_reference is NumericalReference.COMPLEX_STEP:
        return 1j * case.tolerance.complex_step
    return _finite_difference_step(case, values)


def _directional_oracle(
    shifted: Callable[[complex], Any],
    step: complex,
    *,
    op: str = "",
    input_is_real: bool = True,
) -> list[Any]:
    """Differentiate ``shifted``, a call moved by its argument along a probe, at zero.

    Callers promote low-precision values first, so the oracle runs above their
    roundoff. A complex step reads the imaginary part of one evaluation; it is
    only declared for real-analytic calls. Central differences of a gauged
    decomposition ``op`` are aligned to its unshifted outputs.
    """
    if isinstance(step, complex):
        return [np.imag(to_numpy(leaf)) / step.imag for leaf in _leaves(shifted(step))]
    upper, lower = shifted(step), shifted(-step)
    if op in _GAUGED_OPS:
        reference = shifted(0.0)
        upper, lower = (
            _align_gauge(op, reference, candidate, input_is_real=input_is_real)
            for candidate in (upper, lower)
        )
    return [
        (to_numpy(high) - to_numpy(low)) / (2.0 * step)
        for high, low in zip(_leaves(upper), _leaves(lower), strict=True)
    ]


def _numerical_tolerances(case: InvocationCase) -> tuple[float, float]:
    if case.numerical_reference is NumericalReference.COMPLEX_STEP:
        return case.tolerance.complex_step_rtol, case.tolerance.complex_step_atol
    return (
        case.tolerance.finite_difference_rtol,
        case.tolerance.finite_difference_atol,
    )


def _assert_close(
    actual: Any,
    expected: Any,
    *,
    rtol: float,
    atol: float,
    context: str,
) -> None:
    actual_leaves = _leaves(actual)
    expected_leaves = _leaves(expected)
    if len(actual_leaves) != len(expected_leaves):
        msg = f"structure mismatch: {len(actual_leaves)} vs {len(expected_leaves)} leaves{context}"
        raise ConformanceError(msg)
    for position, (left, right) in enumerate(zip(actual_leaves, expected_leaves, strict=True)):
        left_array = np.asarray(to_numpy(left))
        right_array = np.asarray(to_numpy(right))
        if left_array.shape != right_array.shape:
            msg = (
                f"leaf {position} has shape {left_array.shape}, "
                f"expected {right_array.shape}{context}"
            )
            raise ConformanceError(msg)
        if not np.allclose(left_array, right_array, rtol=rtol, atol=atol):
            deviation = np.max(np.abs(left_array - right_array))
            msg = (
                f"leaf {position} differs by {deviation:.3e} "
                f"(rtol={rtol:g} atol={atol:g}){context}"
                f"\n  actual   : {np.ravel(left_array)[:6]}"
                f"\n  expected : {np.ravel(right_array)[:6]}"
            )
            raise ConformanceError(msg)


def _assert_same_metadata(
    actual: Any,
    expected: Any,
    *,
    label: str,
    context: str,
) -> None:
    actual_leaves, actual_treedef = tree_flatten(actual)
    expected_leaves, expected_treedef = tree_flatten(expected)
    if actual_treedef != expected_treedef:
        msg = f"{label} structure differs from the dynamic reference{context}"
        raise ConformanceError(msg)
    for position, (actual_leaf, expected_leaf) in enumerate(
        zip(actual_leaves, expected_leaves, strict=True),
    ):
        actual_array = np.asarray(to_numpy(actual_leaf))
        expected_array = np.asarray(to_numpy(expected_leaf))
        if actual_array.shape != expected_array.shape:
            msg = (
                f"{label} leaf {position} has shape {actual_array.shape}, "
                f"expected {expected_array.shape}{context}"
            )
            raise ConformanceError(msg)
        if actual_array.dtype != expected_array.dtype:
            msg = (
                f"{label} leaf {position} has dtype {actual_array.dtype}, "
                f"expected {expected_array.dtype}{context}"
            )
            raise ConformanceError(msg)


def _directions(case: InvocationCase, values: Sequence[Any], probes: Probes) -> tuple[Any, ...]:
    return tuple(probes(values[index]) for index in case.differentiable_indices)


def _zero_direction(value: Any) -> Any:
    zero = np.zeros_like(np.asarray(value))
    if is_python_number(value):
        return type(value)(zero.reshape(()).item())
    return zero


def _direction_variants(
    case: InvocationCase,
    values: Sequence[Any],
    directions: tuple[Any, ...],
) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    """Probe each input partial independently and the combined differential."""
    if len(directions) == 1:
        return (("argument 0", directions),)
    independent = tuple(
        (
            f"argument {argument_index}",
            tuple(
                direction if position == active_position else _zero_direction(values[index])
                for position, (index, direction) in enumerate(
                    zip(case.differentiable_indices, directions, strict=True)
                )
            ),
        )
        for active_position, argument_index in enumerate(case.differentiable_indices)
    )
    return (*independent, ("combined", directions))


def _seed_for(case: InvocationCase, output: Any, probes: Probes) -> Any:
    # Mirror the output container, including field-named results.
    return tree_map(lambda leaf: wrap_for(case.frontend, probes(to_numpy(leaf))), output)


def _jvp(
    case: InvocationCase,
    values: tuple[Any, ...],
    directions: tuple[Any, ...],
) -> tuple[Any, Any]:
    return ad.jvp(case.call, case.differentiable_indices)(
        *_traced_arguments(case, values),
        tangents=_traced_arguments(case, directions),
        **dict(case.static),
    )


def _law_primal(
    case: InvocationCase,
    values: tuple[Any, ...],
    probes: Probes,
    context: str,
) -> None:
    reference = _invoke(case, values)
    traced, _tangent = _jvp(case, values, _directions(case, values, probes))
    _assert_close(
        traced,
        reference,
        rtol=case.tolerance.primal_rtol,
        atol=case.tolerance.primal_atol,
        context=context,
    )


def _numerical_directional_derivative(
    case: InvocationCase,
    values: tuple[Any, ...],
    directions: tuple[Any, ...],
) -> list[Any]:
    oracle_values = tuple(map(_promote_numerical_reference, values))
    by_index = dict(zip(case.differentiable_indices, directions, strict=True))
    tangents = tuple(
        None if index not in by_index else _promote_numerical_reference(by_index[index])
        for index in range(len(values))
    )
    return _directional_oracle(
        lambda step: _invoke(case, _shifted(oracle_values, tangents, step)),
        _oracle_step(case, oracle_values),
        op=case.op,
        input_is_real=not np.iscomplexobj(values[0]),
    )


def _eigenvalue_permutation(
    reference: np.ndarray[Any, Any],
    candidate: np.ndarray[Any, Any],
) -> tuple[int, ...]:
    size = int(reference.size)
    return min(
        itertools.permutations(range(size)),
        key=lambda order: float(
            np.sum(np.abs(candidate[np.asarray(order)] - reference) ** 2),
        ),
    )


def _unit_phase(overlap: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Return the sign or phase of each overlap, or one where it vanishes."""
    magnitude = np.abs(overlap)
    return np.where(magnitude == 0, 1, overlap / np.where(magnitude == 0, 1, magnitude))


def _align_eigenvectors(
    reference: np.ndarray[Any, Any],
    candidate: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    aligned = np.array(candidate, copy=True)
    aligned *= _unit_phase(np.sum(np.conjugate(aligned) * reference, axis=-2))[..., None, :]
    return aligned


def _align_svd_output(reference: Any, candidate: Any) -> Any:
    """Align paired singular-vector phases to Advect's V-fixed gauge."""
    if not (
        isinstance(reference, tuple)
        and isinstance(candidate, tuple)
        and len(reference) == len(candidate) == 3
    ):
        return candidate
    _reference_u, _reference_s, reference_vh = (np.asarray(leaf) for leaf in reference)
    candidate_u, candidate_s, candidate_vh = (np.asarray(leaf) for leaf in candidate)
    rank = int(candidate_s.shape[-1])
    reference_v = np.swapaxes(np.conjugate(reference_vh[..., :rank, :]), -1, -2)
    candidate_v = np.swapaxes(np.conjugate(candidate_vh[..., :rank, :]), -1, -2)
    phase = _unit_phase(np.sum(np.conjugate(candidate_v) * reference_v, axis=-2))
    aligned_u = np.array(candidate_u, copy=True)
    aligned_vh = np.array(candidate_vh, copy=True)
    aligned_u[..., :rank] *= phase[..., None, :]
    aligned_vh[..., :rank, :] *= np.conjugate(phase)[..., :, None]
    return aligned_u, candidate_s, aligned_vh


def _align_qr_output(reference: Any, candidate: Any) -> Any:
    """Align R's rows, and Q's columns, to the reference's diagonal sign or phase.

    A Householder QR flips a row's sign where its pivot crosses zero, which
    a low-precision difference step can straddle.
    """
    reference_r = np.asarray(to_numpy(reference[-1] if isinstance(reference, tuple) else reference))
    if not isinstance(candidate, tuple):
        candidate = (None, candidate)
    candidate_q, candidate_r = (
        None if leaf is None else np.asarray(to_numpy(leaf)) for leaf in candidate
    )
    phase = _unit_phase(
        np.diagonal(reference_r, axis1=-2, axis2=-1)
        * np.conjugate(np.diagonal(candidate_r, axis1=-2, axis2=-1))
    )
    aligned_r = candidate_r * phase[..., :, None]
    if candidate_q is None:
        return aligned_r
    return candidate_q * np.conjugate(phase)[..., None, :], aligned_r


# Decompositions whose outputs carry an arbitrary order, sign or vector phase.
_GAUGED_OPS = frozenset(
    {
        "array_ext.linalg.eig",
        "array_ext.linalg.eigh",
        "array_ext.linalg.eigvals",
        "array_ext.linalg.qr",
        "array_ext.linalg.qr_r",
        "array_ext.linalg.svd",
    },
)


def _align_gauge(
    op: str,
    reference: Any,
    candidate: Any,
    *,
    input_is_real: bool,
) -> Any:
    """Match the unordered or phase-free outputs of a gauged decomposition ``op``."""
    if op == "array_ext.linalg.svd":
        return _align_svd_output(reference, candidate)
    if op in {"array_ext.linalg.qr", "array_ext.linalg.qr_r"}:
        return _align_qr_output(reference, candidate)

    reference_values = np.asarray(reference[0] if isinstance(reference, tuple) else reference)
    candidate_values = np.asarray(candidate[0] if isinstance(candidate, tuple) else candidate)
    if reference_values.ndim < 1 or candidate_values.shape != reference_values.shape:
        return candidate
    aligned_values = np.array(candidate_values, copy=True)
    aligned_vectors = np.array(candidate[1], copy=True) if isinstance(candidate, tuple) else None
    if op != "array_ext.linalg.eigh":
        for batch_index in np.ndindex(reference_values.shape[:-1]):
            order = np.asarray(
                _eigenvalue_permutation(
                    reference_values[batch_index],
                    candidate_values[batch_index],
                ),
            )
            aligned_values[batch_index] = candidate_values[batch_index][order]
            if aligned_vectors is not None:
                aligned_vectors[batch_index] = aligned_vectors[batch_index][..., order]
    if not isinstance(candidate, tuple):
        return aligned_values

    reference_vectors = np.asarray(reference[1])
    assert aligned_vectors is not None
    if op == "array_ext.linalg.eigh" or (op == "array_ext.linalg.eig" and input_is_real):
        aligned_vectors = _align_eigenvectors(reference_vectors, aligned_vectors)
    return aligned_values, aligned_vectors


def _law_finite_difference(
    case: InvocationCase,
    values: tuple[Any, ...],
    probes: Probes,
    context: str,
) -> None:
    rtol, atol = _numerical_tolerances(case)
    directions = _directions(case, values, probes)
    for label, probe_directions in _direction_variants(case, values, directions):
        numerical = _numerical_directional_derivative(case, values, probe_directions)
        _value, tangent = _jvp(case, values, probe_directions)
        _assert_close(
            tangent,
            numerical,
            rtol=rtol,
            atol=atol,
            context=f"{context}\n  direction: {label}",
        )


def _assert_adjoint(
    cotangent: Any,
    tangent: Any,
    input_cotangent: Any,
    direction: Any,
    *,
    rtol: float,
    atol: float,
    context: str,
) -> None:
    """Require ``Re <v, J u> == Re <J* v, u>`` above the pairings' roundoff.

    Matching NaNs or infinities agree; a lone one does not.
    """
    forward = (_numpy_leaves(cotangent), _numpy_leaves(tangent))
    reverse = (_numpy_leaves(input_cotangent), _numpy_leaves(direction))
    forward_pairing = _real_inner_product(*forward)
    reverse_pairing = _real_inner_product(*reverse)
    scale = max(
        _real_inner_product_magnitude(*forward),
        _real_inner_product_magnitude(*reverse),
        1.0,
    )
    deviation = abs(forward_pairing - reverse_pairing)
    tolerance = atol + rtol * scale
    if not (
        forward_pairing == reverse_pairing
        or (math.isnan(forward_pairing) and math.isnan(reverse_pairing))
        or deviation <= tolerance
    ):
        msg = (
            f"adjoint identity violated by {deviation:.3e} (tolerance {tolerance:.3e}){context}"
            f"\n  <v, J u>    : {forward_pairing!r}"
            f"\n  <J* v, u>   : {reverse_pairing!r}"
        )
        raise ConformanceError(msg)


def _pull_back(
    case: InvocationCase,
    arguments: tuple[Any, ...],
    seed: Any,
    tangent: Any = None,
) -> Any:
    """Apply one pullback to ``seed``, or trace it and push ``tangent`` through."""
    _value, pullback = ad.vjp(case.call, case.differentiable_indices)(
        *arguments,
        **dict(case.static),
    )
    with pullback:
        return pullback(seed) if tangent is None else ad.jvp(pullback)(seed, tangents=tangent)


def _law_adjoint(
    case: InvocationCase,
    values: tuple[Any, ...],
    probes: Probes,
    context: str,
) -> None:
    """Pair ``J u`` with seeds pulled back through the public reverse path.

    The ordinary pullback applies the first seed. A forward transform over a
    second pullback carries the first seed as a traced value and a second
    seed as its tangent, so the law also requires every pullback to be linear
    and traceable unless the case declares, and still raises, a first-order
    boundary.
    """
    value = _transformed_output(case, values)
    arguments = _traced_arguments(case, values)
    directions = _directions(case, values, probes)
    seed, traced_seed = _seed_for(case, value, probes), _seed_for(case, value, probes)
    pairs = [("pullback", seed, _pull_back(case, arguments, seed))]
    try:
        traced_value, traced_tangent = _pull_back(case, arguments, seed, traced_seed)
    except NotImplementedError as refusal:
        if not case.first_order or case.first_order not in str(refusal):
            raise
    else:
        if case.first_order:
            msg = f"the declared first-order pullback now traces{context}"
            raise ConformanceError(msg)
        pairs += [
            ("traced pullback value", seed, traced_value),
            ("traced pullback tangent", traced_seed, traced_tangent),
        ]

    for label, probe_directions in _direction_variants(case, values, directions):
        _value, tangent = _jvp(case, values, probe_directions)
        for seed_label, cotangent, input_cotangent in pairs:
            _assert_adjoint(
                cotangent,
                tangent,
                input_cotangent,
                probe_directions,
                rtol=case.tolerance.adjoint_rtol,
                atol=case.tolerance.adjoint_atol,
                context=f"{context}\n  direction   : {label}\n  cotangent   : {seed_label}",
            )


def _law_dependence(
    case: InvocationCase,
    values: tuple[Any, ...],
    probes: Probes,
    context: str,
) -> None:
    """Check an explicit domain-backed promise of locally nonzero activity."""
    arguments = _traced_arguments(case, values)
    for index in sorted(case.dependence_indices):
        active = False
        for _attempt in range(3):
            direction = wrap_for(case.frontend, probes(values[index]))
            _value, tangent = ad.jvp(case.call, (index,))(
                *arguments,
                tangents=(direction,),
                **dict(case.static),
            )
            if any(np.any(np.asarray(to_numpy(leaf)) != 0) for leaf in _leaves(tangent)):
                active = True
                break
        if not active:
            name = case.arguments[index].name
            msg = (
                f"argument '{name}' promised a locally nonzero derivative but "
                f"three independent probes were zero{context}"
            )
            raise ConformanceError(msg)


def _law_structure(
    case: InvocationCase,
    values: tuple[Any, ...],
    probes: Probes,
    context: str,
) -> None:
    seed = _seed_for(case, _transformed_output(case, values), probes)
    cotangents = _pull_back(case, _traced_arguments(case, values), seed)
    for position, index in enumerate(case.differentiable_indices):
        primal = np.asarray(values[index])
        cotangent = np.asarray(to_numpy(cotangents[position]))
        name = case.arguments[index].name
        if cotangent.shape != primal.shape:
            msg = (
                f"cotangent for '{name}' has shape {cotangent.shape}, "
                f"expected {primal.shape}{context}"
            )
            raise ConformanceError(msg)
        if cotangent.dtype != primal.dtype:
            msg = (
                f"cotangent for '{name}' has dtype {cotangent.dtype}, "
                f"expected {primal.dtype}{context}"
            )
            raise ConformanceError(msg)


def _law_dtype(
    case: InvocationCase,
    values: tuple[Any, ...],
    probes: Probes,
    context: str,
) -> None:
    reference = _invoke(case, values)
    traced, tangent = _jvp(case, values, _directions(case, values, probes))
    _assert_same_metadata(tangent, traced, label="tangent", context=context)
    reference_leaves = _leaves(reference)
    traced_leaves = _leaves(traced)
    if len(reference_leaves) != len(traced_leaves):
        msg = f"output structure differs before dtype comparison{context}"
        raise ConformanceError(msg)
    for position, (actual, expected) in enumerate(
        zip(traced_leaves, reference_leaves, strict=True),
    ):
        actual_dtype = np.asarray(to_numpy(actual)).dtype
        expected_dtype = np.asarray(to_numpy(expected)).dtype
        if actual_dtype != expected_dtype:
            msg = (
                f"output leaf {position} has dtype {actual_dtype}, "
                f"expected {expected_dtype}{context}"
            )
            raise ConformanceError(msg)


def _unmutated[T](call: Callable[[], T], context: str, **inputs: Sequence[Any]) -> T:
    """Return ``call()`` after requiring it left each named input's leaves unchanged."""
    snapshots = {
        name: [np.array(to_numpy(leaf), copy=True) for leaf in _leaves(values)]
        for name, values in inputs.items()
    }
    result = call()
    for name, values in inputs.items():
        for position, (leaf, snapshot) in enumerate(
            zip(_leaves(values), snapshots[name], strict=True),
        ):
            current = np.asarray(to_numpy(leaf))
            if current.dtype != snapshot.dtype or not np.array_equal(
                current, snapshot, equal_nan=True
            ):
                msg = f"{name} {position} was mutated{context}"
                raise ConformanceError(msg)
    return result


def _law_no_input_mutation(
    case: InvocationCase,
    values: tuple[Any, ...],
    probes: Probes,
    context: str,
) -> None:
    inputs = _traced_arguments(case, values)
    _unmutated(lambda: case.call(*inputs, **dict(case.static)), context, input=inputs)

    inputs = _traced_arguments(case, values)
    tangents = _traced_arguments(case, _directions(case, values, probes))
    _unmutated(
        lambda: ad.jvp(case.call, case.differentiable_indices)(
            *inputs,
            tangents=tangents,
            **dict(case.static),
        ),
        f"{context}\n  transform: jvp",
        input=inputs,
        tangent=tangents,
    )


def _weighted_sum(output: Any, weights: Any) -> Any:
    """Reduce outputs to the real scalar ``<weights, output>`` in their namespace."""
    total: Any = 0.0
    for leaf, weight in zip(_leaves(output), _leaves(weights), strict=True):
        namespace = getattr(leaf, "__array_namespace__", None)
        xp = namespace() if callable(namespace) else np
        if np.iscomplexobj(to_numpy(weight)):
            total = total + xp.real(xp.sum(leaf * xp.conj(weight)))
        else:
            total = total + xp.sum(leaf * weight)
    return total


def _law_second_order(
    case: InvocationCase,
    values: tuple[Any, ...],
    probes: Probes,
    context: str,
) -> None:
    """Nested derivatives match the dense Hessian and central differences.

    Along the first differentiable argument ``x``, forward-over-reverse
    ``hvp`` of the probe-weighted outputs must equal the dense
    reverse-over-reverse Hessian and a difference of gradients, and
    forward-over-forward must equal a difference of JVPs.
    The differences run above low-precision roundoff, like the first-order
    oracle.
    """
    primary = case.differentiable_indices[0]
    weights = _seed_for(case, _invoke(case, values), probes)
    direction, second = probes(values[primary]), probes(values[primary])
    oracle = tuple(_promote_numerical_reference(value) for value in values)
    step = _finite_difference_step(case, (oracle[primary],))

    def restricted(point: tuple[Any, ...], *, promote: bool) -> tuple[Any, Any]:
        """Return the weighted scalar and the JVP along ``direction``, in ``x``."""
        lift = _promote_numerical_reference if promote else (lambda value: value)
        arguments = list(_traced_arguments(case, point))
        weight = tree_map(lambda leaf: wrap_for(case.frontend, lift(to_numpy(leaf))), weights)
        tangent = wrap_for(case.frontend, lift(direction))

        def call(argument: Any) -> Any:
            arguments[primary] = argument
            return case.call(*arguments, **dict(case.static))

        def scalar(argument: Any) -> Any:
            return _weighted_sum(call(argument), weight)

        def along(argument: Any) -> Any:
            return ad.jvp(call)(argument, tangents=tangent)[1]

        return scalar, along

    def difference(function: Any, vector: Any) -> list[Any]:
        shift = (_promote_numerical_reference(vector),)
        return _directional_oracle(
            lambda moved: function(
                wrap_for(case.frontend, _shifted((oracle[primary],), shift, moved)[0])
            ),
            step,
        )

    scalar, along = restricted(values, promote=False)
    oracle_scalar, oracle_along = restricted(oracle, promote=True)
    x = wrap_for(case.frontend, values[primary])
    _value, curvature = ad.hvp(scalar)(x, vectors=wrap_for(case.frontend, direction))
    _value, forward = ad.jvp(along)(x, tangents=wrap_for(case.frontend, second))
    checks = [
        (
            "directional gradient difference",
            curvature,
            difference(ad.grad(oracle_scalar), direction),
        ),
        ("directional JVP difference", forward, difference(oracle_along, second)),
    ]
    # The dense Hessian needs real inputs and a NumPy-compatible namespace.
    if case.frontend is Frontend.NUMPY and not np.iscomplexobj(values[primary]):
        size = np.size(direction)
        dense = np.reshape(np.asarray(to_numpy(ad.hessian(scalar)(x))), (size, size))
        expected = np.reshape(dense @ np.ravel(direction), np.shape(direction))
        checks.append(("dense Hessian", curvature, expected))
    for label, actual, expected in checks:
        _assert_close(
            actual,
            expected,
            rtol=case.tolerance.finite_difference_rtol,
            atol=case.tolerance.finite_difference_atol,
            context=f"{context}\n  second-order oracle: {label}",
        )


def _serialized(program: ad.StagedProgram, context: str) -> ad.StagedProgram:
    """Round-trip through real JSON, as documented persistence does."""
    payload = program.to_dict()
    restored = ad.StagedProgram.from_dict(json.loads(json.dumps(payload)))
    if restored.to_dict() != payload:
        msg = f"serialized program does not reload to the same record{context}"
        raise ConformanceError(msg)
    return restored


@functools.cache
def _staged(
    case: InvocationCase,
    specs: tuple[ad.ArraySpec, ...],
) -> tuple[ad.StagedProgram, ad.StagedProgram]:
    """Stage one signature and reload it; programs do not depend on values."""

    def call(*arguments: Any) -> Any:
        return case.call(*arguments, **dict(case.static))

    program = ad.stage(call, specs=specs)
    return program, _serialized(program, f"\n  op: {case.op}")


@functools.cache
def _staged_pullback(
    case: InvocationCase,
    specs: tuple[ad.ArraySpec, ...],
) -> tuple[ad.StagedProgram, ad.StagedProgram]:
    """Build and reload the pullback program of the reloaded primal program."""
    program = ad.vjp_program(_staged(case, specs)[1], argnums=case.differentiable_indices)
    return program, _serialized(program, f"\n  op: {case.op}")


def _law_staged(
    case: InvocationCase,
    values: tuple[Any, ...],
    probes: Probes,
    context: str,
) -> None:
    dynamic = _invoke(case, values)
    arguments = _traced_arguments(case, values)
    specs = _stage_specs(values)
    program, restored = _staged(case, specs)
    for label, staged_program in (
        ("compiled primal", program),
        ("serialized primal", restored),
    ):
        staged = _unmutated(
            functools.partial(staged_program, *arguments),
            f"{context}\n  staged transform: {label}",
            input=arguments,
        )
        _assert_close(
            staged,
            dynamic,
            rtol=case.tolerance.primal_rtol,
            atol=case.tolerance.primal_atol,
            context=f"{context}\n  staged transform: {label}",
        )
        _assert_same_metadata(
            staged,
            dynamic,
            label=f"{label} output",
            context=context,
        )

    if (
        not case.differentiable_indices
        or get_registry().get(case.op).non_differentiable_reason is not None
    ):
        return

    # A loaded program is also an ordinary function under dynamic transforms.
    directions = _directions(case, values, probes)
    _value, staged_tangent = ad.jvp(restored, case.differentiable_indices)(
        *arguments,
        tangents=_traced_arguments(case, directions),
    )
    transformed, dynamic_tangent = _jvp(case, values, directions)
    _assert_close(
        staged_tangent,
        dynamic_tangent,
        rtol=case.tolerance.adjoint_rtol,
        atol=case.tolerance.adjoint_atol,
        context=f"{context}\n  staged transform: jvp of serialized primal",
    )
    _assert_same_metadata(
        staged_tangent,
        dynamic_tangent,
        label="jvp of serialized primal tangent",
        context=context,
    )

    cotangent = _seed_for(case, transformed, probes)
    dynamic_arguments = _traced_arguments(case, values)
    dynamic_cotangents = _unmutated(
        functools.partial(_pull_back, case, dynamic_arguments, cotangent),
        f"{context}\n  transform: dynamic vjp",
        input=dynamic_arguments,
        cotangent=cotangent,
    )
    staged_cotangent = tree_map(
        lambda leaf: wrap_for(case.frontend, np.asarray(leaf)) if is_python_number(leaf) else leaf,
        cotangent,
    )
    for label, pullback_program in zip(
        ("compiled vjp", "serialized vjp"),
        _staged_pullback(case, specs),
        strict=True,
    ):
        result = _unmutated(
            functools.partial(pullback_program, *arguments, cotangent=staged_cotangent),
            f"{context}\n  staged transform: {label}",
            input=arguments,
            cotangent=staged_cotangent,
        )
        _assert_close(
            result,
            dynamic_cotangents,
            rtol=case.tolerance.adjoint_rtol,
            atol=case.tolerance.adjoint_atol,
            context=f"{context}\n  staged transform: {label}",
        )
        _assert_same_metadata(
            result,
            dynamic_cotangents,
            label=f"{label} cotangent",
            context=context,
        )


_LAWS = {
    Law.PRIMAL: _law_primal,
    Law.FINITE_DIFFERENCE: _law_finite_difference,
    Law.ADJOINT: _law_adjoint,
    Law.DEPENDENCE: _law_dependence,
    Law.STRUCTURE: _law_structure,
    Law.NO_INPUT_MUTATION: _law_no_input_mutation,
    Law.DTYPE: _law_dtype,
    Law.SECOND_ORDER: _law_second_order,
    Law.STAGED: _law_staged,
}
assert set(_LAWS) == set(Law), "every law needs exactly one implementation"
# These contracts do not depend on probe values, so they keep the anchors.
_PROBE_INDEPENDENT_LAWS = frozenset({Law.PRIMAL, Law.STRUCTURE, Law.NO_INPUT_MUTATION, Law.DTYPE})


def check_law(
    case: InvocationCase,
    law: Law,
    values: tuple[Any, ...],
    *,
    variant: int = 0,
    data: DataObject | None = None,
) -> None:
    """Run one public-transform law on a Hypothesis-drawn invocation.

    With ``data``, laws whose verdict depends on probe values also draw their
    tangent and cotangent probes; otherwise they use the deterministic,
    mutually independent anchors of ``Probes``.
    """
    context = _describe(case, law, variant)
    probes = Probes(None if law in _PROBE_INDEPENDENT_LAWS else data)
    _LAWS[law](case.resolve_variant(variant), values, probes, context)
