"""Public-transform laws for primitive conformance."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import numpy as np

import advect as ad
from advect.core._pytree import tree_flatten, tree_unflatten
from advect.core._registry import get_registry
from advect.testing import _real_inner_product, _real_inner_product_magnitude
from advect_conformance_tests._harness._cases import Law, NumericalReference
from advect_conformance_tests._harness._frontends import is_python_number, to_numpy, wrap_for

if TYPE_CHECKING:
    from collections.abc import Sequence

    from advect_conformance_tests._harness._cases import InvocationCase

__all__ = ["ConformanceError", "check_law"]


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
    return [_promote_numerical_reference(to_numpy(leaf)) for leaf in _leaves(value)]


def _invoke(case: InvocationCase, values: Sequence[Any]) -> Any:
    wrapped = tuple(wrap_for(case.frontend, value) for value in values)
    return case.call(*wrapped, **dict(case.static))


def _traced_arguments(case: InvocationCase, values: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(wrap_for(case.frontend, value) for value in values)


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
    size = max(array.size, 1)
    positions = np.arange(size, dtype=np.float64)
    real = 1.0 + ((positions + ordinal) % 5.0)
    real[1::2] *= -1.0
    direction: np.ndarray[Any, Any]
    if np.issubdtype(array.dtype, np.complexfloating):
        imaginary = 1.0 + ((positions[::-1] + 2 * ordinal) % 7.0)
        direction = real + 1j * imaginary
    else:
        direction = real
    direction = (direction / np.max(np.abs(direction))).reshape(array.shape).astype(array.dtype)
    if is_python_number(value):
        return type(value)(direction.reshape(()).item())
    return direction


def _perturbed(
    values: Sequence[Any],
    directions: Sequence[Any],
    indices: Sequence[int],
    step: complex,
) -> tuple[Any, ...]:
    moved = list(values)
    for position, index in enumerate(indices):
        original = values[index]
        shifted = np.asarray(original) + step * np.asarray(directions[position])
        moved[index] = type(original)(shifted) if is_python_number(original) else shifted
    return tuple(moved)


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


def _directions(case: InvocationCase, values: Sequence[Any], ordinal: int = 0) -> tuple[Any, ...]:
    return tuple(
        _probe_like(values[index], ordinal + position)
        for position, index in enumerate(case.differentiable_indices)
    )


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


def _seed_for(case: InvocationCase, output: Any, ordinal: int = 0) -> Any:
    leaves = [
        wrap_for(case.frontend, _probe_like(to_numpy(leaf), ordinal + position))
        for position, leaf in enumerate(_leaves(output))
    ]
    return leaves[0] if len(leaves) == 1 else tuple(leaves)


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
    directions: tuple[Any, ...],
    context: str,
) -> None:
    reference = _invoke(case, values)
    traced, _tangent = _jvp(case, values, directions)
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
    indices = case.differentiable_indices
    input_is_real = not np.iscomplexobj(values[0])
    reference = _invoke(case, values)
    if case.numerical_reference is NumericalReference.COMPLEX_STEP:
        oracle_values = tuple(_promote_numerical_reference(value) for value in values)
        oracle_directions = tuple(
            _promote_numerical_reference(direction) for direction in directions
        )
        step = case.tolerance.complex_step
        shifted = _invoke(
            case,
            _perturbed(oracle_values, oracle_directions, indices, 1j * step),
        )
        shifted = _align_eigen_output(case.op, reference, shifted, input_is_real=input_is_real)
        return [np.imag(to_numpy(leaf)) / step for leaf in _leaves(shifted)]

    oracle_values = tuple(_promote_numerical_reference(value) for value in values)
    oracle_directions = tuple(_promote_numerical_reference(direction) for direction in directions)
    reference = _invoke(case, oracle_values)
    step = _finite_difference_step(case, oracle_values)
    forward = _invoke(
        case,
        _perturbed(oracle_values, oracle_directions, indices, step),
    )
    backward = _invoke(
        case,
        _perturbed(oracle_values, oracle_directions, indices, -step),
    )
    forward = _align_eigen_output(case.op, reference, forward, input_is_real=input_is_real)
    backward = _align_eigen_output(case.op, reference, backward, input_is_real=input_is_real)
    return [
        (to_numpy(upper) - to_numpy(lower)) / (2.0 * step)
        for upper, lower in zip(_leaves(forward), _leaves(backward), strict=True)
    ]


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


def _align_eigenvectors(
    reference: np.ndarray[Any, Any],
    candidate: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    aligned = np.array(candidate, copy=True)
    overlap = np.sum(np.conjugate(aligned) * reference, axis=-2)
    magnitude = np.abs(overlap)
    phase = np.where(magnitude == 0, 1, overlap / np.where(magnitude == 0, 1, magnitude))
    aligned *= phase[..., None, :]
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
    overlap = np.sum(np.conjugate(candidate_v) * reference_v, axis=-2)
    magnitude = np.abs(overlap)
    phase = np.where(magnitude == 0, 1, overlap / np.where(magnitude == 0, 1, magnitude))
    aligned_u = np.array(candidate_u, copy=True)
    aligned_vh = np.array(candidate_vh, copy=True)
    aligned_u[..., :rank] *= phase[..., None, :]
    aligned_vh[..., :rank, :] *= np.conjugate(phase)[..., :, None]
    return aligned_u, candidate_s, aligned_vh


def _align_eigen_output(
    op: str,
    reference: Any,
    candidate: Any,
    *,
    input_is_real: bool,
) -> Any:
    """Match unordered eigenpairs and align their arbitrary vector phase."""
    if op == "array_ext.linalg.svd":
        return _align_svd_output(reference, candidate)
    if op not in {
        "array_ext.linalg.eig",
        "array_ext.linalg.eigh",
        "array_ext.linalg.eigvals",
    }:
        return candidate

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
    directions: tuple[Any, ...],
    context: str,
) -> None:
    rtol, atol = _numerical_tolerances(case)
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


def _law_adjoint(
    case: InvocationCase,
    values: tuple[Any, ...],
    directions: tuple[Any, ...],
    context: str,
) -> None:
    indices = case.differentiable_indices
    static = dict(case.static)
    value = _invoke(case, values)
    cotangent = _seed_for(case, value)
    _value, pullback = ad.vjp(case.call, indices)(
        *_traced_arguments(case, values),
        **static,
    )
    try:
        input_cotangent = pullback(cotangent)
    finally:
        close = getattr(pullback, "close", None)
        if callable(close):
            close()

    for label, probe_directions in _direction_variants(case, values, directions):
        _value, tangent = _jvp(case, values, probe_directions)
        cotangent_leaves = _numpy_leaves(cotangent)
        tangent_leaves = _numpy_leaves(tangent)
        input_cotangent_leaves = _numpy_leaves(input_cotangent)
        direction_leaves = [to_numpy(direction) for direction in probe_directions]
        forward_pairing = _real_inner_product(cotangent_leaves, tangent_leaves)
        reverse_pairing = _real_inner_product(input_cotangent_leaves, direction_leaves)
        scale = max(
            _real_inner_product_magnitude(cotangent_leaves, tangent_leaves),
            _real_inner_product_magnitude(input_cotangent_leaves, direction_leaves),
            1.0,
        )
        deviation = abs(forward_pairing - reverse_pairing)
        tolerance = case.tolerance.adjoint_atol + case.tolerance.adjoint_rtol * scale
        if deviation > tolerance:
            msg = (
                f"adjoint identity violated by {deviation:.3e} "
                f"(tolerance {tolerance:.3e}){context}"
                f"\n  direction   : {label}"
                f"\n  <v, J u>    : {forward_pairing!r}"
                f"\n  <J* v, u>   : {reverse_pairing!r}"
            )
            raise ConformanceError(msg)


def _law_dependence(
    case: InvocationCase,
    values: tuple[Any, ...],
    context: str,
) -> None:
    """Check an explicit domain-backed promise of locally nonzero activity."""
    arguments = _traced_arguments(case, values)
    for index in sorted(case.dependence_indices):
        active = False
        for ordinal in range(3):
            direction = wrap_for(case.frontend, _probe_like(values[index], ordinal))
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


def _pullback_once(case: InvocationCase, values: tuple[Any, ...]) -> tuple[Any, Any]:
    value, pullback = ad.vjp(case.call, case.differentiable_indices)(
        *_traced_arguments(case, values),
        **dict(case.static),
    )
    try:
        cotangents = pullback(_seed_for(case, value))
    finally:
        close = getattr(pullback, "close", None)
        if callable(close):
            close()
    return value, cotangents


def _law_structure(case: InvocationCase, values: tuple[Any, ...], context: str) -> None:
    _value, cotangents = _pullback_once(case, values)
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
    directions: tuple[Any, ...],
    context: str,
) -> None:
    reference = _invoke(case, values)
    traced, _tangent = _jvp(case, values, directions)
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


def _assert_unchanged(
    values: Sequence[Any],
    snapshots: Sequence[np.ndarray[Any, Any]],
    *,
    context: str,
) -> None:
    for position, (value, snapshot) in enumerate(zip(values, snapshots, strict=True)):
        current = np.asarray(to_numpy(value))
        if current.dtype != snapshot.dtype or not np.array_equal(current, snapshot, equal_nan=True):
            msg = f"input {position} was mutated{context}"
            raise ConformanceError(msg)


def _law_no_input_mutation(
    case: InvocationCase,
    values: tuple[Any, ...],
    directions: tuple[Any, ...],
    context: str,
) -> None:
    provider_inputs = _traced_arguments(case, values)
    provider_snapshots = [np.array(to_numpy(value), copy=True) for value in provider_inputs]
    case.call(*provider_inputs, **dict(case.static))
    _assert_unchanged(provider_inputs, provider_snapshots, context=context)

    traced_inputs = _traced_arguments(case, values)
    traced_snapshots = [np.array(to_numpy(value), copy=True) for value in traced_inputs]
    traced_directions = _traced_arguments(case, directions)
    direction_snapshots = [np.array(to_numpy(value), copy=True) for value in traced_directions]
    ad.jvp(case.call, case.differentiable_indices)(
        *traced_inputs,
        tangents=traced_directions,
        **dict(case.static),
    )
    _assert_unchanged(traced_inputs, traced_snapshots, context=context)
    _assert_unchanged(
        traced_directions,
        direction_snapshots,
        context=f"{context}\n  derivative input: tangent",
    )


def _law_second_order(case: InvocationCase, values: tuple[Any, ...], context: str) -> None:
    indices = case.differentiable_indices
    primary = indices[0]
    weights = _probe_like(_invoke(case, values))

    def scalar(argument: Any) -> Any:
        arguments = list(_traced_arguments(case, values))
        arguments[primary] = argument
        return np.sum(case.call(*arguments, **dict(case.static)) * weights)

    direction = _probe_like(values[primary])
    _value, curvature = ad.hvp(scalar)(values[primary], vectors=direction)
    dense = np.asarray(ad.hessian(scalar)(values[primary]))
    flat = np.reshape(dense, (np.size(direction), np.size(direction)))
    expected = np.reshape(flat @ np.ravel(direction), np.shape(direction))
    _assert_close(
        curvature,
        expected,
        rtol=case.tolerance.finite_difference_rtol,
        atol=case.tolerance.finite_difference_atol,
        context=f"{context}\n  second-order oracle: dense Hessian",
    )

    step = _finite_difference_step(case, (values[primary],))
    gradient = ad.grad(scalar)
    positive_argument = _perturbed((values[primary],), (direction,), (0,), step)[0]
    negative_argument = _perturbed((values[primary],), (direction,), (0,), -step)[0]
    positive = gradient(wrap_for(case.frontend, positive_argument))
    negative = gradient(wrap_for(case.frontend, negative_argument))
    numerical = (np.asarray(positive) - np.asarray(negative)) / (2.0 * step)
    _assert_close(
        curvature,
        numerical,
        rtol=case.tolerance.finite_difference_rtol,
        atol=case.tolerance.finite_difference_atol,
        context=f"{context}\n  second-order oracle: directional gradient difference",
    )


def _law_staged(case: InvocationCase, values: tuple[Any, ...], context: str) -> None:
    dynamic = _invoke(case, values)

    def call(*arguments: Any) -> Any:
        return case.call(*arguments, **dict(case.static))

    arguments = _traced_arguments(case, values)
    argument_snapshots = [np.array(to_numpy(value), copy=True) for value in arguments]
    program = ad.stage(call, specs=_stage_specs(values))
    restored = ad.StagedProgram.from_dict(program.to_dict())
    for label, staged_program in (
        ("compiled primal", program),
        ("serialized primal", restored),
    ):
        staged = staged_program(*arguments)
        _assert_unchanged(
            arguments,
            argument_snapshots,
            context=f"{context}\n  staged transform: {label}",
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

    cotangent = _seed_for(case, dynamic)
    cotangent_leaves, cotangent_treedef = tree_flatten(cotangent)
    staged_cotangent = tree_unflatten(
        cotangent_treedef,
        [
            wrap_for(case.frontend, np.asarray(leaf)) if is_python_number(leaf) else leaf
            for leaf in cotangent_leaves
        ],
    )
    dynamic_arguments = _traced_arguments(case, values)
    dynamic_argument_snapshots = [
        np.array(to_numpy(value), copy=True) for value in dynamic_arguments
    ]
    cotangent_leaves = _leaves(cotangent)
    cotangent_snapshots = [np.array(to_numpy(value), copy=True) for value in cotangent_leaves]
    _dynamic_value, dynamic_pullback = ad.vjp(
        case.call,
        case.differentiable_indices,
    )(
        *dynamic_arguments,
        **dict(case.static),
    )
    try:
        dynamic_cotangents = dynamic_pullback(cotangent)
    finally:
        close = getattr(dynamic_pullback, "close", None)
        if callable(close):
            close()
    _assert_unchanged(
        dynamic_arguments,
        dynamic_argument_snapshots,
        context=f"{context}\n  derivative input: dynamic primal",
    )
    _assert_unchanged(
        cotangent_leaves,
        cotangent_snapshots,
        context=f"{context}\n  derivative input: dynamic cotangent",
    )
    pullback_program = ad.vjp_program(
        restored,
        argnums=case.differentiable_indices,
    )
    staged_cotangent_leaves = _leaves(staged_cotangent)
    staged_cotangent_snapshots = [
        np.array(to_numpy(value), copy=True) for value in staged_cotangent_leaves
    ]
    staged_cotangents = pullback_program(*arguments, cotangent=staged_cotangent)
    _assert_unchanged(
        arguments,
        argument_snapshots,
        context=f"{context}\n  derivative input: compiled-vjp primal",
    )
    _assert_unchanged(
        staged_cotangent_leaves,
        staged_cotangent_snapshots,
        context=f"{context}\n  derivative input: compiled-vjp cotangent",
    )
    restored_pullback = ad.StagedProgram.from_dict(pullback_program.to_dict())
    roundtrip_cotangents = restored_pullback(
        *arguments,
        cotangent=staged_cotangent,
    )
    _assert_unchanged(
        arguments,
        argument_snapshots,
        context=f"{context}\n  derivative input: serialized-vjp primal",
    )
    _assert_unchanged(
        staged_cotangent_leaves,
        staged_cotangent_snapshots,
        context=f"{context}\n  derivative input: serialized-vjp cotangent",
    )
    for label, result in (
        ("compiled vjp", staged_cotangents),
        ("serialized vjp", roundtrip_cotangents),
    ):
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


def check_law(
    case: InvocationCase,
    law: Law,
    values: tuple[Any, ...],
    *,
    variant: int = 0,
) -> None:
    """Run one public-transform law on a Hypothesis-drawn invocation."""
    context = _describe(case, law, variant)
    case = case.resolve_variant(variant)
    directions = _directions(case, values)

    if law is Law.PRIMAL:
        _law_primal(case, values, directions, context)
    elif law is Law.FINITE_DIFFERENCE:
        _law_finite_difference(case, values, directions, context)
    elif law is Law.ADJOINT:
        _law_adjoint(case, values, directions, context)
    elif law is Law.DEPENDENCE:
        _law_dependence(case, values, context)
    elif law is Law.STRUCTURE:
        _law_structure(case, values, context)
    elif law is Law.NO_INPUT_MUTATION:
        _law_no_input_mutation(case, values, directions, context)
    elif law is Law.DTYPE:
        _law_dtype(case, values, directions, context)
    elif law is Law.SECOND_ORDER:
        _law_second_order(case, values, context)
    elif law is Law.STAGED:
        _law_staged(case, values, context)
    else:  # Exhaustive over a closed enum.
        msg = f"Law {law.value} has no implementation"
        raise ConformanceError(msg)
