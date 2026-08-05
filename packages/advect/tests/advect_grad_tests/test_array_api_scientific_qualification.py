"""Backend-neutral scientific workloads across both Advect lifetimes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import array_api_strict as strict
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scripts._support import qualify_array_providers

import advect as ad

if TYPE_CHECKING:
    from collections.abc import Callable


def _scientific_loss(
    signal: object,
    matrix: object,
    right: object,
    *,
    scale: float = 0.25,
) -> object:
    namespace = signal.__array_namespace__()
    spectrum = namespace.fft.fft(signal)
    solution = namespace.linalg.solve(matrix, right)
    spectral_energy = namespace.sum(namespace.real(namespace.conj(spectrum) * spectrum))
    solution_energy = namespace.sum(namespace.real(namespace.conj(solution) * solution))
    return scale * spectral_energy + solution_energy


def _stencil_loss(field: object) -> object:
    namespace = field.__array_namespace__()
    updated = field.copy()
    laplacian = field[2:] - 2 * field[1:-1] + field[:-2]
    updated[1:-1] += 0.1 * laplacian
    return namespace.sum(updated * updated)


def _versioned_update_loss(field: object) -> object:
    namespace = field.__array_namespace__()
    updated = field.copy()
    old_version = updated * updated
    updated[0] = 5.0
    return namespace.sum(old_version + updated)


def _numpy_arrays() -> tuple[object, object, object]:
    return (
        np.asarray([1 + 2j, 2 - 1j, -1 + 0.5j, 0.25 - 0.75j], dtype=np.complex64),
        np.asarray(
            [[3 + 0.1j, 1 - 0.2j], [1 + 0.3j, 2 - 0.1j]],
            dtype=np.complex64,
        ),
        np.asarray([1 + 0.5j, 2 - 0.25j], dtype=np.complex64),
    )


def _strict_arrays() -> tuple[object, object, object]:
    return (
        strict.asarray(
            [1 + 2j, 2 - 1j, -1 + 0.5j, 0.25 - 0.75j],
            dtype=strict.complex64,
        ),
        strict.asarray(
            [[3 + 0.1j, 1 - 0.2j], [1 + 0.3j, 2 - 0.1j]],
            dtype=strict.complex64,
        ),
        strict.asarray([1 + 0.5j, 2 - 0.25j], dtype=strict.complex64),
    )


@pytest.mark.parametrize(
    "factory",
    [_numpy_arrays, _strict_arrays],
    ids=["numpy", "array-api-strict"],
)
def test_scientific_loss_qualifies_dynamic_linear_maps(
    factory: Callable[[], tuple[object, object, object]],
) -> None:
    signal, matrix, right = factory()
    tangents = tuple(value * 0.05 for value in (signal, matrix, right))

    value, gradients = ad.value_and_grad(
        _scientific_loss,
        argnums=(0, 1, 2),
    )(signal, matrix, right)
    jvp_value, directional = ad.jvp(
        _scientific_loss,
        argnums=(0, 1, 2),
    )(signal, matrix, right, tangents=tangents)
    vjp_value, pullback = ad.vjp(
        _scientific_loss,
        argnums=(0, 1, 2),
    )(signal, matrix, right)
    try:
        namespace = value.__array_namespace__()
        reverse_gradients = pullback(namespace.ones_like(value))
    finally:
        pullback.close()

    assert_allclose(np.asarray(jvp_value), np.asarray(value), rtol=1e-6, atol=1e-6)
    assert_allclose(np.asarray(vjp_value), np.asarray(value), rtol=1e-6, atol=1e-6)
    for gradient, reverse_gradient, primal in zip(
        gradients,
        reverse_gradients,
        (signal, matrix, right),
        strict=True,
    ):
        assert gradient.dtype == primal.dtype
        assert type(gradient) is type(primal)
        assert_allclose(
            np.asarray(reverse_gradient),
            np.asarray(gradient),
            rtol=2e-5,
            atol=2e-5,
        )

    epsilon = 1e-3
    plus = tuple(
        primal + epsilon * tangent
        for primal, tangent in zip((signal, matrix, right), tangents, strict=True)
    )
    minus = tuple(
        primal - epsilon * tangent
        for primal, tangent in zip((signal, matrix, right), tangents, strict=True)
    )
    finite_difference = (
        np.asarray(_scientific_loss(*plus)) - np.asarray(_scientific_loss(*minus))
    ) / (2 * epsilon)
    assert_allclose(
        np.asarray(directional),
        finite_difference,
        rtol=3e-3,
        atol=3e-3,
    )


@pytest.mark.parametrize(
    "factory",
    [_numpy_arrays, _strict_arrays],
    ids=["numpy", "array-api-strict"],
)
def test_scientific_loss_qualifies_staging_and_serialized_derivative(
    factory: Callable[[], tuple[object, object, object]],
) -> None:
    signal, matrix, right = factory()
    specs = tuple(ad.ArraySpec(value.shape, value.dtype) for value in (signal, matrix, right))
    primal = ad.stage(_scientific_loss, specs=specs)
    gradient = ad.grad(primal, argnums=(0, 1, 2))
    restored = ad.StagedProgram.from_dict(gradient.to_dict())

    dynamic_value, dynamic_gradients = ad.value_and_grad(
        _scientific_loss,
        argnums=(0, 1, 2),
    )(signal, matrix, right)
    staged_value = primal(signal, matrix, right)
    staged_gradients = gradient(signal, matrix, right)
    restored_gradients = restored(signal, matrix, right)

    assert_allclose(np.asarray(staged_value), np.asarray(dynamic_value), rtol=1e-6, atol=1e-6)
    for staged, restored_value, dynamic, source in zip(
        staged_gradients,
        restored_gradients,
        dynamic_gradients,
        (signal, matrix, right),
        strict=True,
    ):
        assert type(staged) is type(source)
        assert staged.dtype == source.dtype
        assert_allclose(np.asarray(staged), np.asarray(dynamic), rtol=2e-5, atol=2e-5)
        assert_allclose(np.asarray(restored_value), np.asarray(dynamic), rtol=2e-5, atol=2e-5)

    graph_ops = {primal.graph.get_node(node_id).op for node_id in primal.graph.node_ids()}
    assert "array_ext.fft.fft" in graph_ops
    assert "array_ext.linalg.solve" in graph_ops


def test_one_captured_constant_program_is_portable_between_cpu_providers() -> None:
    window = np.asarray([1.0, 0.5, 0.25, 0.125], dtype=np.float32)

    def captured_loss(signal: object) -> object:
        namespace = signal.__array_namespace__()
        transformed = namespace.fft.fft(signal * window)
        return namespace.sum(namespace.real(namespace.conj(transformed) * transformed))

    program = ad.stage(
        captured_loss,
        specs=(ad.ArraySpec((4,), "complex64"),),
    )
    gradient = ad.grad(program)
    restored = ad.StagedProgram.from_dict(gradient.to_dict())

    assert len(program.constants) == 1
    for signal in (_numpy_arrays()[0], _strict_arrays()[0]):
        result = program(signal)
        derivative = gradient(signal)
        restored_derivative = restored(signal)
        assert result.dtype == signal.__array_namespace__().float32
        assert derivative.dtype == signal.dtype
        assert type(derivative) is type(signal)
        assert_allclose(
            np.asarray(restored_derivative),
            np.asarray(derivative),
            rtol=1e-6,
            atol=1e-6,
        )


@pytest.mark.parametrize(
    "field",
    [
        np.arange(8, dtype=np.float32),
        strict.arange(8, dtype=strict.float32),
    ],
    ids=["numpy", "array-api-strict"],
)
def test_functionalized_stencil_matches_across_lifetimes(
    field: object,
) -> None:
    dynamic = ad.grad(_stencil_loss)(field)
    staged = ad.grad(
        ad.stage(
            _stencil_loss,
            specs=(ad.ArraySpec(field.shape, field.dtype),),
        )
    )(field)

    assert type(dynamic) is type(field)
    assert type(staged) is type(field)
    assert dynamic.dtype == field.dtype
    assert staged.dtype == field.dtype
    assert_allclose(np.asarray(staged), np.asarray(dynamic), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "field",
    [
        np.asarray([2.0, 3.0], dtype=np.float64),
        strict.asarray([2.0, 3.0], dtype=strict.float64),
    ],
    ids=["numpy", "array-api-strict"],
)
def test_functionalized_update_preserves_prior_ssa_versions(field: object) -> None:
    value, dynamic = ad.value_and_grad(_versioned_update_loss)(field)
    program = ad.stage(
        _versioned_update_loss,
        specs=(ad.ArraySpec(field.shape, field.dtype),),
    )
    staged_value = program(field)
    staged = ad.grad(program)(field)

    assert_allclose(np.asarray(value), 21.0)
    assert_allclose(np.asarray(staged_value), 21.0)
    assert_allclose(np.asarray(dynamic), np.asarray([4.0, 7.0]))
    assert_allclose(np.asarray(staged), np.asarray([4.0, 7.0]))


@pytest.mark.parametrize(
    "field",
    [
        np.arange(4, dtype=np.float32),
        strict.arange(4, dtype=strict.float32),
    ],
    ids=["numpy", "array-api-strict"],
)
def test_staged_derivative_preserves_complex_weak_scalar_promotion(
    field: object,
) -> None:
    def loss(value: object) -> object:
        namespace = value.__array_namespace__()
        rotated = 1j * value
        return namespace.sum(namespace.real(namespace.conj(rotated) * rotated))

    staged_loss = ad.stage(
        loss,
        specs=(ad.ArraySpec(field.shape, field.dtype),),
    )
    result = ad.grad(staged_loss)(field)

    assert result.dtype == field.dtype
    assert type(result) is type(field)


def test_provider_evidence_stays_focused_on_qualified_results() -> None:
    result = qualify_array_providers._qualify_provider(
        qualify_array_providers._numpy_provider(),
        qualify_array_providers._build_programs(),
        array_api_version="2024.12",
    )

    assert result.report["arrays"]
    assert "timing_us" not in result.report
    assert "memory_before" not in result.report
    assert "memory_after" not in result.report
