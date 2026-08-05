"""Optional single-device CuPy qualification for Advect's Array API path."""

from __future__ import annotations

import numpy as np
import pytest

import advect as ad
from advect.core._array_api_profiles import SUPPORTED_ARRAY_API_VERSIONS

cp = pytest.importorskip("cupy")


def _reduction_dtype(array_api_version: str) -> object:
    return cp.float64 if array_api_version == "2022.12" else cp.float32


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


def _arrays() -> tuple[object, object, object]:
    return (
        cp.asarray([1 + 2j, 2 - 1j, -1 + 0.5j, 0.25 - 0.75j], dtype=cp.complex64),
        cp.asarray(
            [[3 + 0.1j, 1 - 0.2j], [1 + 0.3j, 2 - 0.1j]],
            dtype=cp.complex64,
        ),
        cp.asarray([1 + 0.5j, 2 - 0.25j], dtype=cp.complex64),
    )


def _synchronize() -> None:
    cp.cuda.get_current_stream().synchronize()


@pytest.mark.parametrize("array_api_version", SUPPORTED_ARRAY_API_VERSIONS)
def test_cupy_scientific_workload_round_trips_every_derivative_lifetime(
    array_api_version: str,
) -> None:
    signal, matrix, right = _arrays()
    inputs = (signal, matrix, right)
    tangents = tuple(value * 0.05 for value in inputs)

    dynamic_value, dynamic_gradients = ad.value_and_grad(
        _scientific_loss,
        argnums=(0, 1, 2),
    )(*inputs)
    _jvp_value, directional = ad.jvp(
        _scientific_loss,
        argnums=(0, 1, 2),
    )(*inputs, tangents=tangents)
    _vjp_value, pullback = ad.vjp(
        _scientific_loss,
        argnums=(0, 1, 2),
    )(*inputs)
    try:
        reverse_gradients = pullback(cp.ones_like(dynamic_value))
    finally:
        pullback.close()

    primal = ad.stage(
        _scientific_loss,
        specs=tuple(ad.ArraySpec(value.shape, value.dtype) for value in inputs),
        array_api_version=array_api_version,
    )
    gradient = ad.grad(primal, argnums=(0, 1, 2))
    restored = ad.StagedProgram.from_dict(gradient.to_dict())
    staged_value = primal(*inputs)
    staged_gradients = gradient(*inputs)
    restored_gradients = restored(*inputs)
    _synchronize()

    assert primal.array_api_version == array_api_version
    assert cp.allclose(staged_value, dynamic_value, rtol=1e-6, atol=1e-6)
    assert directional.dtype == cp.float32
    assert directional.device == signal.device
    for source, dynamic, reverse, staged, round_tripped in zip(
        inputs,
        dynamic_gradients,
        reverse_gradients,
        staged_gradients,
        restored_gradients,
        strict=True,
    ):
        assert dynamic.dtype == source.dtype
        assert dynamic.device == source.device
        assert reverse.dtype == source.dtype
        assert reverse.device == source.device
        assert staged.dtype == source.dtype
        assert staged.device == source.device
        assert round_tripped.dtype == source.dtype
        assert round_tripped.device == source.device
        assert cp.allclose(dynamic, reverse, rtol=2e-5, atol=2e-5)
        assert cp.allclose(dynamic, staged, rtol=2e-5, atol=2e-5)
        assert cp.allclose(dynamic, round_tripped, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("array_api_version", SUPPORTED_ARRAY_API_VERSIONS)
def test_cupy_materializes_captured_constant_on_the_runtime_device(
    array_api_version: str,
) -> None:
    window = np.asarray([1.0, 0.5, 0.25, 0.125], dtype=np.float32)

    def captured_loss(signal: object) -> object:
        namespace = signal.__array_namespace__()
        spectrum = namespace.fft.fft(signal * window)
        return namespace.sum(namespace.real(namespace.conj(spectrum) * spectrum))

    signal = _arrays()[0]
    program = ad.stage(
        captured_loss,
        specs=(ad.ArraySpec(signal.shape, signal.dtype),),
        array_api_version=array_api_version,
    )
    gradient = ad.grad(program)
    result = program(signal)
    derivative = gradient(signal)
    _synchronize()

    assert len(program.constants) == 1
    assert result.dtype == _reduction_dtype(array_api_version)
    assert result.device == signal.device
    assert derivative.dtype == cp.complex64
    assert derivative.device == signal.device


@pytest.mark.parametrize("array_api_version", SUPPORTED_ARRAY_API_VERSIONS)
def test_cupy_functionalized_stencil_matches_dynamic_and_staged_gradients(
    array_api_version: str,
) -> None:
    def stencil_loss(field: object) -> object:
        namespace = field.__array_namespace__()
        updated = field.copy()
        laplacian = field[2:] - 2 * field[1:-1] + field[:-2]
        updated[1:-1] += 0.1 * laplacian
        return namespace.sum(updated * updated)

    field = cp.arange(8, dtype=cp.float32)
    dynamic = ad.grad(stencil_loss)(field)
    staged = ad.grad(
        ad.stage(
            stencil_loss,
            specs=(ad.ArraySpec(field.shape, field.dtype),),
            array_api_version=array_api_version,
        )
    )(field)
    _synchronize()

    assert dynamic.dtype == cp.float32
    assert dynamic.device == field.device
    assert staged.dtype == cp.float32
    assert staged.device == field.device
    assert cp.allclose(staged, dynamic, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("array_api_version", SUPPORTED_ARRAY_API_VERSIONS)
def test_cupy_staged_derivative_preserves_complex_weak_scalar_promotion(
    array_api_version: str,
) -> None:
    def loss(field: object) -> object:
        namespace = field.__array_namespace__()
        rotated = 1j * field
        return namespace.sum(namespace.real(namespace.conj(rotated) * rotated))

    field = cp.arange(4, dtype=cp.float32)
    program = ad.stage(
        loss,
        specs=(ad.ArraySpec(field.shape, field.dtype),),
        array_api_version=array_api_version,
    )
    gradient = ad.grad(program)(field)
    _synchronize()

    assert gradient.dtype == cp.float32
    assert gradient.device == field.device
