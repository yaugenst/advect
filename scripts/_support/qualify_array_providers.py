"""Qualify one backend-neutral scientific program across array providers."""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

import array_api_strict as strict
import numpy as np
from numpy.testing import assert_allclose

import advect as ad
import advect.numpy  # Register the first-class NumPy frontend.
from advect.core._array_api.profiles import (
    LATEST_ARRAY_API_VERSION,
    SUPPORTED_ARRAY_API_VERSIONS,
)
from advect.core._native import native_build_info
from scripts._support.evidence import evidence_environment

if TYPE_CHECKING:
    from collections.abc import Callable


_PROVIDER_NAMES = ("numpy", "array-api-strict", "cupy")


@dataclass(frozen=True, slots=True)
class _Provider:
    name: str
    module: object
    version: str
    to_numpy: Callable[[object], np.ndarray]
    synchronize: Callable[[], None]


@dataclass(frozen=True, slots=True)
class _Programs:
    primal: ad.StagedProgram
    gradient: ad.StagedProgram
    restored_gradient: ad.StagedProgram
    captured: ad.StagedProgram
    captured_gradient: ad.StagedProgram
    restored_captured_gradient: ad.StagedProgram
    stencil_gradient: ad.StagedProgram
    weak_complex: ad.StagedProgram
    weak_gradient: ad.StagedProgram


@dataclass(frozen=True, slots=True)
class _ProviderResult:
    report: dict[str, object]
    reference_values: dict[str, np.ndarray]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--providers",
        default="numpy,array-api-strict",
        help=f"comma-separated subset of {','.join(_PROVIDER_NAMES)}",
    )
    parser.add_argument(
        "--array-api-version",
        choices=SUPPORTED_ARRAY_API_VERSIONS,
        default=LATEST_ARRAY_API_VERSION,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/qualification/array-provider-scientific.json"),
    )
    return parser.parse_args()


def _scientific_loss_in_namespace(
    namespace: object,
    signal: object,
    matrix: object,
    right: object,
    *,
    scale: float = 0.25,
) -> object:
    spectrum = namespace.fft.fft(signal)  # type: ignore[attr-defined]
    solution = namespace.linalg.solve(matrix, right)  # type: ignore[attr-defined]
    spectral_energy = namespace.sum(  # type: ignore[attr-defined]
        namespace.real(namespace.conj(spectrum) * spectrum)  # type: ignore[attr-defined]
    )
    solution_energy = namespace.sum(  # type: ignore[attr-defined]
        namespace.real(namespace.conj(solution) * solution)  # type: ignore[attr-defined]
    )
    return scale * spectral_energy + solution_energy


def _scientific_loss(
    signal: object,
    matrix: object,
    right: object,
    *,
    scale: float = 0.25,
) -> object:
    namespace = signal.__array_namespace__()  # type: ignore[attr-defined]
    return _scientific_loss_in_namespace(namespace, signal, matrix, right, scale=scale)


def _captured_loss(signal: object) -> object:
    window = np.asarray([1.0, 0.5, 0.25, 0.125], dtype=np.float32)
    namespace = signal.__array_namespace__()  # type: ignore[attr-defined]
    spectrum = namespace.fft.fft(signal * window)
    return namespace.sum(namespace.real(namespace.conj(spectrum) * spectrum))


def _stencil_loss(field: object) -> object:
    namespace = field.__array_namespace__()  # type: ignore[attr-defined]
    updated = field.copy()  # type: ignore[attr-defined]
    laplacian = field[2:] - 2 * field[1:-1] + field[:-2]  # type: ignore[index,operator]
    updated[1:-1] += 0.1 * laplacian  # type: ignore[index,operator]
    return namespace.sum(updated * updated)


def _weak_complex_loss(field: object) -> object:
    namespace = field.__array_namespace__()  # type: ignore[attr-defined]
    rotated = 1j * field  # type: ignore[operator]
    return namespace.sum(namespace.real(namespace.conj(rotated) * rotated))


def _build_programs(array_api_version: str = LATEST_ARRAY_API_VERSION) -> _Programs:
    science_specs = (
        ad.ArraySpec((4,), "complex64"),
        ad.ArraySpec((2, 2), "complex64"),
        ad.ArraySpec((2,), "complex64"),
    )
    primal = ad.stage(
        _scientific_loss,
        specs=science_specs,
        array_api_version=array_api_version,
    )
    gradient = ad.grad(primal, argnums=(0, 1, 2))
    captured = ad.stage(
        _captured_loss,
        specs=(ad.ArraySpec((4,), "complex64"),),
        array_api_version=array_api_version,
    )
    captured_gradient = ad.grad(captured)
    stencil = ad.stage(
        _stencil_loss,
        specs=(ad.ArraySpec((8,), "float32"),),
        array_api_version=array_api_version,
    )
    weak_loss = ad.stage(
        _weak_complex_loss,
        specs=(ad.ArraySpec((4,), "float32"),),
        array_api_version=array_api_version,
    )
    weak_complex = ad.stage(
        lambda field: 1j * field,
        specs=(ad.ArraySpec((4,), "float32"),),
        array_api_version=array_api_version,
    )
    return _Programs(
        primal=primal,
        gradient=gradient,
        restored_gradient=ad.StagedProgram.from_dict(gradient.to_dict()),
        captured=captured,
        captured_gradient=captured_gradient,
        restored_captured_gradient=ad.StagedProgram.from_dict(captured_gradient.to_dict()),
        stencil_gradient=ad.grad(stencil),
        weak_complex=weak_complex,
        weak_gradient=ad.grad(weak_loss),
    )


def _numpy_provider() -> _Provider:
    return _Provider(
        name="numpy",
        module=np,
        version=np.__version__,
        to_numpy=np.asarray,
        synchronize=lambda: None,
    )


def _strict_provider(array_api_version: str = LATEST_ARRAY_API_VERSION) -> _Provider:
    strict.set_array_api_strict_flags(api_version=array_api_version)
    return _Provider(
        name="array-api-strict",
        module=strict,
        version=strict.__version__,
        to_numpy=np.asarray,
        synchronize=lambda: None,
    )


def _cupy_provider() -> _Provider:
    cupy = importlib.import_module("cupy")

    def synchronize() -> None:
        cupy.cuda.get_current_stream().synchronize()

    return _Provider(
        name="cupy",
        module=cupy,
        version=str(cupy.__version__),
        to_numpy=cupy.asnumpy,
        synchronize=synchronize,
    )


def _providers(
    names: tuple[str, ...],
    array_api_version: str = LATEST_ARRAY_API_VERSION,
) -> tuple[_Provider, ...]:
    factories = {
        "array-api-strict": lambda: _strict_provider(array_api_version),
        "cupy": _cupy_provider,
        "numpy": _numpy_provider,
    }
    unknown = sorted(set(names).difference(factories))
    if unknown:
        msg = f"Unknown providers: {', '.join(unknown)}"
        raise ValueError(msg)
    return tuple(factories[name]() for name in names)


def _fixtures(provider: _Provider) -> tuple[object, object, object]:
    namespace = provider.module
    return (
        namespace.asarray(  # type: ignore[attr-defined]
            [1 + 2j, 2 - 1j, -1 + 0.5j, 0.25 - 0.75j],
            dtype=namespace.complex64,  # type: ignore[attr-defined]
        ),
        namespace.asarray(  # type: ignore[attr-defined]
            [[3 + 0.1j, 1 - 0.2j], [1 + 0.3j, 2 - 0.1j]],
            dtype=namespace.complex64,  # type: ignore[attr-defined]
        ),
        namespace.asarray(  # type: ignore[attr-defined]
            [1 + 0.5j, 2 - 0.25j],
            dtype=namespace.complex64,  # type: ignore[attr-defined]
        ),
    )


def _device(value: object) -> object:
    return getattr(value, "device", "cpu")


def _check_array_contract(value: object, source: object) -> None:
    if type(value) is not type(source):
        msg = f"Provider type changed from {type(source).__name__} to {type(value).__name__}"
        raise AssertionError(msg)
    if value.dtype != source.dtype:  # type: ignore[attr-defined]
        msg = f"Provider dtype changed from {source.dtype} to {value.dtype}"  # type: ignore[attr-defined]
        raise AssertionError(msg)
    if _device(value) != _device(source):
        msg = f"Provider device changed from {_device(source)} to {_device(value)}"
        raise AssertionError(msg)


def _numeric_summary(value: np.ndarray) -> dict[str, object]:
    return {
        "dtype": str(value.dtype),
        "max_abs": float(np.max(np.abs(value), initial=0.0)),
        "shape": list(value.shape),
    }


def _qualify_provider(
    provider: _Provider,
    programs: _Programs,
    *,
    array_api_version: str,
) -> _ProviderResult:
    namespace = provider.module
    inputs = _fixtures(provider)
    tangents = tuple(value * 0.05 for value in inputs)  # type: ignore[operator]

    dynamic_value, dynamic_gradients = ad.value_and_grad(
        _scientific_loss,
        argnums=(0, 1, 2),
    )(*inputs)
    jvp_value, directional = ad.jvp(
        _scientific_loss,
        argnums=(0, 1, 2),
    )(*inputs, tangents=tangents)
    vjp_value, pullback = ad.vjp(
        _scientific_loss,
        argnums=(0, 1, 2),
    )(*inputs)
    try:
        reverse_gradients = pullback(namespace.ones_like(vjp_value))  # type: ignore[attr-defined]
    finally:
        pullback.close()

    staged_value = programs.primal(*inputs)
    staged_gradients = programs.gradient(*inputs)
    restored_gradients = programs.restored_gradient(*inputs)
    provider.synchronize()
    assert_allclose(
        provider.to_numpy(jvp_value),
        provider.to_numpy(dynamic_value),
        rtol=1e-6,
        atol=1e-6,
    )
    assert_allclose(
        provider.to_numpy(vjp_value),
        provider.to_numpy(dynamic_value),
        rtol=1e-6,
        atol=1e-6,
    )
    assert_allclose(
        provider.to_numpy(staged_value),
        provider.to_numpy(dynamic_value),
        rtol=1e-6,
        atol=1e-6,
    )
    for source, dynamic, reverse, staged, restored in zip(
        inputs,
        dynamic_gradients,
        reverse_gradients,
        staged_gradients,
        restored_gradients,
        strict=True,
    ):
        for derivative in (dynamic, reverse, staged, restored):
            _check_array_contract(derivative, source)
            assert_allclose(
                provider.to_numpy(derivative),
                provider.to_numpy(dynamic),
                rtol=2e-5,
                atol=2e-5,
            )

    epsilon = 1e-3
    plus = tuple(
        primal + epsilon * tangent
        for primal, tangent in zip(inputs, tangents, strict=True)  # type: ignore[operator]
    )
    minus = tuple(
        primal - epsilon * tangent
        for primal, tangent in zip(inputs, tangents, strict=True)  # type: ignore[operator]
    )
    finite_difference = (
        provider.to_numpy(programs.primal(*plus)) - provider.to_numpy(programs.primal(*minus))
    ) / (2 * epsilon)
    assert_allclose(
        provider.to_numpy(directional),
        finite_difference,
        rtol=3e-3,
        atol=3e-3,
    )

    signal = inputs[0]
    captured_value = programs.captured(signal)
    captured_gradient = programs.captured_gradient(signal)
    restored_captured_gradient = programs.restored_captured_gradient(signal)
    _check_array_contract(captured_gradient, signal)
    _check_array_contract(restored_captured_gradient, signal)
    assert_allclose(
        provider.to_numpy(restored_captured_gradient),
        provider.to_numpy(captured_gradient),
        rtol=1e-6,
        atol=1e-6,
    )

    field = namespace.arange(  # type: ignore[attr-defined]
        8,
        dtype=namespace.float32,  # type: ignore[attr-defined]
    )
    dynamic_stencil = ad.grad(_stencil_loss)(field)
    staged_stencil = programs.stencil_gradient(field)
    _check_array_contract(dynamic_stencil, field)
    _check_array_contract(staged_stencil, field)
    assert_allclose(
        provider.to_numpy(staged_stencil),
        provider.to_numpy(dynamic_stencil),
        rtol=1e-6,
        atol=1e-6,
    )

    weak_input = namespace.arange(  # type: ignore[attr-defined]
        4,
        dtype=namespace.float32,  # type: ignore[attr-defined]
    )
    weak_complex = programs.weak_complex(weak_input)
    weak_gradient = programs.weak_gradient(weak_input)
    if weak_complex.dtype != namespace.complex64:  # type: ignore[attr-defined]
        msg = f"1j * float32 produced {weak_complex.dtype}, not complex64"  # type: ignore[attr-defined]
        raise AssertionError(msg)
    _check_array_contract(weak_gradient, weak_input)

    host_values = {
        "captured_gradient": provider.to_numpy(captured_gradient),
        "directional": provider.to_numpy(directional),
        "dynamic_value": provider.to_numpy(dynamic_value),
        "stencil_gradient": provider.to_numpy(dynamic_stencil),
        "weak_gradient": provider.to_numpy(weak_gradient),
        **{
            f"gradient_{index}": provider.to_numpy(gradient)
            for index, gradient in enumerate(dynamic_gradients)
        },
    }
    report = {
        "arrays": {name: _numeric_summary(value) for name, value in sorted(host_values.items())},
        "captured_constant": {
            "gradient_device": str(_device(captured_gradient)),
            "result_dtype": str(captured_value.dtype),  # type: ignore[attr-defined]
        },
        "input_device": str(_device(signal)),
        "input_dtype": str(signal.dtype),  # type: ignore[attr-defined]
        "name": provider.name,
        "reported_array_api_version": getattr(
            provider.module,
            "__array_api_version__",
            None,
        ),
        "selected_array_api_version": array_api_version,
        "version": provider.version,
        "weak_scalar_result_dtype": str(weak_complex.dtype),  # type: ignore[attr-defined]
    }
    return _ProviderResult(report=report, reference_values=host_values)


def _program_report(program: ad.StagedProgram) -> dict[str, object]:
    optimization = program.optimization
    return {
        "compile_seconds": program.compile_seconds,
        "constant_bytes": sum(record.bytes for record in program.constants),
        "constant_count": len(program.constants),
        "nodes_after": optimization.nodes_after,
        "nodes_before": optimization.nodes_before,
    }


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main() -> int:
    """Run the provider matrix, compare providers, and write the evidence."""
    arguments = _arguments()
    names = tuple(item.strip() for item in arguments.providers.split(",") if item.strip())
    if not names:
        msg = "Select at least one provider"
        raise ValueError(msg)
    if len(set(names)) != len(names):
        msg = f"Provider selection contains duplicates: {names!r}"
        raise ValueError(msg)
    if any(name not in _PROVIDER_NAMES for name in names):
        msg = f"Providers must be selected from {_PROVIDER_NAMES!r}"
        raise ValueError(msg)

    programs = _build_programs(arguments.array_api_version)
    results = [
        _qualify_provider(
            provider,
            programs,
            array_api_version=arguments.array_api_version,
        )
        for provider in _providers(names, arguments.array_api_version)
    ]
    reference = results[0]
    for result in results[1:]:
        for key, expected in reference.reference_values.items():
            assert_allclose(
                result.reference_values[key],
                expected,
                rtol=2e-5,
                atol=2e-5,
            )

    gradient_payload = programs.gradient.to_dict()
    report = {
        "schema_version": 1,
        "report_kind": "advect.array-provider-qualification",
        "array_api_version": arguments.array_api_version,
        "environment": {
            **evidence_environment(),
            "advect": _package_version("advect"),
            "native": native_build_info(),
        },
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "passed": True,
        "programs": {
            "captured": _program_report(programs.captured),
            "gradient": _program_report(programs.gradient),
            "gradient_artifact_json_bytes": len(
                json.dumps(gradient_payload, sort_keys=True).encode()
            ),
            "primal": _program_report(programs.primal),
            "stencil_gradient": _program_report(programs.stencil_gradient),
        },
        "providers": [result.report for result in results],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)
    return 0
