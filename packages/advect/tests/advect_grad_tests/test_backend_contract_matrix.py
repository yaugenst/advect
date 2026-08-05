"""Shared backend contract matrix checks for runtime/provider invariants."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import advect as ad
import advect.numpy as adnp
from advect.autodiff.api.common import (
    _allocate_hessian_blocks_flat,
    _prepare_higher_order_inputs,
    _require_array_namespace_for_higher_order,
)
from advect.autodiff.rules.array_family.providers import (
    register_array_family_backend_provider,
    resolve_array_family_backend_provider,
)
from advect.core._array_namespace import ResolvedArrayNamespace
from advect.core._errors import HigherOrderNotSupportedError
from advect.core._eval_dispatch import bind_node_evaluator, evaluate_node_value
from advect_grad_tests._backend_contract_helpers import (
    BackendProviderStub,
    StrictHigherOrderNamespace,
)

xp = pytest.importorskip("array_api_strict")


def _numpy_sin(value: object) -> object:
    return np.sin(value)


def _array_api_sin(value: object) -> object:
    namespace = value.__array_namespace__()
    return namespace.sin(value)


def _numpy_energy(value: object) -> object:
    return np.sum(np.sin(value) ** 2)


def _build_backend_case(backend: str) -> tuple[Any, Any]:
    if backend == "numpy":
        runtime = np.asarray([1.0, -2.0], dtype=np.float64)
        return np, runtime
    if backend == "array_api_strict":
        namespace = StrictHigherOrderNamespace(xp)
        runtime = xp.asarray([1.0, -2.0], dtype=xp.float64)
        return namespace, runtime
    msg = f"Unknown backend case: {backend}"
    raise ValueError(msg)


@pytest.mark.parametrize("backend", ["numpy", "array_api_strict"])
def test_provider_resolution_matrix(
    backend: str,
    isolated_provider_registry: None,
) -> None:
    namespace, runtime_value = _build_backend_case(backend)
    provider = BackendProviderStub(backend=backend, namespace=namespace)
    register_array_family_backend_provider(provider)

    resolved = resolve_array_family_backend_provider(runtime_value)
    assert resolved is provider


@pytest.mark.parametrize("backend", ["numpy", "array_api_strict"])
def test_higher_order_namespace_contract_matrix(
    backend: str,
    isolated_provider_registry: None,
) -> None:
    namespace, runtime_value = _build_backend_case(backend)
    register_array_family_backend_provider(
        BackendProviderStub(backend=backend, namespace=namespace)
    )

    array_ns = _require_array_namespace_for_higher_order(args=(runtime_value,), kwargs={})
    _shapes, flat_sizes, primal_dtypes = _prepare_higher_order_inputs(
        array_ns=array_ns,
        args=(runtime_value,),
        argnums=(0,),
    )
    blocks = _allocate_hessian_blocks_flat(
        array_ns=array_ns,
        primal_flat_sizes=flat_sizes,
        primal_dtypes=primal_dtypes,
    )

    expected_size = int(runtime_value.size)
    assert flat_sizes == [expected_size]
    assert blocks[0][0].shape == (expected_size, expected_size)


def test_unresolved_provider_failure_is_typed(
    isolated_provider_registry: None,
) -> None:
    register_array_family_backend_provider(BackendProviderStub(backend="a", namespace=np))
    register_array_family_backend_provider(BackendProviderStub(backend="b", namespace=np))

    with pytest.raises(HigherOrderNotSupportedError, match="runtime array namespace"):
        _ = _require_array_namespace_for_higher_order(args=(object(),), kwargs={})


@pytest.mark.parametrize("backend", ["numpy", "array_api_strict"])
def test_matrix_shape_dtype_property_contract(
    backend: str,
    isolated_provider_registry: None,
) -> None:
    arr = np.asarray([[1.0, -1.0], [0.5, 2.0]], dtype=np.float64)
    namespace, runtime_value = _build_backend_case(backend)
    runtime_value = xp.asarray(arr, dtype=xp.float64) if backend == "array_api_strict" else arr

    register_array_family_backend_provider(
        BackendProviderStub(backend=backend, namespace=namespace)
    )
    array_ns = _require_array_namespace_for_higher_order(args=(runtime_value,), kwargs={})
    _shapes, flat_sizes, primal_dtypes = _prepare_higher_order_inputs(
        array_ns=array_ns,
        args=(runtime_value,),
        argnums=(0,),
    )
    blocks = _allocate_hessian_blocks_flat(
        array_ns=array_ns,
        primal_flat_sizes=flat_sizes,
        primal_dtypes=primal_dtypes,
    )

    expected_dtype = array_ns.result_type(primal_dtypes[0], primal_dtypes[0])
    assert blocks[0][0].dtype == expected_dtype


def test_numpy_authored_transform_stages_and_restores_on_numpy() -> None:
    value = np.asarray([0.25, -0.5], dtype=np.float64)
    tangent = np.ones_like(value)
    primal, actual_tangent = ad.jvp(_numpy_sin)(value, tangents=tangent)

    assert isinstance(primal, np.ndarray)
    assert isinstance(actual_tangent, np.ndarray)
    np.testing.assert_allclose(primal, np.sin(value))
    np.testing.assert_allclose(actual_tangent, np.cos(value))

    program = ad.stage(_numpy_sin, value)
    restored = ad.StagedProgram.from_dict(program.to_dict())
    for candidate in (program, restored):
        actual = candidate(value)
        assert isinstance(actual, np.ndarray)
        np.testing.assert_allclose(actual, np.sin(value))


def test_numpy_authored_specs_program_accepts_ordinary_python_scalar() -> None:
    program = ad.stage(
        _numpy_sin,
        specs=(ad.ArraySpec((), "float64", weak=True),),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())

    for candidate in (program, restored):
        assert candidate(0.5) == pytest.approx(np.sin(0.5))


def test_array_api_authored_program_is_portable_across_cpu_providers() -> None:
    program = ad.stage(
        _array_api_sin,
        specs=(ad.ArraySpec((2,), "float64"),),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())
    values = (
        np.asarray([0.25, -0.5], dtype=np.float64),
        xp.asarray([0.25, -0.5], dtype=xp.float64),
    )

    for value in values:
        expected = value.__array_namespace__().sin(value)
        for candidate in (program, restored):
            actual = candidate(value)
            assert type(actual) is type(expected)
            np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))


def test_numpy_authored_transform_rejects_array_api_strict_dynamically_and_staged() -> None:
    value = xp.asarray([0.25, -0.5], dtype=xp.float64)
    tangent = xp.ones_like(value)

    with pytest.raises(TypeError):
        ad.jvp(_numpy_sin)(value, tangents=tangent)

    program = ad.stage(_numpy_sin, value)
    restored = ad.StagedProgram.from_dict(program.to_dict())
    for candidate in (program, restored):
        with pytest.raises(TypeError, match="NumPy-authored node"):
            candidate(value)
        with pytest.raises(TypeError, match="NumPy-authored node"):
            ad.jvp(lambda traced, selected=candidate: selected(traced))(
                value,
                tangents=tangent,
            )


def test_numpy_authored_specs_program_rejects_array_api_strict_at_replay() -> None:
    value = xp.asarray([0.25, -0.5], dtype=xp.float64)
    program = ad.stage(
        _numpy_sin,
        specs=(ad.ArraySpec(value.shape, value.dtype),),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())

    for candidate in (program, restored):
        with pytest.raises(
            TypeError,
            match=r"got 'array_api_strict'",
        ):
            candidate(value)


def test_unbound_numpy_evaluator_checks_provider_before_fallback_dispatch() -> None:
    value = xp.asarray([0.25, -0.5], dtype=xp.float64)
    context = ResolvedArrayNamespace(xp, "2024.12")
    attrs = {"_advect_backend": "numpy"}
    evaluator = bind_node_evaluator("unbound.operation", attrs)

    with pytest.raises(TypeError, match="NumPy-authored node"):
        evaluator((value,), context, None)
    with pytest.raises(TypeError, match="NumPy-authored node"):
        evaluate_node_value(
            "unbound.operation",
            (value,),
            attrs,
            namespace=context,
        )


def test_nested_staged_differentiation_retains_numpy_frontend() -> None:
    spec = ad.ArraySpec((4,), "float64")
    value = np.arange(4.0)
    expected = 2 * np.sin(value) * np.cos(value)
    programs = (
        ad.grad(ad.stage(_numpy_energy, specs=(spec,))),
        ad.stage(ad.grad(_numpy_energy), specs=(spec,)),
    )

    for program in programs:
        restored = ad.StagedProgram.from_dict(program.to_dict())
        for candidate in (program, restored):
            actual = candidate(value)
            assert isinstance(actual, np.ndarray)
            np.testing.assert_allclose(actual, expected)


def test_untraced_advect_numpy_delegates_numpy_coercion() -> None:
    values = [0.0, np.pi / 2]

    assert adnp.sin is np.sin
    actual = adnp.sin(values)

    assert isinstance(actual, np.ndarray)
    np.testing.assert_allclose(actual, np.asarray([0.0, 1.0]))
