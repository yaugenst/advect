"""End-to-end compatibility path for every published NumPy minor."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad
from advect.autodiff._ephemeral import trace_call
from advect.numpy._array_function.registry import ARRAY_FUNCTION_RUNTIME
from advect.numpy._profiles import numpy_minor
from advect.numpy._support_contract import numpy_support_declarations

_UPSTREAM_REMOVED_IN_NUMPY_24 = ("in1d", "trapz")


def test_installed_numpy_minor_runs_dynamic_staged_and_serialized_derivatives() -> None:
    array_api_version = np.__array_api_version__
    value = np.asarray([0.2, -0.3, 0.5], dtype=np.float64)

    def loss(x: object) -> object:
        return np.sum(np.sin(x) * x + x * x)

    expected = np.sin(value) + value * np.cos(value) + 2 * value
    assert_allclose(ad.grad(loss)(value), expected)

    trace = trace_call(
        loss,
        args=(value,),
        kwargs={},
        argnums=(0,),
        argnames=None,
    )
    try:
        assert trace.array_api_version == array_api_version
    finally:
        trace.tape.release_payloads()

    primal = ad.stage(
        loss,
        specs=(ad.ArraySpec(value.shape, value.dtype),),
        array_api_version=array_api_version,
    )
    gradient = ad.grad(primal)
    pullback = ad.vjp_program(primal)
    restored_primal = ad.StagedProgram.from_dict(primal.to_dict())
    restored_gradient = ad.StagedProgram.from_dict(gradient.to_dict())

    assert primal.array_api_version == array_api_version
    assert gradient.array_api_version == array_api_version
    assert pullback.array_api_version == array_api_version
    assert_allclose(restored_primal(value), loss(value))
    assert_allclose(restored_gradient(value), expected)
    assert_allclose(pullback(value, cotangent=np.asarray(1.0)), expected)


@pytest.mark.parametrize("name", _UPSTREAM_REMOVED_IN_NUMPY_24)
def test_legacy_alias_registration_and_publication_follow_installed_numpy(name: str) -> None:
    function = np.__dict__.get(name)
    available = callable(function)

    assert (available and function in ARRAY_FUNCTION_RUNTIME.handlers) is available

    path = f"numpy.{name}"
    declared = {
        declaration.callable
        for declaration in numpy_support_declarations()
        if declaration.kind == "function"
    }
    published = {
        str(row["callable"])
        for row in ad.support_catalog()["extensions"]["numpy"]["functions"]
        if row["kind"] == "function"
    }
    assert (path in published) is (available and path in declared)


@pytest.mark.skipif(
    not all(callable(np.__dict__.get(name)) for name in _UPSTREAM_REMOVED_IN_NUMPY_24),
    reason="NumPy 2.4 removed in1d and trapz",
)
def test_numpy_20_to_23_legacy_aliases_execute_through_the_tracer() -> None:
    in1d = np.__dict__["in1d"]
    trapz = np.__dict__["trapz"]
    value = np.asarray([0.5, 2.0, 3.0])
    tangent = np.ones_like(value)

    in1d_primal, in1d_tangent = ad.jvp(
        lambda current: in1d(current, np.asarray([0.5, 3.0])),
    )(value, tangents=tangent)
    with pytest.warns(DeprecationWarning, match=r"`trapz` is deprecated"):
        trapz_primal, trapz_tangent = ad.jvp(trapz)(value, tangents=tangent)

    np.testing.assert_array_equal(in1d_primal, np.asarray([True, False, True]))
    assert_allclose(in1d_tangent, np.zeros_like(value))
    assert_allclose(trapz_primal, np.trapezoid(value))
    assert_allclose(trapz_tangent, 2.0)


@pytest.mark.parametrize("version", ["1.26.4", "2.5.0", "3.0.0"])
def test_numpy_minor_rejects_unsupported_versions(version: str) -> None:
    with pytest.raises(TypeError, match=r"supports NumPy >=2\.0,<2\.5"):
        numpy_minor(version)
