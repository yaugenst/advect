"""End-to-end compatibility path for every published NumPy minor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad
from advect.autodiff._ephemeral import trace_call
from advect.core._array_api.profiles import LATEST_ARRAY_API_VERSION
from advect.numpy._array_function.registry import ARRAY_FUNCTION_HANDLERS
from advect.numpy._profiles import numpy_minor
from advect.numpy._support_contract import numpy_support_declarations

if TYPE_CHECKING:
    from collections.abc import Callable

_UPSTREAM_REMOVED_IN_NUMPY_24 = ("in1d", "trapz")


def test_installed_numpy_minor_runs_dynamic_staged_and_serialized_derivatives() -> None:
    array_api_version = min(np.__array_api_version__, LATEST_ARRAY_API_VERSION)
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

    primal = ad.stage(loss, value)
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

    assert (available and function in ARRAY_FUNCTION_HANDLERS) is available

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
    reason="NumPy 2.4 and newer removed in1d and trapz",
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


@pytest.mark.parametrize("rounding", [np.ceil, np.floor, np.trunc])
@pytest.mark.parametrize("dtype", [np.bool_, np.int8, np.uint16, np.int64])
def test_staged_rounding_keeps_the_installed_numpy_dtype(rounding: np.ufunc, dtype: type) -> None:
    """NumPy 2.0 rounds integers in a floating loop; NumPy 2.1 keeps their dtype."""
    value = np.array([1, 0, 1], dtype=dtype)
    expected = rounding(value)

    program = ad.stage(
        rounding,
        specs=(ad.ArraySpec(value.shape, value.dtype),),
        array_api_version=min(np.__array_api_version__, LATEST_ARRAY_API_VERSION),
    )
    staged = program(value)

    assert staged.dtype == expected.dtype
    assert_allclose(staged, expected)


@pytest.mark.parametrize("function", [np.var, np.std])
def test_statically_masked_full_variance_runs_in_every_lifetime(
    function: Callable[..., object],
) -> None:
    """A static mask counts to a NumPy scalar, which NumPy 2.0's np.astype rejects."""
    value = np.array([0.5, 1.5, 2.5, 4.0], dtype=np.float32)
    mask = np.array([True, False, True, True])

    def reduce(x: object) -> object:
        return function(x, where=mask)

    expected = reduce(value)
    primal, _tangent = ad.jvp(reduce)(value, tangents=np.ones_like(value))
    staged = ad.stage(reduce, value)(value)

    for actual in (primal, staged):
        assert np.asarray(actual).dtype == expected.dtype
        assert_allclose(actual, expected, rtol=1e-6)


@pytest.mark.parametrize("query", [np.float64(0.7), np.asarray(0.7), 0.7])
def test_interp_differentiates_at_a_scalar_query(query: object) -> None:
    """A 0-d query compares to a bool scalar, which NumPy 2.0's np.astype rejects."""

    def interpolate(x: object) -> object:
        return np.interp(x, np.array([0.0, 1.0, 2.0]), np.array([0.5, -1.0, 2.0]))

    assert_allclose(ad.grad(interpolate)(query), -1.5)
    assert_allclose(ad.hessian(interpolate)(query), 0.0)


@pytest.mark.parametrize("version", ["1.26.4", "2.6.0", "3.0.0"])
def test_numpy_minor_rejects_unsupported_versions(version: str) -> None:
    with pytest.raises(TypeError, match=r"supports NumPy >=2\.0,<2\.6"):
        numpy_minor(version)
