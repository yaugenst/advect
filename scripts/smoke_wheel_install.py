"""Smoke-test one installed Advect wheel under one dependency profile."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

import numpy as np
from numpy.testing import assert_allclose
from numpy.typing import NDArray

import advect as ad
from advect import _native_core

if TYPE_CHECKING:
    from xarray import DataArray

type SmokeProfile = Literal["array-api", "base", "scientific"]
type FloatArray = NDArray[np.float64]
type ArraySolver = Callable[[Callable[[FloatArray], FloatArray], FloatArray], FloatArray]
type ScalarSolver = Callable[[Callable[[float], float], float], float]


class _ArrayApiNamespace(Protocol):
    def sum(self, value: object, /) -> object:
        """Sum an Array API value."""


class _ArrayApiValue(Protocol):
    @property
    def dtype(self) -> object:
        """Return the provider dtype."""

    def __array_namespace__(self) -> _ArrayApiNamespace:
        """Return the Array API namespace."""

    def __mul__(self, other: object, /) -> object:
        """Multiply two Array API values."""


def _loss(value: FloatArray) -> object:
    return np.sum(np.sin(value) ** 2)


def _array_api_loss(value: _ArrayApiValue) -> object:
    return value.__array_namespace__().sum(value * value)


def _labeled_loss(value: DataArray) -> object:
    return (value * value).sum()


def _require(*, condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _parse_args() -> SmokeProfile:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "profile",
        choices=("array-api", "base", "scientific"),
        help="Optional dependency profile installed beside the Advect wheel.",
    )
    return cast("SmokeProfile", parser.parse_args().profile)


def _require_module(name: str, *, present: bool) -> None:
    found = importlib.util.find_spec(name) is not None
    _require(
        condition=found is present,
        message=f"Expected module {name!r} to be {'installed' if present else 'absent'}",
    )


def _check_distribution() -> None:
    version = metadata.version("advect")
    module_path = Path(ad.__file__ or "").resolve()
    environment_path = Path(sys.prefix).resolve()
    _require(
        condition=module_path.is_relative_to(environment_path),
        message=f"Advect imported from outside the smoke environment: {module_path}",
    )
    _require(
        condition=_native_core.__build_profile__ == "release",
        message=f"Expected a release native extension, got {_native_core.__build_profile__!r}",
    )
    _require(
        condition=_native_core.__version__ == version,
        message="Native extension and Advect distribution versions differ",
    )


def _check_core_runtime() -> None:
    value = np.linspace(-0.5, 0.5, 8, dtype=np.float64)
    expected = 2 * np.sin(value) * np.cos(value)
    gradient = cast("FloatArray", ad.grad(_loss)(value))
    assert_allclose(gradient, expected)

    # Example-based staging negotiates the newest Array API revision served by
    # the installed NumPy minor. A spec-only stage intentionally defaults to
    # Advect's newest revision and would therefore not be a lower-bound smoke.
    primal = cast("ad.StagedProgram", ad.stage(_loss, value))
    gradient_program = ad.grad(primal)
    payload = json.loads(json.dumps(gradient_program.to_dict(), sort_keys=True))
    restored = ad.StagedProgram.from_dict(payload)
    assert_allclose(restored(value), expected)


def _check_base_profile() -> None:
    _require_module("array_api_compat", present=True)
    _require_module("scipy", present=False)
    _require_module("xarray", present=False)


def _check_scientific_profile() -> None:
    _require_module("array_api_compat", present=True)
    _require_module("scipy", present=True)
    _require_module("xarray", present=True)

    xr = importlib.import_module("xarray")
    scipy_special = importlib.import_module("scipy.special")
    importlib.import_module("advect.xarray")
    special = importlib.import_module("advect.scipy.special")
    optimize = importlib.import_module("advect.scipy.optimize")
    sparse_linalg = importlib.import_module("advect.scipy.sparse.linalg")

    erf = cast("Callable[[FloatArray], FloatArray]", special.erf)
    root_solver = cast("Callable[[], ArraySolver]", optimize.root_solver)()
    linear_solver = cast("Callable[..., ArraySolver]", sparse_linalg.gmres_solver)(
        rtol=1e-12,
        atol=1e-12,
    )

    def residual(solution: FloatArray, data: FloatArray) -> FloatArray:
        return erf(solution) - data

    special_value = np.array(0.25, dtype=np.float64)
    expected_special_gradient = 2 / np.sqrt(np.pi) * np.exp(-(special_value**2))
    special_gradient = cast("FloatArray", ad.grad(erf)(special_value))
    assert_allclose(
        special_gradient,
        expected_special_gradient,
    )

    solve_root = ad.implicit_root(
        residual,
        solve=root_solver,
        linear_solve=linear_solver,
    )
    parameters = np.array([0.1, -0.3])
    initial = np.array([0.1, -0.3])

    def implicit_loss(data: FloatArray) -> object:
        return np.sum(solve_root(data, initial=initial))

    implicit_gradient = cast("FloatArray", ad.grad(implicit_loss)(parameters))
    implicit_solution = scipy_special.erfinv(parameters)
    expected_implicit_gradient = np.sqrt(np.pi) / 2 * np.exp(implicit_solution**2)
    assert_allclose(implicit_gradient, expected_implicit_gradient, rtol=1e-9, atol=1e-9)

    scalar_root_solver = cast("Callable[[], ScalarSolver]", optimize.root_solver)()
    scalar_linear_solver = cast("Callable[..., ScalarSolver]", sparse_linalg.gmres_solver)(
        rtol=1e-12,
        atol=1e-12,
    )

    def scalar_residual(solution: float, parameter: float) -> float:
        return solution * solution - parameter

    scalar_root = ad.implicit_root(
        scalar_residual,
        solve=scalar_root_solver,
        linear_solve=scalar_linear_solver,
    )

    def scalar_loss(parameter: float) -> object:
        return scalar_root(parameter, initial=1.0)

    scalar_gradient = cast("float", ad.grad(scalar_loss)(4.0))
    _require(
        condition=type(scalar_gradient) is float and bool(np.isclose(scalar_gradient, 0.25)),
        message=f"Python-scalar implicit gradient changed category/value: {scalar_gradient!r}",
    )

    data_array = cast("Callable[..., DataArray]", xr.DataArray)
    field = data_array(
        np.arange(6.0).reshape(2, 3),
        dims=("y", "x"),
        coords={"y": [10, 20], "x": [1, 2, 3]},
        name="field",
        attrs={"units": "V"},
    )
    labeled_gradient = cast("DataArray", ad.grad(_labeled_loss)(field))
    assert_identical = cast("Callable[[DataArray, DataArray], None]", xr.testing.assert_identical)
    assert_identical(labeled_gradient, field.copy(data=2.0 * field.data))


def _check_array_api_profile() -> None:
    _require_module("array_api_compat", present=True)
    _require_module("scipy", present=False)
    _require_module("xarray", present=False)

    # No provider import: `import advect` already configured the built-in
    # fallback, which is exactly what this profile verifies.
    strict = importlib.import_module("array_api_strict")
    asarray = cast("Callable[..., _ArrayApiValue]", strict.asarray)
    value = asarray([1.0, -2.0, 3.0], dtype=strict.float32)
    gradient = cast("_ArrayApiValue", ad.grad(_array_api_loss)(value))

    _require(
        condition=type(gradient) is type(value),
        message="Array API gradient changed provider",
    )
    _require(
        condition=gradient.dtype == value.dtype,
        message="Array API gradient changed dtype",
    )
    assert_allclose(np.asarray(gradient), np.array([2.0, -4.0, 6.0]))


def main() -> int:
    """Exercise the installed wheel under one dependency profile."""
    profile = _parse_args()
    _check_distribution()
    _check_core_runtime()
    if profile == "base":
        _check_base_profile()
    elif profile == "scientific":
        _check_scientific_profile()
    else:
        _check_array_api_profile()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
