#!/usr/bin/env python3
"""Smoke-test one installed Advect wheel under one dependency profile."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from importlib import metadata
from pathlib import Path
from typing import Literal, cast

import numpy as np
from numpy.testing import assert_allclose

import advect as ad
from advect import _native_core

type SmokeProfile = Literal["array-api", "base", "scientific"]


def _loss(value: object) -> object:
    return np.sum(np.sin(value) ** 2)


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
    assert_allclose(ad.grad(_loss)(value), expected)

    # Example-based staging negotiates the newest Array API revision served by
    # the installed NumPy minor. A spec-only stage intentionally defaults to
    # Advect's newest revision and would therefore not be a lower-bound smoke.
    primal = ad.stage(_loss, value)
    gradient = ad.grad(primal)
    payload = json.loads(json.dumps(gradient.to_dict(), sort_keys=True))
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

    special_value = np.array(0.25, dtype=np.float64)
    expected_special_gradient = 2 / np.sqrt(np.pi) * np.exp(-(special_value**2))
    assert_allclose(
        ad.grad(lambda value: special.erf(value))(special_value),
        expected_special_gradient,
    )

    solve_root = ad.implicit_root(
        lambda solution, parameters: special.erf(solution) - parameters,
        solve=optimize.root_solver(),
        linear_solve=sparse_linalg.gmres_solver(rtol=1e-12, atol=1e-12),
    )
    parameters = np.array([0.1, -0.3])
    initial = np.array([0.1, -0.3])
    implicit_gradient = ad.grad(lambda data: np.sum(solve_root(data, initial=initial)))(parameters)
    implicit_solution = scipy_special.erfinv(parameters)
    expected_implicit_gradient = np.sqrt(np.pi) / 2 * np.exp(implicit_solution**2)
    assert_allclose(implicit_gradient, expected_implicit_gradient, rtol=1e-9, atol=1e-9)

    scalar_root = ad.implicit_root(
        lambda solution, parameter: solution * solution - parameter,
        solve=optimize.root_solver(),
        linear_solve=sparse_linalg.gmres_solver(rtol=1e-12, atol=1e-12),
    )
    scalar_gradient = ad.grad(lambda parameter: scalar_root(parameter, initial=1.0))(4.0)
    _require(
        condition=type(scalar_gradient) is float and np.isclose(scalar_gradient, 0.25),
        message=f"Python-scalar implicit gradient changed category/value: {scalar_gradient!r}",
    )

    field = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("y", "x"),
        coords={"y": [10, 20], "x": [1, 2, 3]},
        name="field",
        attrs={"units": "V"},
    )
    labeled_gradient = ad.grad(lambda data: (data * data).sum())(field)
    xr.testing.assert_identical(labeled_gradient, field.copy(data=2.0 * field.data))


def _check_array_api_profile() -> None:
    _require_module("array_api_compat", present=True)
    _require_module("scipy", present=False)
    _require_module("xarray", present=False)

    # No provider import: `import advect` already configured the built-in
    # fallback, which is exactly what this profile verifies.
    strict = importlib.import_module("array_api_strict")
    value = strict.asarray([1.0, -2.0, 3.0], dtype=strict.float32)
    gradient = ad.grad(
        lambda data: data.__array_namespace__().sum(data * data),  # type: ignore[attr-defined]
    )(value)

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
