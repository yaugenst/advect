"""Additional contracts for Advect's dynamic-only xarray integration."""

from __future__ import annotations

import builtins
import runpy
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

import advect as ad
import advect.xarray as advect_xarray


def test_complex_dataarray_jvp_preserves_supported_static_metadata() -> None:
    packed = np.array(("sample",), dtype=[("label", object)])[()]
    value = xr.DataArray(
        np.array([1.0 + 2.0j, 3.0 + 4.0j]),
        attrs={
            "window": slice(1, None, 2),
            "date": date(2026, 1, 2),
            "packed": packed,
            "nested": [{"value": np.int64(2)}],
            "object_array": np.array([{"kind": "sample"}], dtype=object),
        },
    )
    tangent = xr.ones_like(value) * (1.0 + 1.0j)
    scale = 2.0 + 1.0j

    output, output_tangent = ad.jvp(lambda field: scale * field)(value, tangents=tangent)

    xr.testing.assert_identical(output, scale * value)
    xr.testing.assert_identical(output_tangent, scale * tangent)


@pytest.mark.parametrize(
    ("attrs", "message"),
    [
        ({1: "value"}, "attribute keys must be strings"),
        ({"value": object()}, "unsupported type object"),
        (
            {"value": np.array([(1, 2)], dtype=[("left", "i4"), ("right", "i4")])},
            "structured metadata arrays are not supported",
        ),
    ],
)
def test_dataarray_rejects_unsupported_static_attributes(
    attrs: dict[Any, Any],
    message: str,
) -> None:
    value = xr.DataArray(np.ones(1), attrs=attrs)

    with pytest.raises(TypeError, match=message):
        ad.pytree.tree_flatten(value)


def test_dataarray_rejects_a_traced_attribute() -> None:
    value = xr.DataArray(np.arange(2.0), dims="x", coords={"x": [0, 1]})

    with pytest.raises(TypeError, match="found a traced value at attrs\\['scale'\\]"):
        ad.jvp(lambda field: field.assign_attrs(scale=field.data[0]))(
            value,
            tangents=xr.ones_like(value),
        )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (xr.DataArray(np.arange(2.0)), r"Invalid xarray\.DataArray pytree metadata"),
        (xr.Dataset({"field": ("x", [1.0, 2.0])}), r"Invalid xarray\.Dataset pytree metadata"),
    ],
)
def test_xarray_treedefs_validate_metadata(value: object, message: str) -> None:
    leaves, treedef = ad.pytree.tree_flatten(value)

    with pytest.raises(TypeError, match=message):
        ad.pytree.tree_unflatten(replace(treedef, aux_data=None), leaves)


@pytest.mark.parametrize("missing_name", ["xarray", "transitive_dependency"])
def test_xarray_import_reports_only_the_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
    missing_name: str,
) -> None:
    original_import = builtins.__import__
    missing = ModuleNotFoundError(f"No module named {missing_name!r}", name=missing_name)

    def import_with_missing_dependency(
        name: str,
        globals_: dict[str, Any] | None = None,
        locals_: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "advect.xarray._pytree":
            raise missing
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_with_missing_dependency)
    module_path = Path(advect_xarray.__file__)

    if missing_name == "xarray":
        with pytest.raises(ModuleNotFoundError, match=r"pip install 'advect\[xarray\]'"):
            runpy.run_path(str(module_path))
    else:
        with pytest.raises(ModuleNotFoundError) as error:
            runpy.run_path(str(module_path))
        assert error.value is missing
