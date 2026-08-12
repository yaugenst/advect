"""Additional contracts for Advect's dynamic-only xarray integration."""

from __future__ import annotations

import builtins
import runpy
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

import advect as ad
import advect.xarray as advect_xarray


def test_complex_dataarray_jvp_preserves_rich_static_metadata() -> None:
    packed = np.array(("sample",), dtype=[("label", object)])[()]
    value = xr.DataArray(
        np.array([1.0 + 2.0j, 3.0 + 4.0j]),
        dims="x",
        coords={"x": [1, 2], "label": "sample"},
        name="field",
        attrs={
            "window": slice(1, None, 2),
            "date": date(2026, 1, 2),
            "datetime": datetime(2026, 1, 2, 3, 4, tzinfo=UTC),
            "duration": timedelta(seconds=3),
            "packed": packed,
            "nested": [{"value": np.int64(2)}],
            "object_array": np.array([{"kind": "sample"}], dtype=object),
        },
    )
    tangent = xr.ones_like(value) * (1.0 + 1.0j)
    scale = 2.0 + 1.0j

    output, output_tangent = ad.jvp(lambda field: scale * field)(
        value,
        tangents=tangent,
    )

    xr.testing.assert_identical(output, scale * value)
    xr.testing.assert_identical(output_tangent, scale * tangent)


def test_dataset_jvp_and_vjp_preserve_each_variable_metadata() -> None:
    value = xr.Dataset(
        {
            "field": xr.DataArray([1.0, 2.0], dims="x", attrs={"units": "V"}),
            "weight": xr.DataArray([3.0, 4.0], dims="x", attrs={"role": "weight"}),
        },
        coords={"x": [10, 20]},
        attrs={"source": "simulation"},
    )
    tangent = value.copy(data={"field": [0.1, 0.2], "weight": [0.3, 0.4]})

    output, output_tangent = ad.jvp(lambda dataset: 3.0 * dataset)(
        value,
        tangents=tangent,
    )
    primal, pullback = ad.vjp(lambda dataset: 3.0 * dataset)(value)
    try:
        input_cotangent = pullback(xr.ones_like(primal))
    finally:
        pullback.close()

    xr.testing.assert_identical(output, 3.0 * value)
    xr.testing.assert_identical(output_tangent, 3.0 * tangent)
    xr.testing.assert_identical(primal, output)
    xr.testing.assert_identical(input_cotangent, xr.ones_like(value) * 3.0)


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


def test_dataarray_rejects_a_tracer_nested_in_coordinate_metadata() -> None:
    class FakeTracer:
        def _advect_snapshot(self) -> tuple[int, object]:
            return 0, object()

    coordinate = np.empty(1, dtype=object)
    coordinate[0] = {"nested": FakeTracer()}
    value = xr.DataArray(
        np.ones(1),
        dims="x",
        coords={"label": ("x", coordinate)},
    )

    with pytest.raises(TypeError, match="found traced coordinate 'label'"):
        ad.pytree.tree_flatten(value)


def test_xarray_treedefs_validate_metadata_and_leaf_counts() -> None:
    dataarray = xr.DataArray(np.arange(2.0), dims="x")
    dataarray_leaves, dataarray_tree = ad.pytree.tree_flatten(dataarray)

    with pytest.raises(TypeError, match=r"Invalid xarray\.DataArray pytree metadata"):
        ad.pytree.tree_unflatten(replace(dataarray_tree, aux_data=None), dataarray_leaves)

    dataarray_extra_leaf = replace(
        dataarray_tree,
        children=dataarray_tree.children * 2,
        num_leaves=2,
    )
    with pytest.raises(ValueError, match="requires exactly one data leaf"):
        ad.pytree.tree_unflatten(
            dataarray_extra_leaf,
            [dataarray.data, dataarray.data],
        )

    dataset = xr.Dataset({"field": ("x", [1.0, 2.0]), "weight": ("x", [3.0, 4.0])})
    dataset_leaves, dataset_tree = ad.pytree.tree_flatten(dataset)
    _, matching_tree = ad.pytree.tree_flatten(dataset.copy(deep=True))
    assert dataset_tree == matching_tree

    with pytest.raises(TypeError, match=r"Invalid xarray\.Dataset pytree metadata"):
        ad.pytree.tree_unflatten(replace(dataset_tree, aux_data=None), dataset_leaves)

    dataset_extra_leaf = replace(
        dataset_tree,
        children=(*dataset_tree.children, dataset_tree.children[0]),
        num_leaves=3,
    )
    with pytest.raises(ValueError, match="data-variable count does not match"):
        ad.pytree.tree_unflatten(
            dataset_extra_leaf,
            [*dataset_leaves, dataset_leaves[0]],
        )


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
