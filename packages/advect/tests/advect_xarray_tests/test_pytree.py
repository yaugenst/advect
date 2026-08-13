"""Tests for lossless xarray pytree flattening and reconstruction."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import advect as ad
import advect.xarray  # Importing the optional package registers its nodes.


class _DataArraySubclass(xr.DataArray):
    __slots__ = ()


def test_dataarray_subclass_pytree_round_trip_preserves_concrete_type() -> None:
    value = _DataArraySubclass([1.0, 2.0], dims="x", coords={"x": [10, 20]})
    leaves, tree = ad.pytree.tree_flatten(value)
    rebuilt = ad.pytree.tree_unflatten(tree, leaves)

    assert type(rebuilt) is type(value)
    xr.testing.assert_identical(rebuilt, value)


def test_dataarray_pytree_round_trip_preserves_labels_and_metadata() -> None:
    value = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("y", "x"),
        coords={
            "y": xr.DataArray([10, 20], dims="y", attrs={"axis": "vertical"}),
            "x": [1, 2, 3],
            "material": (("y", "x"), [["a", "b", "c"], ["d", "e", "f"]]),
        },
        name="field",
        attrs={"units": "V", "scale": np.float32(2.0)},
    )

    leaves, tree = ad.pytree.tree_flatten(value)
    rebuilt = ad.pytree.tree_unflatten(tree, leaves)

    assert len(leaves) == 1
    xr.testing.assert_identical(rebuilt, value)


def test_dataset_pytree_round_trip_preserves_variable_metadata() -> None:
    value = xr.Dataset(
        data_vars={
            "field": xr.DataArray(
                np.arange(6.0).reshape(2, 3),
                dims=("y", "x"),
                attrs={"units": "V"},
            ),
            "weight": xr.DataArray([2.0, 3.0], dims="y", attrs={"role": "weight"}),
        },
        coords={"y": [10, 20], "x": [1, 2, 3]},
        attrs={"source": "simulation"},
    )

    leaves, tree = ad.pytree.tree_flatten(value)
    rebuilt = ad.pytree.tree_unflatten(tree, leaves)

    assert len(leaves) == 2
    xr.testing.assert_identical(rebuilt, value)


def test_dataset_pytree_identity_preserves_mapping_order() -> None:
    value = xr.Dataset(
        data_vars={"field": ("x", [1.0, 2.0]), "weight": ("x", [3.0, 4.0])},
        coords={"x": [10, 20], "label": "sample"},
    )
    variables_reordered = value[["weight", "field"]]
    coordinates_reordered = value.drop_vars(["x", "label"]).assign_coords(
        label=value.coords["label"], x=value.coords["x"]
    )

    _leaves, tree = ad.pytree.tree_flatten(value)
    _variable_leaves, variables_tree = ad.pytree.tree_flatten(variables_reordered)
    _coordinate_leaves, coordinates_tree = ad.pytree.tree_flatten(coordinates_reordered)

    assert value.identical(variables_reordered)
    assert value.identical(coordinates_reordered)
    assert tree != variables_tree
    assert tree != coordinates_tree


def test_dataarray_rejects_nondifferentiable_data_dtype() -> None:
    value = xr.DataArray(
        np.array([1, 2, 3]),
        dims="x",
        coords={"x": [10, 20, 30]},
    )

    with pytest.raises(
        TypeError,
        match="differentiable data must have a floating or complex dtype",
    ):
        ad.pytree.tree_flatten(value)


def test_dataset_identifies_nondifferentiable_data_variable() -> None:
    value = xr.Dataset(
        data_vars={
            "field": ("x", np.array([1.0, 2.0])),
            "material": ("x", np.array(["air", "oxide"])),
        },
        coords={"x": [0, 1]},
    )

    with pytest.raises(
        TypeError,
        match="differentiable data must have a floating or complex dtype",
    ) as error:
        ad.pytree.tree_flatten(value)
    assert "Dataset data variable 'material'" in str(error.value)


def test_pytree_metadata_is_snapshotted_and_has_stable_equality() -> None:
    original = xr.DataArray(
        np.arange(3.0),
        dims="x",
        coords={"x": xr.DataArray([1, 2, 3], dims="x", attrs={"kind": "index"})},
        attrs={"config": {"window": [1, 2]}},
    )
    expected = original.copy(deep=True)
    _leaves, tree = ad.pytree.tree_flatten(original)
    _other_leaves, other_tree = ad.pytree.tree_flatten(expected)

    original.attrs["config"]["window"][0] = 99
    original.coords["x"] = [42, 2, 3]
    rebuilt = ad.pytree.tree_unflatten(tree, [expected.data])

    assert tree == other_tree
    xr.testing.assert_identical(rebuilt, expected)


def test_nested_traced_coordinate_metadata_is_rejected() -> None:
    value = xr.DataArray(np.ones(1), dims="x")

    def add_traced_coordinate(field: xr.DataArray) -> xr.DataArray:
        coordinate = np.empty(1, dtype=object)
        coordinate[0] = {"nested": [field.data[0]]}
        return field.assign_coords(label=("x", coordinate))

    with pytest.raises(TypeError, match="found traced coordinate 'label'"):
        ad.jvp(add_traced_coordinate)(value, tangents=xr.ones_like(value))


def test_multiindex_coordinate_is_an_explicit_boundary() -> None:
    value = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
    ).stack(sample=("x", "y"))

    with pytest.raises(TypeError, match="MultiIndex coordinate"):
        ad.pytree.tree_flatten(value)
