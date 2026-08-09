"""Tests for lossless xarray pytree flattening and reconstruction."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import advect as ad
import advect.xarray  # Importing the optional package registers its nodes.


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


def test_traced_coordinate_metadata_is_rejected() -> None:
    class FakeTracer:
        def _advect_snapshot(self) -> tuple[int, object]:
            return 0, object()

    coordinate = np.empty(2, dtype=object)
    coordinate[:] = [FakeTracer(), FakeTracer()]
    value = xr.DataArray(
        np.ones(2),
        dims="x",
        coords={"label": ("x", coordinate)},
    )

    with pytest.raises(
        TypeError, match="coordinates, dimensions, names, and attributes are static"
    ):
        ad.pytree.tree_flatten(value)


def test_multiindex_coordinate_is_an_explicit_boundary() -> None:
    value = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
    ).stack(sample=("x", "y"))

    with pytest.raises(TypeError, match="MultiIndex coordinate"):
        ad.pytree.tree_flatten(value)
