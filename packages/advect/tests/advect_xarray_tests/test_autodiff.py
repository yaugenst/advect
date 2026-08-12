"""Tests for dynamic differentiation across the xarray pytree boundary."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import advect as ad
import advect.xarray  # Importing the optional package registers its nodes.


def _field() -> xr.DataArray:
    return xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("y", "x"),
        coords={"y": [10, 20], "x": [1, 2, 3]},
        name="field",
        attrs={"units": "V"},
    )


def test_dataarray_grad_and_value_and_grad_preserve_labels() -> None:
    field = _field()

    def energy(value: xr.DataArray) -> xr.DataArray:
        interior = value.transpose("x", "y").isel(x=slice(1, None))
        return (interior * interior).sum(("x", "y"))

    value, gradient = ad.value_and_grad(energy)(field)
    expected_gradient = field.copy(data=np.array([[0.0, 2.0, 4.0], [0.0, 8.0, 10.0]]))

    assert isinstance(value, xr.DataArray)
    assert value.item() == pytest.approx(46.0)
    xr.testing.assert_identical(gradient, expected_gradient)
    xr.testing.assert_identical(ad.grad(energy)(field), expected_gradient)


def test_dataarray_jvp_and_vjp_preserve_output_and_input_coordinates() -> None:
    field = _field()
    tangent = xr.ones_like(field)

    def project(value: xr.DataArray) -> xr.DataArray:
        return 3.0 * value.transpose("x", "y").isel(x=slice(1, None))

    output, output_tangent = ad.jvp(project)(field, tangents=tangent)
    expected_output = project(field)
    expected_tangent = xr.ones_like(expected_output) * 3.0

    xr.testing.assert_identical(output, expected_output)
    xr.testing.assert_identical(output_tangent, expected_tangent)

    primal, pullback = ad.vjp(project)(field)
    cotangent = xr.ones_like(primal)
    try:
        input_cotangent = pullback(cotangent)
    finally:
        pullback.close()
    expected_input_cotangent = field.copy(data=np.array([[0.0, 3.0, 3.0], [0.0, 3.0, 3.0]]))

    xr.testing.assert_identical(primal, expected_output)
    xr.testing.assert_identical(input_cotangent, expected_input_cotangent)


def test_jvp_and_vjp_reject_mismatched_seed_coordinates() -> None:
    field = _field()
    mismatched_seed = xr.ones_like(field).assign_coords(x=[4, 5, 6])

    with pytest.raises(ValueError, match="JVP tangent pytree structure"):
        ad.jvp(lambda value: 3.0 * value)(field, tangents=mismatched_seed)

    output, pullback = ad.vjp(lambda value: 3.0 * value)(field)
    mismatched_cotangent = xr.ones_like(output).assign_coords(x=[4, 5, 6])
    with pytest.raises(ValueError, match="Cotangent pytree structure"):
        pullback(mismatched_cotangent)


def test_linearize_reuses_labeled_forward_and_reverse_map() -> None:
    field = _field()

    def project(value: xr.DataArray) -> xr.DataArray:
        return 3.0 * value.transpose("x", "y")

    output, linear = ad.linearize(project, field)
    try:
        first_tangent = linear(xr.ones_like(field))
        second_tangent = linear(xr.full_like(field, 2.0))
        input_cotangent = linear.pullback(xr.ones_like(output))
    finally:
        linear.close()

    xr.testing.assert_identical(output, project(field))
    xr.testing.assert_identical(first_tangent, 3.0 * xr.ones_like(output))
    xr.testing.assert_identical(second_tangent, 6.0 * xr.ones_like(output))
    xr.testing.assert_identical(input_cotangent, xr.full_like(field, 3.0))


def test_dataset_autodiff_preserves_each_data_variable() -> None:
    dataset = xr.Dataset(
        data_vars={
            "field": xr.DataArray(
                np.arange(6.0).reshape(2, 3),
                dims=("y", "x"),
                attrs={"units": "V"},
            ),
            "weight": xr.DataArray(
                np.array([2.0, 3.0]),
                dims="y",
                attrs={"role": "weight"},
            ),
        },
        coords={"y": [10, 20], "x": [1, 2, 3]},
        attrs={"source": "simulation"},
    )

    def loss(value: xr.Dataset) -> xr.DataArray:
        return (value["field"] * value["field"]).sum() + (value["weight"] * value["weight"]).sum()

    gradient = ad.grad(loss)(dataset)
    expected = dataset.copy(
        data={
            "field": 2.0 * dataset["field"].data,
            "weight": 2.0 * dataset["weight"].data,
        }
    )
    xr.testing.assert_identical(gradient, expected)

    tangent = xr.ones_like(dataset)
    output, output_tangent = ad.jvp(lambda value: 3.0 * value)(dataset, tangents=tangent)
    primal, pullback = ad.vjp(lambda value: 3.0 * value)(dataset)
    try:
        input_cotangent = pullback(xr.ones_like(primal))
    finally:
        pullback.close()

    xr.testing.assert_identical(output, 3.0 * dataset)
    xr.testing.assert_identical(output_tangent, 3.0 * tangent)
    xr.testing.assert_identical(primal, output)
    xr.testing.assert_identical(input_cotangent, 3.0 * tangent)


def test_xarray_alignment_runs_normally_inside_dynamic_trace() -> None:
    left = xr.DataArray(
        np.array([1.0, 2.0]),
        dims="x",
        coords={"x": [0, 1]},
        name="left",
    )
    right = xr.DataArray(
        np.array([10.0, 20.0]),
        dims="x",
        coords={"x": [1, 2]},
        name="right",
    )

    def loss(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
        return ((a + b).fillna(0.0) ** 2).sum()

    left_gradient, right_gradient = ad.grad(loss, argnums=(0, 1))(left, right)

    xr.testing.assert_identical(
        left_gradient,
        left.copy(data=np.array([0.0, 24.0])),
    )
    xr.testing.assert_identical(
        right_gradient,
        right.copy(data=np.array([24.0, 0.0])),
    )


def test_data_dependent_output_coordinates_are_rejected() -> None:
    field = _field().isel(y=0)

    def label_from_data(value: xr.DataArray) -> xr.DataArray:
        return value.assign_coords(label=("x", value.data))

    with pytest.raises(
        TypeError, match="coordinates, dimensions, names, and attributes are static"
    ):
        ad.vjp(label_from_data)(field)


def test_staging_rejects_xarray_until_custom_pytree_codecs_exist() -> None:
    with pytest.raises(TypeError, match="Stage the numerical leaves or raw-array kernel"):
        ad.stage(lambda value: value + 1, xr.DataArray(np.asarray(2.0)))


def test_stage_raw_data_and_reattach_labels_outside_program() -> None:
    field = _field()
    program = ad.stage(lambda data: 2.0 * data, field.data)

    transformed = field.copy(data=program(field.data))

    xr.testing.assert_identical(transformed, field.copy(data=2.0 * field.data))
