"""Tests for the optional xarray pytree registration, round trip, and metadata boundary."""

from __future__ import annotations

import builtins
import copy
import re
import runpy
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hypothesis.extra.numpy as npst
import numpy as np
import pytest
import xarray as xr
import xarray.testing.strategies as xrst
from hypothesis import example, given, settings, strategies as st

import advect as ad
import advect.xarray  # Importing the optional package registers its nodes.

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterator, Mapping


class _DataArraySubclass(xr.DataArray):
    __slots__ = ()


_DATA_DTYPES = st.sampled_from(
    tuple(np.dtype(name) for name in ("float32", "float64", "complex64", "complex128"))
)
_COORDINATE_DTYPES = xrst.pandas_index_dtypes() | xrst.supported_dtypes()


@st.composite
def _coordinates(draw: st.DrawFn, sizes: Mapping[Hashable, int]) -> dict[Hashable, xr.Variable]:
    """Draw index coordinates for some dimensions and at most one other coordinate."""
    coordinates: dict[Hashable, xr.Variable] = {}
    if not sizes:
        return coordinates
    for dim in draw(xrst.unique_subset_of(list(sizes))):
        values = draw(npst.arrays(xrst.pandas_index_dtypes(), sizes[dim], unique=True))
        coordinates[dim] = xr.Variable(dim, values, attrs=draw(xrst.attrs()))
    if draw(st.booleans()):
        name = draw(xrst.names().filter(lambda name: name not in sizes))
        dims = draw(xrst.unique_subset_of(sizes, min_size=1))
        coordinates[name] = draw(xrst.variables(dims=st.just(dims), dtype=_COORDINATE_DTYPES))
    return coordinates


@st.composite
def _labeled_containers(draw: st.DrawFn) -> xr.DataArray | xr.Dataset:
    sizes = draw(xrst.dimension_sizes(max_dims=2, max_side=3))
    coordinates = draw(_coordinates(sizes))
    if draw(st.booleans()):
        container_type = draw(st.sampled_from((xr.DataArray, _DataArraySubclass)))
        data = draw(xrst.variables(dims=st.just(sizes), dtype=_DATA_DTYPES))
        return container_type(data, coords=coordinates, name=draw(st.none() | xrst.names()))
    unused = xrst.names().filter(lambda name: name not in sizes and name not in coordinates)
    names = draw(st.lists(unused, min_size=1, max_size=3, unique=True))
    variable_dims = xrst.unique_subset_of(sizes) if sizes else st.just(sizes)
    return xr.Dataset(
        {name: draw(xrst.variables(dims=variable_dims, dtype=_DATA_DTYPES)) for name in names},
        coords=coordinates,
        attrs=draw(xrst.attrs()),
    )


def _arrays_eq_cannot_match(value: object, path: str) -> Iterator[str]:
    """Yield the paths of arrays in ``value`` whose ``==`` against a copy is not true."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _arrays_eq_cannot_match(item, f"{path}[{key!r}]")
    elif isinstance(value, np.ndarray):
        try:
            matches = bool(value == value.copy())
        except ValueError:  # The truth value of several elements is ambiguous.
            matches = False
        if not matches:
            yield path


def _incomparable_attribute_arrays(container: xr.DataArray | xr.Dataset) -> list[str]:
    """Return the paths of arrays nested in dict attributes that ``==`` cannot match."""
    variables = {"": container}
    variables |= {f"coords[{name!r}].": coord for name, coord in container.coords.items()}
    if isinstance(container, xr.Dataset):
        variables |= {f"data_vars[{name!r}].": array for name, array in container.data_vars.items()}
    return [
        path
        for prefix, variable in variables.items()
        for key, value in variable.attrs.items()
        if isinstance(value, dict)
        for path in _arrays_eq_cannot_match(value, f"{prefix}attrs[{key!r}]")
    ]


def _real_total(container: xr.DataArray | xr.Dataset) -> xr.DataArray:
    arrays = container.data_vars.values() if isinstance(container, xr.Dataset) else (container,)
    return sum(array.real.sum(skipna=False) for array in arrays)  # type: ignore[return-value]


def _ones_like(container: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
    if isinstance(container, xr.Dataset):
        return container.copy(
            data={name: np.ones_like(array.data) for name, array in container.data_vars.items()}
        )
    return container.copy(data=np.ones_like(container.data))


@given(value=_labeled_containers())
@example(  # Only the multi-element array is incomparable, not the size-1 one before it.
    value=xr.DataArray(
        [1.0, 2.0], dims="x", attrs={"config": {"step": np.array([2]), "window": np.array([1, 2])}}
    )
)
@example(
    value=_DataArraySubclass(
        np.arange(6.0).reshape(2, 3),
        dims=("y", "x"),
        coords={
            "y": xr.DataArray([10, 20], dims="y", attrs={"axis": "vertical"}),
            "x": [1, 2, 3],
            "material": (("y", "x"), [["a", "b", "c"], ["d", "e", "f"]]),
        },
        name="field",
        attrs={"missing": np.array(["NaT"], dtype="datetime64[Y]"), "scale": np.float32(2.0)},
    )
)
@example(  # pandas.Index warns for float16, so non-index coordinates must not build one.
    value=xr.DataArray(
        np.zeros(1, dtype=np.float32),
        dims="d",
        coords={"d": np.array([0], dtype=np.int32), "c": ("d", np.array([0.0], dtype=np.float16))},
    )
)
@settings(deadline=None)
def test_labeled_pytrees_round_trip_compare_reflexively_and_relabel_gradients(
    value: xr.DataArray | xr.Dataset,
) -> None:
    """Labeled metadata round-trips, equals itself, and labels exact gradients.

    xarray compares arrays nested in dict attributes with ``==``, so such
    metadata may fail to equal its copy; that comparison must then raise the
    TypeError naming every array ``==`` cannot match to its copy, and only
    those. The gradient of the linear real total is exactly one everywhere, so
    no tolerance is involved; ``skipna=False`` keeps NaN entries active.
    """
    leaves, treedef = ad.pytree.tree_flatten(value)
    assert len(leaves) == (len(value.data_vars) if isinstance(value, xr.Dataset) else 1)
    rebuilt = ad.pytree.tree_unflatten(treedef, leaves)
    assert type(rebuilt) is type(value)
    try:
        equal: bool | str = treedef == ad.pytree.tree_flatten(copy.deepcopy(value))[1]
    except TypeError as error:
        equal = str(error)
    if isinstance(equal, str):
        # Generated keys and names are letters and digits, so ", " only
        # separates the named paths.
        listed = re.search(r" at (.*): it compares", equal)
        assert listed is not None, equal
        assert sorted(listed[1].split(", ")) == sorted(_incomparable_attribute_arrays(value))
        return
    assert equal
    xr.testing.assert_identical(rebuilt, value)

    with np.errstate(all="ignore"):
        gradient = ad.grad(_real_total)(value)
    expected = _ones_like(value)
    assert type(gradient) is type(value)
    xr.testing.assert_identical(gradient, expected)
    gradient_leaves, _treedef = ad.pytree.tree_flatten(gradient)
    expected_leaves, _treedef = ad.pytree.tree_flatten(expected)
    assert [leaf.dtype for leaf in gradient_leaves] == [leaf.dtype for leaf in expected_leaves]


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
            "missing": np.array(["NaT"], dtype="datetime64[Y]"),
            "arrays": [np.array([1, 2]), [np.array([np.nan])]],
        },
    )
    tangent = value.copy(data=np.ones_like(value.data) * (1.0 + 1.0j))
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


def _field(**attrs: Any) -> xr.DataArray:
    return xr.DataArray(np.array([1.0, 2.0]), dims="x", attrs=attrs)


_WINDOW = {"window": np.array([1, 2])}


@pytest.mark.parametrize(
    ("value", "path"),
    [
        (_field(config=_WINDOW), r"attrs\['config'\]\['window'\]"),
        (
            _field(config={"missing": np.array(["NaT"], dtype="datetime64[Y]")}),
            r"attrs\['config'\]\['missing'\]",
        ),
        (_field(pair=(np.array([1, 2]),)), r"attrs\['pair'\]\[0\]"),
        (_field(items=[_WINDOW]), r"attrs\['items'\]\[0\]\['window'\]"),
        (
            _field(table=np.array([_WINDOW], dtype=object)),
            r"attrs\['table'\]\.flat\[0\]\['window'\]",
        ),
        (
            _field().assign_coords(lat=("x", [5.0, 6.0], {"config": _WINDOW})),
            r"coords\['lat'\]\.attrs\['config'\]\['window'\]",
        ),
        (
            xr.Dataset({"field": ("x", [1.0, 2.0], {"config": _WINDOW})}),
            r"data_vars\['field'\]\.attrs\['config'\]\['window'\]",
        ),
        (
            _field(scale={"factor": np.array(2), "offset": np.array([5])}, config=_WINDOW),
            r"attrs\['config'\]\['window'\]",
        ),
        (
            _field(config={"labels": np.array(["p", "q"], dtype=object)}),
            r"attrs\['config'\]\['labels'\]",
        ),
        (
            _field(config={"table": np.array([_WINDOW], dtype=object)}),
            r"attrs\['config'\]\['table'\]\.flat\[0\]\['window'\]",
        ),
    ],
    ids=[
        "dict",
        "nat",
        "tuple",
        "list-of-dict",
        "object-array",
        "coordinate",
        "dataset-variable",
        "after-comparable-arrays",
        "object-array-in-dict",
        "object-array-element-in-dict",
    ],
)
def test_transforms_name_metadata_arrays_xarray_cannot_compare(
    value: xr.DataArray | xr.Dataset,
    path: str,
) -> None:
    """Arrays xarray compares with ``==`` fail comparisons by name, not by accident.

    Without the named error these raised NumPy's ambiguous-truth ValueError, or
    a structure mismatch for NaT metadata that never equals itself. The
    innermost incomparable array is named first, and comparable ones never.
    """
    message = f"cannot compare the metadata arrays? at {path}[,:]"
    with pytest.raises(TypeError, match=message):
        ad.jvp(lambda field: 2.0 * field)(value, tangents=value)

    _output, pullback = ad.vjp(lambda field: 2.0 * field)(value)
    try:
        with pytest.raises(TypeError, match=message):
            pullback(value)
    finally:
        pullback.close()


def test_metadata_arrays_xarray_compares_with_eq_work_until_compared() -> None:
    field = _field(config=_WINDOW)
    leaves, treedef = ad.pytree.tree_flatten(field)
    rebuilt = ad.pytree.tree_unflatten(treedef, leaves)
    np.testing.assert_array_equal(rebuilt.attrs["config"]["window"], [1, 2])

    # A plain-array output never compares the input metadata.
    gradient = ad.grad(lambda value: (value.data * value.data).sum())(field)
    np.testing.assert_array_equal(gradient.data, [2.0, 4.0])

    # xarray shares index-coordinate attributes between copies, and a 0-d
    # array compares unambiguously, so these comparisons succeed.
    # Older xarray versions drop attrs during arithmetic unless requested.
    with xr.set_options(keep_attrs=True):
        for value in (
            _field().assign_coords(x=("x", [1, 2], {"config": _WINDOW})),
            _field(config={"window": np.array(2)}),
        ):
            _output, tangent = ad.jvp(lambda field: 2.0 * field)(value, tangents=value)
            xr.testing.assert_identical(tangent, 2.0 * value)
            _output, pullback = ad.vjp(lambda field: 2.0 * field)(value)
            try:
                xr.testing.assert_identical(pullback(value), 2.0 * value)
            finally:
                pullback.close()


def test_every_metadata_array_eq_cannot_compare_is_named() -> None:
    """Name every such array, and no comparable one.

    Which array a comparison reaches depends on attribute order and on xarray
    sharing index-coordinate attributes between copies.
    """
    value = xr.Dataset(
        {"field": ("x", [1.0, 2.0], {"scale": {"factor": np.array(2), "offset": np.array([5])}})},
        coords={"x": ("x", [1, 2], {"config": _WINDOW})},
        attrs={"config": {"missing": np.array(["NaT"], dtype="datetime64[Y]")}},
    )
    message = (
        "xarray cannot compare the metadata arrays at "
        "coords['x'].attrs['config']['window'], attrs['config']['missing']: "
    )

    with pytest.raises(TypeError, match=re.escape(message)):
        ad.jvp(lambda field: 2.0 * field)(value, tangents=value)


def test_comparable_metadata_arrays_are_not_blamed_for_other_mismatches() -> None:
    """A NaN name never equals itself; the 0-d array beside it compares fine."""
    value = xr.DataArray(
        np.array([1.0, 2.0]),
        dims="x",
        name=np.float64("nan"),
        attrs={"config": {"window": np.array(2)}},
    )

    with pytest.raises(ValueError, match="JVP tangent pytree structure does not match"):
        ad.jvp(lambda field: 2.0 * field)(value, tangents=value)

    _output, pullback = ad.vjp(lambda field: 2.0 * field)(value)
    try:
        with pytest.raises(ValueError, match="Cotangent pytree structure does not match"):
            pullback(value)
    finally:
        pullback.close()


def test_dataarray_rejects_a_traced_attribute() -> None:
    value = xr.DataArray(np.arange(2.0), dims="x", coords={"x": [0, 1]})

    with pytest.raises(TypeError, match="found a traced value at attrs\\['scale'\\]"):
        ad.jvp(lambda field: field.assign_attrs(scale=field.data[0]))(
            value,
            tangents=xr.ones_like(value),
        )


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
    module_path = Path(advect.xarray.__file__)

    if missing_name == "xarray":
        with pytest.raises(ModuleNotFoundError, match=r"pip install 'advect\[xarray\]'"):
            runpy.run_path(str(module_path))
    else:
        with pytest.raises(ModuleNotFoundError) as error:
            runpy.run_path(str(module_path))
        assert error.value is missing
