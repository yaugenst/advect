"""Additional public contracts for provider-neutral abstract staging."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import array_api_strict as strict
import numpy as np
import pytest

import advect as ad

if TYPE_CHECKING:
    from collections.abc import Callable


def test_dtype_metadata_variants_drive_staged_control_flow() -> None:
    def promote_if_metadata_matches(value: Any) -> Any:
        namespace = value.__array_namespace__()
        result_dtype = namespace.result_type(value, namespace.float64, 1)
        float_info = namespace.finfo(namespace.complex64)
        int_info = namespace.iinfo(namespace.uint8)
        matches = (
            namespace.isdtype(value.dtype, ("real floating", namespace.int32))
            and not namespace.can_cast(namespace.float64, value.dtype)
            and float_info.bits == 32
            and int_info.max == 255
        )
        return namespace.astype(value if matches else -value, result_dtype)

    value = strict.asarray([1.0, -2.0], dtype=strict.float32)
    program = ad.stage(
        promote_if_metadata_matches,
        specs=(ad.ArraySpec(value.shape, value.dtype),),
    )

    for staged in (program, ad.StagedProgram.from_dict(program.to_dict())):
        actual = staged(value)
        assert actual.dtype == strict.float64
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(value))


@pytest.mark.parametrize(
    ("query", "message"),
    [
        (lambda namespace, _value: namespace.result_type(), "requires at least one argument"),
        (
            lambda namespace, _value: namespace.isdtype("object", "numeric"),
            "Unsupported staged dtype",
        ),
        (
            lambda namespace, _value: namespace.can_cast("object", namespace.float32),
            "Unsupported dtype pair",
        ),
        (
            lambda namespace, _value: namespace.finfo(namespace.int32),
            "requires a floating-point dtype",
        ),
        (
            lambda namespace, _value: namespace.iinfo(namespace.float32),
            "requires an integer dtype",
        ),
    ],
    ids=["result-type-empty", "isdtype-invalid", "can-cast-invalid", "finfo-int", "iinfo-float"],
)
def test_invalid_dtype_metadata_is_reported_while_staging(
    query: Callable[[Any, Any], object],
    message: str,
) -> None:
    def invalid(value: Any) -> Any:
        query(value.__array_namespace__(), value)
        return value

    with pytest.raises(TypeError, match=message):
        ad.stage(invalid, specs=(ad.ArraySpec((2,), "float32"),))


def test_asarray_stages_nested_tracers_with_copy_and_dtype_conversion() -> None:
    def matrix(value: Any) -> Any:
        namespace = value.__array_namespace__()
        return namespace.asarray(
            [[value[0], value[1]], [value[1], value[0]]],
            dtype=namespace.float64,
            copy=True,
        )

    value = strict.asarray([1.0, 2.0], dtype=strict.float32)
    program = ad.stage(matrix, specs=(ad.ArraySpec(value.shape, value.dtype),))

    for staged in (program, ad.StagedProgram.from_dict(program.to_dict())):
        actual = staged(value)
        assert actual.dtype == strict.float64
        np.testing.assert_array_equal(np.asarray(actual), [[1.0, 2.0], [2.0, 1.0]])

    device_program = ad.stage(
        lambda item: item.__array_namespace__().asarray(item, device="cuda:0", copy=True),
        specs=(ad.ArraySpec((2,), "float32"),),
    )
    cast_node = next(
        device_program.graph.get_node(node_id)
        for node_id in device_program.graph.node_ids()
        if device_program.graph.get_node(node_id).op == "array.astype"
    )
    assert cast_node.attrs["_advect_device"] == "cuda:0"


@pytest.mark.parametrize(
    ("constructor", "error", "message"),
    [
        (
            lambda namespace, _value: namespace.asarray([1, 2], copy=False),
            ValueError,
            r"copy=False.*sequence",
        ),
        (
            lambda namespace, value: namespace.asarray(value, copy="yes"),
            TypeError,
            "copy must be a bool or None",
        ),
        (
            lambda namespace, _value: namespace.asarray([[1, 2], [3]]),
            ValueError,
            "rectangular nested sequence",
        ),
        (
            lambda namespace, value: namespace.asarray(value, order="C"),
            TypeError,
            "supports only asarray",
        ),
    ],
    ids=["sequence-no-copy", "invalid-copy", "ragged-sequence", "unknown-option"],
)
def test_asarray_rejects_impossible_or_invalid_requests(
    constructor: Callable[[Any, Any], object],
    error: type[Exception],
    message: str,
) -> None:
    def invalid(value: Any) -> Any:
        return constructor(value.__array_namespace__(), value)

    with pytest.raises(error, match=message):
        ad.stage(invalid, specs=(ad.ArraySpec((2,), "float32"),))


@pytest.mark.parametrize(
    ("operation", "data", "expected"),
    [
        (
            lambda namespace, value: namespace.cumulative_sum(value, include_initial=True),
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 3.0, 6.0],
        ),
        (
            lambda namespace, value: namespace.cumulative_prod(
                value,
                axis=1,
                include_initial=True,
            ),
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[1.0, 1.0, 2.0, 6.0], [1.0, 4.0, 20.0, 120.0]],
        ),
    ],
    ids=["sum-default-axis", "prod-explicit-axis"],
)
def test_cumulative_include_initial_stages_the_identity_slice(
    operation: Callable[[Any, Any], object],
    data: object,
    expected: object,
) -> None:
    def cumulative(value: Any) -> Any:
        return operation(value.__array_namespace__(), value)

    value = strict.asarray(data, dtype=strict.float32)
    program = ad.stage(cumulative, specs=(ad.ArraySpec(value.shape, value.dtype),))

    np.testing.assert_array_equal(np.asarray(program(value)), expected)


@pytest.mark.parametrize(
    ("operation", "error", "message"),
    [
        (
            lambda namespace, value: namespace.cumulative_sum(value, include_initial=True),
            ValueError,
            "require axis=",
        ),
        (
            lambda namespace, value: namespace.cumulative_prod(
                value,
                1,
                axis=0,
                include_initial=True,
            ),
            TypeError,
            "received 'axis' twice",
        ),
    ],
    ids=["missing-matrix-axis", "duplicate-axis"],
)
def test_cumulative_include_initial_validates_axis_contract(
    operation: Callable[[Any, Any], object],
    error: type[Exception],
    message: str,
) -> None:
    def invalid(value: Any) -> Any:
        return operation(value.__array_namespace__(), value)

    with pytest.raises(error, match=message):
        ad.stage(invalid, specs=(ad.ArraySpec((2, 3), "float32"),))


def test_diff_stages_boundaries_and_the_zero_order_identity() -> None:
    def differences(value: Any) -> tuple[Any, Any]:
        namespace = value.__array_namespace__()
        return (
            namespace.diff(value, axis=1, prepend=0.0, append=9.0),
            namespace.diff(value, n=0, axis=1),
        )

    value = strict.reshape(strict.arange(6, dtype=strict.float32), (2, 3))
    program = ad.stage(differences, specs=(ad.ArraySpec(value.shape, value.dtype),))
    expected = np.asarray(value)
    for staged in (program, ad.StagedProgram.from_dict(program.to_dict())):
        scalar_boundaries, identity = staged(value)
        np.testing.assert_array_equal(
            np.asarray(scalar_boundaries),
            np.diff(expected, axis=1, prepend=0.0, append=9.0),
        )
        np.testing.assert_array_equal(np.asarray(identity), expected)


@pytest.mark.parametrize(
    ("operation", "error", "message"),
    [
        (
            lambda namespace, value: namespace.diff(value, 1, n=2),
            TypeError,
            "received 'n' twice",
        ),
        (
            lambda namespace, value: namespace.diff(value, period=2),
            TypeError,
            "does not support.*period",
        ),
        (
            lambda namespace, value: namespace.diff(value, n=True),
            ValueError,
            "non-negative integer",
        ),
        (
            lambda namespace, value: namespace.diff(value, axis=2),
            ValueError,
            "out of bounds",
        ),
    ],
    ids=["duplicate-n", "unknown-option", "boolean-n", "invalid-axis"],
)
def test_diff_validates_static_controls(
    operation: Callable[[Any, Any], object],
    error: type[Exception],
    message: str,
) -> None:
    def invalid(value: Any) -> Any:
        return operation(value.__array_namespace__(), value)

    with pytest.raises(error, match=message):
        ad.stage(invalid, specs=(ad.ArraySpec((2, 3), "float32"),))


@pytest.mark.parametrize(
    ("index", "error", "message"),
    [
        ((Ellipsis, Ellipsis), IndexError, "Only one ellipsis"),
        ((0, 0, 0), IndexError, "Too many indices"),
        (3, IndexError, "out of bounds"),
        ("row", ad.TracingError, "Basic indexing supports only"),
        (slice(None, None, 0), ValueError, "slice step cannot be zero"),
    ],
    ids=["multiple-ellipsis", "too-many", "integer-out-of-bounds", "string", "zero-step"],
)
def test_public_staged_indexing_reports_invalid_basic_indices(
    index: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        ad.stage(lambda value: value[index], specs=(ad.ArraySpec((2, 3), "float32"),))
