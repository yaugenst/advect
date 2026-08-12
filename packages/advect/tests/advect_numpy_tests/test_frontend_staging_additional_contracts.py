"""Additional public contracts for NumPy's staged lifetime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import advect as ad

if TYPE_CHECKING:
    from collections.abc import Callable


def _assert_same(actual: object, expected: object) -> None:
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_same(actual_item, expected_item)
        return
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def _assert_staged_round_trip(
    function: Callable[..., object],
    values: tuple[np.ndarray[Any, Any], ...],
) -> None:
    program = ad.stage(
        function,
        specs=tuple(ad.ArraySpec(value.shape, value.dtype) for value in values),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())
    expected = function(*values)

    for staged in (program, restored):
        _assert_same(staged(*values), expected)


def test_staged_gradient_supports_scalar_and_coordinate_spacings() -> None:
    values = np.arange(1.0, 13.0).reshape(3, 4)
    rows = np.array([0.0, 1.0, 3.0])
    columns = np.array([0.0, 0.5, 2.0, 5.0])

    _assert_staged_round_trip(
        lambda array: np.gradient(array, 2.0, axis=(0, 1), edge_order=2),
        (values,),
    )
    _assert_staged_round_trip(
        lambda array: np.gradient(array, rows, columns, axis=(0, 1), edge_order=2),
        (values,),
    )
    _assert_staged_round_trip(
        lambda array, spacing: np.gradient(array, spacing, axis=1, edge_order=2),
        (values, columns),
    )


def test_staged_gradient_rejects_coordinate_spacing_with_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="must be one-dimensional and match axis 1 length 3"):
        ad.stage(
            lambda array: np.gradient(array, np.ones(2), axis=1),
            specs=(ad.ArraySpec((2, 3), "float64"),),
        )


def test_staged_average_preserves_weight_and_result_metadata() -> None:
    values = np.arange(1.0, 13.0).reshape(3, 4)
    weights = np.array([1.0, 2.0, 3.0, 4.0])

    _assert_staged_round_trip(np.average, (values,))
    _assert_staged_round_trip(
        lambda array, weight: np.average(array, 1, weight, True),  # noqa: FBT003
        (values, weights),
    )


@pytest.mark.parametrize(
    ("operation", "error", "match"),
    [
        (
            lambda array: np.average(array, weights=np.ones(3)),
            TypeError,
            "Axis must be specified",
        ),
        (
            lambda array: np.average(array, axis=1, weights=np.ones(2)),
            ValueError,
            "Shape of weights must be consistent",
        ),
    ],
)
def test_staged_average_rejects_incompatible_weight_shapes(
    operation: Callable[[Any], Any],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        ad.stage(operation, specs=(ad.ArraySpec((2, 3), "float64"),))


@pytest.mark.parametrize("operation", [np.var, np.nanstd])
def test_staged_variance_preserves_nan_for_nonpositive_degrees_of_freedom(
    operation: Callable[..., Any],
) -> None:
    values = np.arange(1.0, 13.0, dtype=np.float32).reshape(3, 4)
    mask = np.array([[True], [False], [True]])

    with pytest.warns(RuntimeWarning):
        _assert_staged_round_trip(
            lambda array, where: operation(
                array,
                axis=1,
                dtype=np.float32,
                where=where,
                ddof=1,
                keepdims=True,
            ),
            (values, mask),
        )


def test_staged_variance_accepts_live_mean_and_correction_operands() -> None:
    values = np.arange(1.0, 13.0).reshape(3, 4)
    mean = np.mean(values, axis=1, keepdims=True)
    correction = np.asarray(1.0)

    _assert_staged_round_trip(
        lambda array, supplied_mean, supplied_correction: np.var(
            array,
            axis=1,
            mean=supplied_mean,
            correction=supplied_correction,
            keepdims=True,
        ),
        (values, mean, correction),
    )


@pytest.mark.parametrize("operation", [np.cumulative_sum, np.cumulative_prod])
def test_staged_one_dimensional_cumulative_initial_uses_the_default_axis(
    operation: Callable[..., Any],
) -> None:
    values = np.arange(1.0, 5.0, dtype=np.float32)

    _assert_staged_round_trip(
        lambda array: operation(array, dtype=np.float64, include_initial=True),
        (values,),
    )


def test_staged_extrema_with_where_requires_an_initial_value() -> None:
    with pytest.raises(TypeError, match="with where= requires initial="):
        ad.stage(
            lambda array: np.max(array, axis=1, where=array > 0),
            specs=(ad.ArraySpec((2, 3), "float64"),),
        )


@pytest.mark.parametrize(
    "operation",
    [
        lambda array: np.linalg.pinv(array, 1e-8),
        lambda array: np.linalg.pinv(array, rtol=1e-8),
        lambda array: np.linalg.pinv(array, rtol=1e-8, hermitian=True),
        lambda array: np.linalg.pinv(array, hermitian=np.bool_(1)),
    ],
    ids=("positional-rcond", "rtol", "hermitian", "numpy-bool"),
)
def test_staged_pinv_preserves_tolerance_and_hermitian_controls(
    operation: Callable[[Any], Any],
) -> None:
    matrix = np.array([[3.0, 1.0], [1.0, 2.0]])

    _assert_staged_round_trip(operation, (matrix,))


def test_staged_creation_preserves_positional_and_default_metadata() -> None:
    anchor = np.zeros(1)
    fill = np.asarray(2.0)

    def create(like: Any, value: Any) -> tuple[Any, ...]:
        return (
            np.full(
                (2, 3),
                value,
                np.float32,
                "F",
                device="cpu",
                like=like,
            ),
            np.zeros((2, 3), like=like),
            np.ones((2, 3), like=like),
        )

    _assert_staged_round_trip(create, (anchor, fill))


def test_staged_take_reshape_and_diff_replay_public_metadata() -> None:
    values = np.arange(12.0).reshape(3, 4)
    indices = np.array([[3, 0], [2, 1], [0, 0]])

    def transform(array: Any, positions: Any) -> tuple[Any, ...]:
        return (
            np.take_along_axis(array, positions, -1),
            np.reshape(array, (4, 3), order="F", copy=True),
            np.diff(array, 2, 0, np.ones((1, 4)), np.full((1, 4), 2.0)),
        )

    _assert_staged_round_trip(transform, (values, indices))


def test_staged_compress_rejects_a_non_vector_condition() -> None:
    with pytest.raises(ValueError, match="condition must be a 1-d array"):
        ad.stage(
            lambda array: np.compress(np.asarray(1, dtype=bool), array),
            specs=(ad.ArraySpec((2, 3), "float64"),),
        )


def test_staged_out_rejects_concrete_and_cross_trace_destinations() -> None:
    with pytest.raises(ad.MutationError, match="requires one owned staged array destination"):
        ad.stage(
            lambda array: np.sum(array, out=np.zeros(())),
            specs=(ad.ArraySpec((3,), "float64"),),
        )

    def outer(array: Any) -> Any:
        destination = array.copy()
        ad.stage(
            lambda inner: np.add(inner, 1.0, out=destination),
            specs=(ad.ArraySpec((3,), "float64"),),
        )
        return array

    with pytest.raises(ad.TracingError, match="array from another trace"):
        ad.stage(outer, specs=(ad.ArraySpec((3,), "float64"),))


@pytest.mark.parametrize(
    ("operation", "error", "match"),
    [
        (
            lambda array: np.eye(2, order="F", like=array),
            TypeError,
            "supports only order='C'",
        ),
        (
            lambda array: array.astype(np.float32, casting="banana"),
            ValueError,
            "invalid casting rule",
        ),
        (
            lambda array: array.astype(np.float32, subok=1),
            TypeError,
            "subok must be a bool",
        ),
        (
            lambda array: array.astype(np.int32, casting="safe"),
            TypeError,
            "according to the 'safe' rule",
        ),
        (
            lambda array: np.linalg.matrix_power(array, 2),
            ValueError,
            "requires square matrices",
        ),
        (
            lambda array: np.diff(array, n=-1),
            ValueError,
            "non-negative integer",
        ),
    ],
    ids=(
        "eye-order",
        "astype-casting",
        "astype-subok",
        "astype-safe",
        "matrix-power-shape",
        "diff-order",
    ),
)
def test_staged_metadata_validation_matches_numpy_frontend_contracts(
    operation: Callable[[Any], Any],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        ad.stage(operation, specs=(ad.ArraySpec((2, 3), "float64"),))
