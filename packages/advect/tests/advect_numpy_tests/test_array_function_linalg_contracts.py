"""Public contracts for material NumPy linalg call forms and controls."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import pytest

import advect as ad

ArrayFunction = Callable[..., Any]


def _assert_jvp_matches_directional_difference(
    function: ArrayFunction,
    primals: tuple[np.ndarray[Any, Any], ...],
    tangents: tuple[np.ndarray[Any, Any], ...],
    *,
    rtol: float = 2e-5,
    atol: float = 2e-6,
) -> None:
    argnums = 0 if len(primals) == 1 else tuple(range(len(primals)))
    tangent_input: object = tangents[0] if len(tangents) == 1 else tangents
    primal, tangent = ad.jvp(function, argnums=argnums)(
        *primals,
        tangents=tangent_input,
    )

    step = 1e-6
    plus = function(
        *(value + step * direction for value, direction in zip(primals, tangents, strict=True))
    )
    minus = function(
        *(value - step * direction for value, direction in zip(primals, tangents, strict=True))
    )
    expected = function(*primals)
    primal_leaves, primal_tree = ad.pytree.tree_flatten(primal)
    expected_leaves, expected_tree = ad.pytree.tree_flatten(expected)
    tangent_leaves, tangent_tree = ad.pytree.tree_flatten(tangent)
    plus_leaves, plus_tree = ad.pytree.tree_flatten(plus)
    minus_leaves, minus_tree = ad.pytree.tree_flatten(minus)
    assert primal_tree == expected_tree
    assert tangent_tree == plus_tree
    assert tangent_tree == minus_tree

    for actual, expected_leaf in zip(primal_leaves, expected_leaves, strict=True):
        np.testing.assert_allclose(actual, expected_leaf, rtol=rtol, atol=atol)
    for actual, upper, lower in zip(
        tangent_leaves,
        plus_leaves,
        minus_leaves,
        strict=True,
    ):
        np.testing.assert_allclose(
            actual,
            (np.asarray(upper) - np.asarray(lower)) / (2 * step),
            rtol=rtol,
            atol=atol,
        )


def _assert_staged_matches(
    function: ArrayFunction,
    values: tuple[np.ndarray[Any, Any], ...],
) -> None:
    program = ad.stage(
        function,
        specs=tuple(ad.ArraySpec(value.shape, value.dtype) for value in values),
    )
    expected = function(*values)
    expected_leaves, expected_tree = ad.pytree.tree_flatten(expected)
    for staged in (program, ad.StagedProgram.from_dict(program.to_dict())):
        actual_leaves, actual_tree = ad.pytree.tree_flatten(staged(*values))
        assert actual_tree == expected_tree
        for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
            np.testing.assert_allclose(actual_leaf, expected_leaf)


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x: np.linalg.norm(
                x,
                ord=np.int64(1),
                axis=np.int64(0),
                keepdims=True,
            ),
            id="numpy-integer-ord-and-axis",
        ),
        pytest.param(
            lambda x: np.linalg.norm(x, ord=np.float64(2), axis=(0, 1)),
            id="numpy-float-ord-and-tuple-axis",
        ),
    ],
)
def test_norm_accepts_numpy_scalar_and_sequence_controls(function: ArrayFunction) -> None:
    value = np.array([[3.0, 1.0], [1.0, 2.0]])
    direction = np.array([[0.2, -0.1], [0.3, 0.4]])

    _assert_jvp_matches_directional_difference(function, (value,), (direction,))


@pytest.mark.parametrize(
    ("function", "value"),
    [
        pytest.param(
            lambda x: np.linalg.pinv(x, 1e-8, np.bool_(0)),
            np.array([[1.0, 0.2], [0.3, 1.4], [1.2, -0.4]]),
            id="positional-rcond-and-hermitian",
        ),
        pytest.param(
            lambda x: np.linalg.pinv(x, rtol=1e-8, hermitian=True),
            np.array([[3.0, 1.0], [1.0, 2.0]]),
            id="rtol-and-hermitian",
        ),
    ],
)
def test_pinv_accepts_each_static_tolerance_spelling(
    function: ArrayFunction,
    value: np.ndarray[Any, Any],
) -> None:
    direction = np.linspace(0.05, 0.2, value.size).reshape(value.shape)
    if value.ndim == 2 and value.shape[0] == value.shape[1] and np.array_equal(value, value.T):
        direction = (direction + direction.T) / 2

    _assert_jvp_matches_directional_difference(
        function,
        (value,),
        (direction,),
        rtol=2e-4,
        atol=2e-5,
    )


def test_matrix_rank_handles_vector_empty_and_hermitian_inputs() -> None:
    cases = (
        (np.array([2.0, 0.0, 0.0]), {"tol": 1e-8}),
        (np.empty((2, 0, 3)), {}),
        (np.array([[3.0, 1.0], [1.0, 2.0]]), {"rtol": 1e-8, "hermitian": True}),
    )

    for value, kwargs in cases:
        function = partial(np.linalg.matrix_rank, **kwargs)
        primal, tangent = ad.jvp(function)(
            value,
            tangents=np.ones_like(value),
        )
        np.testing.assert_array_equal(primal, np.linalg.matrix_rank(value, **kwargs))
        np.testing.assert_array_equal(tangent, np.zeros_like(primal))


def test_eigh_upper_triangle_preserves_tuple_outputs_across_lifetimes() -> None:
    value = np.array([[3.0, 1.0], [1.0, 2.0]])
    direction = np.array([[0.2, -0.1], [-0.1, 0.3]])

    def function(x: Any) -> Any:
        return tuple(np.linalg.eigh(x, UPLO="u"))

    primal, tangent = ad.jvp(function)(value, tangents=direction)
    eigenvalues, eigenvectors = primal
    eigenvalue_tangent, eigenvector_tangent = tangent
    reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    reconstructed_tangent = (
        eigenvector_tangent @ np.diag(eigenvalues) @ eigenvectors.T
        + eigenvectors @ np.diag(eigenvalue_tangent) @ eigenvectors.T
        + eigenvectors @ np.diag(eigenvalues) @ eigenvector_tangent.T
    )
    assert ad.pytree.tree_flatten(primal)[1] == ad.pytree.tree_flatten(tangent)[1]
    np.testing.assert_allclose(reconstructed, value)
    np.testing.assert_allclose(reconstructed_tangent, direction)
    _assert_staged_matches(function, (value,))


def test_eigvalsh_accepts_lowercase_upper_triangle_control_dynamically() -> None:
    value = np.array([[3.0, 1.0], [1.0, 2.0]])
    direction = np.array([[0.2, -0.1], [-0.1, 0.3]])

    _assert_jvp_matches_directional_difference(
        lambda x: np.linalg.eigvalsh(x, UPLO="u"),
        (value,),
        (direction,),
    )


def test_qr_static_modes_preserve_their_output_contracts() -> None:
    value = np.array([[1.0, 0.2], [0.3, 1.4], [1.2, -0.4]])

    def complete(x: Any) -> Any:
        return tuple(np.linalg.qr(x, mode="complete"))

    _assert_staged_matches(complete, (value,))
    q_value, r_value = complete(value)
    np.testing.assert_allclose(q_value @ r_value, value)

    def reduced(x: Any) -> Any:
        return tuple(np.linalg.qr(x, mode=None))

    primal, tangent = ad.jvp(reduced)(value, tangents=np.zeros_like(value))
    assert isinstance(primal, tuple)
    assert isinstance(tangent, tuple)
    np.testing.assert_allclose(primal[0] @ primal[1], value)
    for leaf in tangent:
        np.testing.assert_array_equal(leaf, np.zeros_like(leaf))

    def r_only(x: Any) -> Any:
        return np.linalg.qr(x, mode="r")

    _assert_jvp_matches_directional_difference(
        r_only,
        (value,),
        (np.full_like(value, 0.1),),
        rtol=2e-4,
        atol=2e-5,
    )
    _assert_staged_matches(r_only, (value,))


def test_cross_honors_independent_input_and_output_axes() -> None:
    left = np.arange(24.0).reshape(3, 2, 4)
    right = np.flip(left, axis=0) + 1.0

    _assert_jvp_matches_directional_difference(
        lambda a, b: np.cross(a, b, axisa=0, axisb=0, axisc=0),
        (left, right),
        (np.full_like(left, 0.1), np.full_like(right, -0.2)),
    )


@pytest.mark.parametrize(
    ("function", "left", "right"),
    [
        pytest.param(
            lambda a, b: np.tensordot(a, b, axes=([2, 0], [0, 1])),
            np.arange(24.0).reshape(2, 3, 4),
            np.arange(40.0).reshape(4, 2, 5),
            id="explicit-axis-lists",
        ),
        pytest.param(
            lambda a, b: np.tensordot(a, b, axes=1),
            np.arange(24.0).reshape(2, 3, 4),
            np.arange(20.0).reshape(4, 5),
            id="integer-axis-count",
        ),
        pytest.param(
            lambda a, b: np.tensordot(a, b, axes=(1, 0)),
            np.arange(6.0).reshape(2, 3),
            np.arange(12.0).reshape(3, 4),
            id="axis-pair",
        ),
    ],
)
def test_tensordot_axes_forms_trace_and_stage(
    function: ArrayFunction,
    left: np.ndarray[Any, Any],
    right: np.ndarray[Any, Any],
) -> None:
    _assert_jvp_matches_directional_difference(
        function,
        (left, right),
        (np.full_like(left, 0.1), np.full_like(right, -0.2)),
    )
    _assert_staged_matches(function, (left, right))


@pytest.mark.parametrize(
    ("function", "values"),
    [
        pytest.param(
            lambda x: np.einsum("ii->i", x, optimize=True, dtype=np.float64),
            (np.array([[3.0, 1.0], [1.0, 2.0]]),),
            id="diagonal-and-dtype",
        ),
        pytest.param(
            lambda x: np.einsum("ij->i", x, dtype=np.float64),
            (np.array([[1.0, 0.2], [0.3, 1.4], [1.2, -0.4]]),),
            id="single-operand-reduction",
        ),
        pytest.param(
            lambda a, b: np.einsum(a, [0, 1], b, [1, 2], [0, 2], optimize="greedy"),
            (
                np.array([[3.0, 1.0], [1.0, 2.0]]),
                np.array([[1.0, 2.0], [0.5, -1.0]]),
            ),
            id="sublist-with-explicit-output",
        ),
        pytest.param(
            lambda a, b: np.einsum(a, [0, 1], b, [1, 2]),
            (
                np.array([[3.0, 1.0], [1.0, 2.0]]),
                np.array([[1.0, 2.0], [0.5, -1.0]]),
            ),
            id="sublist-with-implicit-output",
        ),
    ],
)
def test_einsum_material_call_forms_match_directional_differences(
    function: ArrayFunction,
    values: tuple[np.ndarray[Any, Any], ...],
) -> None:
    tangents = tuple(np.linspace(0.05, 0.2, value.size).reshape(value.shape) for value in values)
    _assert_jvp_matches_directional_difference(function, values, tangents)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        pytest.param(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]), id="vector-vector"),
        pytest.param(
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            np.array([4.0, 5.0]),
            id="matrix-vector",
        ),
        pytest.param(
            np.array([4.0, 5.0, 6.0]),
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            id="vector-matrix",
        ),
    ],
)
def test_dot_shape_families_trace_and_stage(
    left: np.ndarray[Any, Any],
    right: np.ndarray[Any, Any],
) -> None:
    _assert_jvp_matches_directional_difference(
        np.dot,
        (left, right),
        (np.full_like(left, 0.1), np.full_like(right, -0.2)),
    )
    _assert_staged_matches(np.dot, (left, right))


@pytest.mark.parametrize(
    ("function", "value", "match"),
    [
        pytest.param(
            lambda x: np.linalg.norm(x, axis=object()),
            np.eye(2),
            r"np\.linalg\.norm\(axis=",
            id="norm-axis-type",
        ),
        pytest.param(
            lambda x: np.linalg.norm(x, ord=object()),
            np.eye(2),
            r"np\.linalg\.norm\(ord=",
            id="norm-ord-type",
        ),
        pytest.param(
            lambda x: np.linalg.pinv(x, rcond=1e-6, rtol=1e-5),
            np.eye(2),
            "only one of rcond= and rtol=",
            id="pinv-conflicting-tolerances",
        ),
        pytest.param(
            np.linalg.matrix_rank,
            np.array(1.0),
            "ndim >= 1",
            id="matrix-rank-scalar",
        ),
        pytest.param(
            lambda x: np.linalg.matrix_rank(x, tol=1e-6, rtol=1e-5),
            np.eye(2),
            "cannot receive both tol and rtol",
            id="matrix-rank-conflicting-tolerances",
        ),
        pytest.param(
            lambda x: np.linalg.matrix_rank(x, hermitian=True),
            np.ones((2, 3)),
            "hermitian=True.*square",
            id="matrix-rank-nonsquare-hermitian",
        ),
        pytest.param(
            lambda x: np.linalg.eigh(x, UPLO="middle"),
            np.eye(2),
            r"eigh\(UPLO=.*Use 'L' or 'U'",
            id="eigh-uplo",
        ),
        pytest.param(
            lambda x: np.linalg.eigvalsh(x, UPLO="middle"),
            np.eye(2),
            r"eigvalsh\(UPLO=.*Use 'L' or 'U'",
            id="eigvalsh-uplo",
        ),
        pytest.param(
            lambda x: np.linalg.qr(x, mode="raw"),
            np.eye(2),
            "can change the output arity",
            id="qr-output-arity-mode",
        ),
        pytest.param(
            lambda x: np.linalg.cholesky(x, upper=1),
            np.eye(2),
            "upper= must be a bool",
            id="cholesky-upper-type",
        ),
        pytest.param(
            lambda x: np.einsum("ii->ij", x),
            np.eye(2),
            "syntax parsing failed",
            id="einsum-invalid-output-label",
        ),
        pytest.param(
            np.einsum,
            np.eye(2),
            "sublist form requires at least one operand",
            id="einsum-missing-labels",
        ),
        pytest.param(
            lambda x: np.einsum("ij->i", x, banana=True),
            np.eye(2),
            "kwargs not supported.*banana",
            id="einsum-unsupported-keyword",
        ),
    ],
)
def test_linalg_rejects_ambiguous_or_invalid_static_controls(
    function: ArrayFunction,
    value: np.ndarray[Any, Any],
    match: str,
) -> None:
    with pytest.raises(ad.TracingError, match=match):
        ad.jvp(function)(value, tangents=np.ones_like(value))


@pytest.mark.parametrize("tolerance_name", ["rcond", "rtol"])
def test_pinv_rejects_dynamic_rank_tolerances(tolerance_name: str) -> None:
    value = np.array([[3.0, 1.0], [1.0, 2.0]])
    tolerance = np.array(1e-6)

    def function(matrix: Any, dynamic_tolerance: Any) -> Any:
        return np.linalg.pinv(matrix, **{tolerance_name: dynamic_tolerance})

    with pytest.raises(ad.TracingError, match=rf"pinv {tolerance_name}= must be static"):
        ad.jvp(function, argnums=(0, 1))(
            value,
            tolerance,
            tangents=(np.ones_like(value), np.zeros_like(tolerance)),
        )


def test_lstsq_underdetermined_system_has_empty_residuals() -> None:
    matrix = np.array([[1.0, 0.2, -0.1], [0.3, 1.4, 0.5]])
    right = np.array([0.5, -1.0])
    matrix_tangent = np.full_like(matrix, 0.1)
    right_tangent = np.full_like(right, -0.2)

    def function(a: Any, b: Any) -> Any:
        return np.linalg.lstsq(a, b, rcond=None)

    _assert_jvp_matches_directional_difference(
        function,
        (matrix, right),
        (matrix_tangent, right_tangent),
        rtol=2e-4,
        atol=2e-5,
    )
    primal, tangent = ad.jvp(function, argnums=(0, 1))(
        matrix,
        right,
        tangents=(matrix_tangent, right_tangent),
    )
    assert primal[1].shape == (0,)
    assert tangent[1].shape == (0,)


@pytest.mark.parametrize(
    ("function", "value", "match"),
    [
        pytest.param(
            lambda matrix: np.linalg.lstsq(matrix, np.ones(2), rcond=None),
            np.ones((2, 2, 2)),
            "matrix must be two-dimensional",
            id="matrix-rank",
        ),
        pytest.param(
            lambda right: np.linalg.lstsq(np.eye(2), right, rcond=None),
            np.ones(3),
            "right-hand side must match",
            id="right-shape",
        ),
    ],
)
def test_lstsq_rejects_invalid_core_shapes(
    function: ArrayFunction,
    value: np.ndarray[Any, Any],
    match: str,
) -> None:
    with pytest.raises(ad.TracingError, match=match):
        ad.jvp(function)(value, tangents=np.ones_like(value))


def test_lstsq_rejects_a_dynamic_rank_tolerance() -> None:
    matrix = np.array([[1.0, 0.2], [0.3, 1.4], [1.2, -0.4]])
    right = np.array([0.5, -1.0, 2.0])
    tolerance = np.array(1e-6)

    def function(a: Any, dynamic_tolerance: Any) -> Any:
        return np.linalg.lstsq(a, right, rcond=dynamic_tolerance)

    with pytest.raises(ad.TracingError, match="lstsq rcond must be static"):
        ad.jvp(function, argnums=(0, 1))(
            matrix,
            tolerance,
            tangents=(np.ones_like(matrix), np.zeros_like(tolerance)),
        )
