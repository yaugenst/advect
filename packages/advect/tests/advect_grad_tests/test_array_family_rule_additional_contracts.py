"""Public contracts for less common array-family derivative-rule branches."""

from __future__ import annotations

import warnings
from typing import Any

import array_api_strict as strict
import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad


def _tree_inner(left: Any, right: Any) -> float:
    left_leaves, left_treedef = ad.pytree.tree_flatten(left)
    right_leaves, right_treedef = ad.pytree.tree_flatten(right)
    assert left_treedef == right_treedef
    return sum(
        float(np.real(np.vdot(np.asarray(left_leaf), np.asarray(right_leaf))))
        for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True)
    )


def _assert_tree_allclose(actual: Any, expected: Any) -> None:
    actual_leaves, actual_treedef = ad.pytree.tree_flatten(actual)
    expected_leaves, expected_treedef = ad.pytree.tree_flatten(expected)
    assert actual_treedef == expected_treedef
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        assert_allclose(actual_leaf, expected_leaf, rtol=2e-10, atol=2e-10)


def _assert_adjoint(
    function: Any,
    primals: tuple[Any, ...],
    tangents: Any,
    cotangent: Any,
    *,
    argnums: int | tuple[int, ...] = 0,
) -> Any:
    output, output_tangent = ad.jvp(function, argnums=argnums)(
        *primals,
        tangents=tangents,
    )
    reverse_output, pullback = ad.vjp(function, argnums=argnums)(*primals)
    try:
        input_cotangent = pullback(cotangent)
    finally:
        pullback.close()

    _assert_tree_allclose(reverse_output, output)
    assert_allclose(
        _tree_inner(cotangent, output_tangent),
        _tree_inner(input_cotangent, tangents),
        rtol=2e-9,
        atol=2e-10,
    )
    return input_cotangent


def test_concatenate_axis_none_restores_each_input_shape() -> None:
    left = np.arange(6.0).reshape(2, 3)
    right = np.arange(4.0).reshape(2, 2)
    left_tangent = np.linspace(-0.3, 0.4, left.size).reshape(left.shape)
    right_tangent = np.linspace(0.2, -0.5, right.size).reshape(right.shape)
    cotangent = np.linspace(-0.6, 0.7, left.size + right.size)

    gradients = _assert_adjoint(
        lambda a, b: np.concatenate((a, b), axis=None),
        (left, right),
        (left_tangent, right_tangent),
        cotangent,
        argnums=(0, 1),
    )
    assert_allclose(gradients[0], cotangent[: left.size].reshape(left.shape))
    assert_allclose(gradients[1], cotangent[left.size :].reshape(right.shape))


def test_rollaxis_negative_start_has_the_inverse_pullback() -> None:
    value = np.arange(24.0).reshape(2, 3, 4)
    tangent = np.linspace(-0.4, 0.5, value.size).reshape(value.shape)
    output = np.rollaxis(value, 0, -1)
    cotangent = np.linspace(0.7, -0.3, output.size).reshape(output.shape)

    _assert_adjoint(lambda x: np.rollaxis(x, 0, -1), (value,), tangent, cotangent)


@pytest.mark.parametrize("pad_width", [2, (1, 2)], ids=["integer", "pair"])
def test_constant_pad_width_forms_have_crop_pullbacks(pad_width: Any) -> None:
    value = np.arange(6.0).reshape(2, 3)
    tangent = np.linspace(-0.2, 0.3, value.size).reshape(value.shape)

    def function(x: Any) -> Any:
        return np.pad(x, pad_width, mode="constant", constant_values=2.5)

    output = function(value)
    cotangent = np.linspace(-0.5, 0.6, output.size).reshape(output.shape)
    _assert_adjoint(function, (value,), tangent, cotangent)


@pytest.mark.parametrize(
    ("function", "value"),
    [
        (lambda x: np.diff(x, n=0), np.arange(4.0)),
        (lambda x: np.diff(x, n=2), np.arange(2.0)),
        (lambda x: np.diff(x, prepend=0.0), np.arange(4.0)),
    ],
    ids=["identity", "empty-output", "scalar-prepend"],
)
def test_diff_edge_forms_obey_the_adjoint_law(function: Any, value: np.ndarray[Any, Any]) -> None:
    tangent = np.linspace(-0.3, 0.4, value.size)
    output = function(value)
    cotangent = np.linspace(0.2, -0.5, output.size)

    _assert_adjoint(function, (value,), tangent, cotangent)


def test_second_order_gradient_edges_obey_the_adjoint_law() -> None:
    value = np.arange(10.0).reshape(2, 5)
    tangent = np.linspace(-0.3, 0.4, value.size).reshape(value.shape)
    cotangent = np.linspace(0.5, -0.2, value.size).reshape(value.shape)

    def function(x: Any) -> Any:
        return np.gradient(x, axis=1, edge_order=2)

    _assert_adjoint(function, (value,), tangent, cotangent)


def test_single_sample_interp_differentiates_the_only_sample_value() -> None:
    queries = np.array([-1.0, 0.5, 2.0])
    sample_positions = np.array([0.5])
    sample_value = np.array([1.2])
    tangent = np.array([-0.3])
    cotangent = np.array([0.2, -0.4, 0.7])

    def function(fp: Any) -> Any:
        return np.interp(queries, sample_positions, fp)

    gradient = _assert_adjoint(function, (sample_value,), tangent, cotangent)
    assert_allclose(gradient, np.array([np.sum(cotangent)]))


@pytest.mark.parametrize("operation", [np.inner, np.kron], ids=["inner", "kron"])
def test_scalar_bilinear_operations_obey_the_complex_adjoint_law(operation: Any) -> None:
    left = np.array(1.2 + 0.3j)
    right = np.array(-0.4 + 0.7j)
    left_tangent = np.array(0.2 - 0.1j)
    right_tangent = np.array(-0.3 + 0.25j)
    cotangent = np.array(0.4 + 0.15j)

    _assert_adjoint(
        operation,
        (left, right),
        (left_tangent, right_tangent),
        cotangent,
        argnums=(0, 1),
    )


def test_linspace_single_endpoint_depends_only_on_start() -> None:
    start = np.array(1.2)
    stop = np.array(-0.4)
    start_tangent = np.array(0.3)
    stop_tangent = np.array(-0.2)
    cotangent = np.array([0.7])

    gradient = _assert_adjoint(
        lambda a, b: np.linspace(a, b, num=1, endpoint=True),
        (start, stop),
        (start_tangent, stop_tangent),
        cotangent,
        argnums=(0, 1),
    )
    assert_allclose(gradient[0], 0.7)
    assert_allclose(gradient[1], 0.0)


def test_real_input_imaginary_part_has_a_zero_pullback() -> None:
    value = np.array([1.0, -2.0, 3.0])
    tangent = np.array([0.2, -0.4, 0.1])
    cotangent = np.array([0.5, -0.3, 0.7])

    gradient = _assert_adjoint(np.imag, (value,), tangent, cotangent)
    assert_allclose(gradient, np.zeros_like(value))


@pytest.mark.parametrize("mode", ["wrap", "clip"])
def test_take_modes_scatter_repeated_indices_on_a_middle_axis(mode: str) -> None:
    value = np.arange(24.0).reshape(2, 3, 4)
    tangent = np.linspace(-0.3, 0.5, value.size).reshape(value.shape)
    indices = np.array([4, -1, 1])

    def function(x: Any) -> Any:
        return np.take(x, indices, axis=1, mode=mode)

    output = function(value)
    cotangent = np.linspace(0.4, -0.2, output.size).reshape(output.shape)
    _assert_adjoint(function, (value,), tangent, cotangent)


def test_take_along_axis_pullback_reduces_broadcast_source_dimensions() -> None:
    value = np.array([[1.0, 2.0, 3.0]])
    tangent = np.array([[0.2, -0.4, 0.7]])
    indices = np.array([[0, 2], [1, 1]])

    def function(x: Any) -> Any:
        return np.take_along_axis(x, indices, axis=1)

    cotangent = np.array([[0.5, -0.3], [0.7, 0.2]])
    gradient = _assert_adjoint(function, (value,), tangent, cotangent)
    assert_allclose(gradient, np.array([[0.5, 0.9, -0.3]]))


def test_single_point_rfft_obeys_the_real_adjoint_law() -> None:
    value = np.array([1.2])
    tangent = np.array([-0.3])
    cotangent = np.array([0.4 - 0.2j])

    _assert_adjoint(np.fft.rfft, (value,), tangent, cotangent)


@pytest.mark.filterwarnings(
    "ignore:`axes` should not be `None` if `s` is not `None`:DeprecationWarning"
)
def test_array_api_irfftn_shape_defaults_to_the_trailing_axis_and_has_a_portable_pullback() -> None:
    value = strict.asarray(
        [[1.0 + 0.2j, -0.4 + 0.1j, 0.7 - 0.3j]],
        dtype=strict.complex128,
    )
    tangent = strict.asarray(
        [[0.2 - 0.1j, -0.3 + 0.4j, 0.5 + 0.2j]],
        dtype=strict.complex128,
    )

    def function(x: Any) -> Any:
        return x.__array_namespace__().fft.irfftn(x, s=(4,), axes=None, norm="ortho")

    output, output_tangent = ad.jvp(function)(value, tangents=tangent)
    assert output.shape == (1, 4)
    assert output_tangent.shape == output.shape
    assert_allclose(
        np.asarray(output_tangent),
        np.asarray(strict.fft.irfftn(tangent, s=(4,), axes=(-1,), norm="ortho")),
    )

    cotangent = strict.asarray([[0.4, -0.2, 0.6, 0.1]], dtype=strict.float64)
    _assert_adjoint(function, (value,), tangent, cotangent)


@pytest.mark.filterwarnings(
    "ignore:`axes` should not be `None` if `s` is not `None`:DeprecationWarning"
)
def test_array_api_fftn_shape_crop_has_a_portable_zero_padding_pullback() -> None:
    value = strict.asarray(
        [[1.0 + 0.2j, -0.4 + 0.1j, 0.7 - 0.3j, 0.2 + 0.5j, -0.6 + 0.4j]],
        dtype=strict.complex128,
    )
    tangent = strict.asarray(
        [[0.2 - 0.1j, -0.3 + 0.4j, 0.5 + 0.2j, -0.1 - 0.2j, 0.3 + 0.1j]],
        dtype=strict.complex128,
    )

    def function(x: Any) -> Any:
        return x.__array_namespace__().fft.fftn(x, s=(3,), axes=None, norm="ortho")

    cotangent = strict.asarray(
        [[0.4 - 0.2j, -0.3 + 0.1j, 0.7 + 0.2j]],
        dtype=strict.complex128,
    )
    gradient = _assert_adjoint(function, (value,), tangent, cotangent)
    assert gradient.shape == value.shape
    assert_allclose(np.asarray(gradient)[..., 3:], 0.0)


def test_batched_matrix_vector_product_obeys_the_complex_adjoint_law() -> None:
    matrix = np.arange(12.0).reshape(2, 2, 3) / 7 + 0.1j
    vector = np.array([0.5 - 0.2j, -0.7 + 0.1j, 1.2 + 0.3j])
    matrix_tangent = np.linspace(-0.2, 0.3, matrix.size).reshape(matrix.shape) + 0.05j
    vector_tangent = np.array([0.2 - 0.1j, -0.3 + 0.2j, 0.4 + 0.1j])
    cotangent = np.array([[0.5 - 0.2j, -0.3 + 0.1j], [0.7 + 0.2j, 0.1 - 0.4j]])

    _assert_adjoint(
        np.matmul,
        (matrix, vector),
        (matrix_tangent, vector_tangent),
        cotangent,
        argnums=(0, 1),
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (np.array(1.2 + 0.3j), np.array(-0.4 + 0.7j)),
        (np.array(1.2 + 0.3j), np.array([0.5 - 0.2j, -0.7 + 0.1j])),
        (np.array([0.5 - 0.2j, -0.7 + 0.1j]), np.array(1.2 + 0.3j)),
    ],
    ids=["scalar-scalar", "scalar-array", "array-scalar"],
)
def test_dot_zero_dimensional_forms_obey_the_complex_adjoint_law(left: Any, right: Any) -> None:
    left_tangent = np.ones_like(left) * (0.2 - 0.1j)
    right_tangent = np.ones_like(right) * (-0.3 + 0.25j)
    cotangent = np.ones_like(np.dot(left, right)) * (0.4 + 0.15j)

    _assert_adjoint(
        np.dot,
        (left, right),
        (left_tangent, right_tangent),
        cotangent,
        argnums=(0, 1),
    )


def test_einsum_explicit_path_and_selected_input_preserve_the_adjoint() -> None:
    left = np.arange(12.0).reshape(2, 2, 3) / 7 + 0.1j
    right = np.arange(12.0, 24.0).reshape(2, 3, 2) / 9 - 0.2j
    tangent = np.linspace(-0.3, 0.4, left.size).reshape(left.shape) + 0.05j
    path = ["einsum_path", (0, 1)]

    def function(a: Any, b: Any) -> Any:
        return np.einsum("...ij,...jk->...ik", a, b, optimize=path)

    cotangent = np.linspace(0.4, -0.2, 8).reshape(2, 2, 2) - 0.1j
    _assert_adjoint(function, (left, right), tangent, cotangent, argnums=0)


def test_higher_order_advanced_indexing_reports_the_scatter_boundary() -> None:
    value = np.array([0.5, -0.2, 1.1, 0.7])

    def loss(x: Any) -> Any:
        selected = x[np.array([0, 0, 2])]
        return np.sum(selected * selected)

    with pytest.raises(NotImplementedError, match="Higher-order pullbacks for advanced indexing"):
        ad.hessian(loss)(value)


def test_empty_mean_reports_the_undefined_pullback() -> None:
    value = np.empty((0, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        output, pullback = ad.vjp(lambda x: np.mean(x, axis=0))(value)
    try:
        with pytest.raises(RuntimeError, match="mean derivative received an empty reduction axis"):
            pullback(np.ones_like(output))
    finally:
        pullback.close()


def test_tall_complete_qr_reports_its_nonunique_derivative() -> None:
    value = np.array([[2.0, 0.3], [0.2, 1.5], [-0.1, 0.4]])
    tangent = np.linspace(-0.3, 0.4, value.size).reshape(value.shape)

    def function(x: Any) -> Any:
        return np.linalg.qr(x, mode="complete")

    with pytest.raises(NotImplementedError, match="provider-dependent null-space"):
        ad.jvp(function)(value, tangents=tangent)

    output, pullback = ad.vjp(function)(value)
    try:
        cotangent = tuple(np.ones_like(leaf) for leaf in output)
        with pytest.raises(NotImplementedError, match="provider-dependent null-space"):
            pullback(cotangent)
    finally:
        pullback.close()


def test_hermitian_svd_reports_its_derivative_boundary() -> None:
    value = np.array([[3.0, 0.4], [0.4, 1.6]])
    tangent = np.array([[0.2, -0.1], [-0.1, 0.3]])

    def function(x: Any) -> Any:
        return np.linalg.svd(x, hermitian=True)

    with pytest.raises(NotImplementedError, match="requires hermitian=False"):
        ad.jvp(function)(value, tangents=tangent)

    output, pullback = ad.vjp(function)(value)
    try:
        cotangent = tuple(np.ones_like(leaf) for leaf in output)
        with pytest.raises(NotImplementedError, match="requires hermitian=False"):
            pullback(cotangent)
    finally:
        pullback.close()


def test_rectangular_full_svd_reports_nonunique_singular_vector_derivatives() -> None:
    value = np.array([[2.0, 0.3, -0.4], [0.2, 1.4, 0.7]])
    tangent = np.linspace(-0.3, 0.4, value.size).reshape(value.shape)

    def function(x: Any) -> Any:
        return np.linalg.svd(x, full_matrices=True)

    with pytest.raises(NotImplementedError, match="completed null-space basis"):
        ad.jvp(function)(value, tangents=tangent)

    output, pullback = ad.vjp(function)(value)
    try:
        cotangent = tuple(np.ones_like(leaf) for leaf in output)
        with pytest.raises(NotImplementedError, match="completed null-space basis"):
            pullback(cotangent)
    finally:
        pullback.close()


@pytest.mark.parametrize("operation", [np.var, np.std], ids=["variance", "standard-deviation"])
def test_degenerate_reduction_reports_its_undefined_jvp(operation: Any) -> None:
    value = np.array([1.0])
    tangent = np.array([0.2])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(NotImplementedError, match="requires count > ddof"):
            ad.jvp(lambda x: operation(x, ddof=1))(value, tangents=tangent)
