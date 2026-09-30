"""Linear-algebra derivative contracts across the NumPy and Array API frontends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import array_api_strict as strict
import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, example, given
from numpy.testing import assert_allclose

import advect as ad

if TYPE_CHECKING:
    from collections.abc import Callable


_RECTANGULAR = strict.asarray(
    [[1.4, 0.2], [0.3, 1.1], [0.5, -0.4]],
    dtype=strict.float64,
)
_RECTANGULAR_TANGENT = strict.asarray(
    [[0.1, -0.2], [0.3, 0.1], [-0.1, 0.25]],
    dtype=strict.float64,
)


def _pinv_loss(value: Any) -> Any:
    namespace = value.__array_namespace__()
    inverse = namespace.linalg.pinv(value)
    return namespace.sum(inverse * inverse)


def _qr_loss(value: Any) -> Any:
    namespace = value.__array_namespace__()
    q, r = namespace.linalg.qr(value, mode="reduced")
    q_weights = namespace.asarray(
        [[0.7, -0.2], [0.1, 0.4], [-0.3, 0.9]],
        dtype=value.dtype,
    )
    r_weights = namespace.asarray(
        [[0.5, -0.1], [0.2, 0.8]],
        dtype=value.dtype,
    )
    return namespace.sum(q * q_weights) + namespace.sum(r * r_weights)


def test_complex_vecdot_gradient_uses_the_real_inner_product_convention() -> None:
    value = strict.asarray(
        [[1.0 + 0.5j, -0.2 + 0.7j], [0.3 - 0.4j, 1.2 + 0.1j]],
        dtype=strict.complex128,
    )
    weights = strict.asarray(
        [[0.5 - 0.25j, 0.8 + 0.2j], [-0.1 + 0.6j, 0.4 - 0.3j]],
        dtype=strict.complex128,
    )

    def loss(argument: Any) -> Any:
        namespace = argument.__array_namespace__()
        products = namespace.vecdot(argument, weights, axis=-1)
        return namespace.sum(namespace.real(products))

    dynamic = ad.grad(loss)(value)
    staged = ad.grad(ad.stage(loss, specs=(ad.ArraySpec(value.shape, value.dtype),)))(value)

    assert_allclose(np.asarray(dynamic), np.asarray(weights), rtol=1e-12, atol=1e-12)
    assert_allclose(np.asarray(staged), np.asarray(dynamic), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("loss", [_pinv_loss, _qr_loss], ids=["pinv", "qr"])
def test_traceable_linalg_pullbacks_support_higher_order_differentiation(
    loss: Callable[[Any], Any],
) -> None:
    _value, product = ad.hvp(loss)(_RECTANGULAR, vectors=_RECTANGULAR_TANGENT)
    epsilon = 1e-5
    expected = (
        ad.grad(loss)(_RECTANGULAR + epsilon * _RECTANGULAR_TANGENT)
        - ad.grad(loss)(_RECTANGULAR - epsilon * _RECTANGULAR_TANGENT)
    ) / (2 * epsilon)

    assert_allclose(np.asarray(product), np.asarray(expected), rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("lifetime", ["dynamic", "staged"])
@pytest.mark.parametrize("dtype", [strict.float32, strict.float64], ids=["float32", "float64"])
def test_trace_tangent_takes_the_requested_dtype(lifetime: str, dtype: Any) -> None:
    """``linalg.trace(dtype=)`` sums the tangent in the dtype of its primal."""
    value = strict.asarray(np.arange(12, dtype=np.float64).reshape(3, 4))
    tangent_array = np.linspace(-1.0, 1.0, 12).reshape(3, 4)

    def function(x: Any) -> Any:
        return x.__array_namespace__().linalg.trace(x, offset=1, dtype=dtype)

    if lifetime == "staged":
        program = ad.stage(function, specs=(ad.ArraySpec(value.shape, value.dtype),))
        function = ad.StagedProgram.from_dict(program.to_dict())
    output, output_tangent = ad.jvp(function)(value, tangents=strict.asarray(tangent_array))

    assert output.dtype == dtype
    assert output_tangent.dtype == dtype
    assert_allclose(np.asarray(output_tangent), np.trace(tangent_array, offset=1), rtol=1e-6)


def _upper_triangle_inputs() -> tuple[
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
]:
    value = np.array(
        [
            [4.0, 1.0 + 0.4j, -0.3 + 0.2j],
            [11.0 - 7.0j, 2.0, 0.5 - 0.6j],
            [-9.0 + 3.0j, 8.0 + 2.0j, -1.0],
        ],
        dtype=complex,
    )
    tangent = np.array(
        [
            [0.4, -0.2 + 0.1j, 0.3 - 0.05j],
            [13.0 - 2.0j, -0.1, 0.25 + 0.2j],
            [-5.0 + 6.0j, 7.0 - 4.0j, 0.2],
        ],
        dtype=complex,
    )
    return value, tangent


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(lambda x: np.linalg.pinv(x, hermitian=True), id="pinv"),
        pytest.param(
            lambda x: np.linalg.svd(x, compute_uv=False, hermitian=True),
            id="svdvals",
        ),
    ],
)
def test_hermitian_pinv_and_singular_values_read_the_lower_triangle(function: Any) -> None:
    """``hermitian=True`` differentiates the matrix NumPy reads from ``UPLO='L'``."""
    value, tangent = _upper_triangle_inputs()
    value = value.conj().T

    step = 1e-6
    central = (function(value + step * tangent) - function(value - step * tangent)) / (2 * step)
    _output, directional = ad.jvp(function)(value, tangents=tangent)
    assert_allclose(directional, central, rtol=2e-6, atol=2e-7)

    cotangent = np.linspace(-0.6, 0.9, directional.size).reshape(directional.shape)
    _output, pullback = ad.vjp(function)(value)
    try:
        gradient = pullback(cotangent.astype(directional.dtype))
    finally:
        pullback.close()

    assert_allclose(
        np.real(np.vdot(cotangent, directional)),
        np.real(np.vdot(gradient, tangent)),
        rtol=2e-6,
        atol=2e-7,
    )
    assert_allclose(np.triu(gradient, 1), 0.0, atol=2e-10)


def _namespace(value: Any) -> Any:
    return value.__array_namespace__()


_MATRIX = np.array([[2.0, 0.3, -0.1], [0.4, 1.5, 0.2], [-0.3, 0.1, 1.8]])
_SYMMETRIC = _MATRIX @ _MATRIX.T


@pytest.mark.parametrize(
    ("array_api_loss", "numpy_loss", "value"),
    [
        pytest.param(
            lambda a: _namespace(a).sum(_namespace(a).linalg.svdvals(a) ** 2),
            lambda a: np.sum(np.linalg.svd(a, compute_uv=False) ** 2),
            _MATRIX,
            id="svdvals",
        ),
        pytest.param(
            lambda a: _namespace(a).sum(_namespace(a).linalg.svd(a, full_matrices=False)[0] ** 3),
            lambda a: np.sum(np.linalg.svd(a, full_matrices=False)[0] ** 3),
            _MATRIX,
            id="svd",
        ),
        pytest.param(
            lambda a: _namespace(a).sum(_namespace(a).linalg.eigvalsh(a) ** 3),
            lambda a: np.sum(np.linalg.eigvalsh(a) ** 3),
            _SYMMETRIC,
            id="eigvalsh",
        ),
        pytest.param(
            lambda a: _namespace(a).sum(_namespace(a).linalg.eigh(a)[1][..., 0] ** 4),
            lambda a: np.sum(np.linalg.eigh(a)[1][..., 0] ** 4),
            _SYMMETRIC,
            id="eigh",
        ),
        pytest.param(
            lambda a: _namespace(a).sum(_namespace(a).linalg.cholesky(a) ** 3),
            lambda a: np.sum(np.linalg.cholesky(a) ** 3),
            _SYMMETRIC,
            id="cholesky",
        ),
    ],
)
def test_array_api_decomposition_hvp_builds_identities_in_the_trace(
    array_api_loss: Any,
    numpy_loss: Any,
    value: np.ndarray[Any, Any],
) -> None:
    """Forward-over-reverse mixes rule constants with traced primals."""
    direction = np.array([[0.1, -0.2, 0.3], [0.2, 0.1, -0.4], [0.5, 0.3, 0.2]])
    direction = direction + direction.T

    _gradient, product = ad.hvp(array_api_loss)(
        strict.asarray(value),
        vectors=strict.asarray(direction),
    )
    _reference_gradient, reference = ad.hvp(numpy_loss)(value, vectors=direction)

    assert_allclose(np.asarray(product), reference, rtol=1e-10, atol=1e-12)


def test_array_api_complex_svd_gradient_matches_numpy() -> None:
    """The complex SVD transpose applies its phase gauge under every provider."""
    value = _MATRIX + 1j * _MATRIX.T[::-1]

    def array_api_loss(a: Any) -> Any:
        xp = _namespace(a)
        return xp.sum(xp.real(xp.linalg.svd(a, full_matrices=False)[0]))

    def numpy_loss(a: Any) -> Any:
        return np.sum(np.real(np.linalg.svd(a, full_matrices=False)[0]))

    assert_allclose(
        np.asarray(ad.grad(array_api_loss)(strict.asarray(value))),
        ad.grad(numpy_loss)(value),
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize("argnums", [1, (0, 1)], ids=["rtol", "matrix-and-rtol"])
def test_array_api_pinv_is_locally_constant_in_a_traced_rtol(
    argnums: int | tuple[int, int],
) -> None:
    """A traced ``rtol`` is a second pinv operand whose tangent contributes nothing."""
    tall = np.array([[2.0, 1.0], [1.0, 3.0], [0.5, 0.25]])
    direction = np.linspace(-1.0, 1.0, tall.size).reshape(tall.shape)

    def total(value: Any, rtol: Any) -> Any:
        namespace = _namespace(value)
        return namespace.sum(namespace.linalg.pinv(value, rtol=rtol))

    def directional(value: Any, rtol: Any, matrix_tangent: Any) -> Any:
        rtol_tangent = rtol * 0 + 1
        tangents = rtol_tangent if argnums == 1 else (matrix_tangent, rtol_tangent)
        return ad.jvp(total, argnums=argnums)(value, rtol, tangents=tangents)[1]

    matrix_spec = ad.ArraySpec(tall.shape, np.float64)
    staged = ad.stage(directional, specs=(matrix_spec, ad.ArraySpec((), np.float64), matrix_spec))
    _value, expected = ad.jvp(lambda a: np.sum(np.linalg.pinv(a, rtol=0.1)))(
        tall,
        tangents=direction,
    )

    result = staged(tall, np.array(0.1), direction)

    assert_allclose(result, 0.0 if argnums == 1 else expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize(
    ("function", "value", "match"),
    [
        pytest.param(
            lambda x: np.linalg.qr(x, mode="complete"),
            np.arange(15.0).reshape(5, 3) + np.eye(5, 3),
            "provider-dependent null-space columns",
            id="tall-complete-qr",
        ),
        pytest.param(
            lambda x: np.linalg.svd(x, hermitian=True),
            np.array([[3.0, 1.0], [1.0, 2.0]]),
            "hermitian=False",
            id="hermitian-svd",
        ),
        pytest.param(
            lambda x: np.linalg.svd(x, full_matrices=True),
            np.arange(15.0).reshape(5, 3) + np.eye(5, 3),
            "full_matrices=False",
            id="rectangular-full-svd",
        ),
    ],
)
def test_nonunique_linalg_derivatives_report_public_boundaries(
    function: Any,
    value: np.ndarray[Any, Any],
    match: str,
) -> None:
    """Non-unique provider choices are outside the public derivative contract."""
    tangent = np.linspace(-0.2, 0.4, value.size).reshape(value.shape)
    with pytest.raises(NotImplementedError, match=match):
        ad.jvp(function)(value, tangents=tangent)

    output, pullback = ad.vjp(function)(value)
    try:
        with pytest.raises(NotImplementedError, match=match):
            pullback(type(output)(*(np.ones_like(leaf) for leaf in output)))
    finally:
        pullback.close()


def _cofactor_matrix(value: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    size = value.shape[-1]
    cofactors = np.empty_like(value)
    for row in range(size):
        for column in range(size):
            minor = np.delete(np.delete(value, row, axis=0), column, axis=1)
            cofactors[row, column] = (-1) ** (row + column) * np.linalg.det(minor)
    return cofactors


@st.composite
def _rank_deficient_matrices(draw: st.DrawFn) -> np.ndarray[Any, Any]:
    """Draw a product of an n-by-(n-1) and an (n-1)-by-n factor.

    Entries stay well scaled. For a nearly singular product of tiny entries
    the regular ``inv`` formula overflows, a conditioning limit separate from
    exact singularity.
    """
    size = draw(st.integers(2, 5))
    magnitudes = st.floats(0.125, 2.0)
    elements = st.one_of(st.just(0.0), magnitudes, magnitudes.map(lambda value: -value))
    left = draw(hnp.arrays(np.float64, (size, size - 1), elements=elements))
    right = draw(hnp.arrays(np.float64, (size - 1, size), elements=elements))
    return left @ right


@given(value=_rank_deficient_matrices())
@example(value=np.array([[1.0, 2.0], [2.0, 4.0]]))
@example(value=np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]]))
def test_det_gradient_is_the_cofactor_matrix_at_singular_inputs(
    value: np.ndarray[Any, Any],
) -> None:
    """The determinant is a polynomial, so singular inputs still have a gradient."""
    expected = _cofactor_matrix(value)

    gradient = ad.grad(np.linalg.det)(value)

    scale = max(1.0, float(np.max(np.abs(expected))))
    assert_allclose(gradient, expected, rtol=0, atol=1e-10 * scale)


def _cofactor_derivative(
    value: np.ndarray[Any, Any],
    direction: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Differentiate the cofactor matrix along ``direction``.

    The cofactors are polynomials of degree ``n - 1``, for which this
    five-point stencil is exact up to degree four.
    """
    samples = {step: _cofactor_matrix(value + step * direction) for step in (-2, -1, 1, 2)}
    return (samples[-2] - 8.0 * samples[-1] + 8.0 * samples[1] - samples[2]) / 12.0


type _DirectedPoint = tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]


@st.composite
def _directed_singular_points(draw: st.DrawFn) -> _DirectedPoint:
    """Draw an exactly singular integer product and two integer directions."""
    size = draw(st.integers(2, 4))
    rank = draw(st.integers(0, size - 1))
    elements = st.integers(-3, 3).map(float)
    left = draw(hnp.arrays(np.float64, (size, rank), elements=elements))
    right = draw(hnp.arrays(np.float64, (rank, size), elements=elements))
    value = left @ right
    assume(np.linalg.det(value) == 0.0)
    directions = hnp.arrays(np.float64, value.shape, elements=st.integers(-2, 2).map(float))
    return value, draw(directions), draw(directions)


# The rule takes its singular path only where det is exactly zero. The last
# row is the second minus the first, and every pivot ratio is a power of two,
# so LU eliminates it exactly whatever the LAPACK build's rounding or FMA use.
# No cofactor is zero.
_EXACTLY_SINGULAR = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 1.0], [1.0, -1.0, -2.0]])


def _directed(value: np.ndarray[Any, Any]) -> _DirectedPoint:
    assert np.linalg.det(value) == 0.0
    direction = np.linspace(-1.0, 1.0, value.size).reshape(value.shape)
    return value, direction, np.flip(direction) + 0.5


@given(point=_directed_singular_points())
@example(point=_directed(np.array([[1.0, 2.0], [2.0, 4.0]])))
@example(point=_directed(_EXACTLY_SINGULAR))
@example(point=_directed(np.outer([1.0, 2.0, 4.0], [1.0, -1.0, 2.0])))
def test_det_nested_derivatives_are_exact_at_singular_inputs(point: _DirectedPoint) -> None:
    """Forward-over-reverse and forward-over-forward differentiate the cofactors."""
    value, direction, other = point
    expected = _cofactor_derivative(value, direction)
    scale = max(1.0, float(np.max(np.abs(expected))))

    _gradient, product = ad.hvp(np.linalg.det)(value, vectors=direction)
    _tangent, second = ad.jvp(lambda m: ad.jvp(np.linalg.det)(m, tangents=other)[1])(
        value,
        tangents=direction,
    )

    assert_allclose(product, expected, rtol=0, atol=1e-10 * scale)
    assert_allclose(second, np.sum(expected * other), rtol=0, atol=1e-10 * scale * other.size)


@pytest.mark.parametrize("frontend", ["numpy", "array-api"])
def test_det_derivatives_select_singular_batch_elements(frontend: str) -> None:
    """Singular and regular batch elements each get their exact derivatives."""
    batch = np.stack(
        [
            _EXACTLY_SINGULAR,
            np.eye(3),
            np.array([[2.0, 1.0, 0.0], [0.5, 3.0, 1.0], [1.0, 0.0, 1.0]]),
        ]
    )
    direction = np.linspace(-1.0, 1.0, batch.size).reshape(batch.shape)
    asarray = strict.asarray if frontend == "array-api" else np.asarray

    def total(value: Any) -> Any:
        namespace = _namespace(value) if frontend == "array-api" else np
        return namespace.sum(namespace.linalg.det(value))

    gradient = ad.grad(total)(asarray(batch))
    _gradient, product = ad.hvp(total)(asarray(batch), vectors=asarray(direction))

    assert_allclose(np.asarray(gradient), [_cofactor_matrix(matrix) for matrix in batch])
    assert_allclose(
        np.asarray(product),
        [_cofactor_derivative(*pair) for pair in zip(batch, direction, strict=True)],
        rtol=0,
        atol=1e-12,
    )
