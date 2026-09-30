"""Structural transposes of linear JVP rules: dense-map parity and real-linearity checks."""

from __future__ import annotations

import math
import warnings
from typing import Any

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from numpy.testing import assert_allclose

import advect as ad


@pytest.mark.parametrize("leaf", ["square", "sum", "negative", "transpose"])
def test_structural_transpose_rejects_a_nonlinear_custom_primitive_on_the_tangent(
    leaf: str,
) -> None:
    # A custom primitive whose name ends like a linear built-in is still an
    # opaque operation: its linearity cannot be inferred from the name.
    @ad.primitive(name=f"tests.structural_linearity.nonlinear.{leaf}")
    def square(x: np.ndarray) -> np.ndarray:
        return x * x

    @square.def_transpose
    def square_transpose(
        cotangent: np.ndarray,
        primals: tuple[np.ndarray, ...],
        output: np.ndarray,
    ) -> tuple[np.ndarray]:
        del output
        return (2 * primals[0] * cotangent,)

    @ad.primitive(name=f"tests.structural_linearity.jvp_only_{leaf}")
    def cube(x: np.ndarray) -> np.ndarray:
        return x**3

    @cube.def_jvp
    def cube_jvp(
        output: np.ndarray,
        primals: tuple[np.ndarray, ...],
        tangents: tuple[np.ndarray | None, ...],
    ) -> np.ndarray:
        del output, primals
        tangent = tangents[0]
        assert tangent is not None
        return square(tangent)

    with pytest.raises(ValueError, match="is not real-linear: uses unsupported tangent-dependent"):
        ad.grad(lambda value: np.sum(cube(value)))(np.array([1.0, 2.0]))


type _LinearMap = tuple[str, tuple[int, ...], Any]


def _flattening_map(draw: st.DrawFn) -> _LinearMap:
    """Draw a ravel or reshape whose order depends on a strided or broadcast view."""
    rank = draw(st.integers(1, 3))
    shape = tuple(draw(st.lists(st.integers(1, 3), min_size=rank, max_size=rank)))
    axes = tuple(draw(st.permutations(range(rank))))
    steps = tuple(draw(st.lists(st.sampled_from([1, -1, 2, -2]), min_size=rank, max_size=rank)))
    broadcast_axis = draw(st.none() | st.integers(0, rank))
    reshape = draw(st.booleans())
    order = draw(st.sampled_from("CFA" if reshape else "CFAK"))

    def flatten(v: Any) -> Any:
        view = np.transpose(v, axes)[tuple(slice(None, None, step) for step in steps)]
        if broadcast_axis is not None:
            view = np.expand_dims(view, broadcast_axis)
            extents = list(np.shape(view))
            extents[broadcast_axis] = 2
            view = np.broadcast_to(view, tuple(extents))
        if reshape:
            return np.reshape(view, (-1,), order=order)
        return np.ravel(view, order=order)

    operation = "reshape" if reshape else "ravel"
    label = f"{operation}(order={order}) of {axes=}, {steps=}, {broadcast_axis=}"
    return label, shape, flatten


@st.composite
def _structural_linear_maps(draw: st.DrawFn) -> _LinearMap:
    """Draw a trace, diagonal, diff, gradient or flattening call with its input shape."""
    kind = draw(st.sampled_from(["trace", "diagonal", "diff", "gradient", "flatten"]))
    if kind == "flatten":
        return _flattening_map(draw)
    if kind in {"trace", "diagonal"}:
        rank = draw(st.integers(2, 3))
        shape = tuple(draw(st.lists(st.integers(1, 4), min_size=rank, max_size=rank)))
        axis1, axis2, *_ = draw(st.permutations(range(rank)))
        offset = draw(st.integers(-4, 4))
        operation = np.trace if kind == "trace" else np.diagonal
        return (
            f"{kind}(offset={offset}, axis1={axis1}, axis2={axis2})",
            shape,
            lambda v: operation(v, offset, axis1, axis2),
        )
    if kind == "diff":
        length = draw(st.integers(1, 5))
        order = draw(st.integers(1, 3))
        prepended = draw(st.integers(0, 2))
        appended = draw(st.integers(0, 2))

        def difference(v: Any) -> Any:
            # Trailing input entries become traced prepend and append operands.
            boundaries = {
                name: v[start : start + count]
                for name, start, count in (
                    ("prepend", length, prepended),
                    ("append", length + prepended, appended),
                )
                if count
            }
            return np.diff(v[:length], n=order, **boundaries)

        return (
            f"diff(n={order}, prepend={prepended}, append={appended})",
            (length + prepended + appended,),
            difference,
        )
    edge_order = draw(st.integers(1, 2))
    length = draw(st.integers(edge_order + 1, 6))
    axis = draw(st.integers(0, 1))
    shape = (length, 2) if axis == 0 else (2, length)
    return (
        f"gradient(axis={axis}, edge_order={edge_order})",
        shape,
        lambda v: np.gradient(v, axis=axis, edge_order=edge_order),
    )


def _dense_matrix(function: Any, shape: tuple[int, ...]) -> Any:
    """Evaluate a linear NumPy function on the standard basis."""
    size = math.prod(shape)
    columns = [np.ravel(function(vector)) for vector in np.eye(size).reshape((size, *shape))]
    return np.stack(columns, axis=1).reshape((-1, size))


def _assert_dense_linear_map(
    linear_map: _LinearMap,
    tangent: Any,
    cotangent: Any,
    direction: Any,
) -> None:
    """Check the JVP against the dense map and the pullback against its transpose.

    The pullback is also run inside an enclosing forward trace. Callers pass a
    Fortran-ordered ``tangent`` for a C-ordered primal, so a rule that reads a
    layout-dependent order from the tangent is caught.
    """
    label, shape, function = linear_map
    matrix = _dense_matrix(function, shape)
    value = np.linspace(-1.0, 1.0, math.prod(shape)).reshape(shape)

    _output, output_tangent = ad.jvp(function)(value, tangents=tangent)
    assert_allclose(np.ravel(output_tangent), matrix @ np.ravel(tangent), atol=1e-12, err_msg=label)

    def pull(seed: Any) -> Any:
        _output, pullback = ad.vjp(function)(value)
        return pullback(seed)

    expected = (matrix.T @ np.ravel(cotangent)).reshape(shape)
    expected_tangent = (matrix.T @ np.ravel(direction)).reshape(shape)
    gradient, gradient_tangent = ad.jvp(pull)(cotangent, tangents=direction)

    assert_allclose(pull(cotangent), expected, atol=1e-12, err_msg=label)
    assert_allclose(gradient, expected, atol=1e-12, err_msg=label)
    assert_allclose(gradient_tangent, expected_tangent, atol=1e-12, err_msg=label)


@given(linear_map=_structural_linear_maps(), data=st.data())
def test_structural_linear_pullbacks_are_traceable_transposes(
    linear_map: _LinearMap,
    data: st.DataObject,
) -> None:
    """Each JVP is the dense map and each pullback its transpose, also when traced."""
    _label, shape, function = linear_map
    output_shape = np.shape(function(np.zeros(shape)))
    elements = st.floats(-2.0, 2.0, allow_subnormal=False)
    tangent = data.draw(hnp.arrays(np.float64, shape, elements=elements))
    cotangent = data.draw(hnp.arrays(np.float64, output_shape, elements=elements))
    direction = data.draw(hnp.arrays(np.float64, output_shape, elements=elements))
    _assert_dense_linear_map(linear_map, np.asfortranarray(tangent), cotangent, direction)


@pytest.mark.parametrize(
    "flatten",
    [
        pytest.param(lambda view: np.ravel(view, order="A"), id="ravel-A"),
        pytest.param(lambda view: np.reshape(view, (-1,), order="A"), id="reshape-A"),
        pytest.param(lambda view: np.ravel(view, order="K"), id="ravel-K"),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        pytest.param(np.transpose, id="fortran"),
        pytest.param(lambda v: np.broadcast_to(v[:1], (3, 4)), id="broadcast"),
        pytest.param(lambda v: np.transpose(np.reshape(v, (2, 3, 2)), (1, 2, 0)), id="permuted"),
        pytest.param(lambda v: np.transpose(v)[:, None, :], id="inserted-axis"),
    ],
)
def test_layout_dependent_flattening_reads_the_primal_layout(flatten: Any, layout: Any) -> None:
    """``order='A'`` and ``'K'`` follow the primal view, not the tangent or cotangent."""
    output_size = np.size(layout(np.zeros((3, 4))))
    seeds = np.linspace(-2.0, 2.0, output_size)
    _assert_dense_linear_map(
        ("layout-dependent flattening", (3, 4), lambda v: flatten(layout(v))),
        np.asfortranarray(np.linspace(1.0, 2.0, 12).reshape(3, 4)),
        seeds,
        np.flip(seeds),
    )


def test_diff_pullback_reaches_traced_prepend_and_append_operands() -> None:
    """Traced boundary operands shift the source window and get cotangents."""
    value = np.array([1.0, 4.0, 9.0, 16.0])
    boundary = np.array([0.5])
    weights = np.array([1.0, 2.0, 3.0, 4.0])

    def weighted(x: Any, prepend: Any, append: Any) -> Any:
        return np.sum(np.diff(x, n=2, prepend=prepend, append=append) * weights)

    gradients = ad.grad(weighted, argnums=(0, 1, 2))(value, boundary, boundary)

    # diff([p, x0, ..., x3, a], n=2) pairs weight k with p, x0..x3, a at k..k+2.
    assert_allclose(gradients[0], [0.0, 0.0, 0.0, -5.0])
    assert_allclose(gradients[1], [1.0])
    assert_allclose(gradients[2], [4.0])


def test_empty_second_difference_vjp_returns_zero_source_cotangent() -> None:
    value = np.array([0.2, 1.3])
    output, pullback = ad.vjp(lambda x: np.diff(x, n=2))(value)
    try:
        gradient = pullback(np.empty(0))
    finally:
        pullback.close()

    assert output.shape == (0,)
    assert_allclose(gradient, np.zeros_like(value))


@pytest.mark.parametrize(
    ("function", "cotangent", "expected"),
    [
        pytest.param(
            lambda a: np.convolve(a, [1.0, 2.0, 3.0], mode="valid"),
            [np.inf, 1.0],
            [np.inf, np.inf, np.inf, 1.0],
            id="convolve-valid",
        ),
        pytest.param(
            lambda a: np.convolve(a, [1.0, 2.0, 3.0], mode="same"),
            [np.nan, 0.0, 0.0, 1.0],
            [np.nan, np.nan, 3.0, 2.0],
            id="convolve-same",
        ),
        pytest.param(
            lambda a: a[1:3],
            [np.inf, 1.0],
            [0.0, np.inf, 1.0, 0.0],
            id="getitem",
        ),
        pytest.param(
            lambda a: np.trace(np.reshape(a, (2, 2))),
            np.inf,
            [np.inf, 0.0, 0.0, np.inf],
            id="trace",
        ),
        pytest.param(
            lambda a: np.diagonal(np.reshape(a, (2, 2))),
            [np.inf, 1.0],
            [np.inf, 0.0, 0.0, 1.0],
            id="diagonal",
        ),
        pytest.param(
            lambda a: np.diag(np.reshape(a, (2, 2)), k=-1),
            [np.inf],
            [0.0, 0.0, np.inf, 0.0],
            id="diag-of-matrix",
        ),
    ],
)
def test_zero_padded_cotangents_keep_non_finite_entries_local(
    function: Any,
    cotangent: float | list[float],
    expected: list[float],
) -> None:
    """Exact zeros never multiply a non-finite cotangent entry."""
    value = np.array([1.0, 2.0, 3.0, 4.0])
    seed = np.array(cotangent)

    def pull(seed: Any) -> Any:
        _output, pullback = ad.vjp(function)(value)
        return pullback(seed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        gradient = pull(seed)
        traced_gradient, _tangent = ad.jvp(pull)(seed, tangents=np.ones_like(seed))

    assert_allclose(gradient, expected)
    assert_allclose(traced_gradient, expected)
