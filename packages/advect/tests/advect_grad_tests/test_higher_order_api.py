"""Tests for public higher-order autodiff APIs."""

from __future__ import annotations

import operator
from typing import Any, cast

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings
from numpy.testing import assert_allclose, assert_array_equal

import advect as ad


def _cubic_loss(x: np.ndarray[Any, Any]) -> np.floating[Any]:
    return cast("np.floating[Any]", np.sum(x**3))


def test_hvp_evaluates_stateful_function_once() -> None:
    calls = 0

    def stateful_loss(x: np.ndarray[Any, Any]) -> np.floating[Any]:
        nonlocal calls
        calls += 1
        return cast("np.floating[Any]", calls * np.sum(x**3))

    x = np.array([0.5, -1.2, 2.0], dtype=np.float64)
    v = np.array([1.0, -0.25, 0.7], dtype=np.float64)

    value, product = ad.hvp(stateful_loss)(x, vectors=v)

    assert calls == 1
    assert value == pytest.approx(float(np.sum(x**3)), rel=1e-10, abs=1e-10)
    assert_allclose(product, 6.0 * x * v, rtol=1e-12)


def test_higher_order_singleton_tuple_argnums_preserve_selection_structure() -> None:
    x = np.array([[0.5], [-1.2]], dtype=np.float64)
    vector = np.array([[1.0], [-0.25]], dtype=np.float64)

    _value, bare_product = ad.hvp(_cubic_loss, argnums=0)(x, vectors=vector)
    _value, tuple_product = ad.hvp(_cubic_loss, argnums=(0,))(x, vectors=(vector,))
    bare_hessian = ad.hessian(_cubic_loss, argnums=0)(x)
    tuple_hessian = ad.hessian(_cubic_loss, argnums=(0,))(x)
    bare_diagonal = ad.hessian_diag(_cubic_loss, argnums=0)(x)
    tuple_diagonal = ad.hessian_diag(_cubic_loss, argnums=(0,))(x)

    assert_allclose(bare_product, 6.0 * x * vector, rtol=1e-12)
    assert_allclose(bare_hessian, np.diag(6.0 * x.reshape(-1)).reshape(x.shape * 2), rtol=1e-12)
    assert_allclose(bare_diagonal, 6.0 * x, rtol=1e-12)

    assert isinstance(tuple_product, tuple)
    assert len(tuple_product) == 1
    assert_allclose(tuple_product[0], bare_product, rtol=1e-12)

    assert isinstance(tuple_hessian, tuple)
    assert len(tuple_hessian) == 1
    assert isinstance(tuple_hessian[0], tuple)
    assert len(tuple_hessian[0]) == 1
    assert tuple_hessian[0][0].shape == x.shape + x.shape
    assert_allclose(tuple_hessian[0][0], bare_hessian, rtol=1e-12)

    assert isinstance(tuple_diagonal, tuple)
    assert len(tuple_diagonal) == 1
    assert tuple_diagonal[0].shape == x.shape
    assert_allclose(tuple_diagonal[0], bare_diagonal, rtol=1e-12)


@pytest.mark.parametrize(
    ("primal", "vectors", "expected"),
    [
        pytest.param(("label", None), (None, None), (None, None), id="tuple"),
        pytest.param({"tag": "abc"}, {"tag": None}, {"tag": None}, id="dict"),
    ],
)
def test_hvp_takes_none_for_a_static_leaf(primal: Any, vectors: Any, expected: Any) -> None:
    x = np.array([0.5, -1.2])
    vector = np.array([1.0, -0.25])

    def loss(value: np.ndarray[Any, Any], _static: object) -> np.floating[Any]:
        return _cubic_loss(value)

    _value, product = ad.hvp(loss, argnums=(0, 1))(x, primal, vectors=(vector, vectors))

    assert_allclose(product[0], 6.0 * x * vector, rtol=1e-12)
    assert product[1] == expected
    with pytest.raises(TypeError, match="tangent provided for a static/untraceable input leaf"):
        ad.hvp(loss, argnums=(0, 1))(x, primal, vectors=(vector, primal))


def test_higher_order_transforms_reject_invalid_calls() -> None:
    x = np.array([0.5, -1.2])
    empty = r"Higher-order APIs require at least one selected argument\."

    with pytest.raises(ValueError, match=empty):
        ad.hvp(_cubic_loss, argnums=())(x, vectors=())
    with pytest.raises(ValueError, match=empty):
        ad.hessian(_cubic_loss, argnums=())(x)
    with pytest.raises(ValueError, match=empty):
        ad.hessian_diag(_cubic_loss, argnums=())(x)
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'vectors'"):
        ad.hvp(_cubic_loss)(x)
    with pytest.raises(ValueError, match="hessian requires real input leaves"):
        ad.hessian(lambda value: np.real(np.sum(value)))(x.astype(np.complex128))
    with pytest.raises(ad.AdvectError, match="gradient structure"):
        ad.hessian(lambda tree: np.sum(tree["value"] ** 2))({"value": x})


_UNIT = st.floats(min_value=-1.0, max_value=1.0, width=32)
# Up to 24 selected coordinates cross the 16-seed pullback batch bound.
_X_SHAPES = st.one_of(
    st.just(()),
    st.tuples(st.integers(0, 20)),
    st.tuples(st.just(2), st.integers(0, 10)),
)


@given(data=st.data())
@settings(deadline=None, max_examples=settings.default.max_examples // 4)
def test_dense_hessians_and_hvp_match_the_closed_form(data: st.DataObject) -> None:
    dtype = data.draw(st.sampled_from([np.float32, np.float64]))
    x: Any = data.draw(hnp.arrays(dtype, _X_SHAPES, elements=_UNIT))
    y = data.draw(hnp.arrays(dtype, st.integers(0, 4).map(lambda size: (size,)), elements=_UNIT))
    python_scalar = x.shape == () and data.draw(st.booleans())
    if python_scalar:
        x = float(x)
    shapes = (np.shape(x), y.shape)
    sizes = (int(np.size(x)), y.size)
    weights = np.linspace(0.5, 1.5, sizes[0]).reshape(shapes[0]).astype(np.asarray(x).dtype)
    coupling = np.linspace(-1.0, 1.0, sizes[1]).astype(dtype)
    vectors = (
        float(data.draw(_UNIT))
        if python_scalar
        else data.draw(hnp.arrays(dtype, shapes[0], elements=_UNIT)),
        data.draw(hnp.arrays(dtype, shapes[1], elements=_UNIT)),
    )

    def objective(a: Any, b: Any) -> Any:
        return (
            np.sum(np.tanh(a) ** 2 * weights)
            + np.sum(a * weights) * np.sum(b * coupling)
            + np.sum(b**3)
        )

    flat_x = np.asarray(x, dtype=np.float64).reshape(-1)
    flat_weights = weights.astype(np.float64).reshape(-1)
    flat_coupling = coupling.astype(np.float64)
    sech2 = 1.0 / np.cosh(flat_x) ** 2
    expected = (
        (
            np.diag(2.0 * flat_weights * sech2 * (sech2 - 2.0 * np.tanh(flat_x) ** 2)),
            np.outer(flat_weights, flat_coupling),
        ),
        (np.outer(flat_coupling, flat_weights), np.diag(6.0 * y.astype(np.float64))),
    )
    # Float32 inputs trace their primal in float32, so the float64 results sit
    # within a few float32 eps (absolute) of the closed form; the worst
    # measured error is about 5 eps (6.3e-7), well inside the 64 eps bound.
    tolerance = 64 * np.finfo(np.float32).eps if dtype is np.float32 else 1e-12

    hessian = ad.hessian(objective, argnums=(0, 1))(x, y)
    diagonal = ad.hessian_diag(objective, argnums=(0, 1))(x, y)
    _value, product = ad.hvp(objective, argnums=(0, 1))(x, y, vectors=vectors)

    for row in range(2):
        for column in range(2):
            block = hessian[row][column]
            if not (python_scalar and row == column == 0):
                # Storage is promoted to float64 whatever the input precision.
                assert isinstance(block, np.ndarray)
                assert block.dtype == np.float64
                assert block.shape == shapes[row] + shapes[column]
            assert_allclose(
                np.reshape(block, (sizes[row], sizes[column])),
                expected[row][column],
                rtol=0,
                atol=tolerance,
            )
        if python_scalar and row == 0:
            assert type(hessian[0][0]) is type(diagonal[0]) is type(product[0]) is float
        else:
            assert isinstance(diagonal[row], np.ndarray)
            assert diagonal[row].dtype == np.float64
            assert diagonal[row].shape == shapes[row]
        assert_array_equal(
            np.reshape(diagonal[row], -1),
            np.diag(np.reshape(hessian[row][row], (sizes[row], sizes[row]))),
        )
        dense_product = sum(
            np.reshape(hessian[row][column], (sizes[row], sizes[column]))
            @ np.reshape(vectors[column], -1).astype(np.float64)
            for column in range(2)
        )
        assert np.shape(product[row]) == shapes[row]
        assert_allclose(
            np.reshape(product[row], -1),
            dense_product,
            rtol=0,
            atol=tolerance * (1.0 + sum(np.sum(np.abs(vector)) for vector in vectors)),
        )


def _update_tail(update: Any) -> Any:
    def through_view(target: Any, operand: Any) -> Any:
        tail = target[1:]
        update(tail, operand[1:])
        return target

    return through_view


@pytest.mark.parametrize(
    ("augmented", "pure"),
    [
        (operator.iadd, operator.add),
        (operator.isub, operator.sub),
        (operator.imul, operator.mul),
        (operator.itruediv, operator.truediv),
        (_update_tail(operator.imul), lambda a, b: np.concatenate([a[:1], a[1:] * b[1:]])),
    ],
    ids=["add", "subtract", "multiply", "divide", "multiply-view"],
)
def test_augmented_assignment_composes_with_second_order_transforms(
    augmented: Any, pure: Any
) -> None:
    """The functionalized in-place update records a pure op in an outer trace."""
    value = np.array([0.5, -1.5, 2.0])

    def mutated(a: Any) -> Any:
        return np.sum(augmented(a.copy(), a * a + 2.0) ** 3)

    def functional(a: Any) -> Any:
        return np.sum(pure(a, a * a + 2.0) ** 3)

    assert_allclose(ad.hessian(mutated)(value), ad.hessian(functional)(value), rtol=1e-12)


def _write_all(a: Any) -> Any:
    out = np.zeros_like(a)
    out[:] = np.sin(a)
    return out


def _write_one(a: Any) -> Any:
    out = np.zeros_like(a)
    out[1] = np.sin(a[1])
    return out


def _add_one(a: Any) -> Any:
    out = np.zeros_like(a)
    out[1:] += np.sin(a[1:])
    return out


@pytest.mark.parametrize(
    ("write", "selected"),
    [(_write_all, slice(None)), (_write_one, slice(1, 2)), (_add_one, slice(1, None))],
    ids=["set-slice", "set-element", "add-slice"],
)
def test_forward_over_forward_differentiates_a_write_into_a_constant(
    write: Any, selected: slice
) -> None:
    """The inner tangent of a written value belongs to the outer trace."""
    value = np.array([0.3, -1.2, 2.0])
    tangent = np.array([1.0, 2.0, 3.0])

    _value, curvature = ad.jvp(lambda a: ad.jvp(write)(a, tangents=tangent)[1])(
        value, tangents=tangent
    )

    expected = np.zeros_like(value)
    expected[selected] = (-np.sin(value) * tangent**2)[selected]
    assert_allclose(curvature, expected, rtol=1e-12)


def _scalar_updates(a: Any) -> Any:
    b = a.copy()
    b += 0.1
    b *= 0.3
    b -= 0.7
    b[1:] /= 0.9
    return b


@pytest.mark.parametrize("depth", ["traced", "nested"])
def test_augmented_assignment_with_a_python_scalar_matches_eager_numpy(depth: str) -> None:
    """A Python-scalar operand keeps NumPy's weak promotion in a traced update."""
    value = np.linspace(-1.0, 1.0, 1001, dtype=np.float32)
    ones = np.ones_like(value)

    def primal(a: Any) -> Any:
        return ad.jvp(_scalar_updates)(a, tangents=ones)[0]

    traced = ad.jvp(primal)(value, tangents=ones)[0] if depth == "nested" else primal(value)

    assert traced.dtype == np.float32
    assert_array_equal(traced, _scalar_updates(value))


def test_augmented_assignment_with_a_weak_scalar_tracer_matches_eager_numpy() -> None:
    """A selected Python-float operand updates a float32 array as eager NumPy does."""
    value = np.linspace(-1.0, 1.0, 1001, dtype=np.float32)

    def shifted(a: Any, scale: Any) -> Any:
        b = a.copy()
        b += scale
        b *= scale
        return b

    traced = ad.jvp(shifted, argnums=(0, 1))(value, 0.1, tangents=(np.ones_like(value), 1.0))[0]

    assert traced.dtype == np.float32
    assert_array_equal(traced, shifted(value, 0.1))


def test_a_copied_python_scalar_is_a_strong_float64_array_in_nested_transforms() -> None:
    # A Python float has no copy(), so the lifted scalar copies as a 0-d float64
    # array, which promotes as a strong value (NEP 50); staging agrees.
    def scaled(v: Any) -> Any:
        w = v.copy()
        w += 1.0
        return w * w * np.float32(2)

    value, gradient = ad.value_and_grad(scaled)(3.0)
    hvp_value, product = ad.hvp(scaled)(3.0, vectors=1.0)
    staged = ad.stage(scaled, 3.0)(3.0)

    assert value.dtype == hvp_value.dtype == staged.dtype == np.float64
    assert (value, gradient, hvp_value, product, staged) == (32.0, 16.0, 32.0, 4.0, 32.0)
