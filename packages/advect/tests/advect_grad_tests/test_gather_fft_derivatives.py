"""Gather and FFT derivative contracts beyond the conformance draws."""

from __future__ import annotations

import tracemalloc
from typing import TYPE_CHECKING, Any

import array_api_strict as strict
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as hnp
from numpy.testing import assert_allclose

import advect as ad

if TYPE_CHECKING:
    from collections.abc import Callable


def _round_tripped_staged_gradient(
    loss: Callable[..., Any],
    *values: Any,
    argnums: int = 0,
) -> tuple[Any, Any, Any]:
    specs = tuple(ad.ArraySpec(value.shape, value.dtype) for value in values)
    program = ad.stage(loss, specs=specs)
    gradient = ad.grad(program, argnums=argnums)
    restored = ad.StagedProgram.from_dict(gradient.to_dict())
    return (
        ad.grad(loss, argnums=argnums)(*values),
        gradient(*values),
        restored(*values),
    )


def test_take_along_axis_gradients_accumulate_duplicate_indices() -> None:
    value = strict.asarray(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        dtype=strict.float64,
    )
    indices = strict.asarray([[2, 0, 2], [1, 1, 3]], dtype=strict.int64)
    weights = strict.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=strict.float64)
    expected = np.asarray([[2.0, 0.0, 4.0, 0.0], [0.0, 9.0, 0.0, 6.0]])

    def loss(argument: Any, index: Any) -> Any:
        namespace = argument.__array_namespace__()
        gathered = namespace.take_along_axis(argument, index, axis=1)
        return namespace.sum(gathered * weights)

    for actual in _round_tripped_staged_gradient(loss, value, indices):
        assert type(actual) is type(value)
        assert actual.dtype == value.dtype
        assert_allclose(np.asarray(actual), expected, rtol=1e-12, atol=1e-12)


def test_take_along_axis_pullback_reduces_broadcast_axes_and_duplicates() -> None:
    value = strict.asarray([[1.0, 2.0, 3.0]], dtype=strict.float64)
    indices = strict.asarray([[0, 0], [2, 0]], dtype=strict.int64)

    def loss(argument: Any) -> Any:
        namespace = argument.__array_namespace__()
        return namespace.sum(namespace.take_along_axis(argument, indices, axis=1))

    for actual in _round_tripped_staged_gradient(loss, value):
        assert_allclose(np.asarray(actual), np.asarray([[3.0, 0.0, 1.0]]))


_LONG_AXIS = 200
_INT8_INDICES = [100, 127, -1, -128, 5, 5]


def _numpy_take_reads_uint64() -> bool:
    try:
        np.take(np.zeros(1), np.zeros(1, dtype=np.uint64))
    except TypeError:
        return False
    return True


@pytest.mark.parametrize("staged", [False, True], ids=["dynamic", "staged"])
@pytest.mark.parametrize(
    ("namespace", "mode", "indices", "dtype"),
    [
        pytest.param(np, "along", [3, 199, 0, 5, 5], "uint64", id="uint64-take_along_axis"),
        pytest.param(np, "along", _INT8_INDICES, "int8", id="int8-take_along_axis"),
        pytest.param(np, "raise", _INT8_INDICES, "int8", id="int8-take"),
        pytest.param(np, "wrap", _INT8_INDICES, "int8", id="int8-take-wrap"),
        pytest.param(np, "clip", _INT8_INDICES, "int8", id="int8-take-clip"),
        pytest.param(
            np,
            "wrap",
            [2**64 - 1, 3, 5, 5],
            "uint64",
            id="uint64-take-wrap",
            marks=pytest.mark.skipif(
                not _numpy_take_reads_uint64(), reason="NumPy 2.0 take rejects uint64 indices"
            ),
        ),
        pytest.param(
            strict, "along", [3, 199, 0, 5, 5], "uint64", id="array-api-uint64-take_along_axis"
        ),
        pytest.param(
            strict, "along", [100, 127, 0, 5, 5], "int8", id="array-api-int8-take_along_axis"
        ),
        pytest.param(strict, "raise", [100, 127, 0, 5, 5], "int8", id="array-api-int8-take"),
    ],
)
def test_gather_derivatives_read_an_integer_index_as_the_provider_does(
    namespace: Any, mode: str, indices: list[int], dtype: str, *, staged: bool
) -> None:
    """Gathers read any integer index as ``intp``, and so do their derivatives.

    Index arithmetic in a narrow index dtype overflows on a long axis. NumPy
    promotes ``uint64`` with ``int64`` flat offsets to ``float64``, and Array
    API promotion rejects that pair rather than widening it.
    """

    def select(v: Any) -> Any:
        xp = np if namespace is np else v.__array_namespace__()
        index = xp.asarray(indices, dtype=getattr(xp, dtype))
        if mode == "along":
            return xp.take_along_axis(v, index, axis=0)
        return np.take(v, index, mode=mode) if xp is np else xp.take(v, index, axis=0)

    def cube(v: Any) -> Any:
        xp = np if namespace is np else v.__array_namespace__()
        return xp.sum(select(v) ** 3)

    def maybe_stage(function: Any) -> Any:
        return ad.stage(function, primal) if staged else function

    positions = np.asarray(select(namespace.asarray(np.arange(_LONG_AXIS))))
    value = np.linspace(0.1, 2.0, _LONG_AXIS)
    tangent = np.linspace(-1.0, 1.0, _LONG_AXIS)
    primal = namespace.asarray(value, dtype=namespace.float64)
    direction = namespace.asarray(tangent, dtype=namespace.float64)

    forward = ad.jvp(maybe_stage(select))(primal, tangents=direction)[1]
    gradient = ad.grad(maybe_stage(cube))(primal)
    _, product = ad.hvp(maybe_stage(cube))(primal, vectors=direction)

    def scatter(contributions: np.ndarray) -> np.ndarray:
        result = np.zeros(_LONG_AXIS)
        np.add.at(result, positions, contributions)
        return result

    np.testing.assert_array_equal(np.asarray(forward), tangent[positions])
    assert_allclose(np.asarray(gradient), scatter(3.0 * value[positions] ** 2))
    assert_allclose(np.asarray(product), scatter(6.0 * value[positions] * tangent[positions]))


@pytest.mark.parametrize("axis", [0, -1])
@pytest.mark.parametrize("indices", [0, [0, -1, 0]], ids=["scalar", "vector"])
def test_take_along_the_axis_of_a_rank_zero_input_gathers_its_element(
    indices: Any, axis: int
) -> None:
    """NumPy takes along the axis 0 or -1 of a 0-D array from its one element."""
    value = np.asarray(2.0)

    def gather(x: Any) -> Any:
        return np.sum(np.take(x, indices, axis=axis) ** 2)

    count = np.size(indices)
    tangent = np.asarray(3.0)

    assert float(ad.grad(gather)(value)) == 4.0 * count
    assert float(ad.jvp(gather)(value, tangents=tangent)[1]) == 12.0 * count
    assert float(ad.hessian(gather)(value)) == 2.0 * count


@pytest.mark.parametrize(
    "derivative",
    [
        pytest.param(ad.grad(lambda v: np.sum(np.sort(v) * v)), id="sort-grad"),
        pytest.param(ad.grad(np.median), id="median-grad"),
        pytest.param(lambda x: ad.jvp(lambda v: np.partition(v, 7))(x, tangents=x), id="partition"),
        pytest.param(
            lambda x: ad.jvp(lambda v: np.take_along_axis(v, np.argsort(v), 0))(x, tangents=x),
            id="take_along_axis",
        ),
    ],
)
def test_ordering_derivatives_gather_in_linear_memory(derivative: Any) -> None:
    """A one-hot gather basis would hold the square of the axis length."""
    value = np.random.default_rng(0).normal(size=4096)
    derivative(value)
    tracemalloc.start()
    try:
        derivative(value)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert peak < 64 * value.nbytes


_POSITIONS = np.random.default_rng(1).integers(0, 4096, size=4096)
_ROWS = np.random.default_rng(2).integers(0, 1024, size=(4, 1024))


def _take_loss(v: Any) -> Any:
    return np.sum(np.take(v, _POSITIONS) ** 2)


def _strict_take_loss(v: Any) -> Any:
    xp = v.__array_namespace__()
    return xp.sum(xp.take(v, xp.asarray(_POSITIONS), axis=0) ** 2)


def _rows_loss(v: Any) -> Any:
    return np.sum(np.take_along_axis(np.reshape(v, (4, 1024)), _ROWS, axis=1) ** 2)


@pytest.mark.parametrize(
    "prepare",
    [
        pytest.param(lambda x: ad.stage(ad.grad(_take_loss), x), id="staged-take"),
        pytest.param(
            lambda x: (
                lambda v: ad.vjp_program(ad.stage(_take_loss, x))(v, cotangent=np.asarray(1.0))
            ),
            id="vjp-program-take",
        ),
        pytest.param(lambda x: ad.stage(ad.grad(_rows_loss), x), id="staged-take_along_axis"),
        pytest.param(
            lambda x: ad.stage(ad.grad(lambda v: np.sum(np.sort(v) * v)), x), id="staged-sort"
        ),
        pytest.param(
            lambda _x: lambda v: ad.hvp(_take_loss)(v, vectors=v), id="traced-cotangent-take"
        ),
        pytest.param(
            lambda x: (
                lambda v: ad.stage(ad.grad(_strict_take_loss), strict.asarray(x))(strict.asarray(v))
            ),
            id="array-api-staged-take",
        ),
    ],
)
def test_staged_and_nested_gathers_scatter_in_linear_memory(prepare: Any) -> None:
    """Reverse mode scatters a gather's cotangent without a one-hot basis.

    The basis would hold the product of the gathered and source lengths,
    here 4096 squared, whether it is staged or recorded under an outer trace.
    """
    value = np.random.default_rng(0).normal(size=4096)
    derivative = prepare(value)
    derivative(value)
    tracemalloc.start()
    try:
        derivative(value)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert peak < 64 * value.nbytes


@st.composite
def _gathers(draw: st.DrawFn) -> tuple[np.ndarray, Callable[[Any], Any]]:
    """Draw a source and one NumPy gather of it, with repeated and negative indices."""
    kind = draw(st.sampled_from(("take", "take_along_axis", "sort")))
    shape = draw(hnp.array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=4))
    value = draw(hnp.arrays(np.float64, shape, elements=st.floats(-2.0, 2.0)))
    axis = draw(st.integers(-len(shape), len(shape) - 1))
    length = shape[axis]
    if kind == "sort":
        return value, lambda v: np.sort(v, axis=axis)
    if kind == "take_along_axis":
        index_shape = tuple(
            draw(st.integers(0, 3))
            if dimension == axis % len(shape)
            else draw(st.sampled_from((1, size)) if size > 1 else st.integers(1, 3))
            for dimension, size in enumerate(shape)
        )
        along = draw(hnp.arrays(np.int64, index_shape, elements=st.integers(-length, length - 1)))
        return value, lambda v: np.take_along_axis(v, along, axis=axis)
    mode = draw(st.sampled_from(("raise", "wrap", "clip")))
    take_axis = draw(st.sampled_from((axis, None)))
    extent = value.size if take_axis is None else length
    bounds = (-extent, extent - 1) if mode == "raise" else (-3 * extent, 3 * extent)
    index_shape = draw(hnp.array_shapes(min_dims=0, max_dims=2, min_side=0, max_side=3))
    indices = draw(hnp.arrays(np.int64, index_shape, elements=st.integers(*bounds)))
    return value, lambda v: np.take(v, indices, axis=take_axis, mode=mode)


@given(gather=_gathers())
@settings(deadline=None)
def test_staged_gather_gradients_equal_dynamic_ones(
    gather: tuple[np.ndarray, Callable[[Any], Any]],
) -> None:
    """Both lifetimes scatter a gather's cotangent through one operation, bit for bit."""
    value, select = gather

    def loss(v: Any) -> Any:
        return np.sum(np.sin(select(v)))

    np.testing.assert_array_equal(ad.stage(ad.grad(loss), value)(value), ad.grad(loss)(value))


@given(
    value=hnp.arrays(
        np.float64,
        hnp.array_shapes(min_dims=1, max_dims=2, min_side=1, max_side=5),
        elements=st.floats(-2.0, 2.0),
    ),
    data=st.data(),
)
@settings(deadline=None)
def test_array_api_gather_gradients_scatter_without_add_at(
    value: np.ndarray, data: st.DataObject
) -> None:
    """A provider without ``add.at`` scatters identically in both lifetimes.

    Its sorted-run sums agree with NumPy's ``add.at`` up to rounding.
    """
    axis = data.draw(st.integers(0, value.ndim - 1), label="axis")
    count = data.draw(st.integers(0, 8), label="count")
    positions = data.draw(
        hnp.arrays(np.int64, (count,), elements=st.integers(0, value.shape[axis] - 1)),
        label="positions",
    )

    def loss(v: Any) -> Any:
        xp = v.__array_namespace__()
        return xp.sum(xp.sin(xp.take(v, xp.asarray(positions), axis=axis)))

    source = strict.asarray(value)
    dynamic = ad.grad(loss)(source)
    np.testing.assert_array_equal(
        np.asarray(ad.stage(ad.grad(loss), source)(source)), np.asarray(dynamic)
    )
    assert_allclose(
        np.asarray(dynamic),
        ad.grad(lambda v: np.sum(np.sin(np.take(v, positions, axis=axis))))(value),
        rtol=1e-14,
        atol=1e-14,
    )


@pytest.mark.filterwarnings("ignore:`axes` should not be `None`:DeprecationWarning")
def test_array_api_irfftn_explicit_shape_uses_runtime_output_shape() -> None:
    """Explicit inverse lengths determine the Array API result shape."""
    value = strict.asarray([[1.0 + 0.0j, 0.4 - 0.2j, -0.3 + 0.0j]], dtype=strict.complex128)
    tangent = strict.asarray(
        [[0.2 + 0.1j, -0.5 + 0.3j, 0.7 - 0.4j]],
        dtype=strict.complex128,
    )
    cotangent = strict.asarray([[0.6, -0.4, 0.2, 0.9]], dtype=strict.float64)

    def function(x: Any) -> Any:
        return x.__array_namespace__().fft.irfftn(x, s=(4,), axes=None, norm="ortho")

    output, output_tangent = ad.jvp(function)(value, tangents=tangent)
    assert type(output) is type(value)
    assert output.shape == (1, 4)
    assert_allclose(
        np.asarray(output),
        np.fft.irfftn(np.asarray(value), s=(4,), axes=None, norm="ortho"),
        rtol=2e-9,
        atol=2e-10,
    )

    reverse_output, pullback = ad.vjp(function)(value)
    try:
        gradient = pullback(cotangent)
    finally:
        pullback.close()

    assert type(gradient) is type(value)
    assert_allclose(np.asarray(reverse_output), np.asarray(output), rtol=2e-9, atol=2e-10)
    assert_allclose(
        np.real(np.vdot(np.asarray(cotangent), np.asarray(output_tangent))),
        np.real(np.vdot(np.asarray(gradient), np.asarray(tangent))),
        rtol=2e-9,
        atol=2e-10,
    )
