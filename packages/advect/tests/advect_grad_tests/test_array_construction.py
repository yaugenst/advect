"""Contracts for explicit traced array construction and scalar extraction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad
import advect.numpy as advect_numpy
from advect.core import EscapedTracerError, TracingError
from advect.numpy import array as numpy_array, asarray as numpy_asarray


def test_asarray_preserves_a_direct_tracer_and_dtype_conversion() -> None:
    value = np.array([1.0, 2.0], dtype=np.float64)

    gradient = ad.grad(
        lambda x: np.sum(ad.asarray(x, dtype=np.float32) ** 2),
    )(value)

    assert gradient.dtype == value.dtype
    assert_allclose(gradient, 2 * value)


def test_array_constructs_owned_values_that_can_be_functionally_mutated() -> None:
    value = np.array([1.0, 2.0])

    def loss(x: np.ndarray) -> Any:
        result = ad.array(x)
        result += 2
        return np.sum(result)

    assert_allclose(ad.grad(loss)(value), np.ones_like(value))


def test_array_constructs_rectangular_nested_tracer_sequences() -> None:
    value = np.array([1.0, 2.0])

    def loss(x: np.ndarray) -> Any:
        matrix = ad.array([[x[0], x[1]], [x[1], 2 * x[0]]])
        return np.sum(matrix)

    assert_allclose(ad.grad(loss)(value), np.array([3.0, 2.0]))


def test_array_constructs_a_sequence_from_a_lifted_python_scalar() -> None:
    gradient = ad.grad(lambda x: np.sum(ad.array([x, 2 * x])))(3.0)

    assert gradient == pytest.approx(3.0)


def test_asarray_copy_false_rejects_sequence_construction() -> None:
    with pytest.raises(ValueError, match="avoid a copy"):
        ad.grad(lambda x: np.sum(ad.asarray([x[0]], copy=False)))(np.array([1.0]))


def test_explicit_constructors_are_available_from_the_numpy_frontend() -> None:
    value = np.array([1.0, 2.0])
    assert_allclose(ad.grad(lambda x: np.sum(numpy_array(x)))(value), np.ones_like(value))
    assert_allclose(ad.grad(lambda x: np.sum(numpy_asarray(x)))(value), np.ones_like(value))


@pytest.mark.parametrize("constructor", [np.array, np.asarray, np.asanyarray])
def test_numpy_like_constructors_preserve_nested_traced_values(
    constructor: Any,
) -> None:
    value = np.array([1.0, 2.0])

    def loss(x: np.ndarray) -> Any:
        matrix = constructor(
            [[x[0], x[1]], [x[1], 2 * x[0]]],
            dtype=np.float32,
            like=x,
        )
        return np.sum(matrix)

    assert_allclose(ad.grad(loss)(value), np.array([3.0, 2.0]))


def test_numpy_array_like_preserves_ndmin_dtype_and_order() -> None:
    value = np.array([1.0, 2.0])

    primal, tangent = ad.jvp(
        lambda x: np.array(
            [[x[0], x[1]], [x[1], 2 * x[0]]],
            dtype=np.float32,
            order="F",
            ndmin=3,
            like=x,
        )
    )(value, tangents=np.ones_like(value))

    assert primal.shape == (1, 2, 2)
    assert primal.dtype == np.dtype(np.float32)
    assert primal.flags.f_contiguous
    assert_allclose(tangent, np.array([[[1.0, 1.0], [1.0, 2.0]]], dtype=np.float32))


def test_numpy_asarray_like_preserves_direct_tracer_identity() -> None:
    observations: list[bool] = []

    def loss(x: np.ndarray) -> Any:
        converted = np.asarray(x, device="cpu", copy=False, like=x)
        observations.append(converted is x)
        return np.sum(converted)

    value = np.array([1.0, 2.0])
    assert_allclose(ad.grad(loss)(value), np.ones_like(value))
    assert observations == [True]


def test_numpy_array_like_creates_an_owned_copy() -> None:
    def loss(x: np.ndarray) -> Any:
        converted = np.array(x, like=x)
        converted += 2
        return np.sum(converted)

    value = np.array([1.0, 2.0])
    assert_allclose(ad.grad(loss)(value), np.ones_like(value))
    assert_allclose(value, np.array([1.0, 2.0]))


def test_numpy_like_constructor_of_constants_has_zero_derivative() -> None:
    value = np.array([1.0, 2.0])
    primal, tangent = ad.jvp(
        lambda x: np.array([3.0, 4.0], like=x),
    )(value, tangents=np.ones_like(value))

    assert_allclose(primal, np.array([3.0, 4.0]))
    assert_allclose(tangent, np.zeros(2))


def test_numpy_like_constructor_rejects_an_unavoidable_copy() -> None:
    with pytest.raises(ValueError, match="avoid copy"):
        ad.grad(
            lambda x: np.sum(np.asarray([x[0]], copy=False, like=x)),
        )(np.array([1.0]))


def test_advect_numpy_is_a_transparent_numpy_facade() -> None:
    overridden = {"array", "asanyarray", "asarray"}
    for name in dir(np):
        if name.startswith("_") or name in overridden:
            continue
        assert getattr(advect_numpy, name) is getattr(np, name)
    assert_allclose(advect_numpy.arange(3.0), np.arange(3.0))

    value = np.array([1.0, 2.0])
    gradient = ad.grad(
        lambda x: advect_numpy.sum(advect_numpy.array([x[0], 2 * x[1]])),
    )(value)
    assert_allclose(gradient, np.array([1.0, 2.0]))


def test_advect_numpy_constructor_facade_survives_staging() -> None:
    value = np.array([1.0, 2.0])
    program = ad.stage(
        lambda x: advect_numpy.array([x[0], 2 * x[1]]).sum(),
        specs=(ad.ArraySpec(value.shape, value.dtype),),
    )

    assert_allclose(ad.grad(program)(value), np.array([1.0, 2.0]))


def test_unsupported_numpy_coercion_recommends_like() -> None:
    with pytest.raises(TracingError, match=r"np\.array\(values, like=x\)"):
        ad.grad(
            lambda x: np.sum(np.array([x[0], x[1]])),
        )(np.array([1.0, 2.0]))


def test_item_keeps_size_one_results_differentiable() -> None:
    value = np.array([1.0, 2.0])

    gradient = ad.grad(lambda x: np.sum(x * x).item())(value)

    assert_allclose(gradient, 2 * value)


def test_item_supports_flat_and_tuple_indices() -> None:
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])

    assert_allclose(
        ad.grad(lambda x: x.item(2))(matrix),
        np.array([[0.0, 0.0], [1.0, 0.0]]),
    )
    assert_allclose(
        ad.grad(lambda x: x.item((2,)))(matrix),
        np.array([[0.0, 0.0], [1.0, 0.0]]),
    )
    assert_allclose(
        ad.grad(lambda x: x.item((0, 1)))(matrix),
        np.array([[0.0, 1.0], [0.0, 0.0]]),
    )
    assert_allclose(
        ad.grad(lambda x: x.item(1, 0))(matrix),
        np.array([[0.0, 0.0], [1.0, 0.0]]),
    )


def test_item_rejects_ambiguous_array() -> None:
    with pytest.raises(ValueError, match="size 1"):
        ad.grad(lambda x: x.item())(np.array([1.0, 2.0]))


def test_item_rejects_an_incorrect_coordinate_rank() -> None:
    with pytest.raises(ValueError, match="incorrect number of indices"):
        ad.grad(lambda x: x.item(0, 0, 0))(np.ones((2, 2)))


def test_item_and_nested_array_construction_survive_staged_differentiation() -> None:
    value = np.array([1.0, 2.0])
    spec = ad.ArraySpec(value.shape, value.dtype)
    program = ad.stage(
        lambda x: ad.array([[x[0], x[1]]]).sum().item(),
        specs=(spec,),
    )
    gradient_program = ad.grad(program)

    assert_allclose(program(value), np.sum(value))
    assert_allclose(gradient_program(value), np.ones_like(value))
    assert_allclose(
        ad.StagedProgram.from_dict(gradient_program.to_dict())(value),
        np.ones_like(value),
    )


@pytest.mark.parametrize("constructor", [np.array, np.asarray, np.asanyarray])
def test_numpy_like_constructors_survive_staged_differentiation(
    constructor: Any,
) -> None:
    value = np.array([1.0, 2.0])
    spec = ad.ArraySpec(value.shape, value.dtype)
    program = ad.stage(
        lambda x: constructor(
            [[x[0], x[1]], [x[1], 2 * x[0]]],
            dtype=np.float32,
            like=x,
        )
        .sum()
        .item(),
        specs=(spec,),
    )
    gradient_program = ad.grad(program)

    assert program(value) == pytest.approx(7.0)
    for candidate in (
        gradient_program,
        ad.StagedProgram.from_dict(gradient_program.to_dict()),
    ):
        assert_allclose(candidate(value), np.array([3.0, 2.0]))


def test_staged_unsupported_numpy_coercion_recommends_like() -> None:
    with pytest.raises(TracingError, match=r"np\.array\(values, like=x\)"):
        ad.stage(
            lambda x: np.array([x[0], x[1]]),
            specs=(ad.ArraySpec((2,), "float64"),),
        )


def test_coordinate_item_survives_staged_differentiation() -> None:
    value = np.arange(4.0).reshape(2, 2)
    program = ad.stage(
        lambda x: x.item(1, 0),
        specs=(ad.ArraySpec(value.shape, value.dtype),),
    )

    assert program(value) == value.item(1, 0)
    assert_allclose(
        ad.grad(program)(value),
        np.array([[0.0, 0.0], [1.0, 0.0]]),
    )


def test_array_and_item_remain_array_api_provider_neutral() -> None:
    strict = pytest.importorskip("array_api_strict")
    value = strict.asarray([1.0, 2.0], dtype=strict.float32)

    gradient = ad.grad(
        lambda x: ad.array([[x[0], x[1]], [x[1], x[0]]]).sum().item(),
    )(value)

    assert type(gradient) is type(value)
    assert gradient.dtype == value.dtype
    assert_allclose(np.asarray(gradient), np.array([2.0, 2.0]))


def test_trace_inspection_and_stop_gradient_are_explicit() -> None:
    observations: list[tuple[bool, bool]] = []

    def loss(x: np.ndarray) -> Any:
        stopped = ad.stop_gradient(x)
        observations.append((ad.is_traced(x), ad.is_traced(stopped)))
        return np.sum(x * stopped)

    value = np.array([2.0, 3.0])
    gradient = ad.grad(loss)(value)

    assert observations == [(True, False)]
    assert_allclose(gradient, value)
    assert not ad.is_traced(value)


def test_stop_gradient_preserves_registered_pytree_structure() -> None:
    auxiliary: list[object] = []

    def loss(tree: dict[str, np.ndarray]) -> Any:
        stopped = ad.stop_gradient(tree)
        auxiliary.append(stopped)
        return np.sum(tree["x"])

    value = {"x": np.array([1.0, 2.0])}
    gradient = ad.grad(loss)(value)

    assert_allclose(gradient["x"], np.ones(2))
    assert isinstance(auxiliary[0], dict)
    assert_allclose(auxiliary[0]["x"], value["x"])  # type: ignore[index]


def test_stop_gradient_rejects_abstract_staging() -> None:
    with pytest.raises(TracingError, match="abstract staged values"):
        ad.stage(
            lambda x: ad.stop_gradient(x),
            specs=(ad.ArraySpec((2,), "float64"),),
        )


def test_is_traced_does_not_read_an_escaped_payload() -> None:
    escaped: list[object] = []

    def loss(x: np.ndarray) -> Any:
        escaped.append(x)
        return np.sum(x)

    ad.grad(loss)(np.array([1.0, 2.0]))

    assert ad.is_traced(escaped[0])
    with pytest.raises(EscapedTracerError, match="escaped"):
        ad.stop_gradient(escaped[0])
