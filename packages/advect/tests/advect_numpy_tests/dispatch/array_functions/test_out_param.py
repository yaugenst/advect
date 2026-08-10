"""Generic functionalization contracts for NumPy array-function ``out=``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given, strategies as st

import advect as ad
from advect.numpy._abstract_calls import can_cast_dtype


def _directional_difference(
    function: Any,
    values: tuple[np.ndarray, ...],
    directions: tuple[np.ndarray, ...],
) -> np.ndarray:
    epsilon = 1e-6
    positive = tuple(
        value + epsilon * direction for value, direction in zip(values, directions, strict=True)
    )
    negative = tuple(
        value - epsilon * direction for value, direction in zip(values, directions, strict=True)
    )
    return (function(*positive) - function(*negative)) / (2 * epsilon)


@pytest.mark.parametrize(
    ("pure", "write"),
    [
        (
            lambda x: np.sum(
                x,
                axis=1,
                dtype=np.float64,
                keepdims=True,
                initial=0.25,
                where=x > -0.5,
            ),
            lambda x, out: np.sum(
                x,
                axis=1,
                dtype=np.float64,
                out=out,
                keepdims=True,
                initial=0.25,
                where=x > -0.5,
            ),
        ),
        (
            lambda x: np.mean(
                x,
                axis=0,
                dtype=np.float64,
                keepdims=True,
                where=x > -0.5,
            ),
            lambda x, out: np.mean(
                x,
                axis=0,
                dtype=np.float64,
                out=out,
                keepdims=True,
                where=x > -0.5,
            ),
        ),
        (
            lambda x: np.cumsum(x, axis=1, dtype=np.float64),
            lambda x, out: np.cumsum(x, axis=1, dtype=np.float64, out=out),
        ),
        (
            lambda x: np.cumprod(x, axis=1, dtype=np.float64),
            lambda x, out: np.cumprod(x, axis=1, dtype=np.float64, out=out),
        ),
        (
            lambda x: np.take(x, [0, 2], axis=1, mode="wrap"),
            lambda x, out: np.take(x, [0, 2], axis=1, out=out, mode="wrap"),
        ),
        (
            lambda x: np.fft.fft(x, axis=1, norm="ortho"),
            lambda x, out: np.fft.fft(x, axis=1, norm="ortho", out=out),
        ),
        (
            lambda x: np.round(x, decimals=3),
            lambda x, out: np.round(x, decimals=3, out=out),
        ),
    ],
    ids=["sum", "mean", "cumsum", "cumprod", "take", "fft", "round"],
)
def test_single_input_array_function_out_matches_pure_call_and_jvp(
    pure: Any,
    write: Any,
) -> None:
    value = np.array([[0.7, 1.2, 1.8], [1.1, 0.8, 1.4]])
    direction = np.array([[0.2, -0.3, 0.5], [-0.1, 0.4, 0.25]])
    identities: list[bool] = []

    def into_out(x: Any) -> Any:
        expected_shape = pure(x)
        destination = np.zeros_like(expected_shape)
        result = write(x, destination)
        identities.append(result is destination)
        return destination

    primal, tangent = ad.jvp(into_out)(value, tangents=direction)
    expected = pure(value)
    finite_difference = _directional_difference(pure, (value,), (direction,))

    assert identities == [True]
    np.testing.assert_allclose(primal, expected)
    np.testing.assert_allclose(tangent, finite_difference, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    ("call", "shape"),
    [
        (
            lambda x, y, out: np.stack(
                (x, y),
                axis=1,
                out=out,
                casting="same_kind",
            ),
            (3, 2),
        ),
        (lambda x, y, out: np.outer(x, y, out=out), (3, 3)),
        (
            lambda x, y, out: np.einsum(
                "i,j->ij",
                x,
                y,
                out=out,
                optimize=True,
                casting="safe",
            ),
            (3, 3),
        ),
    ],
    ids=["stack", "outer", "einsum"],
)
def test_multi_input_array_function_out_differentiates_every_array_operand(
    call: Any,
    shape: tuple[int, ...],
) -> None:
    left = np.array([0.4, 1.2, -0.7])
    right = np.array([1.1, -0.3, 0.8])
    left_tangent = np.array([0.2, 0.1, -0.4])
    right_tangent = np.array([-0.3, 0.5, 0.25])

    def apply(x: Any, y: Any) -> Any:
        destination = np.empty(shape, dtype=np.float64, like=x)
        result = call(x, y, destination)
        assert result is destination
        return destination

    primal, tangent = ad.jvp(apply, argnums=(0, 1))(
        left,
        right,
        tangents=(left_tangent, right_tangent),
    )
    expected = call(left, right, np.empty_like(primal))
    finite_difference = _directional_difference(
        lambda x, y: call(x, y, np.empty_like(primal)),
        (left, right),
        (left_tangent, right_tangent),
    )

    np.testing.assert_allclose(primal, expected)
    np.testing.assert_allclose(tangent, finite_difference, rtol=2e-6, atol=2e-6)


@given(
    rows=st.integers(min_value=1, max_value=4),
    columns=st.integers(min_value=1, max_value=5),
)
def test_reduction_out_handles_shapes_and_unsafe_destination_casts(
    rows: int,
    columns: int,
) -> None:
    value = np.arange(rows * columns, dtype=np.float64).reshape(rows, columns) / 7

    def apply(x: Any) -> Any:
        destination = np.zeros_like(np.sum(x, axis=1), dtype=np.float32)
        result = np.sum(x, axis=1, dtype=np.float64, out=destination)
        assert result is destination
        return destination

    primal, tangent = ad.jvp(apply)(value, tangents=np.ones_like(value))

    np.testing.assert_allclose(primal, np.sum(value, axis=1).astype(np.float32))
    np.testing.assert_allclose(tangent, np.full(rows, columns, dtype=np.float32))
    assert primal.dtype == np.dtype(np.float32)


def test_array_function_out_to_integer_has_zero_derivative() -> None:
    value = np.array([0.7, 1.2, -0.4])

    def loss(x: Any) -> Any:
        destination = np.zeros((), dtype=np.int64, like=x)
        np.sum(x, out=destination)
        return destination.astype(np.float64)

    primal, tangent = ad.jvp(loss)(value, tangents=np.ones_like(value))

    np.testing.assert_allclose(primal, np.sum(value).astype(np.int64))
    np.testing.assert_allclose(tangent, 0.0)
    np.testing.assert_allclose(ad.grad(loss)(value), np.zeros_like(value))


def test_clip_out_where_preserves_masked_destination_in_both_modes() -> None:
    value = np.array([-2.0, -0.4, 0.7, 2.5])
    direction = np.array([0.3, -0.2, 0.4, 0.1])
    mask = np.array([True, False, True, False])

    def apply(x: Any) -> Any:
        destination = np.ones_like(x, dtype=np.float32) * 7
        result = np.clip(
            x,
            -1.0,
            1.0,
            out=destination,
            where=mask,
            casting="unsafe",
        )
        assert result is destination
        return destination

    primal, tangent = ad.jvp(apply)(value, tangents=direction)
    expected = np.full(value.shape, 7, dtype=np.float32)
    np.clip(
        value,
        -1.0,
        1.0,
        out=expected,
        where=mask,
        casting="unsafe",
    )

    np.testing.assert_allclose(primal, expected)
    np.testing.assert_allclose(
        tangent,
        np.where(mask & (value > -1) & (value < 1), direction, 0),
    )
    np.testing.assert_allclose(
        ad.grad(lambda x: np.sum(apply(x)))(value),
        np.where(mask & (value > -1) & (value < 1), 1.0, 0.0),
    )


def test_clip_out_rejects_explicit_ufunc_loop_selection() -> None:
    value = np.array([-2.0, -0.4, 0.7, 2.5])

    def apply(x: Any) -> Any:
        destination = np.empty_like(x, dtype=np.float32)
        return np.clip(
            x,
            -1.0,
            1.0,
            out=destination,
            dtype=np.float32,
            casting="unsafe",
        )

    with pytest.raises(ad.TracingError, match=r"dtype=.*loop selection"):
        ad.jvp(apply)(value, tangents=np.ones_like(value))
    with pytest.raises(ad.TracingError, match=r"dtype=.*staged out="):
        ad.stage(apply, specs=(ad.ArraySpec(value.shape, value.dtype),))


def test_positional_array_function_out_preserves_identity() -> None:
    value = np.arange(6.0).reshape(2, 3)

    def apply(x: Any) -> Any:
        destination = np.zeros_like(np.sum(x, axis=1))
        result = np.sum(x, 1, None, destination)
        assert result is destination
        return destination

    np.testing.assert_allclose(ad.jvp(apply)(value, tangents=np.ones_like(value))[0], [3, 12])


@pytest.mark.parametrize("positional", [False, True], ids=["keyword", "positional"])
def test_uninspectable_dot_out_forms_preserve_identity_and_derivatives(
    positional: object,
) -> None:
    assert isinstance(positional, bool)
    left = np.array([[1.0, 2.0], [3.0, 4.0]])
    right = np.array([[2.0, 0.0], [1.0, 3.0]])

    def apply(x: Any, y: Any) -> Any:
        destination = np.empty((2, 2), dtype=np.float64, order="C", like=x)
        result = np.dot(x, y, destination) if positional else np.dot(x, y, out=destination)
        assert result is destination
        return np.sum(destination)

    primal, tangent = ad.jvp(apply, argnums=(0, 1))(
        left,
        right,
        tangents=(np.ones_like(left), np.zeros_like(right)),
    )
    np.testing.assert_allclose(primal, np.sum(np.dot(left, right)))
    np.testing.assert_allclose(tangent, np.sum(np.dot(np.ones_like(left), right)))
    grad_left, grad_right = ad.grad(apply, argnums=(0, 1))(left, right)
    np.testing.assert_allclose(grad_left, np.ones((2, 2)) @ right.T)
    np.testing.assert_allclose(grad_right, left.T @ np.ones((2, 2)))

    program = ad.stage(
        apply,
        specs=(
            ad.ArraySpec(left.shape, left.dtype),
            ad.ArraySpec(right.shape, right.dtype),
        ),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())
    for staged in (program, restored):
        np.testing.assert_allclose(staged(left, right), np.sum(np.dot(left, right)))


@pytest.mark.parametrize("positional", [False, True], ids=["keyword", "positional"])
def test_uninspectable_concatenate_out_forms_stage_and_differentiate(
    positional: object,
) -> None:
    assert isinstance(positional, bool)
    left = np.array([1.0, 2.0])
    right = np.array([3.0, 4.0])

    def apply(x: Any, y: Any) -> Any:
        destination = np.empty((4,), dtype=np.float64, like=x)
        result = (
            np.concatenate((x, y), 0, destination)
            if positional
            else np.concatenate((x, y), axis=0, out=destination)
        )
        assert result is destination
        return destination

    primal, tangent = ad.jvp(apply, argnums=(0, 1))(
        left,
        right,
        tangents=(np.ones_like(left), np.zeros_like(right)),
    )
    np.testing.assert_allclose(primal, np.concatenate((left, right)))
    np.testing.assert_allclose(tangent, np.array([1.0, 1.0, 0.0, 0.0]))

    program = ad.stage(
        apply,
        specs=(
            ad.ArraySpec(left.shape, left.dtype),
            ad.ArraySpec(right.shape, right.dtype),
        ),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())
    for staged in (program, restored):
        np.testing.assert_allclose(staged(left, right), np.concatenate((left, right)))


def test_array_function_out_uses_upstream_shape_and_casting_errors() -> None:
    value = np.arange(4.0)

    def invalid_fft_dtype(x: Any) -> Any:
        destination = np.zeros_like(x, dtype=np.float64)
        return np.fft.fft(x, out=destination)

    with pytest.raises(TypeError, match="casting rule"):
        ad.jvp(invalid_fft_dtype)(value, tangents=np.ones_like(value))

    def invalid_stack_cast(x: Any) -> Any:
        destination = np.zeros((2, 4), dtype=np.int32, like=x)
        return np.stack((x, x), out=destination, casting="same_kind")

    with pytest.raises(TypeError, match="same_kind"):
        ad.jvp(invalid_stack_cast)(value, tangents=np.ones_like(value))


def test_array_function_out_validation_never_mutates_traced_operands() -> None:
    value = np.array([3.0, 1.0, 2.0, 4.0])
    original = value.copy()

    def invalid_median(x: Any) -> Any:
        destination = np.empty((), dtype=x.dtype, like=x)
        return np.median(x, out=destination, overwrite_input=True)

    with pytest.raises(ad.TracingError, match="overwrite_input=True"):
        ad.jvp(invalid_median)(value, tangents=np.ones_like(value))

    np.testing.assert_array_equal(value, original)


def test_stageable_array_function_out_round_trips() -> None:
    def apply(x: Any) -> Any:
        destination = np.zeros_like(np.sum(x, axis=1))
        result = np.sum(x, axis=1, out=destination)
        assert result is destination
        return destination

    program = ad.stage(apply, specs=(ad.ArraySpec((2, 3), "float64"),))
    restored = ad.StagedProgram.from_dict(program.to_dict())
    value = np.arange(6.0).reshape(2, 3)

    np.testing.assert_allclose(restored(value), np.sum(value, axis=1))


def test_staged_reduction_out_preserves_where_and_unsafe_destination_cast() -> None:
    def apply(x: Any) -> Any:
        destination = np.zeros_like(np.sum(x, axis=1), dtype=np.float32)
        result = np.sum(
            x,
            axis=1,
            dtype=np.float64,
            out=destination,
            where=x > 1.5,
        )
        assert result is destination
        return destination

    program = ad.stage(apply, specs=(ad.ArraySpec((2, 3), "float64"),))
    restored = ad.StagedProgram.from_dict(program.to_dict())
    value = np.arange(6.0).reshape(2, 3)
    expected = np.zeros(2, dtype=np.float32)
    np.sum(
        value,
        axis=1,
        dtype=np.float64,
        out=expected,
        where=value > 1.5,
    )

    for staged in (program, restored):
        actual = staged(value)
        np.testing.assert_allclose(actual, expected)
        assert actual.dtype == np.dtype(np.float32)


def test_staged_dot_out_requires_exact_dtype_and_c_layout() -> None:
    def valid(x: Any) -> Any:
        destination = np.empty((), dtype=x.dtype, order="C", like=x)
        result = np.dot(x, x, out=destination)
        assert result is destination
        return destination

    value = np.arange(4, dtype=np.float32)
    program = ad.stage(valid, specs=(ad.ArraySpec(value.shape, value.dtype),))
    restored = ad.StagedProgram.from_dict(program.to_dict())
    for staged in (program, restored):
        np.testing.assert_allclose(staged(value), np.dot(value, value))

    def wrong_dtype(x: Any) -> Any:
        destination = np.empty((), dtype=np.float64, order="C", like=x)
        return np.dot(x, x, out=destination)

    with pytest.raises(ValueError, match="output array is not acceptable"):
        ad.jvp(wrong_dtype)(value, tangents=np.ones_like(value))
    with pytest.raises(ValueError, match="output array is not acceptable"):
        ad.stage(wrong_dtype, specs=(ad.ArraySpec(value.shape, value.dtype),))

    def wrong_layout(x: Any) -> Any:
        destination = np.empty((2, 2), dtype=x.dtype, order="F", like=x)
        return np.dot(x, x, out=destination)

    matrix = value.reshape(2, 2)
    with pytest.raises(ValueError, match="C layout"):
        ad.stage(wrong_layout, specs=(ad.ArraySpec(matrix.shape, matrix.dtype),))


def test_staged_take_out_matches_numpy_casting_policy() -> None:
    def narrowing(x: Any) -> Any:
        destination = np.empty((2,), dtype=np.float16, like=x)
        result = np.take(x, [0, 2], out=destination)
        assert result is destination
        return destination

    value = np.arange(4, dtype=np.float32)
    dynamic, _tangent = ad.jvp(narrowing)(value, tangents=np.ones_like(value))
    np.testing.assert_allclose(dynamic, np.take(value, [0, 2]).astype(np.float16))
    program = ad.stage(narrowing, value)
    restored = ad.StagedProgram.from_dict(program.to_dict())
    for staged in (program, restored):
        actual = staged(value)
        np.testing.assert_allclose(actual, np.take(value, [0, 2]).astype(np.float16))
        assert actual.dtype == np.dtype(np.float16)

    def widening(x: Any) -> Any:
        destination = np.empty((2,), dtype=np.float64, like=x)
        return np.take(x, [0, 2], out=destination)

    try:
        expected = np.take(value, [0, 2], out=np.empty((2,), dtype=np.float64))
    except TypeError:
        with pytest.raises(TypeError, match="according to the rule 'safe'"):
            ad.jvp(widening)(value, tangents=np.ones_like(value))
        with pytest.raises(TypeError, match="according to the rule 'safe'"):
            ad.stage(widening, value)
    else:
        dynamic, _tangent = ad.jvp(widening)(value, tangents=np.ones_like(value))
        np.testing.assert_allclose(dynamic, expected)
        staged = ad.stage(widening, value)(value)
        np.testing.assert_allclose(staged, expected)
        assert dynamic.dtype == expected.dtype
        assert staged.dtype == expected.dtype


def test_compress_out_matches_numpy_casting_policy() -> None:
    value = np.arange(4, dtype=np.float32)
    condition = np.array([True, False, True, False])

    def narrowing(x: Any) -> Any:
        destination = np.empty((2,), dtype=np.float16, like=x)
        result = np.compress(condition, x, out=destination)
        assert result is destination
        return destination

    expected = np.compress(condition, value).astype(np.float16)
    dynamic, _tangent = ad.jvp(narrowing)(value, tangents=np.ones_like(value))
    np.testing.assert_allclose(dynamic, expected)
    program = ad.stage(narrowing, value)
    np.testing.assert_allclose(program(value), expected)

    def widening(x: Any) -> Any:
        destination = np.empty((2,), dtype=np.float64, like=x)
        return np.compress(condition, x, out=destination)

    try:
        expected = np.compress(condition, value, out=np.empty((2,), dtype=np.float64))
    except TypeError:
        with pytest.raises(TypeError, match="according to the rule 'safe'"):
            ad.jvp(widening)(value, tangents=np.ones_like(value))
        with pytest.raises(TypeError, match="according to the rule 'safe'"):
            ad.stage(widening, value)
    else:
        dynamic, _tangent = ad.jvp(widening)(value, tangents=np.ones_like(value))
        np.testing.assert_allclose(dynamic, expected)
        staged = ad.stage(widening, value)(value)
        np.testing.assert_allclose(staged, expected)
        assert dynamic.dtype == expected.dtype
        assert staged.dtype == expected.dtype


def test_staged_out_tuple_matches_numpy_function_category() -> None:
    def clip_tuple(x: Any) -> Any:
        destination = np.zeros_like(x)
        np.clip(x, -1, 1, out=(destination,))
        return destination

    value = np.array([-2.0, 0.5, 3.0])
    dynamic, tangent = ad.jvp(clip_tuple)(value, tangents=np.ones_like(value))
    np.testing.assert_allclose(dynamic, np.clip(value, -1, 1))
    np.testing.assert_allclose(tangent, np.array([0.0, 1.0, 0.0]))
    program = ad.stage(clip_tuple, specs=(ad.ArraySpec(value.shape, value.dtype),))
    restored = ad.StagedProgram.from_dict(program.to_dict())
    for staged in (program, restored):
        np.testing.assert_allclose(staged(value), np.clip(value, -1, 1))

    def round_tuple(x: Any) -> Any:
        destination = np.zeros_like(x)
        return np.round(x, out=(destination,))

    with pytest.raises(ad.TracingError, match="tuple destination"):
        ad.jvp(round_tuple)(value, tangents=np.ones_like(value))
    with pytest.raises(ad.MutationError, match="tuple destination"):
        ad.stage(round_tuple, specs=(ad.ArraySpec(value.shape, value.dtype),))


def test_array_function_out_remains_traceable_at_second_order() -> None:
    value = np.array([0.4, -0.7, 1.2])

    def loss(x: Any) -> Any:
        destination = np.zeros_like(np.sum(x * x))
        np.sum(x * x, out=destination)
        return destination

    np.testing.assert_allclose(ad.grad(loss)(value), 2 * value)
    np.testing.assert_allclose(ad.hessian(loss)(value), 2 * np.eye(value.size))


def test_astype_inexact_vjp_uses_the_static_target_dtype() -> None:
    value = np.array([1.0, 2.0], dtype=np.float64)

    gradient = ad.grad(lambda x: np.sum(x.astype(np.float32)))(value)

    np.testing.assert_allclose(gradient, np.ones_like(value))


@pytest.mark.parametrize(
    ("order", "casting", "subok", "copy"),
    [
        ("C", "same_kind", False, True),
        ("F", "safe", True, True),
        ("A", "unsafe", False, False),
        ("K", "equiv", False, False),
    ],
)
def test_astype_supports_the_full_ndarray_control_surface(
    order: str,
    casting: str,
    subok: object,
    copy: object,
) -> None:
    assert isinstance(subok, bool)
    assert isinstance(copy, bool)
    value = np.arange(6.0, dtype=np.float64).reshape(2, 3)
    direction = np.linspace(0.1, 0.6, 6).reshape(2, 3)

    def apply(x: Any) -> Any:
        return x.astype(
            np.float64,
            order=order,
            casting=casting,
            subok=subok,
            copy=copy,
        )

    primal, tangent = ad.jvp(apply)(value, tangents=direction)
    expected = value.astype(
        np.float64,
        order=order,
        casting=casting,
        subok=subok,
        copy=copy,
    )

    np.testing.assert_allclose(primal, expected)
    np.testing.assert_allclose(tangent, direction)
    assert primal.flags.f_contiguous == expected.flags.f_contiguous

    program = ad.stage(apply, specs=(ad.ArraySpec((2, 3), "float64"),))
    restored = ad.StagedProgram.from_dict(program.to_dict())
    replayed = restored(value)
    np.testing.assert_allclose(replayed, expected)
    assert replayed.flags.f_contiguous == expected.flags.f_contiguous


@pytest.mark.parametrize(
    "casting",
    ["no", "equiv", "safe", "same_kind", "unsafe"],
)
def test_staged_casting_authority_matches_numpy_for_every_supported_dtype_pair(
    casting: str,
) -> None:
    dtypes = (
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    )

    for source in dtypes:
        for target in dtypes:
            assert can_cast_dtype(source, target, casting=casting) is bool(
                np.can_cast(source, target, casting=casting)
            )
