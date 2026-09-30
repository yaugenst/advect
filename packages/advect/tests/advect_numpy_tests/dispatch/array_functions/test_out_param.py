"""Generic functionalization contracts for NumPy array-function ``out=``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import array_api_strict as strict
import numpy as np
import pytest
from hypothesis import example, given, strategies as st
from hypothesis.extra import numpy as hnp

import advect as ad
from advect.core._array_api.profiles import LATEST_ARRAY_API_VERSION
from advect.numpy._abstract_calls import can_cast_dtype
from advect_numpy_tests._assertions import (
    assert_jvp_matches_central_difference,
    assert_spellings_agree,
    assert_staged_round_trip,
    assert_tree_close,
    seeded_like,
)

if TYPE_CHECKING:
    from collections.abc import Callable


_SINGLE = {"f": np.dtype(np.float32), "c": np.dtype(np.complex64)}
# Single-input array functions whose out= only receives the result, and
# whether that spelling stages (round with decimals is dynamic-only).
_OUT_FORMS: dict[str, tuple[Callable[..., Any], bool]] = {
    "sum": (
        lambda x, **out: np.sum(
            x, axis=1, dtype=np.float64, keepdims=True, initial=0.25, where=x > 0.9, **out
        ),
        True,
    ),
    "mean": (lambda x, **out: np.mean(x, axis=0, dtype=np.float64, keepdims=True, **out), True),
    "std": (lambda x, **out: np.std(x, axis=1, **out), True),
    "prod": (lambda x, **out: np.prod(x, axis=0, **out), True),
    "max": (lambda x, **out: np.max(x, axis=0, **out), True),
    "cumsum": (lambda x, **out: np.cumsum(x, axis=1, dtype=np.float64, **out), True),
    "cumprod": (lambda x, **out: np.cumprod(x, axis=1, dtype=np.float64, **out), True),
    "take": (lambda x, **out: np.take(x, [0, 2], axis=1, mode="wrap", **out), True),
    "fft": (lambda x, **out: np.fft.fft(x, axis=1, norm="ortho", **out), True),
    "round": (lambda x, **out: np.round(x, decimals=1, **out), False),
    "clip": (lambda x, **out: np.clip(x, 0.8, 1.5, **out), True),
    "matmul": (lambda x, **out: np.matmul(x, x.T, **out), True),
}
_MATRIX = np.array([[0.7, 1.2, 1.8], [1.1, 0.8, 1.4]])


@given(
    name=st.sampled_from(sorted(_OUT_FORMS)),
    single=st.booleans(),
    traced_destination=st.booleans(),
    value=hnp.arrays(
        np.float64,
        hnp.array_shapes(min_dims=2, max_dims=2, max_side=4),
        elements=st.floats(min_value=0.5, max_value=2.0),
    ),
)
@example(name="sum", single=True, traced_destination=False, value=_MATRIX)
@example(name="mean", single=False, traced_destination=False, value=_MATRIX)
@example(name="cumsum", single=False, traced_destination=False, value=_MATRIX)
@example(name="cumprod", single=False, traced_destination=False, value=_MATRIX)
@example(name="take", single=False, traced_destination=False, value=_MATRIX)
@example(name="fft", single=False, traced_destination=False, value=_MATRIX)
@example(name="round", single=False, traced_destination=False, value=_MATRIX)
def test_array_function_out_writes_the_cast_pure_result(
    name: str,
    *,
    single: bool,
    traced_destination: bool,
    value: np.ndarray[Any, Any],
) -> None:
    """``f(x, out=d)`` is ``d`` rebound to ``f(x).astype(d.dtype)`` in every lifetime."""
    function, stages = _OUT_FORMS[name]
    result = function(value)
    dtype = _SINGLE[result.dtype.kind] if single else result.dtype

    def write(x: Any) -> Any:
        out = np.full(result.shape, 7.0, like=x)
        out = (out * np.mean(x) if traced_destination else out).astype(dtype)
        assert function(x, out=out) is out
        return out

    primal = assert_spellings_agree(
        write, lambda x: function(x).astype(dtype), (value,), argnums=(0,)
    )
    # NumPy may round intermediate results in the destination dtype.
    rtol = 4 * float(np.finfo(dtype).eps)
    assert_tree_close(primal, write(value), rtol=rtol)
    if stages:
        assert_staged_round_trip(write, value, rtol=rtol)


@pytest.mark.parametrize("single", [False, True], ids=("same-dtype", "cast"))
@pytest.mark.parametrize("name", sorted(name for name, (_, stages) in _OUT_FORMS.items() if stages))
def test_staged_out_lowers_the_same_graph_from_another_providers_examples(
    name: str, *, single: bool
) -> None:
    # Staged code sees the examples' provider dtype objects, which NumPy
    # cannot interpret, so out= validation reads the staged dtypes.
    function, _stages = _OUT_FORMS[name]
    result = function(_MATRIX)
    dtype = _SINGLE[result.dtype.kind] if single else result.dtype
    version = min(np.__array_api_version__, LATEST_ARRAY_API_VERSION)

    def write(x: Any) -> Any:
        out = np.full(result.shape, 7.0, like=x).astype(dtype)
        function(x, out=out)
        return out

    def graph(example: object) -> object:
        program = ad.stage(write, example, array_api_version=version)
        return program.to_dict()["program"]["graph"]

    assert graph(strict.asarray(_MATRIX)) == graph(_MATRIX)


@pytest.mark.parametrize("name", sorted(_OUT_FORMS))
def test_array_function_out_derivatives_match_central_differences(name: str) -> None:
    """Away from mask, clip, tie and rounding edges, ``out=`` matches central differences."""
    function, _stages = _OUT_FORMS[name]

    def write(x: Any) -> Any:
        out = np.zeros_like(function(x))
        assert function(x, out=out) is out
        return out

    value = np.array([[0.7, 1.2, 1.8], [1.1, 0.83, 1.4]])
    assert_jvp_matches_central_difference(write, (value,), (seeded_like(value, name),))


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

    assert_jvp_matches_central_difference(
        apply, (left, right), (left_tangent, right_tangent), rtol=2e-6, atol=2e-6
    )


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


@pytest.mark.parametrize(
    "function",
    [
        lambda x: np.outer(x, x, out=None),
        lambda x: np.einsum("i,j->ij", x, x, out=None),
        lambda x: np.concatenate((x, x), out=None),
        lambda x: np.stack((x, x), out=None),
    ],
    ids=["outer", "einsum", "concatenate", "stack"],
)
def test_array_function_out_none_means_no_destination(function: Any) -> None:
    value = np.array([0.7, -1.2, 1.8])
    direction = np.array([0.2, -0.3, 0.5])

    assert_jvp_matches_central_difference(function, (value,), (direction,), rtol=2e-5, atol=2e-5)


def test_composite_out_needs_a_traced_operand_besides_the_destination() -> None:
    value = np.array([1.0, 2.0, 3.0])

    def choose_into(x: Any) -> Any:
        destination = x * 0.0
        np.choose(np.array([0, 1, 0]), [value, -value], out=destination)
        return destination

    with pytest.raises(ad.TracingError, match="traced operand other than out="):
        ad.jvp(choose_into)(value, tangents=np.ones_like(value))


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


@pytest.mark.parametrize(
    "write",
    [
        lambda x, out: np.take(x, [0, 2], out=out),
        lambda x, out: np.compress([True, False, True, False], x, out=out),
    ],
    ids=("take", "compress"),
)
def test_selection_out_matches_numpy_casting_policy(write: Any) -> None:
    value = np.arange(4, dtype=np.float32)

    def into(dtype: type[np.floating[Any]]) -> Any:
        def apply(x: Any) -> Any:
            destination = np.empty((2,), dtype=dtype, like=x)
            result = write(x, destination)
            assert result is destination
            return destination

        return apply

    narrowing, widening = into(np.float16), into(np.float64)
    dynamic, _tangent = ad.jvp(narrowing)(value, tangents=np.ones_like(value))
    assert_tree_close(dynamic, narrowing(value))
    assert_staged_round_trip(narrowing, value)

    # NumPy versions differ on whether a widening selection out= is safe.
    try:
        expected = widening(value)
    except TypeError:
        with pytest.raises(TypeError, match="according to the rule 'safe'"):
            ad.jvp(widening)(value, tangents=np.ones_like(value))
        with pytest.raises(TypeError, match="according to the rule 'safe'"):
            ad.stage(widening, value)
    else:
        dynamic, _tangent = ad.jvp(widening)(value, tangents=np.ones_like(value))
        assert_tree_close(dynamic, expected)
        assert_staged_round_trip(widening, value)


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


@pytest.mark.parametrize("name", sorted(_OUT_FORMS))
def test_array_function_out_differentiates_under_an_enclosing_transform(name: str) -> None:
    """An enclosing trace sees how ``out=`` depends on the inputs, as without ``out=``."""
    function, _stages = _OUT_FORMS[name]
    value = np.array([[0.7, 1.2, 1.8], [1.1, 0.83, 1.4]])
    direction = seeded_like(value, name)

    def write(x: Any) -> Any:
        out = np.zeros_like(function(x))
        function(x, out=out)
        return out

    def inner_primal(call: Callable[..., Any]) -> Callable[..., Any]:
        return lambda x: ad.jvp(call)(x, tangents=direction)[0]

    def loss(call: Callable[..., Any]) -> Callable[..., Any]:
        return lambda x: np.sum(np.sin(np.real(call(x))) ** 3)

    assert_spellings_agree(inner_primal(write), inner_primal(function), (value,), argnums=(0,))
    np.testing.assert_allclose(ad.hessian(loss(write))(value), ad.hessian(loss(function))(value))


def test_astype_inexact_vjp_uses_the_static_target_dtype() -> None:
    value = np.array([1.0, 2.0], dtype=np.float64)

    gradient = ad.grad(lambda x: np.sum(x.astype(np.float32)))(value)

    np.testing.assert_allclose(gradient, np.ones_like(value))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_astype_function_casts_nested_traces(dtype: type[np.floating[Any]]) -> None:
    value = np.array([0.5, -1.0, 2.0])

    def loss(x: Any) -> Any:
        return np.sum(np.astype(x**3, dtype))

    expected = np.diag(6 * value)
    np.testing.assert_allclose(ad.hessian(loss)(value), expected, rtol=1e-6)
    staged = ad.stage(ad.grad(loss), specs=(ad.ArraySpec(value.shape, value.dtype),))
    np.testing.assert_allclose(staged(value), 3 * value**2, rtol=1e-6)


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
