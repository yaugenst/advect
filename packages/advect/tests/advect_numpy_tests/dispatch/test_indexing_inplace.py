"""Functionalized mutation contracts exercised through dynamic transforms."""

from __future__ import annotations

import math
import operator
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from hypothesis import example, given, strategies as st
from hypothesis.extra import numpy as hnp

import advect as ad
from advect.numpy import TracedArray
from advect.numpy._traced_array_state import user_location
from advect.pytree import tree_leaves, tree_map
from advect_numpy_tests._assertions import (
    assert_adjoint_identity,
    assert_staged_round_trip,
    assert_tree_close,
    seeded_like,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def test_user_location_skips_internal_frames_in_an_installed_wheel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_frame = SimpleNamespace(
        f_code=SimpleNamespace(co_filename="/workspace/model.py", co_name="update"),
        f_lineno=12,
        f_globals={"__name__": "model"},
        f_back=None,
    )
    internal_frame = SimpleNamespace(
        f_code=SimpleNamespace(
            co_filename="/venv/site-packages/advect/numpy/_traced_array_state.py",
            co_name="user_location",
        ),
        f_lineno=45,
        f_globals={"__name__": "advect.numpy._traced_array_state"},
        f_back=user_frame,
    )
    monkeypatch.setattr(sys, "_getframe", lambda _depth: internal_frame)

    location = user_location()

    assert location is not None
    assert location.filename == "/workspace/model.py"
    assert location.lineno == 12
    assert location.function == "update"


def test_augmented_assignment_updates_one_wrapper() -> None:
    observations: list[tuple[bool, bool, int]] = []

    def update(source: Any) -> Any:
        current = source.copy()
        alias = current
        previous_node = current.node_id
        current += 1.0
        observations.append((current is alias, current.node_id != previous_node, current.epoch))
        return alias

    original = np.array([1.0, 2.0, 3.0])
    value, tangent = ad.jvp(update)(original, tangents=np.ones_like(original))

    assert observations == [(True, True, 1)]
    np.testing.assert_array_equal(value, [2.0, 3.0, 4.0])
    np.testing.assert_array_equal(tangent, np.ones_like(original))
    np.testing.assert_array_equal(original, [1.0, 2.0, 3.0])


def test_chained_augmented_assignments_preserve_derivatives() -> None:
    def update(source: Any) -> Any:
        current = source.copy()
        current += 1.0
        current *= 2.0
        current -= 1.0
        return current

    original = np.array([1.0, 2.0, 3.0])
    tangent_in = np.array([0.5, -1.0, 2.0])
    value, tangent = ad.jvp(update)(original, tangents=tangent_in)

    np.testing.assert_array_equal(value, [3.0, 5.0, 7.0])
    np.testing.assert_array_equal(tangent, 2.0 * tangent_in)


@pytest.mark.parametrize(
    ("operation", "value", "other", "expected_tangent"),
    [
        (operator.itruediv, np.asarray([2.0, 4.0]), 2.0, np.asarray([0.5, 0.5])),
        (operator.ifloordiv, np.asarray([2.5, 5.5]), 2.0, np.zeros(2)),
        (operator.imod, np.asarray([2.5, 5.5]), 2.0, np.ones(2)),
        (operator.ipow, np.asarray([2.0, 3.0]), 2.0, np.asarray([4.0, 6.0])),
        (
            operator.imatmul,
            np.eye(2),
            np.asarray([[2.0, 0.0], [0.0, 3.0]]),
            np.asarray([[2.0, 3.0], [2.0, 3.0]]),
        ),
    ],
    ids=["divide", "floor-divide", "remainder", "power", "matmul"],
)
def test_supported_augmented_operators_functionalize_without_mutating_inputs(
    operation: Callable[[Any, Any], Any],
    value: np.ndarray,
    other: object,
    expected_tangent: np.ndarray,
) -> None:
    original = value.copy()

    def apply(array: Any) -> Any:
        result = array.copy()
        operation(result, other)
        return result

    primal, tangent = ad.jvp(apply)(value, tangents=np.ones_like(value))
    expected = original.copy()
    operation(expected, other)

    np.testing.assert_allclose(primal, expected)
    np.testing.assert_allclose(tangent, expected_tangent)
    np.testing.assert_array_equal(value, original)


@pytest.mark.parametrize(
    ("operation", "other", "name"),
    [
        (operator.iand, np.asarray([1, 6]), "bitwise_and"),
        (operator.ior, np.asarray([1, 6]), "bitwise_or"),
        (operator.ixor, np.asarray([1, 6]), "bitwise_xor"),
        (operator.ilshift, 1, "left_shift"),
        (operator.irshift, 1, "right_shift"),
    ],
)
def test_nondifferentiable_augmented_operators_report_the_canonical_operation(
    operation: Callable[[Any, Any], Any],
    other: object,
    name: str,
) -> None:
    def apply(array: Any) -> Any:
        result = array.copy()
        operation(result, other)
        return result

    value = np.asarray([3, 5])
    with pytest.raises(ad.NoJVPError, match=rf"array\.{name}"):
        ad.jvp(apply)(value, tangents=np.zeros_like(value))


@pytest.mark.parametrize(
    ("operation", "augmented"),
    [(operator.lshift, operator.ilshift), (operator.rshift, operator.irshift)],
    ids=("left_shift", "right_shift"),
)
def test_shift_operators_trace_like_their_augmented_forms(
    operation: Callable[[Any, Any], Any],
    augmented: Callable[[Any, Any], Any],
) -> None:
    def apply(array: Any) -> tuple[Any, Any]:
        integers = (array * 4).astype(np.int64)
        shifted = integers.copy()
        augmented(shifted, 1)
        return operation(integers, 1), shifted

    value = np.asarray([0.5, 1.0, 2.25])
    primal, pullback = ad.vjp(apply)(value)
    pullback.close()

    expected = operation((value * 4).astype(np.int64), 1)
    for actual in primal:
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "name",
    [
        "iadd",
        "isub",
        "imul",
        "itruediv",
        "ifloordiv",
        "imod",
        "ipow",
        "imatmul",
        "iand",
        "ior",
        "ixor",
        "ilshift",
        "irshift",
    ],
)
def test_augmented_operators_carry_their_own_names_for_tracebacks(name: str) -> None:
    method = getattr(TracedArray, f"__{name}__")

    assert method.__name__ == f"__{name}__"
    assert method.__qualname__ == f"TracedArray.__{name}__"


def test_input_augmented_assignment_is_rejected_before_changing_caller_data() -> None:
    original = np.array([1.0, 2.0, 3.0])

    def mutate(parameter: Any) -> Any:
        parameter += 1.0
        return parameter

    with pytest.raises(ad.MutationError, match="Cannot mutate traced input 'parameter'"):
        ad.jvp(mutate)(original, tangents=np.ones_like(original))
    np.testing.assert_array_equal(original, [1.0, 2.0, 3.0])


def test_input_item_assignment_is_rejected_before_changing_caller_data() -> None:
    original = np.array([1.0, 2.0, 3.0])

    def mutate(parameter: Any) -> Any:
        parameter[0] = 99.0
        return parameter

    with pytest.raises(ad.MutationError, match="Cannot mutate traced input 'parameter'"):
        ad.jvp(mutate)(original, tangents=np.ones_like(original))
    np.testing.assert_array_equal(original, [1.0, 2.0, 3.0])


def test_escaped_owned_array_rejects_later_mutation() -> None:
    escaped: list[Any] = []

    def own(source: Any) -> Any:
        current = source.copy()
        escaped.append(current)
        return current

    source = np.arange(3.0)
    ad.jvp(own)(source, tangents=np.ones_like(source))

    with pytest.raises(ad.TracingError, match="escaped its Advect transform"):
        escaped[0][0] = 10.0
    with pytest.raises(ad.TracingError, match="escaped its Advect transform"):
        escaped[0].__iadd__(1.0)


@pytest.mark.parametrize("update", ["add", "assign", "direct"])
def test_nested_traces_reject_functional_updates_from_an_outer_recorder(update: str) -> None:
    inner_value = np.asarray([3.0, 4.0])

    def outer(outer_value: Any) -> Any:
        def inner(value: Any) -> Any:
            result = value.copy()
            if update == "add":
                result[:1] += outer_value[:1]
            elif update == "assign":
                result[0] = outer_value[0]
            else:
                result += outer_value
            return np.sum(result)

        return ad.grad(inner)(inner_value)

    with pytest.raises(ad.TracingError, match="different trace context"):
        ad.jvp(outer)(np.asarray([1.0, 2.0]), tangents=np.ones(2))


def test_item_assignment_reads_numpy_integers_as_basic_indices() -> None:
    def update(source: Any) -> Any:
        current = source.copy()
        current[np.int64(1)] = 8.0
        current[np.intp(-1)] += source[np.int32(0)]
        return current

    original = np.array([1.0, 2.0, 3.0])
    value, tangent = ad.jvp(update)(original, tangents=np.array([2.0, 3.0, 4.0]))

    np.testing.assert_array_equal(value, [1.0, 8.0, 4.0])
    np.testing.assert_array_equal(tangent, [2.0, 0.0, 6.0])


def test_stencil_reads_views_then_updates_the_base_slice() -> None:
    def step(source: Any) -> Any:
        current = source.copy()
        laplacian = current[2:] - 2.0 * current[1:-1] + current[:-2]
        current[1:-1] += 0.25 * laplacian
        return current

    original = np.array([0.0, 1.0, 4.0, 10.0, 18.0, 29.0])
    tangent_in = np.linspace(-0.5, 0.5, original.size)
    expected = original.copy()
    expected[1:-1] += 0.25 * (original[2:] - 2.0 * original[1:-1] + original[:-2])
    expected_tangent = tangent_in.copy()
    expected_tangent[1:-1] += 0.25 * (tangent_in[2:] - 2.0 * tangent_in[1:-1] + tangent_in[:-2])

    value, tangent = ad.jvp(step)(original, tangents=tangent_in)

    np.testing.assert_allclose(value, expected)
    np.testing.assert_allclose(tangent, expected_tangent)


def test_named_basic_view_augmented_assignment_updates_its_base() -> None:
    def update(source: Any) -> Any:
        current = source.copy()
        middle = current[1:-1]
        middle += 1.0
        middle *= 2.0
        return current, middle

    source = np.arange(5.0)
    (value, middle), (tangent, middle_tangent) = ad.jvp(update)(
        source,
        tangents=np.ones_like(source),
    )

    np.testing.assert_array_equal(value, [0.0, 4.0, 6.0, 8.0, 4.0])
    np.testing.assert_array_equal(middle, [4.0, 6.0, 8.0])
    np.testing.assert_array_equal(tangent, [1.0, 2.0, 2.0, 2.0, 1.0])
    np.testing.assert_array_equal(middle_tangent, [2.0, 2.0, 2.0])


_GRID = np.arange(12.0).reshape(3, 4)
_SHAPES = hnp.array_shapes(min_dims=0, max_dims=3, min_side=0, max_side=4)
_KEYED_SHAPES = hnp.array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=4)


def _values(shape: tuple[int, ...]) -> np.ndarray[Any, Any]:
    return np.asarray(np.arange(1.0, math.prod(shape) + 1.0).reshape(shape) / 2)


@st.composite
def _indexed(draw: st.DrawFn) -> tuple[np.ndarray[Any, Any], Any, str]:
    """Draw an array and a basic, integer-array or boolean key into it."""
    kind = draw(st.sampled_from(("basic", "integer", "boolean")))
    shape = draw(_SHAPES if kind == "basic" else _KEYED_SHAPES)
    if kind == "basic":
        key = draw(hnp.basic_indices(shape, allow_newaxis=True, allow_ellipsis=True))
    elif kind == "integer":
        results = hnp.array_shapes(min_dims=0, max_dims=2, max_side=3)
        key = draw(hnp.integer_array_indices(shape, result_shape=results))
    else:
        key = draw(hnp.arrays(np.bool_, shape[: draw(st.integers(1, len(shape)))]))
    return _values(shape), key, kind


@given(indexed=_indexed(), traced_key=st.booleans())
@example(indexed=(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1, 0]), "integer"), traced_key=True)
# A list, or a tuple nested in the key, is an advanced index sequence.
@example(indexed=(_GRID, (slice(None), [3, 1]), "integer"), traced_key=False)
@example(indexed=(_GRID, (0, (1, 2)), "integer"), traced_key=False)
@example(indexed=(_GRID, ((0, 2), (1, 3)), "integer"), traced_key=False)
def test_getitem_gathers_tangents_and_scatter_adds_cotangents(
    indexed: tuple[np.ndarray[Any, Any], Any, str],
    *,
    traced_key: bool,
) -> None:
    value, key, kind = indexed
    # A traced discrete key only selects: its tangent and cotangent are zero.
    arguments = (value, key) if traced_key and kind != "basic" else (value,)
    argnums = tuple(range(len(arguments)))

    def gather(array: Any, *selector: Any) -> Any:
        return array[selector[0] if selector else key]

    def traced_gather(*current: Any) -> Any:
        assert all(isinstance(leaf, TracedArray) for leaf in tree_leaves(current))
        return gather(*current)

    direction = seeded_like(value, "direction")
    key_zeros = tuple(tree_map(np.zeros_like, argument) for argument in arguments[1:])
    primal, tangent = ad.jvp(traced_gather, argnums=argnums)(
        *arguments, tangents=(direction, *key_zeros)
    )
    assert_tree_close(primal, value[key], rtol=0.0)
    assert_tree_close(tangent, direction[key], rtol=0.0)
    cotangent = seeded_like(primal, "cotangent")
    _, pullback = ad.vjp(traced_gather, argnums=argnums)(*arguments)
    try:
        gradient, *key_cotangents = pullback(cotangent)
    finally:
        pullback.close()
    expected = np.zeros_like(value)
    np.add.at(expected, key, cotangent)
    assert_tree_close(gradient, expected, rtol=1e-12, atol=1e-12)
    assert_tree_close(tuple(key_cotangents), key_zeros, rtol=0.0)
    if kind == "basic":
        assert_staged_round_trip(gather, value, rtol=0.0)
    else:
        specs = tree_map(lambda leaf: ad.ArraySpec(leaf.shape, leaf.dtype), arguments)
        with pytest.raises(ad.TracingError, match="Basic indexing supports only"):
            ad.stage(gather, specs=specs)


def _update(array: Any, key: Any, update: str, operand: float) -> Any:
    result = array.copy()
    if update == "set":
        result[key] = operand
    elif update == "add":
        result[key] += operand
    elif update == "multiply":
        result[key] *= operand
    else:
        # NumPy returns a copy, not a view, for a key that selects one element.
        view = result[key]
        view += operand
    return result


@st.composite
def _basic_indexed(draw: st.DrawFn) -> tuple[np.ndarray[Any, Any], Any]:
    shape = draw(_SHAPES)
    return _values(shape), draw(hnp.basic_indices(shape, allow_newaxis=True, allow_ellipsis=True))


@given(indexed=_basic_indexed(), update=st.sampled_from(("set", "add", "multiply", "view-add")))
@example(indexed=(np.array([1.0, 2.0, 3.0]), slice(1, None)), update="set")
@example(indexed=(np.array([0.0, 1.0, 4.0, 9.0, 16.0]), slice(1, -1)), update="add")
@example(indexed=(_GRID, 0), update="view-add")
@example(indexed=(_GRID, (slice(None), 1)), update="view-add")
@example(indexed=(_GRID, Ellipsis), update="view-add")
@example(indexed=(_GRID, (None, Ellipsis)), update="view-add")
@example(indexed=(_GRID, slice(None, None, 2)), update="view-add")
@example(indexed=(_GRID, np.int64(1)), update="view-add")
@example(indexed=(_GRID, (slice(np.int64(1), None), np.intp(-1))), update="view-add")
def test_basic_updates_apply_the_same_update_to_tangents(
    indexed: tuple[np.ndarray[Any, Any], Any],
    update: str,
) -> None:
    """``=`` zeroes the selected tangent, ``+=`` keeps it, and ``*=`` scales it."""
    value, key = indexed

    def apply(array: Any) -> Any:
        return _update(array, key, update, 2.5)

    direction = seeded_like(value, "direction")
    primal, tangent = ad.jvp(apply)(value, tangents=direction)
    assert_tree_close(primal, apply(value), rtol=0.0)
    expected = _update(direction, key, update, 2.5 if update == "multiply" else 0.0)
    assert_tree_close(tangent, expected, rtol=0.0)
    assert_adjoint_identity(apply, (value,), (direction,), tangent, argnums=(0,))
    assert_staged_round_trip(apply, value, rtol=0.0)


def test_named_view_update_only_refreshes_the_mutated_view() -> None:
    def update(source: Any) -> Any:
        current = source.copy()
        sibling = current[1:-1]
        middle = current[1:-1]
        middle += 1.0
        return sibling + middle

    source = np.arange(5.0)
    with pytest.raises(ad.MutationError, match="view is stale"):
        ad.jvp(update)(source, tangents=np.ones_like(source))


def test_using_a_view_after_base_update_reports_stale_view() -> None:
    def update(source: Any) -> Any:
        current = source.copy()
        old_view = current[:2]
        current += 1.0
        return old_view + 1.0

    source = np.arange(5.0)
    with pytest.raises(ad.MutationError, match="view is stale"):
        ad.jvp(update)(source, tangents=np.ones_like(source))


def test_whole_cell_epoch_is_conservative_for_disjoint_slices() -> None:
    def update(source: Any) -> Any:
        current = source.copy()
        old_view = current[:2]
        current[4:] = 10.0
        return old_view + 1.0

    source = np.arange(6.0)
    with pytest.raises(ad.MutationError, match="view is stale"):
        ad.jvp(update)(source, tangents=np.ones_like(source))


def test_layout_dependent_reshape_is_always_a_view() -> None:
    def update(source: Any) -> Any:
        current = source.copy()
        reshaped = current.reshape(2, 3)
        current += 1.0
        return reshaped + 1.0

    source = np.arange(6.0)
    with pytest.raises(ad.MutationError, match="view is stale"):
        ad.jvp(update)(source, tangents=np.ones_like(source))


def test_mutation_through_reshape_is_rejected() -> None:
    def update(source: Any) -> Any:
        current = source.copy()
        reshaped = current.reshape(2, 3)
        reshaped += 1.0
        return reshaped

    source = np.arange(6.0)
    with pytest.raises(ad.MutationError, match="Mutation through this traced view"):
        ad.jvp(update)(source, tangents=np.ones_like(source))


def test_item_assignment_through_a_view_suggests_combining_indices() -> None:
    def update(source: Any) -> Any:
        current = source.copy()
        current[0][1] = 3.0
        return current

    source = np.zeros((2, 2))
    with pytest.raises(ad.MutationError, match=r"field\[i, j\]"):
        ad.jvp(update)(source, tangents=np.ones_like(source))


@pytest.mark.parametrize(
    ("shape", "key"),
    [((4,), np.array([0, 2])), ((2, 3), (0, (1, 2)))],
    ids=("index-array", "nested-tuple"),
)
def test_advanced_index_assignment_is_explicitly_rejected(
    shape: tuple[int, ...],
    key: Any,
) -> None:
    def update(source: Any) -> Any:
        current = source.copy()
        current[key] = 1.0
        return current

    source = np.zeros(shape)
    with pytest.raises(ad.TracingError, match="Advanced-index assignment"):
        ad.jvp(update)(source, tangents=np.ones_like(source))


@pytest.mark.parametrize("update", ["add", "multiply"])
def test_stale_view_error_names_the_user_line_of_an_augmented_view_update(update: str) -> None:
    def operation(array: Any) -> Any:
        base = array.copy()
        view = base[:2]
        overlapping = base[1:3]
        if update == "add":
            view += 1.0
        else:
            view *= 2.0
        return overlapping + 1.0

    with pytest.raises(ad.StaleViewError) as caught:
        ad.jvp(operation)(np.arange(4.0), tangents=np.ones(4))

    updated = str(caught.value).split("The base was updated at ", 1)[1]
    assert updated.startswith(__file__)
    assert "in operation" in updated


def _multiply_by_numpy_float(result: Any, _grad: Any) -> None:
    result *= np.sqrt(2.0)


def _divide_by_numpy_size_root(result: Any, _grad: Any) -> None:
    result /= np.sqrt(result.size)


def _add_numpy_float(result: Any, _grad: Any) -> None:
    result += np.float64(1.0)


def _subtract_numpy_array(result: Any, _grad: Any) -> None:
    result -= np.full(3, 0.5)


def _multiply_view_by_numpy_float(result: Any, _grad: Any) -> None:
    result[1:] *= np.float64(2.0)


def _add_numpy_float_at_index(result: Any, _grad: Any) -> None:
    result[1] += np.float64(0.5)


def _rotate_by_numpy_complex(result: Any, _grad: Any) -> None:
    result *= np.exp(0.3j)


def _add_python_list(result: Any, _grad: Any) -> None:
    result += [1.5, 2.5, 3.5]


def _descend_scaled_gradient(result: Any, grad: Any) -> None:
    result -= np.float64(0.1) * grad


def _descend_scaled_gradient_in_view(result: Any, grad: Any) -> None:
    result[1:] -= np.float64(0.1) * grad[1:]


def _rotate_by_staged_complex(result: Any, grad: Any) -> None:
    result *= np.exp(1j * np.float64(0.3)) * grad


def _add_staged_int64(result: Any, grad: Any) -> None:
    result += grad.astype(np.int64)


def _shift_by_numpy_int64(result: Any, _grad: Any) -> None:
    result <<= np.int64(1)


_FLOAT32 = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
_COMPLEX64 = np.asarray([1.0 - 1.0j, 2.0, 3.0 + 0.5j], dtype=np.complex64)
_INT32 = np.asarray([1, 2, 3], dtype=np.int32)


@pytest.mark.parametrize(
    ("augment", "value"),
    [
        pytest.param(_multiply_by_numpy_float, _FLOAT32, id="numpy-scalar"),
        pytest.param(_divide_by_numpy_size_root, _FLOAT32, id="numpy-scalar-divide"),
        pytest.param(_add_numpy_float, _FLOAT32, id="numpy-scalar-add"),
        pytest.param(_subtract_numpy_array, _FLOAT32, id="numpy-array"),
        pytest.param(_multiply_view_by_numpy_float, _FLOAT32, id="numpy-scalar-view"),
        pytest.param(_add_numpy_float_at_index, _FLOAT32, id="numpy-scalar-indexed"),
        pytest.param(_rotate_by_numpy_complex, _COMPLEX64, id="numpy-complex"),
        pytest.param(_add_python_list, _FLOAT32, id="list"),
        pytest.param(_descend_scaled_gradient, _FLOAT32, id="staged-gradient-step"),
        pytest.param(_descend_scaled_gradient_in_view, _FLOAT32, id="staged-gradient-view"),
        pytest.param(_rotate_by_staged_complex, _COMPLEX64, id="staged-complex"),
        pytest.param(_add_staged_int64, _INT32, id="staged-integer"),
        pytest.param(_shift_by_numpy_int64, _INT32, id="numpy-integer-shift"),
    ],
)
def test_staged_augmented_assignment_casts_back_like_eager_numpy(
    augment: Callable[[Any, Any], None],
    value: np.ndarray[Any, Any],
) -> None:
    # NumPy runs `a op= b` as `op(a, b, out=a)`: a promoted result is cast back
    # to a's dtype under same_kind, whether b is concrete or computed.
    def operation(value: Any, grad: Any) -> Any:
        result = value.copy()
        augment(result, grad)
        return result * result

    grad = value[::-1].copy()
    expected = operation(value, grad)
    spec = ad.ArraySpec(value.shape, value.dtype.name)
    program = ad.stage(operation, specs=(spec, spec))
    for staged in (program, ad.StagedProgram.from_dict(program.to_dict())):
        actual = staged(value, grad)
        assert actual.dtype == expected.dtype == value.dtype
        np.testing.assert_array_equal(actual, expected)

    if value.dtype.kind in "fc":
        primal, dynamic_pullback = ad.vjp(operation)(value, grad)
        assert primal.dtype == value.dtype
        cotangent = np.conj(primal) + 1.0
        np.testing.assert_allclose(
            ad.vjp_program(program)(value, grad, cotangent=cotangent),
            dynamic_pullback(cotangent),
            rtol=1e-6,
        )


@pytest.mark.parametrize(
    ("operand", "dtype"),
    [
        pytest.param(lambda _grad: np.float64(1.5), "int32", id="numpy-float-into-int"),
        pytest.param(lambda _grad: np.complex128(1j), "float32", id="numpy-complex-into-float"),
        pytest.param(lambda grad: grad.astype(np.float64), "int32", id="staged-float-into-int"),
        pytest.param(
            lambda grad: grad.astype(np.complex128), "float32", id="staged-complex-into-float"
        ),
    ],
)
def test_staged_augmented_assignment_rejects_what_numpy_cannot_cast_back(
    operand: Callable[[Any], object],
    dtype: str,
) -> None:
    def operation(value: Any, grad: Any) -> Any:
        result = value.copy()
        result += operand(grad)
        return result

    value = np.ones(3, dtype=dtype)
    with pytest.raises(TypeError):
        operation(value, value)
    with pytest.raises(ad.MutationError, match="would change shape or dtype"):
        ad.stage(operation, specs=(ad.ArraySpec((3,), dtype),) * 2)


def test_staged_item_assignment_drops_leading_unit_dimensions_like_numpy() -> None:
    def update(base: Any, replacement: Any) -> Any:
        result = base.copy()
        result[1:, 1:3] = replacement
        return result

    def loss(base: Any, replacement: Any) -> Any:
        return np.sum(update(base, replacement) ** 2)

    base = np.arange(12.0).reshape(3, 4)
    replacement = np.array([[[5.0], [-6.0]]])
    specs = (ad.ArraySpec(base.shape, "float64"), ad.ArraySpec(replacement.shape, "float64"))
    assert_staged_round_trip(update, base, replacement)
    dynamic = ad.grad(loss, argnums=(0, 1))(base, replacement)
    staged = ad.grad(ad.stage(loss, specs=specs), argnums=(0, 1))(base, replacement)
    assert_tree_close(staged, dynamic)
    with pytest.raises(ValueError, match="Cannot assign shape"):
        ad.stage(update, specs=(specs[0], ad.ArraySpec((2, 2, 2), "float64")))


def _assign_at(index: object) -> Callable[[Any, Any], Any]:
    def update(base: Any, replacement: Any) -> Any:
        result = base.copy()
        result[index] = replacement
        return result

    return update


def _base_and_unit_value(shape: tuple[int, ...]) -> tuple[np.ndarray[Any, Any], ...]:
    return np.arange(float(np.prod(shape))).reshape(shape), np.array([[9.0]])


@pytest.mark.parametrize(
    ("shape", "index"),
    [((3, 4), (0, 0, Ellipsis)), ((), Ellipsis), ((4,), (None, 0))],
    ids=("integers-ellipsis", "ellipsis", "new-axis"),
)
def test_staged_view_assignment_drops_leading_unit_dimensions(
    shape: tuple[int, ...],
    index: object,
) -> None:
    assert_staged_round_trip(_assign_at(index), *_base_and_unit_value(shape))


@pytest.mark.parametrize(
    ("shape", "index"),
    [((4,), 2), ((3, 4), (0, 0)), ((), ())],
    ids=("integer", "integers", "empty-tuple"),
)
def test_staged_element_assignment_rejects_unit_sized_arrays_like_numpy(
    shape: tuple[int, ...],
    index: object,
) -> None:
    update = _assign_at(index)
    values = _base_and_unit_value(shape)
    # NumPy stores one element from a scalar, never from a unit-sized array.
    with pytest.raises(ValueError, match="sequence"):
        update(*values)
    with pytest.raises(ValueError, match="sequence"):
        ad.grad(lambda base, value: np.sum(update(base, value)), argnums=1)(*values)
    with pytest.raises(ValueError, match="Cannot assign shape"):
        ad.stage(update, specs=tuple(ad.ArraySpec(value.shape, value.dtype) for value in values))
