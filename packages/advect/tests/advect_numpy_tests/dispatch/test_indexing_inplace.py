"""Functionalized mutation contracts exercised through dynamic transforms."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import advect as ad
from advect.numpy._traced_array_state import user_location


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


def test_basic_item_assignment_overwrites_the_selected_tangent() -> None:
    def update(source: Any) -> Any:
        current = source.copy()
        current[1:] = np.array([8.0, 9.0])
        return current

    original = np.array([1.0, 2.0, 3.0])
    tangent_in = np.array([2.0, 3.0, 4.0])
    value, tangent = ad.jvp(update)(original, tangents=tangent_in)

    np.testing.assert_array_equal(value, [1.0, 8.0, 9.0])
    np.testing.assert_array_equal(tangent, [2.0, 0.0, 0.0])


def test_literal_slice_augmented_assignment_is_differentiable() -> None:
    def update(source: Any) -> Any:
        current = source.copy()
        current[1:-1] += 10.0
        return current

    original = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
    tangent_in = np.linspace(-1.0, 1.0, original.size)
    value, tangent = ad.jvp(update)(original, tangents=tangent_in)

    np.testing.assert_array_equal(value, [0.0, 11.0, 14.0, 19.0, 16.0])
    np.testing.assert_array_equal(tangent, tangent_in)


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


@pytest.mark.parametrize(
    "key",
    [
        0,
        (slice(None), 1),
        Ellipsis,
        (None, Ellipsis),
        slice(None, None, 2),
    ],
    ids=("integer", "tuple", "ellipsis", "newaxis", "step-slice"),
)
def test_named_view_update_supports_each_basic_index_form(key: object) -> None:
    def update(source: Any) -> Any:
        current = source.copy()
        view = current[key]
        view += 2.0
        return current

    source = np.arange(12.0).reshape(3, 4)
    expected = source.copy()
    expected[key] += 2.0

    value, tangent = ad.jvp(update)(source, tangents=np.ones_like(source))

    np.testing.assert_array_equal(value, expected)
    np.testing.assert_array_equal(tangent, np.ones_like(source))


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


def test_advanced_index_assignment_is_explicitly_rejected() -> None:
    def update(source: Any) -> Any:
        current = source.copy()
        current[np.array([0, 2])] = 1.0
        return current

    source = np.zeros(4)
    with pytest.raises(ad.TracingError, match="Advanced-index assignment"):
        ad.jvp(update)(source, tangents=np.ones_like(source))


def test_putmask_functionalizes_flattened_mask_and_repeated_values() -> None:
    mask = np.array([True, False, True, False, True, False])

    def update(source: Any, replacements: Any) -> Any:
        result = source.copy()
        np.putmask(result, mask, replacements)
        return result

    source = np.arange(6.0).reshape(2, 3)
    replacements = np.array([10.0, 20.0])
    source_tangent = np.linspace(-0.5, 0.5, 6).reshape(2, 3)
    replacement_tangent = np.array([0.25, -0.75])

    value, tangent = ad.jvp(update, argnums=(0, 1))(
        source,
        replacements,
        tangents=(source_tangent, replacement_tangent),
    )
    expected = update(source, replacements)
    expected_tangent = update(source_tangent, replacement_tangent)

    np.testing.assert_array_equal(value, expected)
    np.testing.assert_array_equal(tangent, expected_tangent)
    np.testing.assert_array_equal(source, np.arange(6.0).reshape(2, 3))


def test_put_along_axis_broadcasts_indices_outside_indexed_axis() -> None:
    indices = np.array([[[0], [2], [1]]])

    def update(source: Any, replacements: Any) -> Any:
        result = source.copy()
        np.put_along_axis(result, indices, replacements, axis=1)
        return result

    source = np.arange(24.0).reshape(2, 3, 4)
    replacements = np.array([[[100.0], [200.0], [300.0]]])
    source_tangent = np.linspace(-1.0, 1.0, source.size).reshape(source.shape)
    replacement_tangent = np.array([[[0.2], [-0.3], [0.5]]])

    value, tangent = ad.jvp(update, argnums=(0, 1))(
        source,
        replacements,
        tangents=(source_tangent, replacement_tangent),
    )

    np.testing.assert_array_equal(value, update(source, replacements))
    np.testing.assert_array_equal(
        tangent,
        update(source_tangent, replacement_tangent),
    )


@pytest.mark.parametrize("operation", ["copyto", "place", "fill_diagonal", "put"])
def test_numpy_mutation_functions_match_their_linearized_updates(operation: str) -> None:
    def update(source: Any, replacements: Any) -> Any:
        result = source.copy()
        if operation == "copyto":
            np.copyto(result, replacements, where=np.array([[True, False], [False, True]]))
        elif operation == "place":
            np.place(result, np.array([[True, False], [True, True]]), replacements)
        elif operation == "fill_diagonal":
            np.fill_diagonal(result, replacements)
        else:
            np.put(result, [0, 3, 0], replacements, mode="raise")
        return result

    source = np.arange(4.0).reshape(2, 2)
    replacements = (
        np.array([[10.0, 20.0], [30.0, 40.0]])
        if operation == "copyto"
        else np.array([10.0, 20.0, 30.0])
    )
    source_tangent = np.array([[0.1, 0.2], [0.3, 0.4]])
    replacement_tangent = np.full_like(replacements, -0.5)

    value, tangent = ad.jvp(update, argnums=(0, 1))(
        source,
        replacements,
        tangents=(source_tangent, replacement_tangent),
    )

    np.testing.assert_array_equal(value, update(source, replacements))
    np.testing.assert_array_equal(
        tangent,
        update(source_tangent, replacement_tangent),
    )
