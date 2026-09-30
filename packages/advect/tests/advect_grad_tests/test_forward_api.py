"""Contract tests for forward-mode public APIs."""

from __future__ import annotations

import re
from typing import Any

import array_api_strict as strict
import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings
from numpy.testing import assert_allclose

import advect as ad
from _pytree_strategies import pytree_array_mode_tree
from advect.autodiff._ephemeral import _PULLBACK_MANY_BATCH_SIZE, LinearMap
from advect.autodiff.api.forward import _jacobian_forward, _jacobian_reverse


def test_jacobian_scalar_input_preserves_scalar_shape() -> None:
    jacobian_fn = ad.jacobian(lambda x: x * x)
    assert jacobian_fn(3.0) == pytest.approx(6.0)

    jac = jacobian_fn(np.float64(3.0))

    jac_arr = np.asarray(jac)
    assert jac_arr.shape == ()
    assert float(jac_arr) == pytest.approx(6.0)


def test_jacobian_supports_multiple_arguments() -> None:
    left = np.array([1.0, 2.0])
    right = np.array([3.0, 4.0, 5.0])

    left_block, right_block = ad.jacobian(
        lambda x, y: x[:, None] * y[None, :],
        argnums=(0, 1),
    )(left, right)

    expected_left = np.einsum("ik,j->ijk", np.eye(left.size), right)
    expected_right = np.einsum("i,jk->ijk", left, np.eye(right.size))
    assert left_block.shape == (2, 3, 2)
    assert right_block.shape == (2, 3, 3)
    assert_allclose(left_block, expected_left)
    assert_allclose(right_block, expected_right)


def test_jacobian_supports_named_argument_selection() -> None:
    value = np.array([1.0, 2.0, 3.0])
    scale = np.array(2.0)

    actual = ad.jacobian(
        lambda x, *, scale: scale * x,
        argnums=None,
        argnames=("scale",),
    )(value, scale=scale)

    assert set(actual) == {"scale"}
    assert actual["scale"].shape == value.shape
    assert_allclose(actual["scale"], value)


def test_jacobian_of_restored_staged_program_remains_shape_preserving() -> None:
    value = np.array([1.0, 2.0, 3.0])
    spec = ad.ArraySpec(value.shape, value.dtype)
    program = ad.stage(lambda x: x * x, specs=(spec,))
    restored = ad.StagedProgram.from_dict(program.to_dict())

    actual = ad.jacobian(restored)(value)

    assert_allclose(actual, np.diag(2 * value))


def test_jacobian_remains_array_api_provider_neutral() -> None:
    value = strict.asarray([1.0, 2.0, 3.0], dtype=strict.float32)

    actual = ad.jacobian(lambda x: x * x)(value)

    assert type(actual) is type(value)
    assert actual.dtype == value.dtype
    assert actual.device == value.device
    assert_allclose(np.asarray(actual), np.diag([2.0, 4.0, 6.0]))


def test_forward_selected_jacobian_remains_array_api_provider_neutral() -> None:
    value = strict.asarray(2.0, dtype=strict.float64)
    coefficients = strict.asarray([1.0, 2.0, 3.0], dtype=strict.float64)

    actual = ad.jacobian(lambda x: x * coefficients)(value)

    assert type(actual) is type(value)
    assert_allclose(np.asarray(actual), [1.0, 2.0, 3.0])


def test_debug_jacobian_preserves_an_untraceable_pytree_leaf() -> None:
    params = {"weight": np.array(2.0), "label": "fixed"}

    with ad.debug(), pytest.warns(UserWarning, match="untraceable.*label"):
        jacobian = ad.jacobian(lambda tree: tree["weight"] * np.arange(1.0, 4.0))(params)

    assert_allclose(jacobian["weight"], np.arange(1.0, 4.0))
    assert jacobian["label"] is None


def test_jacobian_supports_constant_empty_and_untraceable_boundaries() -> None:
    assert ad.jacobian(lambda _value: 3.0)(2.0) == pytest.approx(0.0)
    assert ad.jacobian(lambda _value: {})(np.ones(2)) == {}
    assert ad.jacobian(lambda _label: 3.0)("fixed") is None
    assert ad.jacobian(lambda _label: np.empty(0))("fixed") is None


@pytest.mark.parametrize(
    ("function", "value", "match"),
    [
        (lambda tree: np.real(tree["z"]), {"z": np.array([1.0 + 2.0j])}, "real inputs"),
        (lambda _value: 1.0j, 2.0, "jacobian requires real outputs"),
    ],
    ids=["complex-input-leaf", "constant-python-complex-output"],
)
def test_jacobian_rejects_complex_leaves_without_guessing_a_dense_convention(
    function: Any, value: object, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        ad.jacobian(function)(value)


def test_zero_input_linear_map_is_reusable() -> None:
    value, linear = ad.linearize(lambda _scalar: 3.0, 2.0, argnums=())

    assert value == pytest.approx(3.0)
    with linear as active:
        assert active(()) == pytest.approx(0.0)
        assert active.pullback(1.0) == ()
        assert active.apply_many(((), ())) == pytest.approx((0.0, 0.0))
        assert active.transpose_many((1.0, 2.0)) == ((), ())


def test_singleton_linear_map_preserves_structure_and_accepts_empty_batches() -> None:
    left = np.array([2.0, 3.0])
    right = np.array([4.0, 5.0])
    value, linear = ad.linearize(
        lambda left, right: left * right,
        left,
        right,
        argnums=(1,),
    )

    assert_allclose(value, left * right)
    with linear as active:
        tangent = active((np.ones_like(right),))
        assert_allclose(tangent, left)
        (gradient,) = active.pullback(np.ones_like(value))
        assert_allclose(gradient, left)
        assert active.apply_many(()) == ()
        assert active.transpose_many(()) == ()


@pytest.mark.parametrize(
    ("cotangent", "error", "match"),
    [
        (np.full(3, 1.0 + 1.0j), TypeError, "cannot have complex dtype"),
        (np.ones(3, dtype=bool), TypeError, "cannot have boolean dtype"),
        (1.0, ValueError, r"expected \(3,\), got \(\)"),
        (np.ones(2), ValueError, r"expected \(3,\), got \(2,\)"),
    ],
    ids=["complex", "bool", "python-scalar", "shape"],
)
def test_linear_map_batch_transpose_validates_cotangents_like_pullback(
    cotangent: object,
    error: type[Exception],
    match: str,
) -> None:
    value = np.array([1.0, 2.0, 3.0])
    _output, linear = ad.linearize(lambda x: x * x, value)

    with linear as active:
        with pytest.raises(error, match=match):
            active.pullback(cotangent)
        with pytest.raises(error, match=match):
            active.transpose_many((np.ones(3), cotangent))
        assert_allclose(active.transpose_many((np.ones(3),))[0], 2.0 * value)


_FINITE = st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_subnormal=False)


@st.composite
def _linear_map_seeds(draw: st.DrawFn) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    value = draw(hnp.arrays(np.float64, st.integers(1, 5), elements=_FINITE))
    # Cross the 16-seed native batch bound, including an empty batch.
    count = draw(st.integers(0, 2 * _PULLBACK_MANY_BATCH_SIZE + 1))
    seed = hnp.arrays(np.float64, value.shape, elements=_FINITE)
    return value, tuple(draw(seed) for _ in range(count))


@given(case=_linear_map_seeds())
@settings(deadline=None)
def test_linear_map_batches_match_single_applications_and_the_jacobian(
    case: tuple[np.ndarray, tuple[np.ndarray, ...]],
) -> None:
    value, seeds = case
    # The reversed product mixes coordinates, so the Jacobian is not symmetric.
    _output, linear = ad.linearize(lambda x: np.tanh(x) * x[::-1], value)
    jacobian = np.diag(value[::-1] / np.cosh(value) ** 2) + np.diag(np.tanh(value))[:, ::-1]

    with linear:
        jvps = linear.apply_many(seeds)
        vjps = linear.transpose_many(seeds)
        assert len(jvps) == len(vjps) == len(seeds)
        for seed, jvp_value, vjp_value in zip(seeds, jvps, vjps, strict=True):
            assert_allclose(jvp_value, linear(seed), rtol=1e-12, atol=1e-12)
            assert_allclose(vjp_value, linear.pullback(seed), rtol=1e-12, atol=1e-12)
            assert_allclose(jvp_value, jacobian @ seed, rtol=1e-12, atol=1e-12)
            assert_allclose(vjp_value, jacobian.T @ seed, rtol=1e-12, atol=1e-12)


_PYTREE_ARRAY_LEAF = hnp.arrays(
    np.float64,
    hnp.array_shapes(min_dims=0, max_dims=2, min_side=0, max_side=3),
    elements=_FINITE,
)


def _is_untraceable(leaf: object) -> bool:
    return leaf is None or isinstance(leaf, str)


def _is_python_number(leaf: object) -> bool:
    return type(leaf) in {int, float}


def _draw_like(data: st.DataObject, reference: object) -> object:
    """Draw a real tangent or cotangent pytree mirroring ``reference``."""
    leaves, treedef = ad.pytree.tree_flatten(reference)
    return ad.pytree.tree_unflatten(
        treedef,
        [
            None
            if _is_untraceable(leaf)
            else data.draw(
                hnp.arrays(np.float64, leaf.shape, elements=_FINITE)
                if isinstance(leaf, np.ndarray)
                else _FINITE
            )
            for leaf in leaves
        ],
    )


def _pytree_map(tree: object) -> dict[str, object]:
    """Map every traceable leaf leafwise, through an alias, a reduction, and a constant."""
    leaves = [leaf for leaf in ad.pytree.tree_leaves(tree) if not _is_untraceable(leaf)]
    return {
        "y": tuple(np.tanh(leaf) * leaf for leaf in leaves),
        "alias": (leaves[0], leaves[0]),
        "total": sum(np.sum(leaf * leaf) for leaf in leaves),
        "const": 3.0,
    }


def _assert_tree_matches(actual: object, expected: object, reference: object) -> None:
    """Match ``expected`` values and the leaf kinds of the primal ``reference``."""
    actual_leaves, treedef = ad.pytree.tree_flatten(actual)
    assert treedef == ad.pytree.tree_flatten(reference)[1]
    for actual_leaf, expected_leaf, reference_leaf in zip(
        actual_leaves,
        ad.pytree.tree_leaves(expected),
        ad.pytree.tree_leaves(reference),
        strict=True,
    ):
        if _is_untraceable(reference_leaf):
            assert actual_leaf is None
            continue
        if _is_python_number(reference_leaf):
            assert type(actual_leaf) is float
        else:
            assert np.shape(actual_leaf) == np.shape(reference_leaf)
            assert np.asarray(actual_leaf).dtype == np.float64
        # Only summation order differs: at most 63 terms of magnitude <= 40.
        assert_allclose(actual_leaf, expected_leaf, rtol=1e-12, atol=1e-10)


@given(data=st.data())
@settings(deadline=None, max_examples=settings.default.max_examples // 4)
def test_linear_map_applies_and_transposes_mixed_pytrees(data: st.DataObject) -> None:
    tree = data.draw(pytree_array_mode_tree(array_leaf=_PYTREE_ARRAY_LEAF, max_leaves=6))
    primal = {"tree": tree, "label": "static", "missing": None}
    input_leaves, input_treedef = ad.pytree.tree_flatten(primal)
    traced = [index for index, leaf in enumerate(input_leaves) if not _is_untraceable(leaf)]
    values = [np.asarray(input_leaves[index], dtype=np.float64) for index in traced]
    slopes = [np.tanh(value) + value / np.cosh(value) ** 2 for value in values]

    def expected_tangent(tangent: object) -> dict[str, object]:
        seeds = [ad.pytree.tree_leaves(tangent)[index] for index in traced]
        return {
            "y": tuple(slope * seed for slope, seed in zip(slopes, seeds, strict=True)),
            "alias": (seeds[0], seeds[0]),
            "total": sum(np.sum(2.0 * x * seed) for x, seed in zip(values, seeds, strict=True)),
            "const": 0.0,
        }

    def expected_gradient(cotangent: dict[str, Any]) -> object:
        leaves: list[object] = [None] * len(input_leaves)
        alias = cotangent["alias"][0] + cotangent["alias"][1]
        for index, slope, seed, x in zip(traced, slopes, cotangent["y"], values, strict=True):
            aliased = alias if index == traced[0] else 0.0
            leaves[index] = slope * seed + 2.0 * x * cotangent["total"] + aliased
        return ad.pytree.tree_unflatten(input_treedef, leaves)

    output, linear = ad.linearize(_pytree_map, primal)
    tangents = tuple(_draw_like(data, primal) for _ in range(data.draw(st.integers(0, 2))))
    cotangents = tuple(_draw_like(data, output) for _ in range(data.draw(st.integers(0, 2))))
    with linear:
        for tangent, batched in zip(tangents, linear.apply_many(tangents), strict=True):
            _assert_tree_matches(linear(tangent), expected_tangent(tangent), output)
            _assert_tree_matches(batched, expected_tangent(tangent), output)
        for cotangent, batched in zip(cotangents, linear.transpose_many(cotangents), strict=True):
            _assert_tree_matches(linear.pullback(cotangent), expected_gradient(cotangent), primal)
            _assert_tree_matches(batched, expected_gradient(cotangent), primal)

        # One invalid cotangent leaf fails a batch exactly as it fails pullback.
        output_leaves, output_treedef = ad.pytree.tree_flatten(output)
        index = data.draw(st.integers(0, len(output_leaves) - 1))
        shape = np.shape(output_leaves[index])
        invalid_leaves = ad.pytree.tree_leaves(_draw_like(data, output))
        invalid_leaves[index] = data.draw(
            st.sampled_from(
                [np.zeros((*shape, 2)), np.zeros(shape, dtype=bool), np.full(shape, 1j)]
                + ([1.0] if shape else [])
            )
        )
        invalid = ad.pytree.tree_unflatten(output_treedef, invalid_leaves)
        with pytest.raises((TypeError, ValueError)) as single:
            linear.pullback(invalid)
        with pytest.raises(single.type, match=re.escape(str(single.value))):
            linear.transpose_many((_draw_like(data, output), invalid))


@st.composite
def _dense_map_case(draw: st.DrawFn) -> tuple[np.ndarray, ...]:
    # Up to 40 seeds cross the 16-seed batch bound in forward (wide) and
    # reverse (tall) assembly; equal sizes exercise the square mode choice.
    inputs = draw(st.integers(0, 40))
    outputs = draw(st.one_of(st.just(inputs), st.integers(0, 40)))
    weights = np.cos(0.7 * np.arange(outputs)[:, None] + 0.3 * np.arange(inputs))
    weights /= np.sqrt(max(inputs, 1))
    value, tangent = (draw(hnp.arrays(np.float64, inputs, elements=_FINITE)) for _ in range(2))
    cotangent = draw(hnp.arrays(np.float64, outputs, elements=_FINITE))
    return weights, value, tangent, cotangent


@given(case=_dense_map_case())
@settings(deadline=None, max_examples=settings.default.max_examples // 4)
def test_jacobian_assembly_matches_jvp_vjp_and_the_closed_form(
    case: tuple[np.ndarray, ...],
) -> None:
    weights, value, tangent, cotangent = case

    def function(x: np.ndarray) -> object:
        return np.tanh(weights @ x)

    jacobian = ad.jacobian(function)(value)
    _output, jvp_value = ad.jvp(function)(value, tangents=tangent)
    _output, pullback = ad.vjp(function)(value)

    assert jacobian.shape == weights.shape
    assert jacobian.dtype == np.float64
    expected = weights / np.cosh(weights @ value)[:, None] ** 2
    # Entries are at most 1 and products sum at most 40 terms.
    assert_allclose(jacobian, expected, rtol=1e-12, atol=1e-12)
    assert_allclose(jacobian @ tangent, jvp_value, rtol=1e-12, atol=1e-12)
    assert_allclose(cotangent @ jacobian, pullback(cotangent), rtol=1e-12, atol=1e-12)


_COEFFICIENTS = np.arange(1.0, 4.0)


def _mixed_outputs(tree: dict[str, Any]) -> dict[str, object]:
    """Map to empty, input-shaped and float64 leaves; only ``first`` skips inputs."""
    leaves = ad.pytree.tree_leaves(tree)
    total = sum(np.sum(np.tanh(leaf)) * (position + 1) for position, leaf in enumerate(leaves))
    return {
        "empty": np.zeros(0) * total,
        "first": np.sin(leaves[0]),
        "vector": np.cos(_COEFFICIENTS * total),
    }


def _basis_tangent(leaves: list[Any], index: int, coordinate: int) -> list[Any]:
    tangents = [np.zeros_like(leaf) if isinstance(leaf, np.ndarray) else 0.0 for leaf in leaves]
    if isinstance(leaves[index], np.ndarray):
        tangents[index].reshape(-1)[coordinate] = 1.0
    else:
        tangents[index] = 1.0
    return tangents


@given(data=st.data())
@settings(deadline=None, max_examples=settings.default.max_examples // 4)
def test_forward_and_reverse_jacobians_assemble_the_same_pytree(data: st.DataObject) -> None:
    arrays = data.draw(
        st.lists(
            hnp.arrays(
                st.sampled_from([np.float32, np.float64]),
                hnp.array_shapes(min_dims=0, max_dims=2, min_side=0, max_side=3),
                elements=st.floats(min_value=-2.0, max_value=2.0, width=32),
            ),
            min_size=1,
            max_size=3,
        )
    )
    tree: dict[str, Any] = {f"x{index}": array for index, array in enumerate(arrays)}
    if data.draw(st.booleans()):
        tree["python"] = data.draw(_FINITE)
    leaves, treedef = ad.pytree.tree_flatten(tree)
    value, linear = ad.linearize(_mixed_outputs, tree)
    output_leaves, output_treedef = ad.pytree.tree_flatten(value)
    with linear:
        # Oracle columns: one unbatched application per input coordinate.
        columns = [
            [
                ad.pytree.tree_leaves(
                    linear(
                        ad.pytree.tree_unflatten(treedef, _basis_tangent(leaves, index, coordinate))
                    )
                )
                for coordinate in range(np.size(leaf))
            ]
            for index, leaf in enumerate(leaves)
        ]
        jacobians = [
            assemble(linear, output_leaves=output_leaves, output_treedef=output_treedef)
            for assemble in (_jacobian_forward, _jacobian_reverse)
        ]
    jacobians.append(ad.jacobian(_mixed_outputs)(tree))

    expected_treedef = ad.pytree.tree_flatten(ad.pytree.tree_map(lambda _leaf: tree, value))[1]
    # Float32 leaves set the working precision of every block.
    eps = max(np.finfo(getattr(leaf, "dtype", float)).eps for leaf in leaves)
    for jacobian in jacobians:
        blocks, blocks_treedef = ad.pytree.tree_flatten(jacobian)
        assert blocks_treedef == expected_treedef
        for position, block in enumerate(blocks):
            output_index, input_index = divmod(position, len(leaves))
            output_shape = np.shape(output_leaves[output_index])
            leaf = leaves[input_index]
            # Each block lives in its input's tangent-space dtype.
            dtype = np.dtype(getattr(leaf, "dtype", np.float64))
            if output_shape == () and not isinstance(leaf, np.ndarray):
                assert type(block) is float
            assert np.shape(block) == output_shape + np.shape(leaf)
            assert np.asarray(block).dtype == dtype
            expected = [
                np.asarray(column[output_index], dtype=np.float64).reshape(-1)
                for column in columns[input_index]
            ]
            expected_block = np.stack(expected, axis=-1) if expected else np.zeros((0,))
            # Entries are at most 3 * 4 = 12; the worst measured error is 16 eps.
            assert_allclose(
                np.reshape(block, -1), np.reshape(expected_block, -1), rtol=0, atol=64 * eps
            )


def test_jacobian_chooses_modes_by_shape_and_square_trace_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"forward": 0, "reverse": 0}
    original_forward = LinearMap._apply_seed_tables_many
    original_reverse = LinearMap._transpose_seed_tables_many

    def tracked_forward(
        self: LinearMap,
        tangent_seed_sets: tuple[dict[int, object], ...],
    ) -> tuple[object, ...]:
        calls["forward"] += 1
        return original_forward(self, tangent_seed_sets)

    def tracked_reverse(
        self: LinearMap,
        output_cotangent_sets: tuple[dict[int, object], ...],
    ) -> tuple[dict[int, object], ...]:
        calls["reverse"] += 1
        return original_reverse(self, output_cotangent_sets)

    monkeypatch.setattr(LinearMap, "_apply_seed_tables_many", tracked_forward)
    monkeypatch.setattr(LinearMap, "_transpose_seed_tables_many", tracked_reverse)

    # Local primitives keep the square cases independent of built-in rule kinds.
    @ad.primitive(name="tests.jacobian.square_jvp_only")
    def jvp_only(x: np.ndarray) -> np.ndarray:
        return 2.0 * x

    @jvp_only.def_jvp
    def jvp_only_jvp(output: object, primals: object, tangents: tuple[Any, ...]) -> object:
        del output, primals
        return 2.0 * tangents[0]

    @ad.primitive(name="tests.jacobian.square_direct")
    def direct(x: np.ndarray) -> np.ndarray:
        return 2.0 * x

    @direct.def_jvp
    def direct_jvp(output: object, primals: object, tangents: tuple[Any, ...]) -> object:
        del output, primals
        return 2.0 * tangents[0]

    @direct.def_transpose
    def direct_transpose(cotangent: Any, primals: object, output: object) -> tuple[object]:
        del primals, output
        return (2.0 * cotangent,)

    wide = ad.jacobian(lambda x: x * np.arange(1.0, 9.0))(np.array(2.0))
    assert_allclose(wide, np.arange(1.0, 9.0))
    assert calls == {"forward": 1, "reverse": 0}

    calls.update(forward=0, reverse=0)
    tall = ad.jacobian(np.sum)(np.arange(8.0))
    assert_allclose(tall, np.ones(8))
    assert calls == {"forward": 0, "reverse": 1}

    calls.update(forward=0, reverse=0)
    square_jvp_first = ad.jacobian(jvp_only)(np.arange(1.0, 9.0))
    assert_allclose(square_jvp_first, 2.0 * np.eye(8))
    assert calls == {"forward": 1, "reverse": 0}

    calls.update(forward=0, reverse=0)
    square_direct_vjp = ad.jacobian(direct)(np.arange(1.0, 9.0))
    assert_allclose(square_direct_vjp, 2.0 * np.eye(8))
    assert calls == {"forward": 0, "reverse": 1}

    calls.update(forward=0, reverse=0)
    many_leaves = ad.jacobian(
        lambda x: tuple(x[index] for index in range(20)),
    )(np.arange(32.0))
    assert len(many_leaves) == 20
    assert_allclose(many_leaves[0], np.eye(32)[0])
    assert_allclose(many_leaves[-1], np.eye(32)[19])
    assert calls == {"forward": 0, "reverse": 2}


def test_wide_jacobian_uses_reverse_for_a_transpose_only_residual_primitive() -> None:
    released: list[object] = []

    @ad.primitive(name="tests.jacobian.transpose_only_residual", residual=True)
    def remote(x: np.ndarray) -> ad.PrimitiveResult[np.ndarray]:
        matrix = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 0.0],
                [0.0, 3.0],
            ]
        )
        return ad.PrimitiveResult(matrix @ x, matrix, release=released.append)

    @remote.def_transpose
    def transpose(
        cotangent: np.ndarray,
        primals: tuple[np.ndarray, ...],
        output: np.ndarray,
        residual: object,
    ) -> tuple[np.ndarray]:
        del primals, output
        return (np.asarray(residual).T @ cotangent,)

    value = np.array([2.0, 5.0])
    actual = ad.jacobian(remote)(value)

    assert_allclose(
        actual,
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 0.0],
                [0.0, 3.0],
            ]
        ),
    )
    assert len(released) == 1


def test_reverse_jacobian_derives_one_structural_transpose_per_seed_group() -> None:
    jvp_calls = 0

    @ad.primitive(name="tests.jacobian.counted_jvp_only")
    def scale(x: np.ndarray) -> np.ndarray:
        return 2.0 * x

    @scale.def_jvp
    def scale_jvp(output: object, primals: object, tangents: tuple[Any, ...]) -> object:
        nonlocal jvp_calls
        jvp_calls += 1
        del output, primals
        return 2.0 * tangents[0]

    actual = ad.jacobian(lambda x: scale(x[:20]))(np.arange(1.0, 33.0))

    assert_allclose(actual, 2.0 * np.eye(20, 32))
    # Twenty cotangent seeds form two 16-seed groups, each transposing one JVP trace.
    assert jvp_calls == 2


def test_forward_selected_jacobian_remains_differentiable() -> None:
    weights = np.arange(1.0, 5.0)
    inner = ad.jacobian(lambda x: x * x * weights)

    actual = ad.jacobian(inner)(2.0)

    assert_allclose(actual, 2.0 * weights)


def test_forward_selected_jacobian_of_an_array_composes_with_other_transforms() -> None:
    # The tall output selects forward mode, whose seeds are basis rows of the
    # traced input when an outer transform or stage is active.
    inner = ad.jacobian(lambda u: np.concatenate([np.tanh(u), u * u, np.sin(u)]))
    value = np.array([0.3, 0.7])
    column_sums = 1.0 / np.cosh(value) ** 2 + 2.0 * value + np.cos(value)
    column_sum_derivative = -2.0 * np.tanh(value) / np.cosh(value) ** 2 + 2.0 - np.sin(value)
    staged = ad.stage(inner, specs=(ad.ArraySpec(value.shape, "float64"),))

    assert_allclose(np.sum(inner(value), axis=0), column_sums)
    assert_allclose(ad.grad(lambda v: np.sum(inner(v)))(value), column_sum_derivative)
    assert_allclose(
        ad.jacobian(lambda v: np.sum(inner(v), axis=0))(value),
        np.diag(column_sum_derivative),
    )
    assert_allclose(staged(value), inner(value))


def test_reverse_selected_jacobian_remains_differentiable() -> None:
    inner = ad.jacobian(lambda x: np.sum(x * x))
    value = np.array([1.0, 2.0, 3.0])

    actual = ad.jacobian(inner)(value)

    assert_allclose(actual, 2.0 * np.eye(value.size))
