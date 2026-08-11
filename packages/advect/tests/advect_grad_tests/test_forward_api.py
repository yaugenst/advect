"""Contract tests for forward-mode public APIs."""

from __future__ import annotations

import array_api_strict as strict
import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad
import advect.autodiff._ephemeral as ephemeral
from advect.autodiff._ephemeral import LinearMap


def test_jacobian_scalar_input_preserves_scalar_shape() -> None:
    jacobian_fn = ad.jacobian(lambda x: x * x)
    assert jacobian_fn(3.0) == pytest.approx(6.0)

    jac = jacobian_fn(np.float64(3.0))

    jac_arr = np.asarray(jac)
    assert jac_arr.shape == ()
    assert float(jac_arr) == pytest.approx(6.0)


def test_jacobian_preserves_tensor_axes() -> None:
    value = np.arange(6.0).reshape(2, 3)

    actual = ad.jacobian(lambda x: x * x)(value)
    expected = np.diag(2 * value.reshape(-1)).reshape(2, 3, 2, 3)

    assert actual.shape == (2, 3, 2, 3)
    assert_allclose(actual, expected)


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


def test_jacobian_returns_the_output_tree_of_input_tree_blocks() -> None:
    parameters = {
        "weight": np.array([1.0, 2.0]),
        "bias": np.array([3.0, 4.0]),
    }

    def function(tree: dict[str, object]) -> dict[str, object]:
        return {
            "vector": tree["weight"] * tree["bias"],
            "scalar": np.sum(tree["weight"] + tree["bias"]),
        }

    actual = ad.jacobian(function)(parameters)

    assert set(actual) == {"vector", "scalar"}
    assert set(actual["vector"]) == {"weight", "bias"}
    assert_allclose(actual["vector"]["weight"], np.diag(parameters["bias"]))
    assert_allclose(actual["vector"]["bias"], np.diag(parameters["weight"]))
    assert_allclose(actual["scalar"]["weight"], np.ones(2))
    assert_allclose(actual["scalar"]["bias"], np.ones(2))


def test_forward_selected_jacobian_preserves_input_and_output_pytrees() -> None:
    parameters = {
        "weight": np.array(2.0),
        "bias": np.array(3.0),
    }
    coefficients = np.arange(1.0, 5.0)

    def function(tree: dict[str, object]) -> dict[str, object]:
        return {
            "weighted": tree["weight"] * coefficients,
            "shifted": tree["bias"] + coefficients,
        }

    actual = ad.jacobian(function)(parameters)

    assert_allclose(actual["weighted"]["weight"], coefficients)
    assert_allclose(actual["weighted"]["bias"], np.zeros_like(coefficients))
    assert_allclose(actual["shifted"]["weight"], np.zeros_like(coefficients))
    assert_allclose(actual["shifted"]["bias"], np.ones_like(coefficients))


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


def test_jacobian_handles_empty_output_axes() -> None:
    value = np.empty((0,), dtype=np.float64)

    actual = ad.jacobian(lambda x: x * x)(value)

    assert actual.shape == (0, 0)


def test_reverse_selected_jacobian_combines_empty_and_nonempty_output_leaves() -> None:
    value = np.arange(3.0)

    actual = ad.jacobian(
        lambda x: {
            "empty": x[:0],
            "sum": np.sum(x),
        }
    )(value)

    assert actual["empty"].shape == (0, 3)
    assert_allclose(actual["sum"], np.ones(3))


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


def test_jacobian_rejects_complex_leaves_without_guessing_a_dense_convention() -> None:
    value = {"z": np.array([1.0 + 2.0j])}

    with pytest.raises(ValueError, match="real inputs"):
        ad.jacobian(lambda tree: np.real(tree["z"]))(value)


def test_linear_map_applies_multiple_jvp_and_vjp_seeds() -> None:
    value = np.array([2.0, 3.0])
    _output, linear = ad.linearize(lambda x: x * x, value)
    seeds = tuple(np.full_like(value, float(index)) for index in range(1, 19))

    try:
        jvps = linear.apply_many(seeds)
        vjps = linear.transpose_many(seeds)
    finally:
        linear.close()

    assert len(jvps) == len(vjps) == 18
    for index, (jvp_value, vjp_value) in enumerate(zip(jvps, vjps, strict=True), start=1):
        expected = 2.0 * value * index
        assert_allclose(jvp_value, expected)
        assert_allclose(vjp_value, expected)


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

    wide = ad.jacobian(lambda x: x * np.arange(1.0, 9.0))(np.array(2.0))
    assert_allclose(wide, np.arange(1.0, 9.0))
    assert calls == {"forward": 1, "reverse": 0}

    calls.update(forward=0, reverse=0)
    tall = ad.jacobian(np.sum)(np.arange(8.0))
    assert_allclose(tall, np.ones(8))
    assert calls == {"forward": 0, "reverse": 1}

    calls.update(forward=0, reverse=0)
    square_jvp_first = ad.jacobian(np.log)(np.arange(1.0, 9.0))
    assert_allclose(square_jvp_first, np.diag(1.0 / np.arange(1.0, 9.0)))
    assert calls == {"forward": 1, "reverse": 0}

    calls.update(forward=0, reverse=0)
    square_direct_vjp = ad.jacobian(np.sin)(np.arange(1.0, 9.0))
    assert_allclose(square_direct_vjp, np.diag(np.cos(np.arange(1.0, 9.0))))
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


def test_reverse_jacobian_derives_one_structural_transpose_per_seed_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structural_traces = 0
    original_trace_call = ephemeral.trace_call

    def tracked_trace_call(*args: object, **kwargs: object) -> object:
        nonlocal structural_traces
        function = args[0]
        if getattr(function, "__name__", None) == "jvp_program":
            structural_traces += 1
        return original_trace_call(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ephemeral, "trace_call", tracked_trace_call)
    value = np.arange(1.0, 33.0)

    actual = ad.jacobian(lambda x: np.log(x[:20]))(value)

    expected = np.zeros((20, 32))
    expected[np.arange(20), np.arange(20)] = 1.0 / value[:20]
    assert_allclose(actual, expected)
    assert structural_traces == 2


def test_forward_selected_jacobian_preserves_the_input_tangent_dtype() -> None:
    value = np.array([2.0], dtype=np.float32)

    actual = ad.jacobian(
        lambda x: x.astype(np.float64) * np.arange(1.0, 5.0),
    )(value)

    assert actual.dtype == value.dtype
    assert_allclose(actual, np.arange(1.0, 5.0)[:, None])


def test_forward_selected_jacobian_remains_differentiable() -> None:
    weights = np.arange(1.0, 5.0)
    inner = ad.jacobian(lambda x: x * x * weights)

    actual = ad.jacobian(inner)(2.0)

    assert_allclose(actual, 2.0 * weights)


def test_reverse_selected_jacobian_remains_differentiable() -> None:
    inner = ad.jacobian(lambda x: np.sum(x * x))
    value = np.array([1.0, 2.0, 3.0])

    actual = ad.jacobian(inner)(value)

    assert_allclose(actual, 2.0 * np.eye(value.size))
