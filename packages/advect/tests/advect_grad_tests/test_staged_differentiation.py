"""Durable reverse-mode transforms over staged programs."""

from __future__ import annotations

import array_api_strict as strict
import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad
from advect.core._array_api.profiles import LATEST_ARRAY_API_VERSION


def test_grad_preserves_staged_lifetime_and_round_trips() -> None:
    calls = 0

    def loss(x: object) -> object:
        nonlocal calls
        calls += 1
        return np.sum(np.sin(x) ** 2)

    spec = ad.ArraySpec((4,), "float64")
    primal = ad.stage(loss, specs=(spec,))
    gradient = ad.grad(primal)

    assert isinstance(gradient, ad.StagedProgram)
    assert calls == 1
    assert gradient.optimization.nodes_after == gradient.graph.node_count
    assert tuple(item.name for item in gradient.optimization.passes) == (
        "dce",
        "simplify",
        "cse",
    )
    assert gradient.optimization.passes[0].removed_nodes > 0
    assert any(
        gradient.graph.get_node(node_id).op == "array.cos" for node_id in gradient.graph.node_ids()
    )

    x = np.arange(4.0)
    expected = 2 * np.sin(x) * np.cos(x)
    assert_allclose(gradient(x), expected)
    assert_allclose(gradient(x + 1), 2 * np.sin(x + 1) * np.cos(x + 1))
    assert calls == 1

    restored = ad.StagedProgram.from_dict(gradient.to_dict())
    assert_allclose(restored(x), expected)


def test_value_and_grad_and_aux_are_staged_outputs() -> None:
    def loss(x: object) -> tuple[object, object]:
        return np.sum(x * x), x + 1

    primal = ad.stage(loss, specs=(ad.ArraySpec((3,), "float64"),))
    gradient = ad.grad(primal, has_aux=True)
    combined = ad.value_and_grad(primal, has_aux=True)
    x = np.arange(3.0)

    assert isinstance(gradient, ad.StagedProgram)
    assert isinstance(combined, ad.StagedProgram)
    grad_value, grad_aux = gradient(x)
    value, combined_gradient, combined_aux = combined(x)
    assert_allclose(grad_value, 2 * x)
    assert_allclose(grad_aux, x + 1)
    assert_allclose(value, np.sum(x * x))
    assert_allclose(combined_gradient, 2 * x)
    assert_allclose(combined_aux, x + 1)


def test_staged_grad_supports_multi_argument_pytrees_and_keywords() -> None:
    def loss(parameters: dict[str, object], right: object, *, scale: object) -> object:
        return np.sum(scale * (parameters["weight"] * right + parameters["bias"]))

    parameters = {
        "weight": np.array([1.0, 2.0, 3.0]),
        "bias": np.array([4.0, 5.0, 6.0]),
    }
    right = np.array([2.0, 3.0, 4.0])
    scale = np.array([0.5, 1.5, 2.0])
    primal = ad.stage(
        loss,
        specs=(
            {
                "weight": ad.ArraySpec(parameters["weight"].shape, parameters["weight"].dtype),
                "bias": ad.ArraySpec(parameters["bias"].shape, parameters["bias"].dtype),
            },
            ad.ArraySpec(right.shape, right.dtype),
        ),
        kw_specs={"scale": ad.ArraySpec(scale.shape, scale.dtype)},
    )

    gradient = ad.grad(
        primal,
        argnums=(0, 1),
        argnames=("scale",),
    )
    positional, named = gradient(parameters, right, scale=scale)

    assert isinstance(gradient, ad.StagedProgram)
    assert_allclose(positional[0]["weight"], scale * right)
    assert_allclose(positional[0]["bias"], scale)
    assert_allclose(positional[1], scale * parameters["weight"])
    assert_allclose(
        named["scale"],
        parameters["weight"] * right + parameters["bias"],
    )
    restored = ad.StagedProgram.from_dict(primal.to_dict())
    restored_gradient = ad.grad(
        restored,
        argnums=(0, 1),
        argnames=("scale",),
    )
    restored_positional, restored_named = restored_gradient(
        parameters,
        right,
        scale=scale,
    )
    assert_allclose(restored_positional[0]["weight"], scale * right)
    assert_allclose(restored_named["scale"], named["scale"])


def test_staged_grad_preserves_unselected_outer_trace_operands() -> None:
    def loss(value: object, weight: object, *, scale: object) -> object:
        return np.sum(scale * weight * value * value)

    value = np.array([1.0, 2.0, 3.0])
    weight = np.array([4.0, 5.0, 6.0])
    scale = np.array([0.5, 1.5, 2.0])
    primal = ad.stage(
        loss,
        specs=(
            ad.ArraySpec(value.shape, value.dtype),
            ad.ArraySpec(weight.shape, weight.dtype),
        ),
        kw_specs={"scale": ad.ArraySpec(scale.shape, scale.dtype)},
    )

    gradient = ad.grad(primal, argnums=0)
    combined = ad.value_and_grad(primal, argnums=0)
    expected = 2 * scale * weight * value

    assert_allclose(gradient(value, weight, scale=scale), expected)
    result, actual = combined(value, weight, scale=scale)
    assert_allclose(result, loss(value, weight, scale=scale))
    assert_allclose(actual, expected)

    restored = ad.StagedProgram.from_dict(gradient.to_dict())
    assert_allclose(restored(value, weight, scale=scale), expected)


def test_vjp_program_is_a_serializable_staged_pullback() -> None:
    def transform(value: object, weight: object) -> dict[str, object]:
        return {
            "field": weight * value,
            "energy": np.sum(np.sin(value)),
        }

    value = np.array([1.0, 2.0, 3.0])
    weight = np.array([4.0, 5.0, 6.0])
    cotangent = {
        "field": np.array([0.5, 1.5, 2.0]),
        "energy": np.array(3.0),
    }
    primal = ad.stage(
        transform,
        specs=(
            ad.ArraySpec(value.shape, value.dtype),
            ad.ArraySpec(weight.shape, weight.dtype),
        ),
    )
    pullback = ad.vjp_program(primal)
    expected = weight * cotangent["field"] + cotangent["energy"] * np.cos(value)

    assert isinstance(pullback, ad.StagedProgram)
    assert_allclose(pullback(value, weight, cotangent=cotangent), expected)

    payload = pullback.to_dict()
    assert payload["format"] == "advect.ssa-program"
    assert payload["version"] == 2
    artifact = payload["program"]
    assert artifact["output_specs"] == [
        {
            "kind": "array",
            "shape": [3],
            "dtype": "float64",
            "device": None,
            "weak": False,
        }
    ]
    restored = ad.StagedProgram.from_dict(payload)
    assert_allclose(restored(value, weight, cotangent=cotangent), expected)


def test_vjp_program_roundtrip_preserves_numpy_full_dispatch_anchor() -> None:
    def fill(value: object) -> object:
        return np.full((2, 3), value, like=value)

    primal = ad.stage(
        fill,
        specs=(ad.ArraySpec((), "float32"),),
        array_api_version=min(np.__array_api_version__, LATEST_ARRAY_API_VERSION),
    )
    pullback = ad.vjp_program(primal)
    restored = ad.StagedProgram.from_dict(pullback.to_dict())
    value = np.asarray(2.5, dtype=np.float32)
    cotangent = np.arange(6, dtype=np.float32).reshape(2, 3)
    expected = np.sum(cotangent, dtype=np.float32)

    for program in (pullback, restored):
        actual = program(value, cotangent=cotangent)
        assert np.asarray(actual).dtype == np.dtype("float32")
        assert_allclose(actual, expected)


def test_vjp_program_roundtrip_preserves_extreme_ldexp_scaling() -> None:
    def scale(value: object, exponent: object) -> object:
        return np.ldexp(value, exponent)

    primal = ad.stage(
        scale,
        specs=(
            ad.ArraySpec((4,), "float64"),
            ad.ArraySpec((4,), "int32"),
        ),
        array_api_version=min(np.__array_api_version__, LATEST_ARRAY_API_VERSION),
    )
    pullback = ad.vjp_program(primal, argnums=0)
    restored = ad.StagedProgram.from_dict(pullback.to_dict())
    value = np.array([np.ldexp(1.0, 100), np.ldexp(1.0, -100), 2.0, 0.5])
    exponent = np.array([-1100, 1100, -1075, 1024], dtype=np.int32)
    cotangent = value.copy()
    expected = np.ldexp(cotangent, exponent)

    assert np.all(np.isfinite(expected))
    assert np.all(expected != 0)
    for program in (pullback, restored):
        actual = program(value, exponent, cotangent=cotangent)
        assert actual.dtype == np.dtype("float64")
        assert_allclose(actual, expected)


def test_vjp_program_supports_multi_argument_and_named_selection() -> None:
    def transform(
        parameters: dict[str, object],
        value: object,
        *,
        scale: object,
    ) -> tuple[object, object]:
        field = scale * (parameters["weight"] * value + parameters["bias"])
        return field, np.sum(field)

    parameters = {
        "weight": np.array([1.0, 2.0, 3.0]),
        "bias": np.array([4.0, 5.0, 6.0]),
    }
    value = np.array([2.0, 3.0, 4.0])
    scale = np.array([0.5, 1.5, 2.0])
    field_cotangent = np.array([2.0, 3.0, 4.0])
    sum_cotangent = np.array(0.25)
    total_cotangent = field_cotangent + sum_cotangent
    cotangent = (field_cotangent, sum_cotangent)

    primal = ad.stage(
        transform,
        specs=(
            {
                "weight": ad.ArraySpec(parameters["weight"].shape, parameters["weight"].dtype),
                "bias": ad.ArraySpec(parameters["bias"].shape, parameters["bias"].dtype),
            },
            ad.ArraySpec(value.shape, value.dtype),
        ),
        kw_specs={"scale": ad.ArraySpec(scale.shape, scale.dtype)},
    )
    pullback = ad.vjp_program(
        primal,
        argnums=(0, 1),
        argnames=("scale",),
    )

    positional, named = pullback(
        parameters,
        value,
        scale=scale,
        cotangent=cotangent,
    )
    assert_allclose(positional[0]["weight"], scale * value * total_cotangent)
    assert_allclose(positional[0]["bias"], scale * total_cotangent)
    assert_allclose(
        positional[1],
        scale * parameters["weight"] * total_cotangent,
    )
    assert_allclose(
        named["scale"],
        (parameters["weight"] * value + parameters["bias"]) * total_cotangent,
    )


def test_vjp_program_preserves_complex_real_adjoint_and_array_api_provider() -> None:
    def conjugate(value: object) -> object:
        xp = value.__array_namespace__()
        return xp.conj(value)

    primal = ad.stage(
        conjugate,
        specs=(ad.ArraySpec((2,), "complex64"),),
    )
    pullback = ad.vjp_program(primal)
    restored = ad.StagedProgram.from_dict(pullback.to_dict())
    value = strict.asarray([1 + 2j, 3 - 4j], dtype=strict.complex64)
    cotangent = strict.asarray([2 - 1j, -0.5 + 3j], dtype=strict.complex64)

    assert_allclose(
        np.asarray(restored(value, cotangent=cotangent)),
        np.conj(np.asarray(cotangent)),
    )


def test_vjp_program_rejects_reserved_keyword_and_nonstaged_callable() -> None:
    with pytest.raises(TypeError, match=r"requires a StagedProgram"):
        ad.vjp_program(lambda value: value)

    primal = ad.stage(
        lambda value, *, cotangent: value * cotangent,
        specs=(ad.ArraySpec((2,), "float64"),),
        kw_specs={"cotangent": ad.ArraySpec((2,), "float64")},
    )
    with pytest.raises(ValueError, match=r"reserves keyword argument 'cotangent'"):
        ad.vjp_program(primal)


def test_staged_grad_compiles_once_at_construction() -> None:
    calls = 0

    def loss(x: object) -> object:
        nonlocal calls
        calls += 1
        return np.sum(x * x)

    primal = ad.stage(loss, specs=(ad.ArraySpec((5,), "float64"),))
    gradient = ad.grad(primal)
    assert calls == 1
    assert primal.graph.node_count > 0
    assert gradient.graph.node_count > 0

    x = np.arange(5.0)
    assert_allclose(gradient(x), 2 * x)
    assert_allclose(gradient(x + 1), 2 * (x + 1))
    assert calls == 1


def test_staged_grad_covers_functionalized_mutation_and_complex_values() -> None:
    def stencil_loss(field: object) -> object:
        field = field.copy()
        laplacian = field[2:] - 2 * field[1:-1] + field[:-2]
        field[1:-1] += 0.1 * laplacian
        return np.sum(field * field)

    field = np.arange(6.0)
    staged_stencil = ad.grad(
        ad.stage(
            stencil_loss,
            specs=(ad.ArraySpec(field.shape, field.dtype),),
        )
    )
    assert_allclose(staged_stencil(field), ad.grad(stencil_loss)(field))

    z = np.array([1 + 2j, -3 + 0.5j], dtype=np.complex64)
    staged_complex = ad.grad(
        ad.stage(
            lambda value: np.sum(np.abs(value) ** 2),
            specs=(ad.ArraySpec(z.shape, z.dtype),),
        )
    )
    assert_allclose(staged_complex(z), 2 * z)


def test_staged_grad_lifts_captured_array_constants() -> None:
    matrix = np.arange(12.0).reshape(3, 4)

    def loss(x: object) -> object:
        return np.sum((matrix @ x) ** 2)

    x = np.arange(4.0)
    gradient = ad.grad(
        ad.stage(
            loss,
            specs=(ad.ArraySpec(x.shape, x.dtype),),
        )
    )

    assert_allclose(gradient(x), ad.grad(loss)(x))
    assert len(gradient.constants) >= 1


def test_staged_grad_remains_provider_portable_array_api_code() -> None:
    def loss(x: object) -> object:
        xp = x.__array_namespace__()
        return xp.sum(xp.sin(x) * x)

    gradient = ad.grad(
        ad.stage(
            loss,
            specs=(ad.ArraySpec((3,), "float64"),),
        )
    )
    x = strict.asarray([1.0, 2.0, 3.0], dtype=strict.float64)
    expected = np.sin([1.0, 2.0, 3.0]) + np.arange(1.0, 4.0) * np.cos([1.0, 2.0, 3.0])

    assert_allclose(np.asarray(gradient(x)), expected)


@pytest.mark.parametrize(
    ("function", "materializes_constants"),
    [
        (lambda value: np.linalg.eig(value)[0], False),
        (lambda value: np.linalg.qr(value, mode="r"), True),
    ],
    ids=("eig", "qr-r"),
)
def test_serialized_staged_vjp_handles_shape_only_rule_values(
    function: object,
    *,
    materializes_constants: bool,
) -> None:
    matrix = np.array(
        [[1.0 + 0.2j, 0.3 - 0.1j], [0.2 + 0.4j, 2.0 - 0.1j]],
        dtype=np.complex128,
    )
    primal = ad.stage(
        function,
        specs=(ad.ArraySpec(matrix.shape, matrix.dtype),),
    )
    pullback = ad.vjp_program(primal)
    output = primal(matrix)
    cotangent = np.ones_like(output)
    _dynamic_output, dynamic_pullback = ad.vjp(function)(matrix)
    expected = dynamic_pullback(cotangent)

    restored = ad.StagedProgram.from_dict(pullback.to_dict())
    assert_allclose(restored(matrix, cotangent=cotangent), expected)
    if materializes_constants:
        assert any(
            pullback.graph.get_node(node_id).op == "advect.const"
            for node_id in pullback.graph.node_ids()
        )


def test_staged_empty_product_gradient_materializes_zero_tangent() -> None:
    def loss(value: object) -> object:
        return np.sum(np.prod(value, axis=-1))

    value = np.empty((3, 0), dtype=np.float64)
    gradient = ad.grad(
        ad.stage(
            loss,
            specs=(ad.ArraySpec(value.shape, value.dtype),),
        )
    )
    restored = ad.StagedProgram.from_dict(gradient.to_dict())

    assert_allclose(restored(value), np.zeros_like(value))


def test_staged_array_api_max_gradient_materializes_selection_grid() -> None:
    def loss(value: object) -> object:
        xp = value.__array_namespace__()
        return xp.sum(xp.max(value, axis=1))

    gradient = ad.grad(
        ad.stage(
            loss,
            specs=(ad.ArraySpec((2, 3), "float64"),),
        )
    )
    restored = ad.StagedProgram.from_dict(gradient.to_dict())
    value = strict.asarray(
        [[1.0, 4.0, 2.0], [5.0, 3.0, 2.0]],
        dtype=strict.float64,
    )

    assert_allclose(
        np.asarray(restored(value)),
        np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    )


def test_staged_grad_keeps_traceable_custom_primitive_atomic() -> None:
    @ad.primitive(name="tests.staged_derivative_custom")
    def primitive(x: object) -> object:
        return x * x

    @primitive.def_abstract
    def abstract(x: ad.AbstractValue) -> ad.ArraySpec:
        return x.spec

    @primitive.def_transpose
    def transpose(
        cotangent: object,
        primals: tuple[object, ...],
        output: object,
    ) -> tuple[object]:
        del output
        return (2 * primals[0] * cotangent,)

    primal = ad.stage(
        lambda x: np.sum(primitive(x)),
        specs=(ad.ArraySpec((3,), "float64"),),
    )
    gradient = ad.grad(primal)
    x = np.array([1.0, 2.0, 3.0])

    assert primitive.op_name in {
        gradient.graph.get_node(node_id).op for node_id in gradient.graph.node_ids()
    }
    assert_allclose(gradient(x), 2 * x)


def test_opaque_residual_is_a_staged_derivative_barrier() -> None:
    @ad.primitive(name="tests.staged_derivative_residual", residual=True)
    def primitive(x: object) -> ad.PrimitiveResult[object]:
        return ad.PrimitiveResult(x * x, x)

    @primitive.def_abstract
    def abstract(x: ad.AbstractValue) -> ad.ArraySpec:
        return x.spec

    @primitive.def_transpose
    def transpose(
        cotangent: object,
        primals: tuple[object, ...],
        output: object,
        residual: object,
    ) -> tuple[object]:
        del primals, output
        return (2 * residual * cotangent,)

    primal = ad.stage(
        lambda x: np.sum(primitive(x)),
        specs=(ad.ArraySpec((3,), "float64"),),
    )
    for transform in (ad.grad, ad.vjp_program):
        with pytest.raises(
            ad.TracingError,
            match=r"opaque residual.*staged or higher-order derivative",
        ):
            transform(primal)
