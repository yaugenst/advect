"""Python scalar ergonomics through the single array-tracer path."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad
from advect.core._protocols import _snapshot_traced


def test_real_scalar_primal_uses_rank_zero_float64_array_tracing() -> None:
    observed: list[tuple[type[Any], tuple[int, ...], np.dtype[Any]]] = []

    def objective(value: Any) -> Any:
        _node_id, payload = _snapshot_traced(value)
        observed.append((type(value), payload.shape, payload.dtype))
        return np.sin(value) + value * value

    gradient = ad.grad(objective)(3)

    assert gradient == pytest.approx(np.cos(3.0) + 6.0)
    assert type(gradient) is float
    tracer_type, shape, dtype = observed[0]
    assert tracer_type.__name__ == "TracedArray"
    assert shape == ()
    assert dtype == np.dtype("float64")


def test_scalar_values_tangents_and_cotangents_unlift_at_public_boundaries() -> None:
    def function(value: Any) -> Any:
        return value * value

    value, gradient = ad.value_and_grad(function)(3.0)
    assert (value, gradient) == pytest.approx((9.0, 6.0))
    assert type(value) is type(gradient) is float

    value, tangent = ad.jvp(function)(3.0, tangents=2)
    assert (value, tangent) == pytest.approx((9.0, 12.0))
    assert type(value) is type(tangent) is float

    value, pullback = ad.vjp(function)(3.0)
    assert type(value) is float
    gradient = pullback(1)
    assert type(gradient) is float
    assert gradient == pytest.approx(6.0)

    value, linear = ad.linearize(function, 3.0)
    with linear:
        first = linear(1)
        second = linear(2.0)
    assert type(value) is type(first) is type(second) is float
    assert (value, first, second) == pytest.approx((9.0, 6.0, 12.0))


def test_scalar_output_pytrees_unlift_without_changing_structure() -> None:
    value, tangent = ad.jvp(lambda x: {"x": x, "square": (x * x,)})(
        2.0,
        tangents=3.0,
    )

    assert value == {"x": 2.0, "square": (4.0,)}
    assert tangent == {"x": 3.0, "square": (12.0,)}
    assert all(type(leaf) is float for leaf in ad.pytree.tree_leaves(value))
    assert all(type(leaf) is float for leaf in ad.pytree.tree_leaves(tangent))


def test_scalar_output_restoration_preserves_unrelated_rank_zero_arrays() -> None:
    constant = np.asarray(5.0, dtype=np.float32)

    value, tangent = ad.jvp(lambda x: {"derived": x * x, "constant": constant})(
        3.0,
        tangents=2.0,
    )

    assert type(value["derived"]) is type(tangent["derived"]) is float
    assert isinstance(value["constant"], np.ndarray)
    assert isinstance(tangent["constant"], np.ndarray)
    assert value["constant"].dtype == tangent["constant"].dtype == np.dtype("float32")
    assert (value["constant"].shape, tangent["constant"].shape) == ((), ())
    assert_allclose(value["constant"], 5.0)
    assert_allclose(tangent["constant"], 0.0)


def test_scalar_output_restoration_is_leaf_specific_for_mixed_selected_inputs() -> None:
    value, linear = ad.linearize(
        lambda scalar, array: {"scalar": scalar * scalar, "array": array * array},
        3.0,
        np.asarray(4.0),
        argnums=(0, 1),
    )
    with linear:
        tangent = linear((2.0, np.asarray(3.0)))

    assert type(value["scalar"]) is type(tangent["scalar"]) is float
    assert type(value["array"]) is not float
    assert type(tangent["array"]) is not float
    assert_allclose(value["array"], 16.0)
    assert_allclose(tangent["array"], 24.0)


def test_scalar_auxiliary_outputs_remain_transparent_sidecars() -> None:
    sidecar = {
        "loss_scale": np.asarray(5.0, dtype=np.float32),
        "iterations": np.asarray(3, dtype=np.int32),
    }

    gradient, grad_aux = ad.grad(lambda x: (x * x, sidecar), has_aux=True)(3.0)
    value, value_gradient, value_aux = ad.value_and_grad(
        lambda x: (x * x, sidecar),
        has_aux=True,
    )(3.0)

    assert (value, gradient, value_gradient) == pytest.approx((9.0, 6.0, 6.0))
    for auxiliary in (grad_aux, value_aux):
        assert isinstance(auxiliary["loss_scale"], np.ndarray)
        assert isinstance(auxiliary["iterations"], np.ndarray)
        assert auxiliary["loss_scale"].dtype == np.dtype("float32")
        assert auxiliary["iterations"].dtype == np.dtype("int32")


def test_weak_scalar_auxiliary_outputs_match_dynamic_and_staged_transforms() -> None:
    def function(value: float) -> tuple[float, float]:
        return value * value, value + 1.0

    dynamic_gradient, dynamic_grad_aux = ad.grad(function, has_aux=True)(3.0)
    dynamic_value, dynamic_value_gradient, dynamic_value_aux = ad.value_and_grad(
        function,
        has_aux=True,
    )(3.0)
    program = ad.stage(
        function,
        specs=(ad.ArraySpec((), "float64", weak=True),),
    )
    staged_gradient, staged_grad_aux = ad.grad(program, has_aux=True)(3.0)
    staged_value, staged_value_gradient, staged_value_aux = ad.value_and_grad(
        program,
        has_aux=True,
    )(3.0)

    results = (
        dynamic_gradient,
        dynamic_grad_aux,
        dynamic_value,
        dynamic_value_gradient,
        dynamic_value_aux,
        staged_gradient,
        staged_grad_aux,
        staged_value,
        staged_value_gradient,
        staged_value_aux,
    )
    assert all(type(result) is float for result in results)
    assert results == pytest.approx((6.0, 4.0, 9.0, 6.0, 4.0) * 2)


def _copy_add_and_square(value: Any) -> Any:
    value = value.copy()
    value += 1.0
    return value * value


def _copy_index_add_and_square(value: Any) -> Any:
    value = value.copy()
    value[...] += 1.0
    return value * value


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(_copy_add_and_square, id="augmented-assignment"),
        pytest.param(_copy_index_add_and_square, id="indexed-augmented-assignment"),
    ],
)
def test_weak_scalar_category_survives_functionalized_mutation(function: Any) -> None:
    value, tangent = ad.jvp(function)(3.0, tangents=2.0)
    grad_value, gradient = ad.value_and_grad(function)(3.0)
    vjp_value, pullback = ad.vjp(function)(3.0)
    pullback_gradient = pullback(1.0)
    linear_value, linear = ad.linearize(function, 3.0)
    with linear:
        linear_tangent = linear(2.0)

    results = (
        value,
        tangent,
        grad_value,
        gradient,
        vjp_value,
        pullback_gradient,
        linear_value,
        linear_tangent,
    )
    assert all(type(result) is float for result in results)
    assert results == pytest.approx((16.0, 16.0, 16.0, 8.0, 16.0, 8.0, 16.0, 16.0))


def test_scalar_boundary_composes_across_nesting_and_argument_selection() -> None:
    third = ad.grad(ad.grad(ad.grad(lambda x: x**4)))(2.0)
    positional = ad.grad(lambda x, y: x * y, argnums=(0, 1))(3.0, 4.0)
    named = ad.grad(
        lambda x, *, scale: x * scale,
        argnums=0,
        argnames=("scale",),
    )(3.0, scale=4.0)

    assert type(third) is float
    assert third == pytest.approx(48.0)
    assert positional == pytest.approx((4.0, 3.0))
    assert named == pytest.approx((4.0, {"scale": 3.0}))


def test_scalar_boundary_covers_dense_derivative_helpers() -> None:
    def function(x: Any) -> Any:
        return x**3

    jacobian = ad.jacobian(function)(2.0)
    value, product = ad.hvp(function)(2.0, vectors=1.5)
    hessian = ad.hessian(function)(2.0)
    diagonal = ad.hessian_diag(function)(2.0)

    assert type(jacobian) is type(value) is type(product) is float
    assert type(hessian) is type(diagonal) is float
    assert jacobian == pytest.approx(12.0)
    assert (value, product) == pytest.approx((8.0, 18.0))
    assert hessian == pytest.approx(12.0)
    assert diagonal == pytest.approx(12.0)


@pytest.mark.parametrize("transform", [ad.hessian, ad.hessian_diag])
def test_scalar_dense_derivatives_ignore_static_keyword_configuration(
    transform: Any,
) -> None:
    def objective(x: Any, *, mode: str, enabled: bool) -> Any:
        assert mode == "cubic"
        return x**3 if enabled else x**2

    derivative = transform(objective)(2.0, mode="cubic", enabled=True)

    assert type(derivative) is float
    assert derivative == pytest.approx(12.0)


@pytest.mark.parametrize("transform", [ad.hessian, ad.hessian_diag])
def test_scalar_dense_derivatives_resolve_multiple_selected_primals_only(
    transform: Any,
) -> None:
    def objective(x: Any, y: Any, *, mode: str) -> Any:
        assert mode == "bilinear"
        return x * x + x * y + y * y

    derivative = transform(objective, argnums=(0, 1))(2.0, 3.0, mode="bilinear")

    if transform is ad.hessian:
        assert_allclose(derivative, ((2.0, 1.0), (1.0, 2.0)))
    else:
        assert derivative == pytest.approx((2.0, 2.0))


def test_staged_scalar_programs_and_derivatives_use_the_same_array_boundary() -> None:
    program = ad.stage(
        lambda value: np.sin(value) + value * value,
        specs=(ad.ArraySpec((), "float64", weak=True),),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())
    gradient = ad.grad(restored)
    value_and_gradient = ad.value_and_grad(restored)

    primal = restored(3.0)
    derivative = gradient(3.0)
    value, derivative_with_value = value_and_gradient(3.0)

    expected_value = np.sin(3.0) + 9.0
    expected_derivative = np.cos(3.0) + 6.0
    assert type(primal) is type(derivative) is float
    assert type(value) is type(derivative_with_value) is float
    assert (primal, value) == pytest.approx((expected_value, expected_value))
    assert (derivative, derivative_with_value) == pytest.approx(
        (expected_derivative, expected_derivative)
    )


def test_scalar_to_vector_jacobian_keeps_its_array_shape() -> None:
    jacobian = ad.jacobian(lambda x: x * np.arange(1.0, 5.0))(2.0)

    assert isinstance(jacobian, np.ndarray)
    assert jacobian.shape == (4,)
    assert_allclose(jacobian, np.arange(1.0, 5.0))


def test_array_callers_keep_rank_zero_array_results() -> None:
    primal = np.asarray(3.0)

    value, gradient = ad.value_and_grad(lambda x: x * x)(primal)

    assert type(value) is not float
    assert type(gradient) is not float
    assert value.shape == gradient.shape == ()
    assert_allclose(value, 9.0)
    assert_allclose(gradient, 6.0)


def test_numpy_scalars_remain_strong_provider_scalars() -> None:
    primal = np.float64(3.0)

    gradient = ad.grad(lambda value: value * value)(primal)
    value, tangent = ad.jvp(
        lambda scalar: scalar * scalar,
    )(primal, tangents=np.float64(2.0))

    assert isinstance(gradient, np.float64)
    assert isinstance(value, np.float64)
    assert isinstance(tangent, np.float64)


@pytest.mark.parametrize(
    ("primal", "message"),
    [
        pytest.param(True, "Boolean", id="bool"),
        pytest.param(1.0 + 2.0j, "complex", id="complex"),
    ],
)
def test_unsupported_python_scalar_primals_fail_at_the_boundary(
    primal: object,
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        ad.grad(lambda value: value)(primal)


@pytest.mark.parametrize("transform_name", ["grad", "value_and_grad"])
def test_nonscalar_output_error_names_the_public_transform(transform_name: str) -> None:
    transform = getattr(ad, transform_name)

    with pytest.raises(ValueError, match=rf"^{transform_name} requires a scalar-valued function"):
        transform(lambda value: np.stack((value, value)))(np.array(1.0))


@pytest.mark.parametrize(
    "tangent",
    [
        pytest.param(True, id="bool"),
        pytest.param(1.0 + 1.0j, id="complex"),
        pytest.param(np.ones((), dtype=bool), id="rank-zero-bool-array"),
        pytest.param(np.asarray(1.0 + 1.0j), id="rank-zero-complex-array"),
        pytest.param(np.ones(2), id="non-scalar"),
    ],
)
def test_scalar_jvp_rejects_invalid_tangents(tangent: object) -> None:
    with pytest.raises((TypeError, ValueError), match="Scalar JVP tangent"):
        ad.jvp(lambda value: value * value)(2.0, tangents=tangent)


@pytest.mark.parametrize(
    "cotangent",
    [
        pytest.param(True, id="bool"),
        pytest.param(np.ones((), dtype=bool), id="rank-zero-bool-array"),
        pytest.param(1.0 + 1.0j, id="complex"),
        pytest.param(np.asarray(1.0 + 1.0j), id="rank-zero-complex-array"),
        pytest.param(np.ones(2), id="non-scalar"),
    ],
)
def test_scalar_vjp_rejects_invalid_cotangents(cotangent: object) -> None:
    _value, pullback = ad.vjp(lambda value: value * value)(2.0)

    with pytest.raises((TypeError, ValueError), match="VJP cotangent"):
        pullback(cotangent)


def test_scalar_vjp_accepts_complex_cotangents_for_complex_outputs() -> None:
    value, pullback = ad.vjp(lambda x: 1j * x)(3.0)

    gradient = pullback(1.0 + 2.0j)

    assert value == pytest.approx(3.0j)
    assert gradient == pytest.approx(2.0)


def test_constant_python_complex_output_supports_linear_transforms() -> None:
    function = lambda _value: 1.0j  # noqa: E731 - compact constant-output fixture

    value, tangent = ad.jvp(function)(3.0, tangents=2.0)
    assert type(value) is complex
    assert type(tangent) is float
    assert value == pytest.approx(1.0j)
    assert tangent == pytest.approx(0.0)

    value, linear = ad.linearize(function, 3.0)
    with linear:
        assert linear(2.0) == pytest.approx(0.0)
    assert type(value) is complex

    value, pullback = ad.vjp(function)(3.0)
    gradient = pullback(2.0 + 3.0j)
    assert type(value) is complex
    assert type(gradient) is float
    assert value == pytest.approx(1.0j)
    assert gradient == pytest.approx(0.0)

    with pytest.raises(ValueError, match="real scalar output"):
        ad.grad(function)(3.0)


def test_vjp_validates_each_structured_output_cotangent() -> None:
    def function(value: Any) -> dict[str, Any]:
        return {"scalar": value * value, "vector": value * np.arange(2.0)}

    _value, pullback = ad.vjp(function)(3.0)
    gradient = pullback({"scalar": 1.0, "vector": np.ones(2)})
    assert gradient == pytest.approx(7.0)

    _value, pullback = ad.vjp(function)(3.0)
    with pytest.raises(ValueError, match=r"expected \(2,\), got \(\)"):
        pullback({"scalar": 1.0, "vector": 1.0})


def test_mixed_weak_scalar_and_strong_rank_zero_preserve_leaf_categories() -> None:
    scalar = 3.0
    array = np.asarray(4.0, dtype=np.float32)

    def function(scalar: Any, array: Any) -> Any:
        return scalar * array

    value, gradients = ad.value_and_grad(function, argnums=(0, 1))(scalar, array)
    assert value.dtype == np.dtype("float32")
    assert type(gradients[0]) is float
    assert type(gradients[1]) is not float
    assert gradients[1].dtype == np.dtype("float32")

    value, tangent = ad.jvp(function, argnums=(0, 1))(
        scalar,
        array,
        tangents=(2.0, np.asarray(3.0, dtype=np.float32)),
    )
    assert value.dtype == tangent.dtype == np.dtype("float32")

    value, pullback = ad.vjp(function, argnums=(0, 1))(scalar, array)
    gradients = pullback(np.asarray(1.0, dtype=np.float32))
    assert value.dtype == np.dtype("float32")
    assert type(gradients[0]) is float
    assert type(gradients[1]) is not float
    assert gradients[1].dtype == np.dtype("float32")

    value, linear = ad.linearize(function, scalar, array, argnums=(0, 1))
    with linear:
        tangent = linear((2.0, np.asarray(3.0, dtype=np.float32)))
    assert value.dtype == tangent.dtype == np.dtype("float32")

    jacobian = ad.jacobian(function, argnums=(0, 1))(scalar, array)
    assert type(jacobian[0]) is float
    assert isinstance(jacobian[1], np.ndarray)
    assert jacobian[1].dtype == np.dtype("float32")


def test_mixed_weak_scalar_higher_order_results_follow_input_columns() -> None:
    scalar = 3.0
    array = np.asarray(4.0, dtype=np.float32)

    def objective(scalar: Any, array: Any) -> Any:
        return scalar * scalar + scalar * array + array * array

    _value, product = ad.hvp(objective, argnums=(0, 1))(
        scalar,
        array,
        vectors=(1.0, np.asarray(1.0, dtype=np.float32)),
    )
    hessian = ad.hessian(objective, argnums=(0, 1))(scalar, array)
    diagonal = ad.hessian_diag(objective, argnums=(0, 1))(scalar, array)

    assert type(product[0]) is float
    assert type(product[1]) is not float
    assert [type(block) is float for row in hessian for block in row] == [
        True,
        False,
        True,
        False,
    ]
    assert type(diagonal[0]) is float
    assert type(diagonal[1]) is not float
    assert_allclose(hessian, ((2.0, 1.0), (1.0, 2.0)))
    assert_allclose(diagonal, (2.0, 2.0))


def test_staged_mixed_scalar_outputs_and_derivatives_round_trip_leaf_categories() -> None:
    scalar_spec = ad.ArraySpec((), "float64", weak=True)
    array_spec = ad.ArraySpec((), "float32")
    scalar = 3.0
    array = np.asarray(4.0, dtype=np.float32)
    outputs = ad.stage(
        lambda scalar, array: {
            "scalar": scalar * scalar,
            "array": array * array,
        },
        specs=(scalar_spec, array_spec),
    )
    loss = ad.stage(
        lambda scalar, array: scalar * array,
        specs=(scalar_spec, array_spec),
    )

    for program in (outputs, ad.StagedProgram.from_dict(outputs.to_dict())):
        result = program(scalar, array)
        assert type(result["scalar"]) is float
        assert type(result["array"]) is not float
        assert result["array"].dtype == np.dtype("float32")

    transforms = (
        ("grad", ad.grad(loss, argnums=(0, 1))),
        ("value_and_grad", ad.value_and_grad(loss, argnums=(0, 1))),
        ("vjp_program", ad.vjp_program(loss, argnums=(0, 1))),
    )
    for name, transform in transforms:
        for program in (transform, ad.StagedProgram.from_dict(transform.to_dict())):
            if name == "vjp_program":
                result = program(
                    scalar,
                    array,
                    cotangent=np.asarray(1.0, dtype=np.float32),
                )
                gradients = result
            else:
                result = program(scalar, array)
                gradients = result[1] if name == "value_and_grad" else result
            assert type(gradients[0]) is float
            assert type(gradients[1]) is not float
            assert gradients[1].dtype == np.dtype("float32")


def test_staged_weak_scalar_execution_normalizes_ints_and_array_only_operations() -> None:
    program = ad.stage(
        lambda value: np.astype(value, np.float32),
        specs=(ad.ArraySpec((), "float64", weak=True),),
    )

    assert type(program(2)) is float
    assert type(program(2.0)) is float
    assert program(2) == pytest.approx(2.0)


def test_staged_vjp_program_accepts_python_cotangent_for_weak_scalar_output() -> None:
    program = ad.stage(
        lambda value: value * value,
        specs=(ad.ArraySpec((), "float64", weak=True),),
    )
    pullback = ad.StagedProgram.from_dict(ad.vjp_program(program).to_dict())

    gradient = pullback(3.0, cotangent=1.0)

    assert type(gradient) is float
    assert gradient == pytest.approx(6.0)


def test_dynamic_transforms_compose_around_serialized_weak_scalar_program() -> None:
    program = ad.StagedProgram.from_dict(
        ad.stage(
            lambda value: value * value,
            specs=(ad.ArraySpec((), "float64", weak=True),),
        ).to_dict()
    )

    value, tangent = ad.jvp(program)(3.0, tangents=2.0)
    vjp_value, pullback = ad.vjp(program)(3.0)
    vjp_gradient = pullback(1.0)
    linear_value, linear = ad.linearize(program, 3.0)
    with linear:
        linear_tangent = linear(2.0)

    assert all(
        type(result) is float
        for result in (value, tangent, vjp_value, vjp_gradient, linear_value, linear_tangent)
    )
    assert (value, tangent, vjp_value, vjp_gradient) == pytest.approx((9.0, 12.0, 9.0, 6.0))
    assert type(ad.jacobian(program)(3.0)) is float
    assert type(ad.hessian(program)(3.0)) is float


@pytest.mark.parametrize("dtype", ["bool", "int64", "complex128"])
def test_staged_derivatives_reject_non_real_weak_scalar_signatures(dtype: str) -> None:
    program = ad.stage(
        lambda value: value * value,
        specs=(ad.ArraySpec((), dtype, weak=True),),
    )

    with pytest.raises(TypeError, match="real floating signature"):
        ad.grad(program)
    with pytest.raises(TypeError, match="real floating signature"):
        ad.value_and_grad(program)
    with pytest.raises(TypeError, match="real floating signature"):
        ad.vjp_program(program)


def test_staged_has_aux_keeps_strong_sidecars_outside_scalar_restoration() -> None:
    program = ad.stage(
        lambda value: (
            value * value,
            np.astype(value, np.float32) * np.asarray(0.0, dtype=np.float32)
            + np.asarray(5.0, dtype=np.float32),
        ),
        specs=(ad.ArraySpec((), "float64", weak=True),),
    )

    gradient, grad_aux = ad.grad(program, has_aux=True)(3.0)
    value, value_gradient, value_aux = ad.value_and_grad(program, has_aux=True)(3.0)

    assert type(gradient) is type(value) is type(value_gradient) is float
    for auxiliary in (grad_aux, value_aux):
        assert type(auxiliary) is not float
        assert auxiliary.dtype == np.dtype("float32")


def test_staged_constant_outputs_execute_differentiate_and_serialize() -> None:
    captured = np.asarray([2.0, 4.0], dtype=np.float32)
    real_program = ad.stage(
        lambda _value: 2.0,
        specs=(ad.ArraySpec((), "float64", weak=True),),
    )
    complex_program = ad.stage(
        lambda _value: 1.0j,
        specs=(ad.ArraySpec((), "float64", weak=True),),
    )
    array_program = ad.stage(
        lambda _value: captured,
        specs=(ad.ArraySpec((), "float64", weak=True),),
    )

    assert type(real_program(3.0)) is float
    assert type(complex_program(3.0)) is complex
    assert real_program(3.0) == pytest.approx(2.0)
    assert complex_program(3.0) == pytest.approx(1.0j)
    assert_allclose(array_program(3.0), captured)
    assert ad.grad(real_program)(3.0) == pytest.approx(0.0)

    value, tangent = ad.jvp(complex_program)(3.0, tangents=2.0)
    assert type(value) is complex
    assert type(tangent) is float
    assert (value, tangent) == pytest.approx((1.0j, 0.0))
    value, pullback = ad.vjp(complex_program)(3.0)
    assert value == pytest.approx(1.0j)
    assert pullback(2.0 + 3.0j) == pytest.approx(0.0)

    for program, expected in (
        (real_program, 2.0),
        (complex_program, 1.0j),
        (array_program, captured),
    ):
        restored = ad.StagedProgram.from_dict(program.to_dict())
        assert_allclose(restored(3.0), expected)


def test_staged_named_and_pytree_scalar_masks_follow_selected_leaves() -> None:
    scalar_spec = ad.ArraySpec((), "float64", weak=True)
    array_spec = ad.ArraySpec((), "float32")
    named = ad.stage(
        lambda array, *, scale: array * scale,
        specs=(array_spec,),
        kw_specs={"scale": scalar_spec},
    )
    nested = ad.stage(
        lambda values: values["scalar"] * values["array"],
        specs=({"scalar": scalar_spec, "array": array_spec},),
    )
    array = np.asarray(4.0, dtype=np.float32)

    named_gradient = ad.grad(named, argnums=None, argnames=("scale",))(
        array,
        scale=3.0,
    )
    nested_gradient = ad.grad(nested)({"scalar": 3.0, "array": array})

    assert type(named_gradient["scale"]) is float
    assert type(nested_gradient["scalar"]) is float
    assert type(nested_gradient["array"]) is not float


def test_array_api_strict_mixed_scalar_staged_composition_preserves_dtype() -> None:
    strict = pytest.importorskip("array_api_strict")

    array = strict.asarray(4.0, dtype=strict.float32)
    program = ad.stage(
        lambda array, scalar: array * scalar,
        specs=(
            ad.ArraySpec((), "float32"),
            ad.ArraySpec((), "float64", weak=True),
        ),
    )

    value, tangent = ad.jvp(program, argnums=(0, 1))(
        array,
        2.0,
        tangents=(strict.asarray(3.0, dtype=strict.float32), 1.0),
    )
    vjp_value, pullback = ad.vjp(program, argnums=(0, 1))(array, 2.0)
    gradients = pullback(strict.asarray(1.0, dtype=strict.float32))

    assert value.dtype == tangent.dtype == strict.float32
    assert vjp_value.dtype == gradients[0].dtype == strict.float32
    assert type(gradients[1]) is float


def test_scalar_boundary_composes_through_checkpoint() -> None:
    function = ad.checkpoint(lambda value: value * value)

    gradient = ad.grad(function)(3.0)
    value, tangent = ad.jvp(function)(3.0, tangents=2.0)
    vjp_value, pullback = ad.vjp(function)(3.0)
    vjp_gradient = pullback(1.0)
    second = ad.grad(ad.grad(function))(3.0)

    assert all(
        type(result) is float
        for result in (gradient, value, tangent, vjp_value, vjp_gradient, second)
    )
    assert (gradient, value, tangent, vjp_value, vjp_gradient, second) == pytest.approx(
        (6.0, 9.0, 12.0, 9.0, 6.0, 2.0)
    )


def test_scalar_boundary_composes_through_implicit_root() -> None:
    root = ad.implicit_root(
        lambda solution, parameter: solution - parameter,
        solve=lambda residual, initial: initial - residual(initial),
        linear_solve=lambda _operator, rhs: rhs,
    )

    def function(parameter: Any) -> Any:
        return root(parameter, initial=0.0)

    gradient = ad.grad(function)(3.0)
    value, tangent = ad.jvp(function)(3.0, tangents=2.0)
    vjp_value, pullback = ad.vjp(function)(3.0)
    vjp_gradient = pullback(1.0)

    assert all(
        type(result) is float for result in (gradient, value, tangent, vjp_value, vjp_gradient)
    )
    assert (gradient, value, tangent, vjp_value, vjp_gradient) == pytest.approx(
        (1.0, 3.0, 2.0, 3.0, 1.0)
    )
