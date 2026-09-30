"""Python scalar ergonomics through the single array-tracer path."""

from __future__ import annotations

from typing import Any

import array_api_strict as strict
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


def _square(value: Any) -> Any:
    return value * value


def _augmented_square(value: Any) -> Any:
    value = value.copy()
    value += 1.0
    return value * value


def _indexed_augmented_square(value: Any) -> Any:
    value = value.copy()
    value[...] += 1.0
    return value * value


def _indexed_set_square(value: Any) -> Any:
    value = value.copy()
    value[...] = value * value
    return value + 1.0


def _numpy_function(value: Any) -> Any:
    return np.square(value) + value


def _serialized_square() -> Any:
    program = ad.stage(_square, specs=(ad.ArraySpec((), "float64", weak=True),))
    return ad.StagedProgram.from_dict(program.to_dict())


def _implicit_identity() -> Any:
    root = ad.implicit_root(
        lambda solution, parameter: solution - parameter,
        solve=lambda residual, initial: initial - residual(initial),
        linear_solve=lambda _operator, rhs: rhs,
    )
    return lambda parameter: root(parameter, initial=0.0)


def _scalar_category(value: object) -> str:
    """Return ``"weak"`` for a Python float, else a rank-zero value's strong dtype."""
    if type(value) is float:
        return "weak"
    assert isinstance(value, np.ndarray | np.generic)
    assert value.shape == ()
    return str(value.dtype)


# Python operators keep a Python scalar weak; a NumPy function or an array method
# such as copy(), which a Python float lacks, returns a strong NumPy value.
@pytest.mark.parametrize(
    ("make_function", "x", "expected", "output_category"),
    [
        pytest.param(lambda: _square, 3.0, (9.0, 6.0, 2.0), "weak", id="square"),
        pytest.param(lambda: lambda value: value**3, 2.0, (8.0, 12.0, 12.0), "weak", id="cube"),
        pytest.param(lambda: _numpy_function, 3.0, (12.0, 7.0, 2.0), "float64", id="numpy"),
        pytest.param(lambda: _augmented_square, 3.0, (16.0, 8.0, 2.0), "float64", id="augmented"),
        pytest.param(
            lambda: _indexed_augmented_square, 3.0, (16.0, 8.0, 2.0), "float64", id="indexed-add"
        ),
        pytest.param(
            lambda: _indexed_set_square, 3.0, (10.0, 6.0, 2.0), "float64", id="indexed-set"
        ),
        pytest.param(_serialized_square, 3.0, (9.0, 6.0, 2.0), "weak", id="serialized-program"),
        pytest.param(lambda: ad.checkpoint(_square), 3.0, (9.0, 6.0, 2.0), "weak", id="checkpoint"),
        pytest.param(_implicit_identity, 3.0, (3.0, 1.0, 0.0), "weak", id="implicit-root"),
    ],
)
def test_every_scalar_transform_returns_python_float_derivatives(
    make_function: Any,
    x: float,
    expected: tuple[float, float, float],
    output_category: str,
) -> None:
    function = make_function()
    value, gradient, second = expected
    # Python int and float tangents and cotangents are both accepted.
    value_and_gradient = ad.value_and_grad(function)(x)
    primal, tangent = ad.jvp(function)(x, tangents=2)
    vjp_value, pullback = ad.vjp(function)(x)
    linear_value, linear = ad.linearize(function, x)
    with linear:
        linear_tangent, linear_gradient = linear(2.0), linear.pullback(1)
    hvp_value, hvp_product = ad.hvp(function)(x, vectors=1.5)

    # Values and forward tangents keep the output's scalar category ...
    outputs = (value_and_gradient[0], primal, tangent, vjp_value, linear_value, linear_tangent)
    assert [_scalar_category(result) for result in (*outputs, hvp_value)] == [output_category] * 7
    assert (*outputs, hvp_value) == pytest.approx(
        (value, value, 2 * gradient, value, value, 2 * gradient, value)
    )
    # ... and derivatives with respect to a Python-scalar input are Python floats.
    derivatives = (
        value_and_gradient[1],
        pullback(1.0),
        linear_gradient,
        ad.jacobian(function)(x),
        ad.grad(ad.grad(function))(x),
        hvp_product,
        ad.hessian(function)(x),
        ad.hessian_diag(function)(x),
    )
    assert [type(result) for result in derivatives] == [float] * 8
    assert derivatives == pytest.approx((gradient,) * 4 + (second, 1.5 * second, second, second))


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


def test_scalar_boundary_composes_across_nesting_and_argument_selection() -> None:
    def directional(x: Any) -> Any:
        return ad.jvp(lambda v: v * v)(x, tangents=1.0)[1]

    third = ad.grad(ad.grad(ad.grad(lambda x: x**4)))(2.0)
    forward_of_forward = ad.jvp(directional)(3.0, tangents=2.0)
    forward_of_reverse = ad.jvp(ad.grad(lambda x: x**3))(3.0, tangents=1.0)
    positional = ad.grad(lambda x, y: x * y, argnums=(0, 1))(3.0, 4.0)
    named = ad.grad(
        lambda x, *, scale: x * scale,
        argnums=0,
        argnames=("scale",),
    )(3.0, scale=4.0)

    assert all(
        type(result) is float for result in (third, *forward_of_forward, *forward_of_reverse)
    )
    assert third == pytest.approx(48.0)
    assert forward_of_forward == pytest.approx((6.0, 4.0))
    assert forward_of_reverse == pytest.approx((27.0, 18.0))
    assert positional == pytest.approx((4.0, 3.0))
    assert named == pytest.approx((4.0, {"scale": 3.0}))


@pytest.mark.parametrize("xp", [np, strict], ids=["numpy", "array_api_strict"])
def test_nested_scalar_pullback_keeps_the_callers_cotangent_strong(xp: Any) -> None:
    # The pullback of a Python-float primal returns a weak scalar; the caller's
    # own cotangent that it passes through stays strong for every later use.
    single = xp.asarray(2.0, dtype=xp.float32)

    def outer(c: Any) -> Any:
        passed = ad.vjp(lambda v: v)(3.0)[1](c)
        shifted = ad.vjp(lambda v: v + 2.0)(3.0)[1](c)
        return c * single, passed, shifted

    c = xp.asarray(2.0)
    (own, passed, shifted), _tangents = ad.jvp(outer)(c, tangents=xp.asarray(1.0))
    gradient = ad.grad(lambda value: outer(value)[0])(c)
    staged_own, _staged_passed, _staged_shifted = ad.stage(outer, c)(c)

    assert own.dtype == gradient.dtype == staged_own.dtype == xp.float64
    assert type(passed) is type(shifted) is float
    assert (float(own), passed, shifted) == (4.0, 2.0, 2.0)
    assert (float(gradient), float(staged_own)) == (2.0, 4.0)


@pytest.mark.parametrize(
    "inner",
    [
        pytest.param(ad.grad(lambda v: v * v), id="grad"),
        pytest.param(lambda x: ad.jvp(lambda v: v * v)(x, tangents=1.0)[1], id="jvp"),
    ],
)
def test_staged_nested_scalar_derivatives_execute(inner: Any) -> None:
    # Staged execution represents only Python-scalar inputs and the values
    # derived from them as weak scalars, so a staged value is never re-marked.
    staged = ad.stage(lambda x: inner(x) * np.float32(2), 3.0)

    assert float(staged(3.0)) == 12.0


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
    # np.sin of a Python float is a strong NumPy scalar; derivatives stay Python floats.
    assert type(primal) is type(value) is type(expected_value) is np.float64
    assert type(derivative) is type(derivative_with_value) is float
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


def _complex_sum(value: Any) -> Any:
    namespace = value.__array_namespace__()
    return namespace.sum(namespace.astype(value, namespace.complex128))


def test_grad_rejects_invalid_primals_and_outputs() -> None:
    # Python complex primals are rejected in test_complex_convention.py.
    with pytest.raises(TypeError, match="Boolean"):
        ad.grad(lambda value: value)(True)  # noqa: FBT003 - the rejected primal
    for transform in (ad.grad, ad.value_and_grad):
        with pytest.raises(ValueError, match=rf"^{transform.__name__} requires a scalar-valued"):
            transform(lambda value: np.stack((value, value)))(np.array(1.0))
    with pytest.raises(ValueError, match="output pytree has 2 leaves"):
        ad.grad(lambda value: (value, value))(2.0)
    with pytest.raises(TypeError, match="got str"):
        ad.grad(lambda _value: "not numeric")(2.0)
    # An integer argnums takes the unary fast path; a tuple the general one.
    for argnums in (0, (0,)):
        with pytest.raises(ValueError, match="real scalar output"):
            ad.grad(_complex_sum, argnums=argnums)(strict.asarray([1.0, 2.0]))


def test_grad_accepts_a_one_leaf_scalar_output_pytree() -> None:
    gradient = ad.grad(lambda value: {"loss": value * value})(np.array(2.0))

    assert_allclose(gradient, 4.0)


def test_grad_preserves_none_for_an_untraceable_input_leaf() -> None:
    gradient = ad.grad(lambda tree: np.sum(tree["value"]))({"value": np.ones(2), "label": "fixed"})

    assert_allclose(gradient["value"], np.ones(2))
    assert gradient["label"] is None


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


@pytest.mark.parametrize(
    "real_part",
    [
        pytest.param(lambda value: value.real, id="attribute"),
        pytest.param(np.real, id="numpy"),
    ],
)
def test_complex_real_projection_transposes_weak_and_rank_zero_scalars(real_part: Any) -> None:
    coefficient = 1.0 + 2.0j

    weak_gradient = ad.grad(lambda value: real_part(coefficient * value))(2.0)
    rank_zero_gradient = ad.grad(lambda value: real_part(coefficient * value))(
        np.asarray(2.0 + 3.0j)
    )

    assert weak_gradient == pytest.approx(coefficient.real)
    assert_allclose(rank_zero_gradient, np.conjugate(coefficient))


@pytest.mark.parametrize(
    "imaginary_part",
    [
        pytest.param(lambda value: value.imag, id="attribute"),
        pytest.param(np.imag, id="numpy"),
    ],
)
def test_complex_imaginary_projection_transposes_weak_and_rank_zero_scalars(
    imaginary_part: Any,
) -> None:
    coefficient = 1.0 + 2.0j

    weak_gradient = ad.grad(lambda value: imaginary_part(coefficient * value))(2.0)
    rank_zero_gradient = ad.grad(lambda value: imaginary_part(coefficient * value))(
        np.asarray(2.0 + 3.0j)
    )

    assert weak_gradient == pytest.approx(coefficient.imag)
    assert_allclose(rank_zero_gradient, coefficient.imag + 1j * coefficient.real)


def test_constant_python_complex_output_supports_linear_transforms() -> None:
    def function(_value: object) -> complex:
        return 1.0j

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
    # A Python float has no astype: the lifted scalar casts as a 0-d float64 array.
    program = ad.stage(
        lambda value: np.astype(value, np.float32),
        specs=(ad.ArraySpec((), "float64", weak=True),),
    )

    for result in (program(2), program(2.0)):
        assert type(result) is np.ndarray
        assert (result.shape, result.dtype) == ((), np.dtype("float32"))
        assert result == pytest.approx(2.0)


def test_staged_vjp_program_accepts_python_cotangent_for_weak_scalar_output() -> None:
    program = ad.stage(
        lambda value: value * value,
        specs=(ad.ArraySpec((), "float64", weak=True),),
    )
    pullback = ad.StagedProgram.from_dict(ad.vjp_program(program).to_dict())

    gradient = pullback(3.0, cotangent=1.0)

    assert type(gradient) is float
    assert gradient == pytest.approx(6.0)


@pytest.mark.parametrize("dtype", [np.float64, np.dtype("float64"), "float64"])
def test_staged_gradient_selects_a_weak_scalar_declared_with_any_dtype_spelling(
    dtype: object,
) -> None:
    program = ad.stage(
        lambda array, scale: np.sum(array * scale),
        specs=(ad.ArraySpec((3,), "float64"), ad.ArraySpec((), dtype, weak=True)),
    )

    gradient = ad.grad(program, argnums=1)(np.array([1.0, 2.0, 3.0]), 2.0)

    assert type(gradient) is float
    assert gradient == pytest.approx(6.0)


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
