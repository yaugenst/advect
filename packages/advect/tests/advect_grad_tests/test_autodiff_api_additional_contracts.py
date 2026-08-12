"""Additional public contracts for dynamic autodiff APIs."""

from __future__ import annotations

from array import array
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad


def test_named_selection_accepts_a_positionally_passed_argument() -> None:
    def objective(value: float, scale: float) -> float:
        return value * scale

    gradient = ad.grad(
        objective,
        argnums=None,
        argnames=("scale",),
    )(2.0, 3.0)

    assert gradient == {"scale": pytest.approx(2.0)}


def test_negative_and_variadic_positional_selections_are_supported() -> None:
    def objective(first: float, *rest: float) -> float:
        return first * rest[0]

    assert ad.grad(objective, argnums=-1)(2.0, 3.0) == pytest.approx(2.0)


def test_variadic_and_named_selections_can_be_combined() -> None:
    def objective(value: float, *coefficients: float, scale: float) -> float:
        return value * coefficients[0] * scale

    positional, named = ad.grad(
        objective,
        argnums=1,
        argnames=("scale",),
    )(2.0, 3.0, scale=4.0)

    assert positional == pytest.approx(8.0)
    assert named == {"scale": pytest.approx(6.0)}


def test_uninspectable_numpy_callable_uses_a_positional_fallback_name() -> None:
    assert ad.grad(np.sin)(np.array(0.2)) == pytest.approx(np.cos(0.2))


def test_invalid_selectors_fail_at_the_public_boundary() -> None:
    def objective(value: float, *extras: float, **options: float) -> float:
        return value + sum(extras) + sum(options.values())

    with pytest.raises(ValueError, match="Argument 'missing' not found"):
        ad.grad(objective, argnums=None, argnames=("missing",))
    with pytest.raises(ValueError, match="cannot select variadic parameter 'extras'"):
        ad.grad(objective, argnums=None, argnames=("extras",))
    with pytest.raises(ValueError, match="cannot select variadic parameter 'options'"):
        ad.grad(objective, argnums=None, argnames=("options",))
    with pytest.raises(ValueError, match="selected by both argnums and argnames"):
        ad.grad(objective, argnums=0, argnames=("value",))(1.0)
    with pytest.raises(IndexError, match="index 2 is out of range"):
        ad.grad(objective, argnums=2)(1.0)
    with pytest.raises(ValueError, match="argnums contains duplicates"):
        ad.grad(objective, argnums=(0, -1))(1.0)
    with pytest.raises(ValueError, match="argnames contains duplicates"):
        ad.grad(objective, argnums=None, argnames=("value", "value"))(1.0)


def test_staged_named_selection_requires_a_keyword_call() -> None:
    program = ad.stage(
        lambda value, *, scale: value * scale,
        specs=(ad.ArraySpec((2,), "float64"),),
        kw_specs={"scale": ad.ArraySpec((), "float64")},
    )

    with pytest.raises(ValueError, match="not provided as a keyword"):
        ad.jacobian(program, argnums=None, argnames=("scale",))(
            np.ones(2),
            np.array(2.0),
        )


def test_staged_transform_rejects_a_missing_named_selection() -> None:
    program = ad.stage(
        lambda value: value * value,
        specs=(ad.ArraySpec((), "float64"),),
    )

    with pytest.raises(ValueError, match="not present in the compiled signature"):
        ad.grad(program, argnums=None, argnames=("missing",))


def test_nested_transforms_preserve_unselected_pytree_and_keyword_dependencies() -> None:
    inner = ad.grad(
        lambda value, coefficients, *, bias: value * (coefficients["scale"] + bias),
        argnums=0,
    )
    outer = ad.grad(
        lambda scale, bias: inner(2.0, {"scale": scale}, bias=bias),
        argnums=(0, 1),
    )

    assert outer(3.0, 4.0) == pytest.approx((1.0, 1.0))


def test_nested_named_selection_tracks_a_positionally_passed_argument() -> None:
    named = ad.grad(
        lambda value, scale: value * scale,
        argnums=(),
        argnames=("scale",),
    )

    assert ad.grad(lambda value: named(value, 3.0)["scale"])(2.0) == pytest.approx(1.0)


def test_debug_jacobian_preserves_an_untraceable_pytree_leaf() -> None:
    params = {"weight": np.array(2.0), "label": "fixed"}

    with ad.debug(), pytest.warns(UserWarning, match="untraceable.*label"):
        jacobian = ad.jacobian(lambda tree: tree["weight"] * np.arange(1.0, 4.0))(params)

    assert_allclose(jacobian["weight"], np.arange(1.0, 4.0))
    assert jacobian["label"] is None


def test_nested_python_complex_input_names_the_unsupported_boundary() -> None:
    params = {"real": 2.0, "complex": 1.0 + 2.0j}

    with pytest.raises(TypeError, match=r"tree\['complex'\].*Python complex scalar"):
        ad.grad(lambda tree: tree["real"] ** 2)(params)


def test_jvp_requires_none_for_an_untraceable_pytree_leaf() -> None:
    params = {"value": np.array([1.0, 2.0]), "label": "fixed"}
    transformed = ad.jvp(lambda tree: 3.0 * tree["value"])

    value, tangent = transformed(
        params,
        tangents={"value": np.ones(2), "label": None},
    )
    assert_allclose(value, np.array([3.0, 6.0]))
    assert_allclose(tangent, np.full(2, 3.0))

    with pytest.raises(TypeError, match="static/untraceable input leaf"):
        transformed(
            params,
            tangents={"value": np.ones(2), "label": "not-none"},
        )


def test_scalar_jvp_rejects_a_nonnumeric_tangent() -> None:
    with pytest.raises(TypeError, match="got str"):
        ad.jvp(lambda value: value * value)(2.0, tangents="one")


def test_jvp_validates_multi_argument_arity_and_leaf_shapes() -> None:
    transformed = ad.jvp(lambda left, right: left + right, argnums=(0, 1))
    left = np.ones(2)
    right = np.ones(2)

    with pytest.raises(TypeError, match="requires tangents as a tuple"):
        transformed(left, right, tangents=np.ones(2))
    with pytest.raises(ValueError, match="tangent arity mismatch"):
        transformed(left, right, tangents=(np.ones(2),))
    with pytest.raises(ValueError, match="tangent shape mismatch"):
        transformed(left, right, tangents=(np.ones(3), np.ones(2)))


def test_jvp_coerces_a_python_tangent_for_a_rank_zero_array() -> None:
    value, tangent = ad.jvp(lambda scalar: scalar * scalar)(
        np.array(2.0),
        tangents=3.0,
    )

    assert_allclose(value, 4.0)
    assert_allclose(tangent, 12.0)


def test_jvp_reports_shape_mismatch_after_coercing_a_python_sequence() -> None:
    with pytest.raises(ValueError, match="shape mismatch after coercion"):
        ad.jvp(lambda value: value * value)(
            np.ones(2),
            tangents=array("d", [1.0, 2.0, 3.0]),
        )


def test_empty_jvp_selection_returns_a_zero_output_tangent() -> None:
    value, tangent = ad.jvp(lambda scalar: scalar * scalar, argnums=())(
        2.0,
        tangents=(),
    )

    assert value == pytest.approx(4.0)
    assert tangent == pytest.approx(0.0)


def test_zero_input_linear_map_is_reusable() -> None:
    value, linear = ad.linearize(lambda _scalar: 3.0, 2.0, argnums=())

    assert value == pytest.approx(3.0)
    with linear as active:
        assert active(()) == pytest.approx(0.0)
        assert active.pullback(1.0) == ()
        assert active.apply_many(((), ())) == pytest.approx((0.0, 0.0))
        assert active.transpose_many((1.0, 2.0)) == ((), ())


def test_singleton_tuple_linearization_preserves_structure_and_empty_batches() -> None:
    left = np.array([2.0, 3.0])
    right = np.array([4.0, 5.0])
    value, linear = ad.linearize(
        lambda x, y: {"product": x * y, "constant": np.array(7.0)},
        left,
        right,
        argnums=(1,),
    )

    assert_allclose(value["product"], left * right)
    with linear as active:
        tangent = active((np.ones_like(right),))
        assert_allclose(tangent["product"], left)
        assert_allclose(tangent["constant"], 0.0)
        assert active.apply_many(()) == ()
        assert active.transpose_many(()) == ()
        (gradient,) = active.transpose()({"product": np.ones_like(right), "constant": None})
        assert_allclose(gradient, left)

    with pytest.raises(RuntimeError, match="closed or consumed"):
        linear(np.ones_like(right))


def test_pullback_context_closes_an_unconsumed_trace() -> None:
    value = np.array([1.0, 2.0])
    _output, pullback = ad.vjp(lambda x: x * x)(value)

    with pullback as active:
        assert active is pullback

    with pytest.raises(RuntimeError, match="closed or consumed"):
        pullback(np.ones_like(value))


def test_repeated_dynamic_transforms_remain_independent() -> None:
    def bilinear(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return left * right + left * right

    left = np.array([2.0, 3.0])
    right = np.array([4.0, 5.0])
    for _iteration in range(2):
        value, tangent = ad.jvp(bilinear, argnums=(0, 1))(
            left,
            right,
            tangents=(np.ones_like(left), np.ones_like(right)),
        )
        assert_allclose(value, 2.0 * left * right)
        assert_allclose(tangent, 2.0 * (left + right))

    for _iteration in range(2):
        value, pullback = ad.vjp(bilinear, argnums=(0, 1))(left, right)
        left_gradient, right_gradient = pullback(np.ones_like(value))
        assert_allclose(left_gradient, 2.0 * right)
        assert_allclose(right_gradient, 2.0 * left)


def test_jvp_reports_a_public_primitive_without_a_forward_rule() -> None:
    @ad.primitive(name="tests.additional_contracts.transpose_only")
    def transpose_only(value: np.ndarray) -> np.ndarray:
        return value * value

    @transpose_only.def_transpose
    def transpose(
        cotangent: np.ndarray,
        primals: tuple[np.ndarray, ...],
        output: np.ndarray,
    ) -> tuple[np.ndarray]:
        del output
        return (2.0 * primals[0] * cotangent,)

    with pytest.raises(ad.NoJVPError, match="no JVP rule is installed"):
        ad.jvp(transpose_only)(np.ones(2), tangents=np.ones(2))


def test_checkpoint_validates_and_preserves_an_ordinary_callable() -> None:
    with pytest.raises(TypeError, match="checkpoint function must be callable"):
        ad.checkpoint(None)  # type: ignore[arg-type]

    def shifted(value: np.ndarray, *, offset: float = 1.0) -> np.ndarray:
        """Shift one value."""
        return value + offset

    wrapped = ad.checkpoint(shifted)

    assert wrapped.__name__ == "shifted"
    assert wrapped.__doc__ == shifted.__doc__
    assert_allclose(wrapped(np.array([1.0, 2.0]), offset=3.0), [4.0, 5.0])


def test_checkpoint_partial_jvp_zero_fills_passive_inputs() -> None:
    @ad.checkpoint
    def affine(
        value: np.ndarray,
        coefficient: np.ndarray,
        *,
        offset: float,
    ) -> np.ndarray:
        return value * coefficient + offset

    value = np.array([1.0, 2.0])
    coefficient = np.array([3.0, 4.0])
    primal, tangent = ad.jvp(affine, argnums=0)(
        value,
        coefficient,
        offset=2.0,
        tangents=np.ones_like(value),
    )

    assert_allclose(primal, value * coefficient + 2.0)
    assert_allclose(tangent, coefficient)


def test_checkpoint_vjp_restores_a_multi_output_pytree() -> None:
    @ad.checkpoint
    def statistics(value: np.ndarray) -> dict[str, np.ndarray]:
        return {"double": 2.0 * value, "sum": np.sum(value)}

    value = np.array([1.0, 2.0, 3.0])
    output, pullback = ad.vjp(statistics)(value)
    gradient = pullback({"double": np.ones_like(value), "sum": np.array(3.0)})

    assert_allclose(output["double"], 2.0 * value)
    assert_allclose(output["sum"], np.sum(value))
    assert_allclose(gradient, np.full_like(value, 5.0))


def test_checkpoint_without_inputs_remains_a_direct_call_inside_a_trace() -> None:
    calls = 0

    @ad.checkpoint
    def constant() -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.array(3.0)

    gradient = ad.grad(lambda value: value * constant())(2.0)

    assert gradient == pytest.approx(3.0)
    assert calls == 1


def test_jacobian_supports_constant_empty_and_untraceable_boundaries() -> None:
    assert ad.jacobian(lambda _value: 3.0)(2.0) == pytest.approx(0.0)
    assert ad.jacobian(lambda _value: {})(np.ones(2)) == {}
    assert ad.jacobian(lambda _label: 3.0)("fixed") is None
    assert ad.jacobian(lambda _label: np.empty(0))("fixed") is None


def test_forward_jacobian_preserves_an_empty_selected_input_leaf() -> None:
    params = {"empty": np.empty(0), "scale": np.array(2.0)}

    jacobian = ad.jacobian(
        lambda tree: tree["scale"] * np.arange(1.0, 4.0) + np.sum(tree["empty"])
    )(params)

    assert jacobian["empty"].shape == (3, 0)
    assert_allclose(jacobian["scale"], np.arange(1.0, 4.0))


def test_jacobian_rejects_a_constant_python_complex_output() -> None:
    with pytest.raises(ValueError, match="jacobian requires real outputs"):
        ad.jacobian(lambda _value: 1.0j)(2.0)


def test_jacobian_requires_a_backend_for_shaped_outputs() -> None:
    class ShapedWithoutNamespace:
        shape = (2,)
        dtype = np.dtype("float64")

    with pytest.raises(RuntimeError, match="output leaves require an array backend"):
        ad.jacobian(lambda _value: ShapedWithoutNamespace())(np.ones(3))


def test_reverse_transforms_validate_output_structure_and_type() -> None:
    with pytest.raises(ValueError, match="output pytree has 2 leaves"):
        ad.grad(lambda value: (value, value))(2.0)
    with pytest.raises(TypeError, match="got str"):
        ad.grad(lambda _value: "not numeric")(2.0)


def test_grad_accepts_a_one_leaf_scalar_output_pytree() -> None:
    gradient = ad.grad(lambda value: {"loss": value * value})(np.array(2.0))

    assert_allclose(gradient, 4.0)


def test_grad_preserves_none_for_an_untraceable_input_leaf() -> None:
    gradient = ad.grad(lambda tree: np.sum(tree["value"]))({"value": np.ones(2), "label": "fixed"})

    assert_allclose(gradient["value"], np.ones(2))
    assert gradient["label"] is None


def test_vjp_rejects_a_vector_cotangent_for_a_python_scalar_output() -> None:
    _value, pullback = ad.vjp(lambda _value: 3.0)(2.0)

    with pytest.raises(ValueError, match=r"expected \(\), got \(2,\)"):
        pullback(np.ones(2))


def test_vjp_reports_a_missing_namespace_when_coercing_a_cotangent() -> None:
    class ShapedWithoutNamespace:
        shape = (2,)
        dtype = np.dtype("float64")

    _value, pullback = ad.vjp(lambda _value: ShapedWithoutNamespace())(2.0)

    with pytest.raises(TypeError, match="no array namespace is available"):
        pullback(1.0)


def test_custom_transpose_may_restore_a_leading_singleton_dimension() -> None:
    class ReshapableContribution:
        def __init__(self, value: np.ndarray) -> None:
            self.value = value
            self.shape = value.shape

        def reshape(self, shape: tuple[int, ...]) -> np.ndarray:
            return self.value.reshape(shape)

    @ad.primitive(name="tests.additional_contracts.reshapable_transpose")
    def reduce_row(value: np.ndarray) -> np.ndarray:
        return np.sum(value, axis=0)

    @reduce_row.def_transpose
    def transpose(
        cotangent: np.ndarray,
        primals: tuple[np.ndarray, ...],
        output: np.ndarray,
    ) -> tuple[ReshapableContribution]:
        del primals, output
        return (ReshapableContribution(cotangent),)

    _value, pullback = ad.vjp(reduce_row)(np.ones((1, 2)))

    assert_allclose(pullback(np.array([2.0, 3.0])), np.array([[2.0, 3.0]]))


def test_dense_hessian_rejects_complex_inputs() -> None:
    with pytest.raises(ValueError, match="hessian requires real input leaves"):
        ad.hessian(lambda value: np.real(np.sum(value)))(np.ones(2, dtype=np.complex128))


def test_dense_hessian_rejects_generic_pytree_inputs() -> None:
    with pytest.raises(ad.AdvectError, match="gradient structure"):
        ad.hessian(lambda tree: np.sum(tree["value"] ** 2))({"value": np.ones(2)})


@pytest.mark.parametrize(
    ("invalid_argument", "message"),
    [
        ("residual", "residual must be callable"),
        ("solve", "solve must be callable"),
        ("linear_solve", "linear_solve must be callable"),
        ("transpose_solve", "transpose_solve must be callable or None"),
    ],
)
def test_implicit_root_rejects_noncallable_callbacks(
    invalid_argument: str,
    message: str,
) -> None:
    callbacks: dict[str, Any] = {
        "residual": lambda solution, params: solution - params,
        "solve": lambda residual, initial: initial - residual(initial),
        "linear_solve": lambda _operator, rhs: rhs,
        "transpose_solve": None,
    }
    callbacks[invalid_argument] = 3

    with pytest.raises(TypeError, match=message):
        ad.implicit_root(**callbacks)


def test_implicit_root_rejects_solution_and_residual_pytree_changes() -> None:
    bad_solution = ad.implicit_root(
        lambda solution, params: solution - params,
        solve=lambda _residual, initial: {"value": initial},
        linear_solve=lambda _operator, rhs: rhs,
    )
    with pytest.raises(TypeError, match="solution pytree structure"):
        bad_solution(np.ones(2), initial=np.zeros(2))

    bad_residual = ad.implicit_root(
        lambda solution, params: {"value": solution - params},
        solve=lambda _residual, initial: initial,
        linear_solve=lambda _operator, rhs: rhs,
    )
    with pytest.raises(TypeError, match="residual pytree structure"):
        bad_residual(np.ones(2), initial=np.zeros(2))


@pytest.mark.parametrize("initial", [1, 1.0 + 2.0j])
def test_implicit_root_accepts_python_scalar_solution_specs(initial: object) -> None:
    root = ad.implicit_root(
        lambda solution, _params: solution - solution,
        solve=lambda _residual, guess: guess,
        linear_solve=lambda _operator, rhs: rhs,
    )

    assert root(None, initial=initial) == initial


def test_implicit_root_jvp_supports_a_partially_seeded_parameter_pytree() -> None:
    def solve_linear(operator: Any, rhs: np.ndarray) -> np.ndarray:
        return rhs / operator(np.ones_like(rhs))

    root = ad.implicit_root(
        lambda solution, params: solution - params["target"],
        solve=lambda residual, initial: initial - residual(initial),
        linear_solve=solve_linear,
    )
    params = {
        "target": np.array([1.0, 2.0]),
        "unused": np.array([3.0, 4.0]),
    }

    value, tangent = ad.jvp(root)(
        params,
        initial=np.zeros(2),
        tangents={"target": np.ones(2), "unused": None},
    )

    assert_allclose(value, params["target"])
    assert_allclose(tangent, np.ones(2))
