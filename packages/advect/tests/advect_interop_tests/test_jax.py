"""JAX bridge qualification."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
_enable_x64 = jax.enable_x64 if hasattr(jax, "enable_x64") else jax.experimental.enable_x64

from advect.interop.jax import wrap  # noqa: E402 - dependency skip precedes adapter import


def test_jax_bridge_infers_outputs_for_eager_value_and_grad() -> None:
    bridged = wrap(lambda value: np.sum(value * value))
    sample = jnp.asarray([1.0, 2.0, -3.0], dtype=jnp.float32)

    direct_value = bridged(sample)
    eager_value, eager_gradient = jax.value_and_grad(bridged)(sample)

    np.testing.assert_allclose(direct_value, 14.0)
    np.testing.assert_allclose(eager_value, 14.0)
    np.testing.assert_allclose(eager_gradient, 2 * sample)


def test_jax_bridge_requires_result_specs_when_staged() -> None:
    bridged = wrap(lambda value: np.sum(value * value))
    sample = jnp.asarray([1.0, 2.0, -3.0], dtype=jnp.float32)
    message = "result_shape_dtypes is required when an Advect bridge is staged"

    with pytest.raises(TypeError, match=message):
        jax.jit(bridged)(sample)
    with pytest.raises(TypeError, match=message):
        jax.jit(jax.value_and_grad(bridged))(sample)
    _, pullback = jax.vjp(bridged, sample)
    with pytest.raises(TypeError, match=message):
        jax.jit(pullback)(jnp.asarray(1.0, dtype=jnp.float32))


def test_jax_bridge_excludes_auxiliary_outputs_from_the_vjp() -> None:
    def operation(value):
        return np.sum(value * value), {
            "active": np.ones((), dtype=np.bool_),
            "signal": np.sin(value),
            "steps": np.asarray(3, dtype=np.int32),
        }

    bridged = wrap(operation, has_aux=True)
    sample = jnp.asarray([1.0, 2.0], dtype=jnp.float32)

    (value, auxiliary), gradient = jax.value_and_grad(
        bridged,
        has_aux=True,
    )(sample)
    auxiliary_gradient = jax.grad(lambda argument: jnp.sum(bridged(argument)[1]["signal"]))(sample)

    np.testing.assert_allclose(value, 5.0)
    np.testing.assert_allclose(gradient, 2 * sample)
    np.testing.assert_allclose(auxiliary["signal"], np.sin(sample))
    np.testing.assert_array_equal(auxiliary["active"], np.ones((), dtype=np.bool_))
    np.testing.assert_array_equal(auxiliary["steps"], 3)
    np.testing.assert_array_equal(auxiliary_gradient, jnp.zeros_like(sample))


def test_jax_bridge_supports_auxiliary_outputs_under_jit() -> None:
    bridged = wrap(
        lambda value: (
            np.sum(value * value),
            {
                "active": np.ones((), dtype=np.bool_),
                "signal": np.sin(value),
                "steps": np.asarray(3, dtype=np.int32),
            },
        ),
        has_aux=True,
        result_shape_dtypes=(
            jax.ShapeDtypeStruct((), np.float32),
            {
                "active": jax.ShapeDtypeStruct((), np.bool_),
                "signal": jax.ShapeDtypeStruct((2,), np.float32),
                "steps": jax.ShapeDtypeStruct((), np.int32),
            },
        ),
    )
    sample = jnp.asarray([1.0, 2.0], dtype=jnp.float32)

    (value, auxiliary), gradient = jax.jit(jax.value_and_grad(bridged, has_aux=True))(sample)

    np.testing.assert_allclose(value, 5.0)
    np.testing.assert_allclose(gradient, 2 * sample)
    np.testing.assert_allclose(auxiliary["signal"], np.sin(sample))
    np.testing.assert_array_equal(auxiliary["active"], np.ones((), dtype=np.bool_))
    np.testing.assert_array_equal(auxiliary["steps"], 3)


def test_jax_bridge_validates_the_auxiliary_contract() -> None:
    sample = jnp.asarray([1.0, 2.0], dtype=jnp.float32)

    with pytest.raises(TypeError, match=r"must be a \(value, aux\) tuple"):
        wrap(lambda value: np.sum(value), has_aux=True)(sample)
    with pytest.raises(TypeError):
        wrap(
            lambda value: (np.sum(value), {"label": "loss"}),
            has_aux=True,
        )(sample)
    with pytest.raises(TypeError, match=r"must be a \(value, aux\) tuple"):
        wrap(
            lambda value: (np.sum(value), None),
            has_aux=True,
            result_shape_dtypes=jax.ShapeDtypeStruct((), np.float32),
        )


def test_jax_bridge_runs_value_and_grad_eagerly_and_under_jit() -> None:
    bridged = wrap(
        lambda value, *, scale: np.sum((value * scale) ** 2),
        result_shape_dtypes=jax.ShapeDtypeStruct((), np.float32),
    )
    sample = jnp.asarray([1.0, 2.0, -3.0], dtype=jnp.float32)
    scale = jnp.asarray(2.0, dtype=jnp.float32)

    def objective(value, factor):
        return bridged(value, scale=factor)

    direct_value = bridged(sample, scale=scale)
    eager_value, eager_gradient = jax.value_and_grad(
        objective,
        argnums=(0, 1),
    )(sample, scale)
    jit_value, jit_gradient = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))(sample, scale)

    np.testing.assert_allclose(direct_value, 56.0)
    np.testing.assert_allclose(eager_value, 56.0)
    np.testing.assert_allclose(eager_gradient[0], 2 * sample * scale**2)
    np.testing.assert_allclose(eager_gradient[1], 2 * scale * np.sum(sample**2))
    np.testing.assert_allclose(jit_value, eager_value)
    np.testing.assert_allclose(jit_gradient[0], eager_gradient[0])
    np.testing.assert_allclose(jit_gradient[1], eager_gradient[1])


@pytest.mark.parametrize(
    "result_shape_dtypes",
    [None, {"field": jax.ShapeDtypeStruct((3,), np.float64)}],
)
def test_jax_bridge_preserves_pytree_arguments_and_outputs(
    result_shape_dtypes,
) -> None:
    with _enable_x64(True):  # noqa: FBT003 - JAX exposes a positional context API
        bridged = wrap(
            lambda parameters, scale: {"field": parameters["field"] * scale},
            result_shape_dtypes=result_shape_dtypes,
        )
        parameters = {"field": jnp.asarray([1.0, -2.0, 0.5])}
        scale = jnp.asarray(1.5)

        parameter_gradient, scale_gradient = jax.grad(
            lambda params, factor: jnp.sum(bridged(params, scale=factor)["field"] ** 2),
            argnums=(0, 1),
        )(parameters, scale)

        np.testing.assert_allclose(
            parameter_gradient["field"],
            2 * parameters["field"] * scale**2,
        )
        np.testing.assert_allclose(
            scale_gradient,
            2 * scale * jnp.sum(parameters["field"] ** 2),
        )


def test_jax_bridge_translates_the_complex_adjoint_convention() -> None:
    with _enable_x64(True):  # noqa: FBT003 - JAX exposes a positional context API
        coefficient = 2.0 + 3.0j
        bridged = wrap(lambda value: coefficient * value)
        sample = jnp.asarray(1.5 + 2.0j, dtype=jnp.complex128)

        bridged_gradient = jax.grad(
            lambda value: jnp.real(bridged(value) * jnp.conj(bridged(value)))
        )(sample)
        native_gradient = jax.grad(
            lambda value: jnp.real((coefficient * value) * jnp.conj(coefficient * value))
        )(sample)

        np.testing.assert_allclose(bridged_gradient, native_gradient)
        np.testing.assert_allclose(bridged_gradient, 39.0 - 52.0j)


def test_jax_bridge_matches_nonholomorphic_complex_outputs() -> None:
    with _enable_x64(True):  # noqa: FBT003 - JAX exposes a positional context API
        bridged = wrap(
            lambda value: (
                np.conjugate(value),
                np.real(value),
                np.imag(value),
                np.abs(value) ** 2,
            ),
            result_shape_dtypes=(
                jax.ShapeDtypeStruct((), np.complex128),
                jax.ShapeDtypeStruct((), np.float64),
                jax.ShapeDtypeStruct((), np.float64),
                jax.ShapeDtypeStruct((), np.float64),
            ),
        )
        sample = jnp.asarray(1.5 + 2.0j, dtype=jnp.complex128)

        def bridged_loss(value):
            conjugate, real, imag, power = bridged(value)
            return jnp.real((1.0 + 2.0j) * conjugate) + 0.3 * real - 0.7 * imag + 0.2 * power

        def native_loss(value):
            return (
                jnp.real((1.0 + 2.0j) * jnp.conj(value))
                + 0.3 * jnp.real(value)
                - 0.7 * jnp.imag(value)
                + 0.2 * jnp.abs(value) ** 2
            )

        np.testing.assert_allclose(
            jax.grad(bridged_loss)(sample),
            jax.grad(native_loss)(sample),
        )


@pytest.mark.parametrize(
    "result_shape_dtypes",
    [None, jax.ShapeDtypeStruct((), np.float64)],
)
def test_jax_reverse_mode_replays_the_pure_advect_function_once(
    result_shape_dtypes,
) -> None:
    with _enable_x64(True):  # noqa: FBT003 - JAX exposes a positional context API
        calls = 0

        def operation(value):
            nonlocal calls
            calls += 1
            return np.sum(value * value)

        bridged = wrap(
            operation,
            result_shape_dtypes=result_shape_dtypes,
        )
        value, gradient = jax.value_and_grad(bridged)(jnp.asarray([1.0, 2.0]))
        value.block_until_ready()
        gradient.block_until_ready()

        assert calls == 2


def test_jax_bridge_rejects_integer_input_leaves() -> None:
    bridged = wrap(lambda value: np.sum(value))
    with pytest.raises(TypeError, match="only NumPy floating and complex arrays"):
        bridged(jnp.asarray([1, 2], dtype=jnp.int32))


def test_jax_bridge_rejects_non_numpy_floating_dtypes() -> None:
    bridged = wrap(lambda value: np.sum(value))
    with pytest.raises(TypeError, match="only NumPy floating and complex arrays"):
        bridged(jnp.asarray([1.0, 2.0], dtype=jnp.bfloat16))


def test_jax_bridge_rejects_integer_output_leaves_eagerly() -> None:
    bridged = wrap(lambda value: np.sum(value, dtype=np.int32))
    with pytest.raises(TypeError, match="only NumPy floating and complex"):
        bridged(jnp.asarray([1.0, 2.0], dtype=jnp.float32))


def test_jax_bridge_rejects_a_wrong_result_dtype() -> None:
    bridged = wrap(
        lambda value: np.sum(value, dtype=np.float32),
        result_shape_dtypes=jax.ShapeDtypeStruct((), np.complex64),
    )

    with pytest.raises(
        jax.errors.JaxRuntimeError,
        match=r"Incorrect output dtype.*Expected: complex64, Actual: float32",
    ):
        bridged(jnp.asarray([1.0], dtype=jnp.float32)).block_until_ready()


def test_jax_bridge_rejects_a_wrong_result_pytree() -> None:
    bridged = wrap(
        lambda value: {"actual": np.sum(value)},
        result_shape_dtypes={"declared": jax.ShapeDtypeStruct((), np.float32)},
    )

    with pytest.raises(
        jax.errors.JaxRuntimeError,
        match="output pytree does not match JAX result_shape_dtypes",
    ):
        bridged(jnp.asarray([1.0], dtype=jnp.float32))["declared"].block_until_ready()
