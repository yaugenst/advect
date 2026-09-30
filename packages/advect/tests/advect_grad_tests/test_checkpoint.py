"""Manual dynamic rematerialization contracts."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad
from advect.core._registry import get_registry


def test_checkpoint_preserves_the_ordinary_callable_contract() -> None:
    with pytest.raises(TypeError, match="checkpoint function must be callable"):
        ad.checkpoint(None)  # type: ignore[arg-type]

    def shifted(value: float, *, offset: float = 1.0) -> float:
        """Shift one value."""
        return value + offset

    wrapped = ad.checkpoint(shifted)
    assert wrapped(2.0, offset=3.0) == 5.0
    assert (wrapped.__name__, wrapped.__doc__) == (shifted.__name__, shifted.__doc__)


def test_checkpoint_recomputes_once_during_reverse() -> None:
    calls = 0

    @ad.checkpoint
    def block(value: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.sin(value) ** 2

    x = np.linspace(-0.5, 0.5, 7)
    gradient = ad.grad(lambda value: np.sum(block(value)))(x)

    assert_allclose(gradient, 2 * np.sin(x) * np.cos(x))
    assert calls == 2


def test_checkpoint_jvp_recomputes_the_region() -> None:
    calls = 0

    @ad.checkpoint
    def block(value: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return value**3

    x = np.array([1.0, 2.0, -3.0])
    value, tangent = ad.jvp(block)(x, tangents=np.ones_like(x))

    assert_allclose(value, x**3)
    assert_allclose(tangent, 3 * x**2)
    assert calls == 2


def test_checkpoint_supports_pytree_multi_argument_gradients() -> None:
    @ad.checkpoint
    def block(values: dict[str, np.ndarray]) -> np.ndarray:
        return values["left"] * values["right"]

    left = np.array([2.0, 3.0])
    right = np.array([4.0, 5.0])
    dleft, dright = ad.grad(
        lambda x, y: np.sum(block({"left": x, "right": y})),
        argnums=(0, 1),
    )(left, right)

    assert_allclose(dleft, right)
    assert_allclose(dright, left)


def test_checkpoint_preserves_output_pytree_structure() -> None:
    @ad.checkpoint
    def block(value: np.ndarray) -> dict[str, np.ndarray]:
        return {"square": value**2, "cube": value**3}

    x = np.array([1.0, 2.0, -3.0])
    gradient = ad.grad(lambda value: np.sum(block(value)["square"] + block(value)["cube"]))(x)

    assert_allclose(gradient, 2 * x + 3 * x**2)


def test_checkpoint_is_atomic_on_the_outer_tape() -> None:
    @ad.checkpoint
    def block(value: np.ndarray) -> np.ndarray:
        return np.sin(value) ** 2

    _value, linear = ad.linearize(lambda value: np.sum(block(value)), np.ones(4))
    try:
        op_names = cast("Any", linear)._trace.tape.op_names
        assert "custom.advect_internal.checkpoint" in op_names
        assert "array.sin" not in op_names
    finally:
        linear.close()


def test_checkpoint_wrappers_share_one_stable_registry_operation() -> None:
    first = ad.checkpoint(lambda value: value**2)
    assert ad.grad(first)(2.0) == 4.0
    registry = get_registry()
    stable_state = len(registry._ops), registry.get_revision()

    second = ad.checkpoint(lambda value: {"square": value**2, "cube": value**3})
    assert ad.grad(lambda value: second(value)["cube"])(2.0) == 12.0

    assert (len(registry._ops), registry.get_revision()) == stable_state


def test_checkpoint_rejects_abstract_staging() -> None:
    @ad.checkpoint
    def block(value: np.ndarray) -> np.ndarray:
        return value * value

    with pytest.raises(ad.TracingError, match="concrete dynamic autodiff only"):
        ad.stage(block, specs=(ad.ArraySpec((2,), "float64"),))


def test_checkpoint_rejects_opaque_residual_primitives() -> None:
    @ad.primitive(name="tests.checkpoint.residual_barrier", residual=True)
    def primitive(value: np.ndarray) -> ad.PrimitiveResult[np.ndarray]:
        return ad.PrimitiveResult(value, object())

    @primitive.def_transpose
    def transpose(
        cotangent: np.ndarray,
        primals: tuple[np.ndarray, ...],
        output: np.ndarray,
        residual: object,
    ) -> tuple[np.ndarray]:
        del primals, output, residual
        return (cotangent,)

    block = ad.checkpoint(primitive)

    with pytest.raises(ad.TracingError, match="residual primitive"):
        ad.grad(lambda value: np.sum(block(value)))(np.ones(2))


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
