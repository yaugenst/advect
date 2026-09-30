"""End-to-end lifetime contracts for exact primitive residuals."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._primitive import Primitive


def _square_with_residual(
    name: str,
    *,
    released: list[object],
    forwards: list[object] | None = None,
    on_transpose: Callable[[object], object] | None = None,
    jvp: bool = True,
) -> Primitive[..., Any]:
    """Square ``x``, keeping ``2 * x`` as the residual its transpose consumes."""

    @ad.primitive(name=name, residual=True)
    def primitive(x: np.ndarray) -> ad.PrimitiveResult[np.ndarray]:
        residual = 2 * x.copy()
        if forwards is not None:
            forwards.append(residual)
        return ad.PrimitiveResult(x * x, residual, release=released.append)

    primitive.def_abstract(lambda x: x.spec)
    if jvp:
        primitive.def_jvp(lambda _output, primals, tangents: 2 * primals[0] * tangents[0])

    @primitive.def_transpose
    def transpose_rule(
        cotangent: np.ndarray,
        primals: tuple[np.ndarray, ...],
        output: np.ndarray,
        residual: object,
    ) -> tuple[np.ndarray]:
        del primals, output
        if on_transpose is not None:
            on_transpose(residual)
        return (cotangent * cast("np.ndarray", residual),)

    return primitive


def test_one_shot_grad_pairs_and_releases_each_exact_residual() -> None:
    released: list[object] = []
    forwards: list[object] = []
    transposes: list[object] = []
    primitive = _square_with_residual(
        "tests.residual.grad_pairing",
        released=released,
        forwards=forwards,
        on_transpose=transposes.append,
    )
    x = np.array([0.5, 1.5])

    gradient = ad.grad(lambda value: np.sum(primitive(value) + 3 * primitive(value + 1)))(x)

    assert_allclose(gradient, 2 * x + 6 * (x + 1))
    assert [id(value) for value in transposes] == [id(forwards[1]), id(forwards[0])]
    assert [id(value) for value in released] == [id(forwards[1]), id(forwards[0])]


def test_direct_call_and_one_shot_jvp_release_without_reverse() -> None:
    released: list[object] = []
    primitive = _square_with_residual(
        "tests.residual.direct_and_jvp",
        released=released,
    )
    x = np.array([1.0, 2.0])

    assert_allclose(primitive(x), x * x)
    assert len(released) == 1

    value, tangent = ad.jvp(primitive)(x, tangents=np.ones_like(x))
    assert_allclose(value, x * x)
    assert_allclose(tangent, 2 * x)
    assert len(released) == 2


def test_reusable_linear_map_retains_until_idempotent_close() -> None:
    released: list[object] = []
    primitive = _square_with_residual(
        "tests.residual.reusable_linear_map",
        released=released,
    )
    x = np.array([1.0, 2.0])

    value, linear = ad.linearize(primitive, x)
    assert_allclose(value, x * x)
    assert released == []
    assert_allclose(linear.pullback(np.ones_like(x)), 2 * x)
    assert_allclose(linear.pullback(2 * np.ones_like(x)), 4 * x)
    batched = linear.transpose_many(
        (
            np.ones_like(x),
            2 * np.ones_like(x),
        )
    )
    assert_allclose(batched[0], 2 * x)
    assert_allclose(batched[1], 4 * x)
    assert released == []

    linear.close()
    assert len(released) == 1
    linear.close()
    assert len(released) == 1
    with pytest.raises(RuntimeError, match="closed or consumed"):
        linear.pullback(np.ones_like(x))


def test_jacobian_reuses_and_releases_one_exact_residual() -> None:
    released: list[object] = []
    forwards: list[object] = []
    transposes: list[object] = []
    primitive = _square_with_residual(
        "tests.residual.jacobian",
        released=released,
        forwards=forwards,
        on_transpose=transposes.append,
    )
    x = np.arange(1.0, 5.0)

    actual = ad.jacobian(primitive)(x)

    assert_allclose(actual, np.diag(2.0 * x))
    assert len(forwards) == len(released) == 1
    assert len(transposes) == x.size
    assert all(residual is forwards[0] for residual in transposes)
    assert released[0] is forwards[0]


def test_vjp_pullback_consumes_and_releases_automatically() -> None:
    released: list[object] = []
    primitive = _square_with_residual(
        "tests.residual.vjp_close",
        released=released,
    )
    x = np.array([1.0, 2.0])

    _value, pullback = ad.vjp(primitive)(x)
    assert isinstance(pullback, ad.Pullback)
    assert released == []
    assert_allclose(pullback(np.ones_like(x)), 2 * x)
    assert len(released) == 1
    with pytest.raises(RuntimeError, match="closed or consumed"):
        pullback(np.ones_like(x))

    close = cast("Any", pullback).close
    close()
    assert len(released) == 1
    close()
    assert len(released) == 1


def test_reentrant_close_rejection_does_not_poison_later_close() -> None:
    released: list[object] = []
    pullback_owner: dict[str, Any] = {}

    def close_during_traversal(_residual: object) -> None:
        with pytest.raises(RuntimeError, match="during traversal"):
            pullback_owner["value"].close()

    primitive = _square_with_residual(
        "tests.residual.reentrant_close",
        released=released,
        on_transpose=close_during_traversal,
    )
    x = np.array([1.0, 2.0])
    _value, pullback = ad.vjp(primitive)(x)
    pullback_owner["value"] = cast("Any", pullback)

    assert_allclose(pullback(np.ones_like(x)), 2 * x)
    assert len(released) == 1

    cast("Any", pullback).close()
    assert len(released) == 1


def test_forward_and_transpose_failures_release_residuals() -> None:
    forward_released: list[object] = []
    forward_primitive = _square_with_residual(
        "tests.residual.forward_failure",
        released=forward_released,
    )
    x = np.array([1.0, 2.0])

    def failing_forward(value: np.ndarray) -> np.ndarray:
        forward_primitive(value)
        msg = "forward failed"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="forward failed"):
        ad.grad(failing_forward)(x)
    assert len(forward_released) == 1

    transpose_released: list[object] = []

    @ad.primitive(name="tests.residual.transpose_failure", residual=True)
    def transpose_primitive(value: np.ndarray) -> ad.PrimitiveResult[np.ndarray]:
        token = object()
        return ad.PrimitiveResult(value, token, release=transpose_released.append)

    @transpose_primitive.def_transpose
    def failing_transpose(
        cotangent: np.ndarray,
        primals: tuple[np.ndarray, ...],
        output: np.ndarray,
        residual: object,
    ) -> tuple[np.ndarray]:
        del cotangent, primals, output, residual
        msg = "transpose failed"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="transpose failed"):
        ad.grad(lambda value: np.sum(transpose_primitive(value)))(x)
    assert len(transpose_released) == 1


def test_unused_residual_node_is_released() -> None:
    released: list[object] = []
    primitive = _square_with_residual(
        "tests.residual.unused_node",
        released=released,
    )
    x = np.array([1.0, 2.0])

    def loss(value: np.ndarray) -> np.ndarray:
        primitive(value)
        return np.sum(value)

    assert_allclose(ad.grad(loss)(x), np.ones_like(x))
    assert len(released) == 1


def test_staged_residual_primitive_stays_atomic_under_grad() -> None:
    released: list[object] = []
    forwards: list[object] = []
    seen_residuals: list[object] = []
    primitive = _square_with_residual(
        "tests.residual.staged_grad",
        released=released,
        forwards=forwards,
        on_transpose=seen_residuals.append,
        jvp=False,
    )
    program = ad.stage(
        primitive,
        specs=(ad.ArraySpec((2,), "float64"),),
    )
    x = np.array([1.0, 2.0])

    gradient = ad.grad(lambda value: np.sum(program(value)))(x)

    assert_allclose(gradient, 2 * x)
    # The implementation ran once, on concrete values rather than tracers.
    assert [type(residual) for residual in forwards] == [np.ndarray]
    assert len(seen_residuals) == 1
    assert released == seen_residuals


def test_none_is_a_valid_exact_residual_payload() -> None:
    released: list[object] = []

    @ad.primitive(name="tests.residual.none_payload", residual=True)
    def primitive(x: np.ndarray) -> ad.PrimitiveResult[np.ndarray]:
        return ad.PrimitiveResult(x, None, release=released.append)

    @primitive.def_transpose
    def transpose_rule(
        cotangent: np.ndarray,
        primals: tuple[np.ndarray, ...],
        output: np.ndarray,
        residual: object,
    ) -> tuple[np.ndarray]:
        del primals, output
        assert residual is None
        return (cotangent,)

    x = np.array([1.0, 2.0])
    assert_allclose(
        ad.grad(lambda value: np.sum(primitive(value)))(x),
        np.ones_like(x),
    )
    assert released == [None]


def test_primitive_result_rejects_a_noncallable_release() -> None:
    with pytest.raises(TypeError, match="release must be callable"):
        ad.PrimitiveResult(output=1, residual=object(), release=1)  # type: ignore[arg-type]
