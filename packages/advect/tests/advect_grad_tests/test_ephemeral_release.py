"""One-shot dynamic transforms release concrete tape payloads."""

from __future__ import annotations

import gc
import sys
import weakref
from typing import TYPE_CHECKING

import numpy as np
import pytest

import advect as ad
from advect.autodiff._ephemeral import linearize_call

if TYPE_CHECKING:
    from collections.abc import Callable


def test_consuming_pullback_releases_values_and_constants() -> None:
    value, linear = linearize_call(
        lambda x: np.sum(np.sin(x) ** 2),
        args=(np.arange(8.0),),
        kwargs={},
        argnums=(0,),
        argnames=None,
        single_argnum=True,
    )
    trace = linear._trace
    before = trace.tape.stats()
    assert before["retained_value_count"] > 0
    assert before["literal_count"] > 0

    gradient = linear._pullback(np.ones_like(value), consume=True)

    np.testing.assert_allclose(gradient, 2 * np.sin(np.arange(8.0)) * np.cos(np.arange(8.0)))
    after = trace.tape.stats()
    assert trace.tape.is_consumed
    assert after["retained_value_count"] == 0
    assert after["literal_count"] == 0
    assert after["residual_count"] == 0


def test_pullback_context_closes_an_unconsumed_trace() -> None:
    value = np.array([1.0, 2.0])
    _output, pullback = ad.vjp(lambda x: x * x)(value)

    with pullback as active:
        assert active is pullback

    with pytest.raises(RuntimeError, match="closed or consumed"):
        pullback(np.ones_like(value))


def test_reverse_only_trace_prunes_zero_use_values_before_pullback() -> None:
    value, linear = linearize_call(
        lambda x: np.sum(x + x),
        args=(np.arange(4.0),),
        kwargs={},
        argnums=(0,),
        argnames=None,
        single_argnum=True,
        reverse_only=True,
    )
    trace = linear._trace
    stats = trace.tape.stats()

    assert stats["reverse_pruned"] is True
    assert stats["retained_value_count"] < stats["node_count"]
    np.testing.assert_array_equal(
        linear._pullback(np.ones_like(value), consume=True),
        2 * np.ones(4),
    )


def _linearize_and_close(value: np.ndarray) -> None:
    _output, linear = ad.linearize(lambda x: np.prod(x) * x, value)
    linear.close()


class _SlottedScale:
    """A hashable callable object that cannot be weakly referenced."""

    __slots__ = ("data",)

    def __init__(self, data: np.ndarray) -> None:
        self.data = data

    def __call__(self, weights: np.ndarray) -> object:
        return np.sum(weights * self.data)


@pytest.mark.parametrize(
    "run",
    [
        pytest.param(ad.grad(np.prod), id="structural-grad"),
        pytest.param(lambda x: ad.vjp(np.cumprod)(x)[1](np.ones_like(x)), id="structural-vjp"),
        pytest.param(ad.hessian(np.prod), id="structural-hessian"),
        pytest.param(ad.jacobian(np.prod), id="reverse-jacobian"),
        pytest.param(ad.jacobian(np.cumprod), id="forward-jacobian"),
        pytest.param(lambda x: ad.jvp(np.std)(x, tangents=np.ones_like(x)), id="jvp"),
        pytest.param(_linearize_and_close, id="closed-linearize"),
        pytest.param(lambda x: ad.grad(lambda w: np.sum(w * x))(np.ones_like(x)), id="closure"),
        pytest.param(lambda x: ad.grad(_SlottedScale(x))(np.ones_like(x)), id="slotted-callable"),
        pytest.param(
            lambda x: ad.value_and_grad(lambda w: (np.sum(w * w), x), has_aux=True)(np.ones(2)),
            id="auxiliary",
        ),
    ],
)
def test_completed_transform_retains_no_caller_values(
    run: Callable[[np.ndarray], object],
) -> None:
    value = np.linspace(0.1, 0.9, 8)
    result = run(value)
    references = [weakref.ref(value)] + [
        weakref.ref(leaf) for leaf in ad.pytree.tree_leaves(result) if isinstance(leaf, np.ndarray)
    ]

    del value, result
    gc.collect()

    assert [reference() for reference in references] == [None] * len(references)


def test_differentiating_a_staged_program_retains_no_reference_to_it() -> None:
    program = ad.stage(
        lambda x: np.sum(np.sin(x) * x),
        specs=(ad.ArraySpec((4,), "float64"),),
    )
    value = np.linspace(0.1, 0.9, 4)
    references = sys.getrefcount(program)

    ad.grad(program)(value)
    ad.vjp(program)(value)[1](1.0)
    ad.jacobian(program)(value)
    gc.collect()

    assert sys.getrefcount(program) == references
