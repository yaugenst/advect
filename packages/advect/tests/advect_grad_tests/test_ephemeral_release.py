"""One-shot dynamic transforms release concrete tape payloads."""

from __future__ import annotations

import numpy as np

from advect.autodiff._ephemeral import linearize_call


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

    gradient = linear._consume_pullback(np.ones_like(value))

    np.testing.assert_allclose(gradient, 2 * np.sin(np.arange(8.0)) * np.cos(np.arange(8.0)))
    after = trace.tape.stats()
    assert trace.tape.is_consumed
    assert after["retained_value_count"] == 0
    assert after["literal_count"] == 0
    assert after["residual_count"] == 0


def test_public_linear_map_remains_reusable() -> None:
    _value, linear = linearize_call(
        lambda x: x * x,
        args=(np.arange(4.0),),
        kwargs={},
        argnums=(0,),
        argnames=None,
        single_argnum=True,
    )
    tangent = np.ones(4)

    first = linear(tangent)
    second = linear(tangent)

    np.testing.assert_array_equal(first, second)


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
        linear._consume_pullback(np.ones_like(value)),
        2 * np.ones(4),
    )
