"""Portable sorting and search qualification."""

from __future__ import annotations

from typing import Any

import array_api_strict as xp
import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad


def _roundtrip(program: ad.StagedProgram) -> ad.StagedProgram:
    return ad.StagedProgram.from_dict(program.to_dict())


@pytest.mark.parametrize("operation", ["argsort", "searchsorted"])
def test_discrete_sorting_operations_roundtrip_but_do_not_differentiate(
    operation: str,
) -> None:
    value = xp.asarray([3.0, 1.0, 2.0], dtype=xp.float64)

    if operation == "argsort":

        def function(argument: object) -> object:
            namespace = argument.__array_namespace__()  # type: ignore[attr-defined]
            return namespace.argsort(argument, stable=True)

        expected = xp.argsort(value, stable=True)
    else:
        queries = xp.asarray([0.5, 2.5, 4.0], dtype=xp.float64)

        def function(argument: object) -> object:
            namespace = argument.__array_namespace__()  # type: ignore[attr-defined]
            return namespace.searchsorted(argument, queries)

        value = xp.sort(value)
        expected = xp.searchsorted(value, queries)

    program = ad.stage(
        function,
        specs=(ad.ArraySpec(value.shape, value.dtype),),
    )

    assert xp.all(function(value) == expected)
    assert xp.all(program(value) == expected)
    assert xp.all(_roundtrip(program)(value) == expected)

    def loss(argument: object) -> object:
        namespace = argument.__array_namespace__()  # type: ignore[attr-defined]
        indices = function(argument)
        return namespace.sum(namespace.astype(indices, argument.dtype))  # type: ignore[attr-defined]

    with pytest.raises(ad.NoVJPError, match="non-differentiable"):
        ad.grad(loss)(value)

    loss_program = ad.stage(
        loss,
        specs=(ad.ArraySpec(value.shape, value.dtype),),
    )
    restored_loss = _roundtrip(loss_program)
    with pytest.raises(ad.NoVJPError, match="non-differentiable"):
        ad.grad(loss_program)
    with pytest.raises(ad.NoVJPError, match="non-differentiable"):
        ad.grad(restored_loss)


def _namespace(value: Any) -> Any:
    return value.__array_namespace__()


@pytest.mark.parametrize(
    ("function", "expected_tangent", "expected_cotangent"),
    [
        pytest.param(
            lambda v: _namespace(v).where(v > 0.0, v, -v),
            lambda v, t: np.sign(v) * t,
            lambda v, g: np.sign(v) * g,
            id="comparison",
        ),
        pytest.param(
            lambda v: _namespace(v).take(v, _namespace(v).argsort(v)),
            lambda v, t: t[np.argsort(v)],
            lambda v, g: g[np.argsort(np.argsort(v))],
            id="argsort",
        ),
        pytest.param(
            lambda v: v * _namespace(v).astype(_namespace(v).argmax(v), v.dtype),
            lambda v, t: np.argmax(v) * t,
            None,
            id="argmax",
        ),
    ],
)
def test_array_api_discrete_outputs_in_forward_and_reverse_mode(
    function: Any,
    expected_tangent: Any,
    expected_cotangent: Any,
) -> None:
    """Discrete results carry no tangent; only a float cast of indices resists reverse mode.

    Forward mode linearizes every operation a tangent reaches, so boolean and
    integer results must pass through with no tangent. Reverse mode asks for a
    cotangent only where one arrives: a boolean result has none, and an index
    that only feeds ``take`` receives none. A cotangent reaching integer
    indices through a float cast keeps the ``NoVJPError`` that the test above
    also pins for staged programs.
    """
    value = np.array([0.3, -1.2, 0.7])
    tangent = np.array([1.0, -0.5, 0.25])
    cotangent = np.array([0.5, 2.0, -1.0])
    argument = xp.asarray(value, dtype=xp.float64)

    output, output_tangent = ad.jvp(function)(
        argument,
        tangents=xp.asarray(tangent, dtype=xp.float64),
    )

    assert type(output_tangent) is type(output)
    assert_allclose(np.asarray(output_tangent), expected_tangent(value, tangent))

    _output, pullback = ad.vjp(function)(argument)
    try:
        if expected_cotangent is None:
            with pytest.raises(ad.NoVJPError, match="non-differentiable"):
                pullback(xp.asarray(cotangent))
            return
        gradient = pullback(xp.asarray(cotangent))
    finally:
        pullback.close()
    assert_allclose(np.asarray(gradient), expected_cotangent(value, cotangent))
