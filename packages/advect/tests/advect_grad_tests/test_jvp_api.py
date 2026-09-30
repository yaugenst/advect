"""Public dynamic forward-mode contracts."""

from __future__ import annotations

import math
from array import array
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad


def _square(value: Any) -> Any:
    return value * value


def _scaled_value(tree: Any) -> Any:
    return 3.0 * tree["value"]


_LABELED = {"value": np.array([1.0, 2.0]), "label": "fixed"}


@pytest.mark.parametrize(
    ("function", "argnums", "primals", "tangents", "expected"),
    [
        (
            lambda x, y: x * y + np.sin(y),
            (0, 1),
            (2.0, 0.5),
            (3.0, 4.0),
            (1.0 + math.sin(0.5), 3.0 * 0.5 + 4.0 * (2.0 + math.cos(0.5))),
        ),
        (lambda x, _y: x * x, 1, (3.0, 4.0), 5.0, (9.0, 0.0)),
        (_square, (), (2.0,), (), (4.0, 0.0)),
        (_square, 0, (np.array(2.0),), 3.0, (4.0, 12.0)),
        (_scaled_value, 0, (_LABELED,), {"value": np.ones(2), "label": None}, ([3, 6], [3, 3])),
    ],
    ids=[
        "multiple-arguments",
        "disconnected-argument",
        "empty-selection",
        "python-tangent-for-rank-zero-array",
        "none-for-untraceable-leaf",
    ],
)
def test_jvp_pushes_forward_selected_tangents(
    function: Any,
    argnums: int | tuple[int, ...],
    primals: tuple[Any, ...],
    tangents: Any,
    expected: tuple[Any, Any],
) -> None:
    value, tangent = ad.jvp(function, argnums=argnums)(*primals, tangents=tangents)

    assert_allclose(value, expected[0], rtol=1e-12)
    assert_allclose(tangent, expected[1], rtol=1e-12)


def test_output_tangents_keep_the_primal_output_precision() -> None:
    # The tangent of a power of a weak Python scalar is a strong float64 array;
    # adding it to a float32 tangent must not widen the float32 output.
    def mixed(x: Any, scale: float) -> Any:
        return np.sum(np.sin(x)) + scale**3

    primals = (np.array([1.0, 2.0], np.float32), 0.7)
    tangents = (np.array([0.1, -0.2], np.float32), 0.5)
    value, tangent = ad.jvp(mixed, argnums=(0, 1))(*primals, tangents=tangents)
    _value, linear = ad.linearize(mixed, *primals, argnums=(0, 1))
    with linear:
        linear_tangents = (linear(tangents), *linear.apply_many((tangents,)))

    expected = np.sum(np.cos(primals[0]) * tangents[0]) + 0.5 * 3 * 0.7**2
    assert type(value) is type(mixed(*primals)) is np.float32
    for other in (tangent, *linear_tangents):
        assert type(other) is np.float32
        assert_allclose(other, expected, rtol=1e-6)


def test_array_jvp_rejects_a_complex_tangent_for_a_real_input() -> None:
    with pytest.raises(TypeError, match="JVP tangent for a real input cannot have complex dtype"):
        ad.jvp(np.sin)(np.ones(2), tangents=np.ones(2, dtype=np.complex128))


_SCALAR_TANGENT = "Scalar JVP tangents must be real numbers or rank-zero real arrays"


@pytest.mark.parametrize(
    ("tangent", "error", "match"),
    [
        (True, TypeError, _SCALAR_TANGENT),
        (1.0 + 1.0j, TypeError, _SCALAR_TANGENT),
        (np.ones((), dtype=bool), TypeError, _SCALAR_TANGENT),
        (np.asarray(1.0 + 1.0j), TypeError, _SCALAR_TANGENT),
        ("one", TypeError, f"{_SCALAR_TANGENT}; got str"),
        (np.ones(2), ValueError, r"Scalar JVP tangent shape mismatch: expected \(\), got \(2,\)"),
    ],
    ids=["bool", "complex", "rank-zero-bool-array", "rank-zero-complex-array", "str", "non-scalar"],
)
def test_scalar_jvp_rejects_invalid_tangents(
    tangent: object,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        ad.jvp(_square)(2.0, tangents=tangent)


def test_jvp_rejects_tangents_that_do_not_match_the_primals() -> None:
    pair = ad.jvp(lambda left, right: left + right, argnums=(0, 1))
    left = right = np.ones(2)

    with pytest.raises(TypeError, match="requires tangents as a tuple"):
        pair(left, right, tangents=np.ones(2))
    with pytest.raises(ValueError, match="tangent arity mismatch"):
        pair(left, right, tangents=(np.ones(2),))
    with pytest.raises(ValueError, match="tangent shape mismatch"):
        pair(left, right, tangents=(np.ones(3), np.ones(2)))
    with pytest.raises(ValueError, match="shape mismatch after coercion"):
        ad.jvp(_square)(np.ones(2), tangents=array("d", [1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="pytree structure"):
        ad.jvp(lambda params: params["x"] * params["x"])({"x": 2.0}, tangents=2.0)
    with pytest.raises(TypeError, match="static/untraceable input leaf"):
        ad.jvp(_scaled_value)(_LABELED, tangents={"value": np.ones(2), "label": "not-none"})
