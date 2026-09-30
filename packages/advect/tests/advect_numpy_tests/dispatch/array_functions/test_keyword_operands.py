"""Keyword and mixed operand spellings bind exactly as NumPy's signatures do."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import advect as ad
from advect_numpy_tests._assertions import (
    assert_jvp_matches_central_difference,
    assert_staged_round_trip,
)

VALUE = np.array([-2.0, -0.7, 0.2, 0.9, 1.8])
DIRECTION = np.array([0.3, -0.4, 0.5, 0.1, -0.2])
LOWER = np.array([-1.0, -1.0, 0.0, 0.5, 0.5])
UPPER = np.array([0.0, 1.0, 1.0, 1.0, 1.5])


OTHER = np.array([0.7, -1.1, 2.3, 0.2, -0.8])
OTHER_DIRECTION = np.array([-0.2, 0.6, 0.1, -0.5, 0.4])
KNOTS = np.array([-2.5, 0.0, 1.0, 3.0])
SPEC = ad.ArraySpec(VALUE.shape, VALUE.dtype)


def _clip_into_out(x: Any) -> Any:
    destination = np.zeros_like(x)
    result = np.clip(x, LOWER, a_max=UPPER, out=destination)
    assert result is destination
    return destination


@pytest.mark.parametrize(
    "clip",
    [
        pytest.param(lambda x: np.clip(x, -1.0, a_max=1.0), id="positional-then-keyword"),
        pytest.param(lambda x: np.clip(x, None, a_max=1.0), id="none-then-keyword"),
        pytest.param(lambda x: np.clip(x, a_min=-1.0, a_max=1.0), id="keyword-bounds"),
        pytest.param(lambda x: np.clip(a=x, a_min=-1.0, a_max=1.0), id="all-keywords"),
        pytest.param(lambda x: np.clip(x, LOWER, a_max=UPPER), id="array-bounds"),
        pytest.param(lambda x: np.clip(x, -1.0, a_max=0.5 * x + 0.6), id="traced-bound"),
        pytest.param(_clip_into_out, id="out"),
        pytest.param(lambda x: np.clip(x, min=-1.0, max=1.0), id="min-max"),
        pytest.param(lambda x: np.clip(x, min=-1.0), id="min-only"),
        pytest.param(lambda x: np.clip(x, max=1.0), id="max-only"),
        pytest.param(np.clip, id="unbounded"),
    ],
)
def test_clip_binds_bounds_from_every_numpy_spelling(clip: Any) -> None:
    assert_jvp_matches_central_difference(clip, (VALUE,), (DIRECTION,), rtol=1e-8, atol=1e-8)
    assert_staged_round_trip(clip, VALUE)


@pytest.mark.parametrize(
    ("clip", "message"),
    [
        pytest.param(lambda x: np.clip(x, 0.0), "only supported", id="positional-a_min-only"),
        pytest.param(lambda x: np.clip(x, a_max=1.0), "only supported", id="keyword-a_max-only"),
        pytest.param(lambda x: np.clip(x, 0.0, 1.0, min=0.0), "mixing", id="positional-and-min"),
        pytest.param(lambda x: np.clip(x, 0.0, a_max=1.0, max=1.0), "mixing", id="mixed-and-max"),
    ],
)
def test_clip_rejects_the_bound_spellings_numpy_rejects_in_every_lifetime(
    clip: Any, message: str
) -> None:
    with pytest.raises(ad.TracingError, match=message):
        ad.jvp(clip)(VALUE, tangents=DIRECTION)
    with pytest.raises(ad.TracingError, match=message):
        ad.stage(clip, specs=(SPEC,))


def test_concatenate_rejects_its_positional_only_operand_by_keyword() -> None:
    def concatenate(x: Any) -> Any:
        return np.concatenate(arrays=(x, x))

    with pytest.raises(TypeError, match="passed as keyword arguments: 'arrays'"):
        ad.jvp(concatenate)(VALUE, tangents=DIRECTION)
    with pytest.raises(TypeError, match="passed as keyword arguments: 'arrays'"):
        ad.stage(concatenate, specs=(SPEC,))


def test_empty_like_binds_its_prototype_by_keyword_as_numpy_does() -> None:
    def fill(x: Any) -> Any:
        destination = np.empty_like(prototype=x, dtype=np.float32)
        destination[...] = x
        return destination

    expected = fill(VALUE)
    primal, tangent = ad.jvp(fill)(VALUE, tangents=DIRECTION)
    np.testing.assert_array_equal(primal, expected)
    np.testing.assert_array_equal(tangent, DIRECTION.astype(np.float32))
    assert_staged_round_trip(fill, VALUE, rtol=0.0)


@pytest.mark.parametrize(
    ("function", "stageable"),
    [
        pytest.param(lambda a, b: np.dot(a=a, b=b), True, id="dot"),
        pytest.param(lambda a, b: np.dot(a, b=b), True, id="dot-mixed"),
        pytest.param(lambda a, b: np.outer(a=a, b=b), True, id="outer"),
        pytest.param(lambda a, b: np.outer(a, b=b), True, id="outer-mixed"),
        pytest.param(lambda a, b: np.kron(a=a, b=b), False, id="kron"),
        pytest.param(lambda a, b: np.cross(a=a[:3], b=b[:3]), True, id="cross"),
        pytest.param(lambda a, b: np.tensordot(a=a, b=b, axes=1), True, id="tensordot"),
        pytest.param(lambda a, b: np.interp(x=a, xp=KNOTS, fp=b[:4]), False, id="interp"),
        pytest.param(lambda a, b: np.interp(a, KNOTS, fp=b[:4]), False, id="interp-mixed"),
        pytest.param(lambda a, b: np.linspace(start=a, stop=b, num=4), False, id="linspace"),
        pytest.param(lambda a, b: np.linspace(a, stop=b, num=4), False, id="linspace-mixed"),
        pytest.param(lambda a, b: np.stack(arrays=(a, b), axis=1), True, id="stack"),
        pytest.param(lambda a, b: np.concatenate((a, b), axis=0), True, id="concatenate"),
    ],
)
def test_keyword_operand_spellings_match_numpy(function: Any, *, stageable: bool) -> None:
    assert_jvp_matches_central_difference(
        function, (VALUE, OTHER), (DIRECTION, OTHER_DIRECTION), rtol=1e-8, atol=1e-8
    )
    if stageable:
        assert_staged_round_trip(function, VALUE, OTHER)
