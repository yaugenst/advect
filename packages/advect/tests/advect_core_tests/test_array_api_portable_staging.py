"""Staged primal coverage for the portable Array API slice."""

from __future__ import annotations

from typing import TYPE_CHECKING

import array_api_strict as strict
import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad

if TYPE_CHECKING:
    from collections.abc import Callable


_SENTINEL = strict.asarray([0.0], dtype=strict.float32)


@pytest.mark.parametrize(
    ("operation", "input_value"),
    [
        (
            lambda xp, _x: xp.arange(1, 8, 2, dtype=xp.float32),
            _SENTINEL,
        ),
        (
            lambda xp, _x: xp.eye(3, 4, k=1, dtype=xp.float64),
            _SENTINEL,
        ),
        (
            lambda xp, _x: xp.full((2, 3), 2.5, dtype=xp.float32),
            _SENTINEL,
        ),
        (
            lambda xp, _x: xp.asarray([[1, 2], [3, 4]], dtype=xp.float32),
            _SENTINEL,
        ),
        (
            lambda xp, x: xp.repeat(x, 2, axis=1),
            strict.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=strict.float32),
        ),
        (
            lambda xp, x: xp.moveaxis(x, (0, 2), (2, 0)),
            strict.asarray(np.arange(24, dtype=np.float32).reshape(2, 3, 4)),
        ),
        (
            lambda xp, x: xp.count_nonzero(x, axis=1, keepdims=True),
            strict.asarray([[0, 2, 0], [1, 3, 4]], dtype=strict.int32),
        ),
        (
            lambda xp, x: xp.cumulative_sum(x, axis=1),
            strict.asarray([[1, 2, 3], [4, 5, 6]], dtype=strict.int8),
        ),
        (
            lambda xp, x: xp.linalg.cross(x, xp.flip(x, axis=-1)),
            strict.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=strict.float64),
        ),
        (
            lambda xp, x: xp.linalg.diagonal(x, offset=1),
            strict.asarray(np.arange(24, dtype=np.float32).reshape(2, 3, 4)),
        ),
    ],
    ids=[
        "arange",
        "eye",
        "full",
        "asarray-sequence",
        "repeat",
        "moveaxis",
        "count-nonzero",
        "cumulative-sum",
        "cross",
        "batched-diagonal",
    ],
)
def test_portable_array_api_staged_specs_and_values_match_strict(
    operation: Callable[[object, object], object],
    input_value: object,
) -> None:
    def function(value: object) -> object:
        return operation(value.__array_namespace__(), value)

    expected = function(input_value)
    program = ad.stage(
        function,
        specs=(ad.ArraySpec(input_value.shape, input_value.dtype),),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())

    for actual in (program(input_value), restored(input_value)):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("literal", [[], [[], []]])
def test_dtype_free_empty_sequences_use_the_reference_default(
    literal: list[object],
) -> None:
    def function(value: object) -> object:
        namespace = value.__array_namespace__()
        return namespace.asarray(literal)

    expected = function(_SENTINEL)
    program = ad.stage(
        function,
        specs=(ad.ArraySpec(_SENTINEL.shape, _SENTINEL.dtype),),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())

    for actual in (program(_SENTINEL), restored(_SENTINEL)):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-7, atol=1e-7)


def test_explicitly_typed_scalar_constant_does_not_round_through_weak_dtype() -> None:
    literal = -0.31

    def function(value: object) -> object:
        namespace = value.__array_namespace__()
        return value + namespace.asarray(literal, dtype=value.dtype)

    input_value = strict.asarray([0.0], dtype=strict.float64)
    expected = function(input_value)
    program = ad.stage(
        function,
        specs=(ad.ArraySpec(input_value.shape, input_value.dtype),),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())

    for actual in (program(input_value), restored(input_value)):
        assert actual.dtype == strict.float64
        assert_allclose(np.asarray(actual), np.asarray(expected), rtol=0.0, atol=0.0)
