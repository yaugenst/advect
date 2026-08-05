"""Dynamic and staged derivative coverage for the portable Array API slice."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import array_api_strict as strict
import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad

if TYPE_CHECKING:
    from collections.abc import Callable


_OUTER_RIGHT = strict.asarray([0.5, -1.0, 2.0], dtype=strict.float64)
_CROSS_RIGHT = strict.asarray(
    [[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]],
    dtype=strict.float64,
)
_TENSORDOT_RIGHT = strict.asarray(
    np.arange(20, dtype=np.float64).reshape(4, 5) / 10,
)
_MOVEAXIS_WEIGHTS_ARRAY = np.arange(1, 25, dtype=np.float64).reshape(3, 4, 2)
_MOVEAXIS_WEIGHTS = strict.asarray(_MOVEAXIS_WEIGHTS_ARRAY)
_TILE_WEIGHTS_ARRAY = np.arange(1, 37, dtype=np.float64).reshape(4, 9)
_TILE_WEIGHTS = strict.asarray(_TILE_WEIGHTS_ARRAY)
_OFFSET_DIAGONAL_GRADIENT = np.asarray(
    [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
)


@pytest.mark.parametrize(
    ("loss", "input_value", "expected_gradient"),
    [
        (
            lambda x: x.__array_namespace__().sum(
                x.__array_namespace__().clip(x, min=-0.5, max=1.5)
            ),
            strict.asarray([-1.0, 0.2, 1.0, 2.0], dtype=strict.float64),
            np.asarray([0.0, 1.0, 1.0, 0.0]),
        ),
        (
            lambda x: x.__array_namespace__().sum(x.__array_namespace__().copysign(x, 1.0)),
            strict.asarray([-1.0, 0.2, 1.0, 2.0], dtype=strict.float64),
            np.asarray([-1.0, 1.0, 1.0, 1.0]),
        ),
        (
            lambda x: x.__array_namespace__().sum(
                x.__array_namespace__().cumulative_prod(x, axis=0)
            ),
            strict.asarray([1.2, 0.7, 1.5, 0.9], dtype=strict.float64),
            np.asarray([3.695, 4.62, 1.596, 1.26]),
        ),
        (
            lambda x: x.__array_namespace__().sum(x.__array_namespace__().diff(x, axis=0, n=2)),
            strict.asarray([0.2, 1.1, -0.4, 2.0, 0.7], dtype=strict.float64),
            np.asarray([1.0, -1.0, 0.0, -1.0, 1.0]),
        ),
        (
            lambda x: x.__array_namespace__().sum(
                x.__array_namespace__().moveaxis(x, 0, 2) * _MOVEAXIS_WEIGHTS
            ),
            strict.asarray(np.arange(24, dtype=np.float64).reshape(2, 3, 4)),
            np.moveaxis(_MOVEAXIS_WEIGHTS_ARRAY, 2, 0),
        ),
        (
            lambda x: x.__array_namespace__().sum(
                x.__array_namespace__().tile(x, (2, 3)) * _TILE_WEIGHTS
            ),
            strict.asarray(np.arange(6, dtype=np.float64).reshape(2, 3)),
            _TILE_WEIGHTS_ARRAY.reshape(2, 2, 3, 3).sum(axis=(0, 2)),
        ),
        (
            lambda x: x.__array_namespace__().sum(
                x.__array_namespace__().linalg.diagonal(x, offset=1)
            ),
            strict.asarray(np.arange(12, dtype=np.float64).reshape(3, 4)),
            _OFFSET_DIAGONAL_GRADIENT,
        ),
        (
            lambda x: x.__array_namespace__().sum(
                x.__array_namespace__().linalg.trace(x, offset=1)
            ),
            strict.asarray(np.arange(12, dtype=np.float64).reshape(3, 4)),
            _OFFSET_DIAGONAL_GRADIENT,
        ),
        (
            lambda x: x.__array_namespace__().sum(
                x.__array_namespace__().linalg.outer(x, _OUTER_RIGHT)
            ),
            strict.asarray([1.0, 2.0, -0.5, 3.0], dtype=strict.float64),
            np.full((4,), 1.5),
        ),
        (
            lambda x: x.__array_namespace__().sum(
                x.__array_namespace__().linalg.cross(x, _CROSS_RIGHT)
            ),
            strict.asarray(
                [[1.0, 2.0, -0.5], [3.0, -1.0, 0.75]],
                dtype=strict.float64,
            ),
            np.asarray([[-3.0, 1.5, 1.5], [1.0, -2.25, 1.25]]),
        ),
        (
            lambda x: x.__array_namespace__().sum(
                x.__array_namespace__().linalg.tensordot(
                    x,
                    _TENSORDOT_RIGHT,
                    axes=1,
                )
            ),
            strict.asarray(np.arange(24, dtype=np.float64).reshape(2, 3, 4) / 10),
            np.broadcast_to(np.asarray([1.0, 3.5, 6.0, 8.5]), (2, 3, 4)),
        ),
    ],
    ids=[
        "clip",
        "copysign",
        "cumulative-prod",
        "diff",
        "moveaxis",
        "tile",
        "diagonal",
        "trace",
        "outer",
        "cross",
        "tensordot",
    ],
)
def test_portable_array_api_gradients_match_across_lifetimes(
    loss: Callable[[Any], Any],
    input_value: Any,
    expected_gradient: np.ndarray[Any, Any],
) -> None:
    dynamic = ad.grad(loss)(input_value)
    staged = ad.grad(
        ad.stage(
            loss,
            specs=(ad.ArraySpec(input_value.shape, input_value.dtype),),
        )
    )(input_value)

    assert type(dynamic) is type(input_value)
    assert type(staged) is type(input_value)
    assert dynamic.dtype == input_value.dtype
    assert staged.dtype == input_value.dtype
    assert_allclose(np.asarray(dynamic), expected_gradient, rtol=1e-10, atol=1e-10)
    assert_allclose(np.asarray(staged), np.asarray(dynamic), rtol=1e-10, atol=1e-10)
