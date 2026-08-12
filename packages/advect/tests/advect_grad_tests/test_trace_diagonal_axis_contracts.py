"""Derivative contracts for NumPy trace and diagonal axis metadata."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad


@pytest.mark.parametrize("operation", [np.trace, np.diagonal], ids=("trace", "diagonal"))
@pytest.mark.parametrize(("axis1", "axis2"), [(0, 2), (2, 0)], ids=("normal", "swapped"))
def test_nonfinal_matrix_axes_survive_serialized_differentiation(
    operation: Any,
    axis1: int,
    axis2: int,
) -> None:
    def function(argument: object) -> object:
        return operation(argument, offset=1, axis1=axis1, axis2=axis2)

    value = np.arange(24.0).reshape(2, 3, 4)
    tangent = np.linspace(-1.0, 1.0, value.size).reshape(value.shape)
    expected = function(value)
    expected_tangent = function(tangent)
    cotangent = np.linspace(0.5, 1.5, np.size(expected)).reshape(np.shape(expected))

    primal = ad.stage(function, specs=(ad.ArraySpec(value.shape, value.dtype),))
    restored_primal = ad.StagedProgram.from_dict(primal.to_dict())
    staged_output, staged_directional = ad.jvp(restored_primal)(value, tangents=tangent)
    assert_allclose(staged_output, expected)
    assert_allclose(staged_directional, expected_tangent)

    restored_pullback = ad.StagedProgram.from_dict(ad.vjp_program(primal).to_dict())
    gradient = restored_pullback(value, cotangent=cotangent)
    assert_allclose(np.vdot(gradient, tangent), np.vdot(cotangent, expected_tangent))
