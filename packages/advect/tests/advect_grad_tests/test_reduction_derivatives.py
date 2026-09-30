"""Reduction and cumulative-scan derivative contracts beyond the conformance draws."""

from __future__ import annotations

import warnings
from typing import Any

import array_api_strict as strict
import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad


@pytest.mark.parametrize(
    ("function", "value", "error", "match"),
    [
        (np.mean, np.empty((0,), dtype=float), RuntimeError, "empty reduction axis"),
        (lambda x: np.var(x, ddof=1), np.array([2.0]), NotImplementedError, "count > ddof"),
        (lambda x: np.std(x, ddof=1), np.array([2.0]), NotImplementedError, "count > ddof"),
    ],
    ids=["empty-mean", "var-ddof", "std-ddof"],
)
def test_undefined_reduction_denominators_fail_in_reverse_mode(
    function: Any,
    value: np.ndarray[Any, Any],
    error: type[Exception],
    match: str,
) -> None:
    """An empty mean or ``count <= ddof`` fails instead of dividing a gradient by zero."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        output, pullback = ad.vjp(function)(value)
    assert np.isnan(output)
    try:
        with pytest.raises(error, match=match):
            pullback(np.array(1.0))
    finally:
        pullback.close()


@pytest.mark.parametrize(
    "function",
    [np.mean, np.average, np.std, np.nanmean, np.nanvar, np.nanstd],
    ids=["mean", "average", "std", "nanmean", "nanvar", "nanstd"],
)
def test_half_precision_reductions_count_long_slices(function: Any) -> None:
    """A count above the largest finite float16 stays finite."""
    value = np.linspace(-1.0, 1.0, 70_000)
    if function.__name__.startswith("nan"):
        value[::1_000] = np.nan
    half = value.astype(np.float16)
    direction = np.linspace(0.0, 1.0, value.size)

    gradient = ad.grad(function)(half)
    reference = ad.grad(function)(half.astype(np.float64))
    _output, tangent = ad.jvp(function)(half, tangents=direction.astype(np.float16))
    _output, reference_tangent = ad.jvp(function)(half.astype(np.float64), tangents=direction)

    assert gradient.dtype == np.float16
    # Most partials are float16 subnormals, whose spacing is 2**-24.
    assert_allclose(gradient, reference, rtol=1e-2, atol=2.0**-23)
    assert_allclose(tangent, reference_tangent, rtol=1e-2)


@pytest.mark.parametrize("axis", [0, -1])
@pytest.mark.parametrize(
    ("namespace", "name"),
    [
        *((np, name) for name in ("sum", "prod", "max", "amin", "nanmean", "nanvar", "cumprod")),
        *((strict, name) for name in ("sum", "prod", "max")),
    ],
)
def test_the_axis_of_a_rank_zero_input_reduces_nothing(
    namespace: Any, name: str, axis: int
) -> None:
    """NumPy's reductions accept the axis 0 or -1 of a 0-D array, and so do their rules."""
    value = namespace.asarray(2.0, dtype=namespace.float64)

    def reduce(x: Any) -> Any:
        xp = np if namespace is np else x.__array_namespace__()
        return xp.sum(getattr(xp, name)(x, axis=axis))

    expected = 0.0 if name == "nanvar" else 1.0
    tangent = namespace.asarray(3.0, dtype=namespace.float64)

    assert float(ad.grad(reduce)(value)) == expected
    assert float(ad.jvp(reduce)(value, tangents=tangent)[1]) == 3.0 * expected


@pytest.mark.parametrize(
    ("function", "shape", "axis"),
    [
        *(
            (function, shape, None)
            for function in (np.cumsum, np.cumprod)
            for shape in ((0, 3), (2, 3))
        ),
        *(
            (function, (), axis)
            # NumPy 2.1 adds cumulative_prod; the rest of the module runs on 2.0.
            for function in (np.cumsum, np.cumprod, getattr(np, "cumulative_prod", None))
            if function is not None
            for axis in (None, 0, -1)
        ),
    ],
)
def test_cumulative_scan_derivatives_read_flattened_inputs_as_vectors(
    function: Any,
    shape: tuple[int, ...],
    axis: int | None,
) -> None:
    """NumPy scans a flattened or 0-d input as a vector, and so do its derivatives."""
    value = np.full(shape, 1.5)
    ones = np.ones(shape)
    expected = function(value, axis=axis)

    def tangent_of(x: Any, t: Any) -> Any:
        return ad.jvp(lambda y: function(y, axis=axis))(x, tangents=t)[1]

    tangent = tangent_of(value, ones)
    # A traced provider's cumulative scan requires an axis, so staging must flatten too.
    staged_tangent = ad.stage(tangent_of, value, ones)(value, ones)
    vector_tangent = ad.jvp(lambda y: function(y, axis=0))(value.ravel(), tangents=ones.ravel())[1]
    gradient = ad.grad(lambda x: np.sum(function(x, axis=axis)))(value)
    staged = ad.stage(ad.grad(lambda x: np.sum(function(x, axis=axis))), value)(value)

    assert tangent.shape == staged_tangent.shape == expected.shape
    assert gradient.shape == staged.shape == shape
    assert_allclose(tangent, vector_tangent)
    assert_allclose(staged_tangent, tangent)
    assert_allclose(staged, gradient)
