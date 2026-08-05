"""End-to-end qualification for the bounded :mod:`scipy.ndimage` frontend."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import ndimage as scipy_ndimage

import advect as ad
from advect.scipy import ndimage

if TYPE_CHECKING:
    from collections.abc import Callable


_LINEAR_NAMES = (
    "gaussian_filter",
    "gaussian_filter1d",
    "uniform_filter",
    "uniform_filter1d",
    "convolve",
    "correlate",
    "convolve1d",
    "correlate1d",
    "laplace",
    "gaussian_laplace",
    "sobel",
    "prewitt",
)
_EXTREMA_NAMES = (
    "maximum_filter",
    "minimum_filter",
    "maximum_filter1d",
    "minimum_filter1d",
    "grey_dilation",
    "grey_erosion",
    "grey_opening",
    "grey_closing",
    "morphological_gradient",
    "morphological_laplace",
    "white_tophat",
    "black_tophat",
)
_RANK_NAMES = ("median_filter", "rank_filter", "percentile_filter")
_PUBLIC_NAMES = (*_LINEAR_NAMES, *_EXTREMA_NAMES, *_RANK_NAMES)

_FIELD = np.array(
    [
        [0.13, 1.27, -0.82, 2.61],
        [3.19, -2.23, 0.47, 1.83],
        [-1.41, 2.07, 4.31, -0.36],
    ],
    dtype=np.float64,
)
_TANGENT = np.array(
    [
        [0.31, -0.17, 0.43, 0.11],
        [-0.29, 0.37, 0.19, -0.41],
        [0.23, -0.07, 0.47, -0.13],
    ],
    dtype=np.float64,
)
_COTANGENT = np.array(
    [
        [-0.21, 0.39, 0.13, -0.31],
        [0.17, -0.43, 0.29, 0.07],
        [0.41, 0.23, -0.11, 0.37],
    ],
    dtype=np.float64,
)
_WEIGHTS_2D = np.array([[0.31, -0.23], [0.17, 0.41]])
_WEIGHTS_1D = np.array([0.29, -0.37, 0.19])
_STRUCTURE = np.array(
    [
        [0.0, 0.13, -0.07],
        [0.19, -0.11, 0.05],
        [-0.17, 0.23, -0.03],
    ]
)
_FOOTPRINT = np.array(
    [
        [False, True, False],
        [True, True, True],
        [False, True, False],
    ]
)
_NO_OUTPUT = object()


def _sample() -> np.ndarray:
    return np.array(_FIELD, copy=True)


def _call(  # noqa: C901, PLR0911 - one compact table for the public surface.
    name: str,
    function: Callable[..., object],
    value: object,
    *,
    output: object = _NO_OUTPUT,
) -> object:
    """Invoke one representative, nontrivial form of every public function."""
    output_kwargs = {} if output is _NO_OUTPUT else {"output": output}
    if name == "gaussian_filter":
        return function(
            value,
            sigma=(0.7, 1.1),
            order=(0, 1),
            mode=("reflect", "nearest"),
            cval=1.3,
            radius=(2, 2),
            **output_kwargs,
        )
    if name == "gaussian_filter1d":
        return function(
            value,
            0.8,
            axis=1,
            order=2,
            mode="mirror",
            cval=-0.7,
            radius=2,
            **output_kwargs,
        )
    if name == "uniform_filter":
        return function(
            value,
            size=(2, 3),
            mode=("nearest", "wrap"),
            cval=0.9,
            origin=(0, -1),
            **output_kwargs,
        )
    if name == "uniform_filter1d":
        return function(
            value,
            3,
            axis=0,
            mode="constant",
            cval=0.9,
            origin=1,
            **output_kwargs,
        )
    if name in {"convolve", "correlate"}:
        return function(
            value,
            _WEIGHTS_2D,
            mode="nearest",
            cval=-0.6,
            origin=(0, -1),
            **output_kwargs,
        )
    if name in {"convolve1d", "correlate1d"}:
        origin = 1 if name == "convolve1d" else -1
        return function(
            value,
            _WEIGHTS_1D,
            axis=1,
            mode="constant",
            cval=0.8,
            origin=origin,
            **output_kwargs,
        )
    if name == "laplace":
        return function(
            value,
            mode="nearest",
            cval=0.4,
            axes=(1,),
            **output_kwargs,
        )
    if name == "gaussian_laplace":
        return function(
            value,
            0.9,
            mode="mirror",
            cval=-0.4,
            axes=(1,),
            radius=2,
            **output_kwargs,
        )
    if name in {"sobel", "prewitt"}:
        axis = 0 if name == "sobel" else 1
        mode = "mirror" if name == "sobel" else "wrap"
        return function(
            value,
            axis=axis,
            mode=mode,
            cval=0.6,
            **output_kwargs,
        )
    if name in {"maximum_filter", "minimum_filter"}:
        return function(
            value,
            size=(3, 2),
            mode=("nearest", "wrap"),
            cval=-7.0,
            origin=(1, -1),
            **output_kwargs,
        )
    if name in {"maximum_filter1d", "minimum_filter1d"}:
        return function(
            value,
            3,
            axis=1,
            mode="mirror",
            cval=-7.0,
            origin=1,
            **output_kwargs,
        )
    if name in {
        "grey_dilation",
        "grey_erosion",
        "grey_opening",
        "grey_closing",
        "morphological_gradient",
        "morphological_laplace",
        "white_tophat",
        "black_tophat",
    }:
        return function(
            value,
            footprint=_FOOTPRINT,
            structure=_STRUCTURE,
            mode="reflect",
            cval=-3.0,
            origin=0,
            **output_kwargs,
        )
    if name == "median_filter":
        return function(
            value,
            footprint=_FOOTPRINT,
            mode="mirror",
            cval=0.7,
            origin=0,
            **output_kwargs,
        )
    if name == "rank_filter":
        return function(
            value,
            2,
            footprint=_FOOTPRINT,
            mode="nearest",
            cval=-0.7,
            origin=0,
            **output_kwargs,
        )
    if name == "percentile_filter":
        return function(
            value,
            65.0,
            footprint=_FOOTPRINT,
            mode="wrap",
            cval=0.7,
            origin=0,
            **output_kwargs,
        )
    message = f"missing test invocation for {name}"
    raise AssertionError(message)


def _parameter_contract(function: Callable[..., object]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (parameter.name, parameter.kind, parameter.default)
        for parameter in inspect.signature(function).parameters.values()
    )


def _directional_difference(function: Callable[[np.ndarray], object], value: np.ndarray) -> object:
    step = 1e-6
    return (function(value + step * _TANGENT) - function(value - step * _TANGENT)) / (2 * step)


def test_public_ndimage_inventory_is_complete() -> None:
    assert len(_PUBLIC_NAMES) == len(set(_PUBLIC_NAMES))
    assert set(_PUBLIC_NAMES) == set(ndimage.__all__)


@pytest.mark.parametrize("name", _PUBLIC_NAMES)
def test_public_ndimage_signatures_match_scipy(name: str) -> None:
    actual = getattr(ndimage, name)
    expected = getattr(scipy_ndimage, name)

    assert _parameter_contract(actual) == _parameter_contract(expected)


@pytest.mark.parametrize("name", _PUBLIC_NAMES)
def test_public_ndimage_functions_match_scipy(name: str) -> None:
    sample = _sample()

    actual = _call(name, getattr(ndimage, name), sample)
    expected = _call(name, getattr(scipy_ndimage, name), sample)

    assert np.asarray(actual).dtype == np.asarray(expected).dtype
    assert_allclose(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("name", _PUBLIC_NAMES)
def test_every_ndimage_function_accepts_output_array_and_dtype(name: str) -> None:
    sample = _sample()
    actual_destination = np.empty(sample.shape, dtype=np.float64)
    expected_destination = np.empty(sample.shape, dtype=np.float64)

    actual = _call(
        name,
        getattr(ndimage, name),
        sample,
        output=actual_destination,
    )
    expected = _call(
        name,
        getattr(scipy_ndimage, name),
        sample,
        output=expected_destination,
    )
    actual_typed = _call(name, getattr(ndimage, name), sample, output=np.float32)
    expected_typed = _call(name, getattr(scipy_ndimage, name), sample, output=np.float32)

    assert actual is actual_destination
    assert expected is expected_destination
    assert_allclose(actual_destination, expected_destination, rtol=0, atol=0)
    assert np.asarray(actual_typed).dtype == np.asarray(expected_typed).dtype
    assert_allclose(actual_typed, expected_typed, rtol=0, atol=0)


@pytest.mark.parametrize("name", _PUBLIC_NAMES)
def test_every_differentiable_ndimage_function_has_numerical_jvp_and_real_adjoint(
    name: str,
) -> None:
    def function(value: object) -> object:
        return _call(name, getattr(ndimage, name), value)

    value, tangent = ad.jvp(function)(_FIELD, tangents=_TANGENT)
    expected_value = _call(name, getattr(scipy_ndimage, name), _FIELD)
    expected_tangent = _directional_difference(
        lambda argument: _call(name, getattr(scipy_ndimage, name), argument),
        _FIELD,
    )
    _value, pullback = ad.vjp(function)(_FIELD)
    cotangent = pullback(_COTANGENT)

    assert_allclose(value, expected_value, rtol=1e-13, atol=1e-13)
    assert_allclose(tangent, expected_tangent, rtol=2e-7, atol=2e-7)
    assert_allclose(
        np.vdot(tangent, _COTANGENT),
        np.vdot(_TANGENT, cotangent),
        rtol=2e-12,
        atol=2e-12,
    )


@pytest.mark.parametrize("name", _PUBLIC_NAMES)
def test_every_ndimage_function_stages_and_serializes(name: str) -> None:
    sample = _sample()

    def function(value: object) -> object:
        return _call(name, getattr(ndimage, name), value)

    expected = _call(name, getattr(scipy_ndimage, name), sample)
    program = ad.stage(
        function,
        specs=(ad.ArraySpec(sample.shape, sample.dtype),),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())

    assert_allclose(program(sample), expected, rtol=1e-13, atol=1e-13)
    assert_allclose(restored(sample), expected, rtol=1e-13, atol=1e-13)

    _value, pullback = ad.vjp(function)(sample)
    dynamic_cotangent = pullback(_COTANGENT)
    pullback_program = ad.vjp_program(program)
    restored_pullback = ad.StagedProgram.from_dict(pullback_program.to_dict())

    assert_allclose(
        pullback_program(sample, cotangent=_COTANGENT),
        dynamic_cotangent,
        rtol=2e-12,
        atol=2e-12,
    )
    assert_allclose(
        restored_pullback(sample, cotangent=_COTANGENT),
        dynamic_cotangent,
        rtol=2e-12,
        atol=2e-12,
    )


@pytest.mark.parametrize(
    "name",
    ["convolve", "correlate", "convolve1d", "correlate1d"],
)
def test_convolution_inputs_and_weights_both_differentiate_and_serialize(name: str) -> None:
    weights = _WEIGHTS_1D if name.endswith("1d") else _WEIGHTS_2D
    weight_tangent = np.linspace(-0.3, 0.4, weights.size).reshape(weights.shape)

    def function(value: object, kernel: object) -> object:
        if name.endswith("1d"):
            return getattr(ndimage, name)(
                value,
                kernel,
                axis=1,
                mode="constant",
                cval=0.8,
                origin=1,
            )
        return getattr(ndimage, name)(
            value,
            kernel,
            mode="constant",
            cval=0.8,
            origin=(0, -1),
        )

    value, tangent = ad.jvp(function, (0, 1))(
        _FIELD,
        weights,
        tangents=(_TANGENT, weight_tangent),
    )
    step = 1e-6
    expected_tangent = (
        function(_FIELD + step * _TANGENT, weights + step * weight_tangent)
        - function(_FIELD - step * _TANGENT, weights - step * weight_tangent)
    ) / (2 * step)
    _value, pullback = ad.vjp(function, (0, 1))(_FIELD, weights)
    input_cotangent, weight_cotangent = pullback(_COTANGENT)

    assert_allclose(
        value,
        function(_FIELD, weights),
        rtol=1e-13,
        atol=1e-13,
    )
    assert_allclose(tangent, expected_tangent, rtol=2e-8, atol=2e-8)
    assert_allclose(
        np.vdot(tangent, _COTANGENT),
        np.vdot(_TANGENT, input_cotangent) + np.vdot(weight_tangent, weight_cotangent),
        rtol=2e-12,
        atol=2e-12,
    )

    program = ad.stage(
        function,
        specs=(
            ad.ArraySpec(_FIELD.shape, _FIELD.dtype),
            ad.ArraySpec(weights.shape, weights.dtype),
        ),
    )
    pullback_program = ad.vjp_program(program, argnums=(0, 1))
    restored = ad.StagedProgram.from_dict(pullback_program.to_dict())
    staged_cotangents = restored(_FIELD, weights, cotangent=_COTANGENT)

    assert_allclose(staged_cotangents[0], input_cotangent, rtol=2e-12, atol=2e-12)
    assert_allclose(staged_cotangents[1], weight_cotangent, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("name", ["convolve", "correlate"])
def test_correlation_preserves_full_rank_unsorted_axes_origin_mapping(name: str) -> None:
    kwargs = {
        "weights": _WEIGHTS_2D,
        "axes": (1, 0),
        "origin": (-1, 0),
        "mode": "constant",
        "cval": 0.7,
    }

    def function(value: object) -> object:
        return getattr(ndimage, name)(value, **kwargs)

    value, directional = ad.jvp(function)(_FIELD, tangents=_TANGENT)
    scipy_function = getattr(scipy_ndimage, name)
    expected_value = scipy_function(_FIELD, **kwargs)
    expected_directional = _directional_difference(
        lambda argument: scipy_function(argument, **kwargs),
        _FIELD,
    )
    _value, pullback = ad.vjp(function)(_FIELD)
    input_cotangent = pullback(_COTANGENT)
    program = ad.stage(function, specs=(ad.ArraySpec(_FIELD.shape, _FIELD.dtype),))
    restored = ad.StagedProgram.from_dict(program.to_dict())

    assert_array_equal(value, expected_value)
    assert_allclose(directional, expected_directional, rtol=2e-9, atol=2e-9)
    assert_allclose(
        np.vdot(directional, _COTANGENT),
        np.vdot(_TANGENT, input_cotangent),
        rtol=2e-12,
        atol=2e-12,
    )
    assert_array_equal(restored(_FIELD), expected_value)


@pytest.mark.parametrize("name", ["gaussian_filter", "uniform_filter", "correlate"])
def test_linear_filter_cval_is_a_live_serializable_operand(name: str) -> None:
    boundary = np.array(0.7)
    boundary_tangent = np.array(-0.4)

    def function(value: object, boundary_value: object) -> object:
        if name == "gaussian_filter":
            return ndimage.gaussian_filter(
                value,
                sigma=(0.8, 1.1),
                mode="constant",
                cval=boundary_value,
                radius=(2, 3),
            )
        if name == "uniform_filter":
            return ndimage.uniform_filter(
                value,
                size=(2, 3),
                mode="constant",
                cval=boundary_value,
                origin=(0, -1),
            )
        return ndimage.correlate(
            value,
            _WEIGHTS_2D,
            mode="constant",
            cval=boundary_value,
            origin=(0, -1),
        )

    value, directional = ad.jvp(function, (0, 1))(
        _FIELD,
        boundary,
        tangents=(_TANGENT, boundary_tangent),
    )
    step = 1e-6
    expected_directional = (
        function(
            _FIELD + step * _TANGENT,
            boundary + step * boundary_tangent,
        )
        - function(
            _FIELD - step * _TANGENT,
            boundary - step * boundary_tangent,
        )
    ) / (2 * step)
    _value, pullback = ad.vjp(function, (0, 1))(_FIELD, boundary)
    input_cotangent, cval_cotangent = pullback(_COTANGENT)
    program = ad.stage(
        function,
        specs=(
            ad.ArraySpec(_FIELD.shape, _FIELD.dtype),
            ad.ArraySpec(boundary.shape, boundary.dtype),
        ),
    )
    pullback_program = ad.vjp_program(program, argnums=(0, 1))
    restored = ad.StagedProgram.from_dict(pullback_program.to_dict())
    staged_cotangents = restored(_FIELD, boundary, cotangent=_COTANGENT)

    assert_allclose(value, function(_FIELD, boundary), rtol=2e-13, atol=2e-13)
    assert_allclose(directional, expected_directional, rtol=2e-8, atol=2e-8)
    assert_allclose(
        np.vdot(directional, _COTANGENT),
        np.vdot(_TANGENT, input_cotangent) + np.vdot(boundary_tangent, cval_cotangent),
        rtol=2e-12,
        atol=2e-12,
    )
    assert_allclose(staged_cotangents[0], input_cotangent, rtol=2e-12, atol=2e-12)
    assert_allclose(staged_cotangents[1], cval_cotangent, rtol=2e-12, atol=2e-12)


def test_complex_correlation_operands_have_serializable_real_adjoints() -> None:
    sample = _FIELD + 1j * np.flip(_FIELD, axis=1)
    tangent = _TANGENT + 1j * np.flip(_TANGENT, axis=0)
    weights = _WEIGHTS_2D + 1j * np.array([[0.11, -0.07], [0.23, -0.13]])
    weight_tangent = np.array([[0.17, -0.29], [0.31, 0.05]]) + 1j * np.array(
        [[-0.19, 0.37], [0.07, -0.11]]
    )
    boundary = np.array(0.7 - 0.2j)
    boundary_tangent = np.array(-0.4 + 0.3j)
    cotangent = _COTANGENT + 1j * np.flip(_COTANGENT, axis=1)

    def function(value: object, kernel: object, boundary_value: object) -> object:
        return ndimage.correlate(
            value,
            kernel,
            mode="constant",
            cval=boundary_value,
            origin=(0, -1),
        )

    value, directional = ad.jvp(function, (0, 1, 2))(
        sample,
        weights,
        boundary,
        tangents=(tangent, weight_tangent, boundary_tangent),
    )
    step = 1e-6
    expected_directional = (
        function(
            sample + step * tangent,
            weights + step * weight_tangent,
            boundary + step * boundary_tangent,
        )
        - function(
            sample - step * tangent,
            weights - step * weight_tangent,
            boundary - step * boundary_tangent,
        )
    ) / (2 * step)
    _value, pullback = ad.vjp(function, (0, 1, 2))(sample, weights, boundary)
    input_cotangent, weight_cotangent, cval_cotangent = pullback(cotangent)
    program = ad.stage(
        function,
        specs=(
            ad.ArraySpec(sample.shape, sample.dtype),
            ad.ArraySpec(weights.shape, weights.dtype),
            ad.ArraySpec(boundary.shape, boundary.dtype),
        ),
    )
    pullback_program = ad.vjp_program(program, argnums=(0, 1, 2))
    restored = ad.StagedProgram.from_dict(pullback_program.to_dict())
    staged_cotangents = restored(sample, weights, boundary, cotangent=cotangent)

    assert_allclose(value, function(sample, weights, boundary), rtol=2e-13, atol=2e-13)
    assert_allclose(directional, expected_directional, rtol=2e-8, atol=2e-8)
    assert_allclose(
        np.real(np.vdot(directional, cotangent)),
        np.real(np.vdot(tangent, input_cotangent))
        + np.real(np.vdot(weight_tangent, weight_cotangent))
        + np.real(np.vdot(boundary_tangent, cval_cotangent)),
        rtol=2e-12,
        atol=2e-12,
    )
    assert_allclose(staged_cotangents[0], input_cotangent, rtol=2e-12, atol=2e-12)
    assert_allclose(staged_cotangents[1], weight_cotangent, rtol=2e-12, atol=2e-12)
    assert_allclose(staged_cotangents[2], cval_cotangent, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize(
    ("mode", "canonical"),
    [
        ("reflect", "reflect"),
        ("grid-mirror", "reflect"),
        ("mirror", "mirror"),
        ("nearest", "nearest"),
        ("wrap", "wrap"),
        ("grid-wrap", "wrap"),
        ("constant", "constant"),
        ("grid-constant", "constant"),
    ],
)
def test_gaussian_modes_and_grid_aliases_have_exact_boundary_adjoints(
    mode: str,
    canonical: str,
) -> None:
    sample = np.array([0.2, 1.1, -0.7, 2.4, 0.6])
    tangent = np.array([0.3, -0.2, 0.5, 0.1, -0.4])
    cotangent = np.array([-0.1, 0.7, 0.2, -0.5, 0.4])

    def function(value: object) -> object:
        return ndimage.gaussian_filter1d(
            value,
            0.9,
            mode=mode,
            cval=1.7,
            radius=2,
        )

    value, directional = ad.jvp(function)(sample, tangents=tangent)
    _value, pullback = ad.vjp(function)(sample)
    actual_cotangent = pullback(cotangent)
    expected_value = scipy_ndimage.gaussian_filter1d(
        sample,
        0.9,
        mode=canonical,
        cval=1.7,
        radius=2,
    )
    expected_directional = scipy_ndimage.gaussian_filter1d(
        tangent,
        0.9,
        mode=canonical,
        cval=0.0,
        radius=2,
    )
    basis = np.eye(sample.size)
    jacobian = np.stack(
        [
            scipy_ndimage.gaussian_filter1d(
                row,
                0.9,
                mode=canonical,
                cval=0.0,
                radius=2,
            )
            for row in basis
        ],
        axis=1,
    )

    assert_allclose(value, expected_value, rtol=2e-13, atol=2e-13)
    assert_allclose(directional, expected_directional, rtol=2e-13, atol=2e-13)
    assert_allclose(actual_cotangent, jacobian.T @ cotangent, rtol=2e-13, atol=2e-13)


def test_gaussian_axes_order_and_radius_sequences_differentiate() -> None:
    sample = np.arange(24, dtype=np.float64).reshape(2, 3, 4) / 7
    tangent = np.linspace(-0.4, 0.5, sample.size).reshape(sample.shape)
    kwargs = {
        "sigma": (0.8, 1.1),
        "order": (1, 2),
        "mode": ("nearest", "mirror"),
        "cval": 1.7,
        "radius": (2, 3),
        "axes": (2, 0),
    }

    value, directional = ad.jvp(lambda x: ndimage.gaussian_filter(x, **kwargs))(
        sample,
        tangents=tangent,
    )
    expected = scipy_ndimage.gaussian_filter(sample, **kwargs)
    tangent_kwargs = {**kwargs, "cval": 0.0}
    expected_directional = scipy_ndimage.gaussian_filter(tangent, **tangent_kwargs)

    assert_allclose(value, expected, rtol=2e-13, atol=2e-13)
    assert_allclose(directional, expected_directional, rtol=2e-13, atol=2e-13)


def test_gaussian_laplace_preserves_scipy_unsorted_full_axes_mapping() -> None:
    kwargs = {
        "sigma": (0.7, 1.1),
        "axes": (1, 0),
        "mode": ("nearest", "wrap"),
        "radius": (2, 3),
    }

    def function(value: object) -> object:
        return ndimage.gaussian_laplace(value, **kwargs)

    value, directional = ad.jvp(function)(_FIELD, tangents=_TANGENT)
    expected_value = scipy_ndimage.gaussian_laplace(_FIELD, **kwargs)
    expected_directional = _directional_difference(
        lambda argument: scipy_ndimage.gaussian_laplace(argument, **kwargs),
        _FIELD,
    )
    program = ad.stage(function, specs=(ad.ArraySpec(_FIELD.shape, _FIELD.dtype),))
    restored = ad.StagedProgram.from_dict(program.to_dict())

    assert_allclose(value, expected_value, rtol=2e-13, atol=2e-13)
    assert_allclose(directional, expected_directional, rtol=2e-7, atol=2e-7)
    assert_allclose(restored(_FIELD), expected_value, rtol=2e-13, atol=2e-13)


def test_extrema_ties_share_gradient_equally_including_reflected_duplicates() -> None:
    sample = np.array([1.0, 1.0, 0.0, 2.0, 2.0])
    tangent = np.array([0.2, -0.3, 0.5, 0.7, -0.1])
    cotangent = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    expected_tangent = np.array(
        [
            (2 * tangent[0] + tangent[1]) / 3,
            (tangent[0] + tangent[1]) / 2,
            tangent[3],
            (tangent[3] + tangent[4]) / 2,
            (tangent[3] + 2 * tangent[4]) / 3,
        ]
    )
    expected_cotangent = np.array(
        [
            2 * cotangent[0] / 3 + cotangent[1] / 2,
            cotangent[0] / 3 + cotangent[1] / 2,
            0.0,
            cotangent[2] + cotangent[3] / 2 + cotangent[4] / 3,
            cotangent[3] / 2 + 2 * cotangent[4] / 3,
        ]
    )

    for function, primal in (
        (ndimage.maximum_filter1d, sample),
        (ndimage.minimum_filter1d, -sample),
    ):

        def call(value: object, fn: Callable[..., object] = function) -> object:
            return fn(value, 3, mode="reflect")

        _value, directional = ad.jvp(call)(primal, tangents=tangent)
        _value, pullback = ad.vjp(call)(primal)

        assert_allclose(directional, expected_tangent, rtol=0, atol=1e-15)
        assert_allclose(pullback(cotangent), expected_cotangent, rtol=0, atol=1e-15)


def test_extrema_ties_include_constant_boundary_slots() -> None:
    sample = np.array([1.0, 1.0, 0.0])
    tangent = np.array([0.2, -0.3, 0.5])
    boundary = np.asarray(1.0)
    boundary_tangent = np.asarray(0.7)
    cotangent = np.array([1.0, 2.0, 3.0])

    def function(value: object, cval: object) -> object:
        return ndimage.maximum_filter(value, size=3, mode="constant", cval=cval)

    _value, directional = ad.jvp(function, argnums=(0, 1))(
        sample,
        boundary,
        tangents=(tangent, boundary_tangent),
    )
    _value, pullback = ad.vjp(function, (0, 1))(sample, boundary)
    input_cotangent, boundary_cotangent = pullback(cotangent)

    assert_allclose(
        directional,
        [
            (boundary_tangent + tangent[0] + tangent[1]) / 3,
            (tangent[0] + tangent[1]) / 2,
            (tangent[1] + boundary_tangent) / 2,
        ],
        rtol=0,
        atol=1e-15,
    )
    assert_allclose(
        input_cotangent,
        [
            cotangent[0] / 3 + cotangent[1] / 2,
            cotangent[0] / 3 + cotangent[1] / 2 + cotangent[2] / 2,
            0.0,
        ],
        rtol=0,
        atol=1e-15,
    )
    assert_allclose(
        boundary_cotangent,
        cotangent[0] / 3 + cotangent[2] / 2,
        rtol=0,
        atol=1e-15,
    )


@pytest.mark.parametrize(
    ("name", "extra"),
    [
        ("median_filter", {}),
        ("rank_filter", {"rank": 1}),
        ("percentile_filter", {"percentile": 50.0}),
    ],
)
def test_rank_filter_ties_share_gradient_equally(name: str, extra: dict[str, object]) -> None:
    sample = np.ones(5)
    tangent = np.array([0.2, -0.3, 0.5, 0.7, -0.1])
    expected = np.array(
        [
            (2 * tangent[0] + tangent[1]) / 3,
            (tangent[0] + tangent[1] + tangent[2]) / 3,
            (tangent[1] + tangent[2] + tangent[3]) / 3,
            (tangent[2] + tangent[3] + tangent[4]) / 3,
            (tangent[3] + 2 * tangent[4]) / 3,
        ]
    )

    function = getattr(ndimage, name)
    if name == "rank_filter":

        def call(value: object) -> object:
            return function(value, extra["rank"], size=3, mode="reflect")

    elif name == "percentile_filter":

        def call(value: object) -> object:
            return function(value, extra["percentile"], size=3, mode="reflect")

    else:

        def call(value: object) -> object:
            return function(value, size=3, mode="reflect")

    _value, directional = ad.jvp(call)(sample, tangents=tangent)

    assert_allclose(directional, expected, rtol=0, atol=1e-15)


def test_nonsymmetric_footprint_preserves_scipy_unsorted_axes_mapping() -> None:
    sample = np.array(
        [
            [[0.2, 1.4, -0.7, 0.9], [2.1, -0.4, 0.6, 1.7]],
            [[-1.1, 0.3, 2.6, -0.2], [0.8, 1.9, -1.5, 0.1]],
            [[1.2, -0.9, 0.5, 2.3], [-0.6, 0.7, 1.1, -1.8]],
        ]
    )
    tangent = np.linspace(-0.5, 0.4, sample.size).reshape(sample.shape)
    cotangent = np.linspace(0.3, -0.2, sample.size).reshape(sample.shape)
    footprint = np.array([[True, False, True], [False, True, False]])
    kwargs = {
        "footprint": footprint,
        "mode": "wrap",
        "axes": (2, 0),
    }

    def function(value: object) -> object:
        return ndimage.maximum_filter(value, **kwargs)

    value, directional = ad.jvp(function)(sample, tangents=tangent)
    expected_value = scipy_ndimage.maximum_filter(sample, **kwargs)
    step = 1e-6
    expected_directional = (
        scipy_ndimage.maximum_filter(sample + step * tangent, **kwargs)
        - scipy_ndimage.maximum_filter(sample - step * tangent, **kwargs)
    ) / (2 * step)
    _value, pullback = ad.vjp(function)(sample)
    input_cotangent = pullback(cotangent)
    program = ad.stage(function, specs=(ad.ArraySpec(sample.shape, sample.dtype),))

    assert_allclose(value, expected_value, rtol=0, atol=0)
    assert_allclose(directional, expected_directional, rtol=2e-9, atol=2e-9)
    assert_allclose(
        np.vdot(directional, cotangent),
        np.vdot(tangent, input_cotangent),
        rtol=2e-12,
        atol=2e-12,
    )
    assert_allclose(program(sample), expected_value, rtol=0, atol=0)


@pytest.mark.parametrize(
    "name",
    ["maximum_filter", "median_filter", "grey_dilation", "grey_erosion"],
)
def test_nonseparable_filters_preserve_full_rank_unsorted_axes_origins(name: str) -> None:
    footprint = np.array([[True, False], [True, True], [False, True]])
    structure = np.array([[0.1, -0.2], [0.3, 0.05], [-0.1, 0.2]])
    kwargs: dict[str, object] = {
        "footprint": footprint,
        "axes": (1, 0),
        "origin": (1, -1),
        "mode": "constant",
        "cval": -4.0,
    }
    if name.startswith("grey_"):
        kwargs["structure"] = structure

    def function(value: object) -> object:
        return getattr(ndimage, name)(value, **kwargs)

    value, directional = ad.jvp(function)(_FIELD, tangents=_TANGENT)
    scipy_function = getattr(scipy_ndimage, name)
    expected_value = scipy_function(_FIELD, **kwargs)
    expected_directional = _directional_difference(
        lambda argument: scipy_function(argument, **kwargs),
        _FIELD,
    )
    _value, pullback = ad.vjp(function)(_FIELD)
    input_cotangent = pullback(_COTANGENT)
    program = ad.stage(function, specs=(ad.ArraySpec(_FIELD.shape, _FIELD.dtype),))

    assert_array_equal(value, expected_value)
    assert_allclose(directional, expected_directional, rtol=2e-9, atol=2e-9)
    assert_allclose(
        np.vdot(directional, _COTANGENT),
        np.vdot(_TANGENT, input_cotangent),
        rtol=2e-12,
        atol=2e-12,
    )
    assert_array_equal(program(_FIELD), expected_value)


@pytest.mark.parametrize(
    "structure",
    [None, np.array([[0.2, -0.1, 0.3], [0.4, 0.05, -0.2]])],
)
def test_grey_dilation_preserves_unsorted_axes_mixed_parity_mapping(
    structure: np.ndarray | None,
) -> None:
    sample = np.sin(1.7 * np.arange(24)).reshape(3, 4, 2)
    tangent = np.linspace(-0.5, 0.4, sample.size).reshape(sample.shape)
    cotangent = np.linspace(0.3, -0.2, sample.size).reshape(sample.shape)
    kwargs = {
        "footprint": np.array([[True, False, True], [True, True, False]]),
        "structure": structure,
        "mode": "nearest",
        "origin": (0, 0),
        "axes": (2, 0),
    }

    def function(value: object) -> object:
        return ndimage.grey_dilation(value, **kwargs)

    value, directional = ad.jvp(function)(sample, tangents=tangent)
    expected_value = scipy_ndimage.grey_dilation(sample, **kwargs)
    step = 1e-6
    expected_directional = (
        scipy_ndimage.grey_dilation(sample + step * tangent, **kwargs)
        - scipy_ndimage.grey_dilation(sample - step * tangent, **kwargs)
    ) / (2 * step)
    _value, pullback = ad.vjp(function)(sample)
    input_cotangent = pullback(cotangent)
    program = ad.stage(function, specs=(ad.ArraySpec(sample.shape, sample.dtype),))

    assert_array_equal(value, expected_value)
    assert_allclose(directional, expected_directional, rtol=2e-9, atol=2e-9)
    assert_allclose(
        np.vdot(directional, cotangent),
        np.vdot(tangent, input_cotangent),
        rtol=2e-12,
        atol=2e-12,
    )
    assert_array_equal(program(sample), expected_value)


@pytest.mark.parametrize(
    ("name", "cval"),
    [("grey_dilation", 5.0), ("grey_erosion", -5.0)],
)
def test_grey_structure_and_cval_are_live_differentiable_operands(
    name: str,
    cval: float,
) -> None:
    sample = np.array([0.2, -0.4, 1.1, 0.3])
    structure = np.array([0.15, -0.2, 0.4])
    input_tangent = np.array([0.2, -0.3, 0.5, 0.7])
    structure_tangent = np.array([-0.1, 0.4, -0.2])
    cval_tangent = np.array(0.6)
    cotangent = np.array([-0.7, 0.2, 0.5, -0.3])
    boundary = np.array(cval)

    def function(value: object, offsets: object, boundary_value: object) -> object:
        return getattr(ndimage, name)(
            value,
            structure=offsets,
            mode="constant",
            cval=boundary_value,
        )

    value, directional = ad.jvp(function, (0, 1, 2))(
        sample,
        structure,
        boundary,
        tangents=(input_tangent, structure_tangent, cval_tangent),
    )
    step = 1e-6
    expected_directional = (
        function(
            sample + step * input_tangent,
            structure + step * structure_tangent,
            boundary + step * cval_tangent,
        )
        - function(
            sample - step * input_tangent,
            structure - step * structure_tangent,
            boundary - step * cval_tangent,
        )
    ) / (2 * step)
    _value, pullback = ad.vjp(function, (0, 1, 2))(sample, structure, boundary)
    input_cotangent, structure_cotangent, cval_cotangent = pullback(cotangent)

    assert_allclose(value, function(sample, structure, boundary), rtol=0, atol=0)
    assert_allclose(directional, expected_directional, rtol=2e-9, atol=2e-9)
    assert_allclose(
        np.vdot(directional, cotangent),
        np.vdot(input_tangent, input_cotangent)
        + np.vdot(structure_tangent, structure_cotangent)
        + np.vdot(cval_tangent, cval_cotangent),
        rtol=2e-12,
        atol=2e-12,
    )

    specs = (
        ad.ArraySpec(sample.shape, sample.dtype),
        ad.ArraySpec(structure.shape, structure.dtype),
        ad.ArraySpec(boundary.shape, boundary.dtype),
    )
    program = ad.stage(function, specs=specs)
    pullback_program = ad.vjp_program(program, argnums=(0, 1, 2))
    restored = ad.StagedProgram.from_dict(pullback_program.to_dict())
    staged_cotangents = restored(sample, structure, boundary, cotangent=cotangent)

    assert_allclose(staged_cotangents[0], input_cotangent, rtol=2e-12, atol=2e-12)
    assert_allclose(staged_cotangents[1], structure_cotangent, rtol=2e-12, atol=2e-12)
    assert_allclose(staged_cotangents[2], cval_cotangent, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize(
    ("name", "sample", "cval", "expected"),
    [
        ("maximum_filter1d", np.array([1.0, 2.0, 3.0]), 10.0, np.array([0.0, 0.5, 0.0])),
        ("minimum_filter1d", np.array([1.0, 2.0, 3.0]), -10.0, np.array([0.0, 0.2, 0.0])),
    ],
)
def test_constant_padding_winners_have_no_input_gradient(
    name: str,
    sample: np.ndarray,
    cval: float,
    expected: np.ndarray,
) -> None:
    tangent = np.array([0.2, -0.3, 0.5])

    def function(value: object) -> object:
        return getattr(ndimage, name)(
            value,
            3,
            mode="constant",
            cval=cval,
        )

    _value, directional = ad.jvp(function)(sample, tangents=tangent)

    assert_allclose(directional, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "name",
    ["gaussian_filter", "convolve", "maximum_filter", "grey_opening", "median_filter"],
)
def test_traced_output_destination_is_replaced_and_serializes(name: str) -> None:
    def function(value: object) -> object:
        destination = (3 * value).copy()
        result = _call(name, getattr(ndimage, name), value, output=destination)
        assert result is destination
        return result

    expected = _call(name, getattr(scipy_ndimage, name), _FIELD)
    value, tangent = ad.jvp(function)(_FIELD, tangents=_TANGENT)
    expected_tangent = _directional_difference(
        lambda argument: _call(name, getattr(scipy_ndimage, name), argument),
        _FIELD,
    )
    program = ad.stage(function, specs=(ad.ArraySpec(_FIELD.shape, _FIELD.dtype),))
    artifact = program.to_dict()
    restored = ad.StagedProgram.from_dict(artifact)
    nodes = artifact["program"]["graph"]["nodes"]
    operations = {node["op"] for node in nodes}

    assert_allclose(value, expected, rtol=1e-13, atol=1e-13)
    assert_allclose(tangent, expected_tangent, rtol=2e-7, atol=2e-7)
    assert_allclose(restored(_FIELD), expected, rtol=1e-13, atol=1e-13)
    assert "array.multiply" not in operations
    assert "advect.copy" not in operations
    assert all("has_destination" not in node["attrs"] for node in nodes)


def test_selection_artifacts_store_only_source_configuration() -> None:
    program = ad.stage(
        lambda value: ndimage.grey_dilation(
            value,
            size=(2, 3),
            axes=(1, 0),
            origin=(0, -1),
        ),
        specs=(ad.ArraySpec(_FIELD.shape, _FIELD.dtype),),
    )
    node = next(
        node
        for node in program.to_dict()["program"]["graph"]["nodes"]
        if node["op"] == "custom.scipy.ndimage.grey_dilation"
    )

    assert "has_footprint" not in node["attrs"]
    assert not any(name.startswith("neighborhood_") for name in node["attrs"])


@pytest.mark.parametrize("output_dtype", [np.float32, np.int16])
def test_output_dtype_is_part_of_the_differentiated_gaussian_program(
    output_dtype: object,
) -> None:
    sample = np.asarray(_FIELD, dtype=np.float64)
    tangent = np.asarray(_TANGENT, dtype=np.float64)

    def function(value: object) -> object:
        return ndimage.gaussian_filter(
            value,
            0.8,
            mode="nearest",
            output=output_dtype,
        )

    value, directional = ad.jvp(function)(sample, tangents=tangent)
    expected_value = scipy_ndimage.gaussian_filter(
        sample,
        0.8,
        mode="nearest",
        output=output_dtype,
    )

    assert np.asarray(value).dtype == np.dtype(output_dtype)
    assert np.asarray(directional).dtype == np.dtype(output_dtype)
    assert_array_equal(value, expected_value)
    if np.issubdtype(np.dtype(output_dtype), np.integer):
        assert_array_equal(directional, np.zeros_like(expected_value))
    else:
        expected_directional = scipy_ndimage.gaussian_filter(
            tangent,
            0.8,
            mode="nearest",
            output=output_dtype,
        )
        assert_allclose(directional, expected_directional, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(
    "name",
    [
        "grey_opening",
        "grey_closing",
        "morphological_gradient",
        "morphological_laplace",
        "white_tophat",
        "black_tophat",
    ],
)
def test_composite_morphology_preserves_scipy_intermediate_output_dtypes(name: str) -> None:
    sample = np.array(
        [
            -66610.03901687,
            -181389.49037109,
            -132165.39369439,
            -160788.31134167,
            107508.95143259,
            -78080.30865503,
            -137997.82632813,
            -35757.50663888,
            -10455.47961271,
        ]
    )
    structure = np.array([-0.01019979, 0.0165294, -0.00897683])

    def function(value: object) -> object:
        destination = np.zeros_like(value, dtype=np.float32)
        return getattr(ndimage, name)(
            value,
            structure=structure,
            output=destination,
            mode="reflect",
        )

    value, _tangent = ad.jvp(function)(sample, tangents=np.ones_like(sample))
    expected = getattr(scipy_ndimage, name)(
        sample,
        structure=structure,
        output=np.empty_like(sample, dtype=np.float32),
        mode="reflect",
    )
    typed, _typed_tangent = ad.jvp(
        lambda value: getattr(ndimage, name)(
            value,
            structure=structure,
            output=np.float32,
            mode="reflect",
        )
    )(sample, tangents=np.ones_like(sample))
    expected_typed = getattr(scipy_ndimage, name)(
        sample,
        structure=structure,
        output=np.float32,
        mode="reflect",
    )

    assert_array_equal(value, expected)
    assert np.asarray(typed).dtype == np.asarray(expected_typed).dtype
    assert_array_equal(typed, expected_typed)


@pytest.mark.parametrize(
    "name",
    [
        "morphological_gradient",
        "morphological_laplace",
        "white_tophat",
        "black_tophat",
    ],
)
def test_composite_morphology_preserves_scipy_inplace_cast_errors(name: str) -> None:
    sample = np.array([0.0, 1.0, 3.0, 2.0])

    def function(value: object) -> object:
        destination = np.zeros_like(value, dtype=np.int16)
        return getattr(ndimage, name)(value, size=3, output=destination)

    with pytest.raises(TypeError, match="Cannot cast ufunc"):
        ad.jvp(function)(sample, tangents=np.ones_like(sample))


@pytest.mark.parametrize("name", ["white_tophat", "black_tophat"])
def test_tophat_dtype_specification_preserves_scipy_inplace_cast_errors(name: str) -> None:
    sample = np.array([0.0, 1.0, 3.0, 2.0])

    with pytest.raises(TypeError, match="Cannot cast ufunc"):
        ad.jvp(lambda value: getattr(ndimage, name)(value, size=3, output=np.int16))(
            sample,
            tangents=np.ones_like(sample),
        )


def test_gaussian_scalar_and_empty_axes_are_not_special_case_footguns() -> None:
    scalar = np.array(2.0)
    field = np.array(_FIELD, copy=True)

    assert_array_equal(ndimage.gaussian_filter(scalar, 1.2), scalar)
    assert_array_equal(ndimage.gaussian_filter(field, (), axes=()), field)

    program = ad.stage(
        lambda value: ndimage.gaussian_filter(value, (), axes=()),
        specs=(ad.ArraySpec(field.shape, field.dtype),),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())

    assert_array_equal(restored(field), field)
