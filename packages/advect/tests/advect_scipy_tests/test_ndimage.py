"""End-to-end qualification for the bounded :mod:`scipy.ndimage` frontend."""

from __future__ import annotations

import inspect
import math
import operator
import re
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

import array_api_strict as strict
import numpy as np
import pytest
from hypothesis import example, given, settings, strategies as st
from hypothesis.extra import numpy as hnp
from numpy.testing import assert_allclose, assert_array_equal
from scipy import ndimage as scipy_ndimage

import advect as ad
from advect.scipy import ndimage

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


_PUBLIC_NAMES = tuple(ndimage.__all__)

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
_GREY_FORM = {
    "footprint": _FOOTPRINT,
    "structure": _STRUCTURE,
    "mode": "reflect",
    "cval": -3.0,
    "origin": 0,
}
#: One representative, nontrivial ``(args, kwargs)`` form of every public function.
_FORMS: dict[str, tuple[tuple[object, ...], dict[str, object]]] = {
    "gaussian_filter": (
        (),
        {
            "sigma": (0.7, 1.1),
            "order": (0, 1),
            "mode": ("reflect", "nearest"),
            "cval": 1.3,
            "radius": (2, 2),
        },
    ),
    "gaussian_filter1d": (
        (0.8,),
        {"axis": 1, "order": 2, "mode": "mirror", "cval": -0.7, "radius": 2},
    ),
    "uniform_filter": (
        (),
        {"size": (2, 3), "mode": ("nearest", "wrap"), "cval": 0.9, "origin": (0, -1)},
    ),
    "uniform_filter1d": ((3,), {"axis": 0, "mode": "constant", "cval": 0.9, "origin": 1}),
    "convolve": ((_WEIGHTS_2D,), {"mode": "nearest", "cval": -0.6, "origin": (0, -1)}),
    "correlate": ((_WEIGHTS_2D,), {"mode": "nearest", "cval": -0.6, "origin": (0, -1)}),
    "convolve1d": ((_WEIGHTS_1D,), {"axis": 1, "mode": "constant", "cval": 0.8, "origin": 1}),
    "correlate1d": ((_WEIGHTS_1D,), {"axis": 1, "mode": "constant", "cval": 0.8, "origin": -1}),
    "laplace": ((), {"mode": "nearest", "cval": 0.4, "axes": (1,)}),
    "gaussian_laplace": ((0.9,), {"mode": "mirror", "cval": -0.4, "axes": (1,), "radius": 2}),
    "sobel": ((), {"axis": 0, "mode": "mirror", "cval": 0.6}),
    "prewitt": ((), {"axis": 1, "mode": "wrap", "cval": 0.6}),
    **dict.fromkeys(
        ("maximum_filter", "minimum_filter"),
        ((), {"size": (3, 2), "mode": ("nearest", "wrap"), "cval": -7.0, "origin": (1, -1)}),
    ),
    **dict.fromkeys(
        ("maximum_filter1d", "minimum_filter1d"),
        ((3,), {"axis": 1, "mode": "mirror", "cval": -7.0, "origin": 1}),
    ),
    **dict.fromkeys(
        (
            "grey_dilation",
            "grey_erosion",
            "grey_opening",
            "grey_closing",
            "morphological_gradient",
            "morphological_laplace",
            "white_tophat",
            "black_tophat",
        ),
        ((), _GREY_FORM),
    ),
    "median_filter": ((), {"footprint": _FOOTPRINT, "mode": "mirror", "cval": 0.7, "origin": 0}),
    "rank_filter": ((2,), {"footprint": _FOOTPRINT, "mode": "nearest", "cval": -0.7, "origin": 0}),
    "percentile_filter": (
        (65.0,),
        {"footprint": _FOOTPRINT, "mode": "wrap", "cval": 0.7, "origin": 0},
    ),
}


def _call(name: str, module: object, value: object, **output: object) -> object:
    """Invoke ``module``'s ``name`` in its ``_FORMS`` form, plus any ``output=``."""
    args, kwargs = _FORMS[name]
    return getattr(module, name)(value, *args, **kwargs, **output)


def _parameter_contract(function: Callable[..., object]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (parameter.name, parameter.kind, parameter.default)
        for parameter in inspect.signature(function).parameters.values()
    )


def test_public_ndimage_inventory_has_unique_names_with_one_form_each() -> None:
    assert len(_PUBLIC_NAMES) == len(set(_PUBLIC_NAMES))
    assert sorted(_FORMS) == sorted(_PUBLIC_NAMES)


@pytest.mark.parametrize("name", _PUBLIC_NAMES)
def test_public_ndimage_signatures_match_scipy(name: str) -> None:
    actual = getattr(ndimage, name)
    expected = getattr(scipy_ndimage, name)

    assert _parameter_contract(actual) == _parameter_contract(expected)


@pytest.mark.parametrize("name", _PUBLIC_NAMES)
def test_public_ndimage_functions_match_scipy(name: str) -> None:
    sample = _FIELD.copy()

    actual = _call(name, ndimage, sample)
    expected = _call(name, scipy_ndimage, sample)

    assert np.asarray(actual).dtype == np.asarray(expected).dtype
    assert_allclose(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("name", _PUBLIC_NAMES)
def test_concrete_ndimage_functions_reject_non_numpy_providers(name: str) -> None:
    with pytest.raises(TypeError, match=rf"ndimage\.{name} supports NumPy arrays only"):
        _call(name, ndimage, strict.asarray(_FIELD))


_RANK_ERROR = "sequence argument must have length equal to input rank"
_AXES_ERROR = "axes must be an integer, iterable of integers, or None"


@pytest.mark.parametrize(
    ("function", "extra_specs", "error", "match"),
    [
        (lambda x: ndimage.maximum_filter(x, size=None), (), RuntimeError, None),
        (
            lambda x: ndimage.maximum_filter(x, footprint=np.ones((3, 3), bool), axes=(0,)),
            (),
            RuntimeError,
            _RANK_ERROR,
        ),
        (lambda x: ndimage.maximum_filter(x, size=(3,)), (), RuntimeError, _RANK_ERROR),
        (lambda x: ndimage.maximum_filter(x, size=3, axes=(0, "bad")), (), ValueError, _AXES_ERROR),
        (lambda x: ndimage.maximum_filter(x, size=3, axes=object()), (), ValueError, _AXES_ERROR),
        (lambda x: ndimage.maximum_filter(x, size=3, axes=(2,)), (), ValueError, "out of range"),
        (lambda x: ndimage.maximum_filter(x, size=3, axes=(0, 0)), (), ValueError, "unique"),
        (lambda x: ndimage.maximum_filter1d(x, 3, axis=2), (), np.exceptions.AxisError, None),
        (
            lambda x: ndimage.maximum_filter(x, size=3, output=object()),
            (),
            TypeError,
            "NumPy dtype specifications only",
        ),
        (
            lambda x: ndimage.maximum_filter(x, size=3, output=np.empty(3)),
            (),
            RuntimeError,
            "output shape not correct",
        ),
        (
            lambda x: ndimage.maximum_filter(x, size=3, output=np.empty_like(_FIELD)),
            (),
            TypeError,
            r"output=.*owned traced array",
        ),
        (
            lambda x, origin: ndimage.maximum_filter(x, size=3, origin=origin),
            (ad.ArraySpec((), np.dtype(np.int64)),),
            TypeError,
            "configuration arguments must be concrete",
        ),
        (
            lambda x, footprint: ndimage.maximum_filter(x, footprint=footprint),
            (ad.ArraySpec((3, 3), np.dtype(bool)),),
            TypeError,
            r"footprint.*must be concrete",
        ),
    ],
    ids=[
        "no-size",
        "footprint-rank",
        "size-rank",
        "text-axis",
        "object-axes",
        "axis-range",
        "repeated-axis",
        "axis-1d-range",
        "object-output",
        "output-shape",
        "concrete-output",
        "traced-origin",
        "traced-footprint",
    ],
)
def test_staging_rejects_invalid_or_dynamic_configuration(
    function: Callable[..., object],
    extra_specs: tuple[ad.ArraySpec, ...],
    error: type[Exception],
    match: str | None,
) -> None:
    with pytest.raises(error, match=match):
        ad.stage(function, specs=(ad.ArraySpec(_FIELD.shape, _FIELD.dtype), *extra_specs))


def test_staging_accepts_scalar_axes_and_a_numpy_scalar_origin() -> None:
    def function(value: object) -> object:
        return ndimage.maximum_filter(value, size=3, axes=1, origin=np.int64(0))

    assert_array_equal(
        ad.stage(function, _FIELD)(_FIELD),
        scipy_ndimage.maximum_filter(_FIELD, size=3, axes=1, origin=np.int64(0)),
    )


#: ``output=`` kinds: dtype specifications, then owned destinations derived from
#: the input, whose prior values and derivatives must not leak into the result.
_OUTPUTS = (
    (np.float32, False),
    (np.int16, False),
    (np.float64, True),
    (np.float32, True),
    (np.int16, True),
)


def _write(name: str, value: object, dtype: type, *, owned: bool) -> object:
    if not owned:
        return _call(name, ndimage, value, output=dtype)
    destination = (3 * value).astype(dtype)
    result = _call(name, ndimage, value, output=destination)
    assert result is destination
    return result


@pytest.mark.parametrize("name", _PUBLIC_NAMES)
def test_every_ndimage_function_writes_output_dtypes_and_arrays(name: str) -> None:
    expected: dict[tuple[type, bool], np.ndarray] = {}
    for dtype, owned in _OUTPUTS:
        try:
            expected[dtype, owned] = _call(
                name,
                scipy_ndimage,
                _FIELD,
                output=np.zeros(_FIELD.shape, dtype) if owned else dtype,
            )
        except TypeError as error:
            # SciPy's in-place composite casts reject some integer outputs.
            write = partial(_write, name, dtype=dtype, owned=owned)
            paths = (write, partial(ad.jvp(write), tangents=_TANGENT), partial(ad.stage, write))
            for path in paths:
                with pytest.raises(type(error), match=re.escape(str(error))):
                    path(_FIELD)

    def function(value: object) -> tuple[object, ...]:
        writes = (_write(name, value, dtype, owned=owned) for dtype, owned in expected)
        return (_call(name, ndimage, value), *writes)

    (_value, *values), (reference, *tangents) = ad.jvp(function)(_FIELD, tangents=_TANGENT)
    staged = ad.stage(function, _FIELD)(_FIELD)[1:]
    concrete = function(_FIELD)[1:]

    for desired, value, tangent, staged_value, concrete_value in zip(
        expected.values(), values, tangents, staged, concrete, strict=True
    ):
        for actual in (value, staged_value, concrete_value):
            assert actual.dtype == desired.dtype
            assert_array_equal(actual, desired)
        assert tangent.dtype == desired.dtype
        if np.issubdtype(desired.dtype, np.inexact):
            tolerance = 16 * np.finfo(desired.dtype).eps
            assert_allclose(tangent, reference, rtol=tolerance, atol=tolerance)
        else:
            assert not np.any(tangent)


def _form(name: str, *args: object, **kwargs: object) -> Callable[..., object]:
    """Call ``module``'s ``name`` on one differentiated input with fixed arguments."""
    return lambda module, value: getattr(module, name)(value, *args, **kwargs)


def _probe(rng: np.random.Generator, like: object) -> np.ndarray:
    values = rng.uniform(-0.5, 0.5, (2, *np.shape(like)))
    return values[0] + 1j * values[1] if np.iscomplexobj(like) else values[0]


#: Distinct within 0.006, so a finite-difference step never crosses a kink.
_VOLUME = np.sin(1.7 * np.arange(24)).reshape(3, 4, 2)
_NONSEPARABLE = {
    "footprint": np.array([[True, False], [True, True], [False, True]]),
    "axes": (1, 0),
    "origin": (1, -1),
    "mode": "constant",
    "cval": -4.0,
}
_MIXED_PARITY = {
    "footprint": np.array([[True, False, True], [True, True, False]]),
    "mode": "nearest",
    "origin": (0, 0),
    "axes": (2, 0),
}
#: ``(form, primals)`` rows whose every primal is differentiated. A form takes
#: the module to call (Advect's or SciPy's reference) and then the primals.
_DERIVATIVE_FORMS = (
    *(pytest.param(partial(_call, name), (_FIELD,), id=name) for name in _PUBLIC_NAMES),
    # SciPy maps full-rank unsorted axes, origins, and per-axis sequences in
    # call order, and partial unsorted axes and mixed-parity windows by axis.
    *(
        pytest.param(
            _form(name, _WEIGHTS_2D, axes=(1, 0), origin=(-1, 0), mode="constant", cval=0.7),
            (_FIELD,),
            id=f"{name}-unsorted-axes",
        )
        for name in ("convolve", "correlate")
    ),
    pytest.param(
        _form("gaussian_laplace", (0.7, 1.1), axes=(1, 0), mode=("nearest", "wrap"), radius=(2, 3)),
        (_FIELD,),
        id="gaussian_laplace-unsorted-axes",
    ),
    pytest.param(
        _form(
            "maximum_filter",
            footprint=np.array([[True, False, True], [False, True, False]]),
            mode="wrap",
            axes=(2, 0),
        ),
        (_VOLUME,),
        id="maximum_filter-nonsymmetric-footprint",
    ),
    *(
        pytest.param(_form(name, **_NONSEPARABLE), (_FIELD,), id=f"{name}-nonseparable")
        for name in ("maximum_filter", "median_filter")
    ),
    *(
        pytest.param(
            _form(
                name, structure=np.array([[0.1, -0.2], [0.3, 0.05], [-0.1, 0.2]]), **_NONSEPARABLE
            ),
            (_FIELD,),
            id=f"{name}-nonseparable",
        )
        for name in ("grey_dilation", "grey_erosion")
    ),
    *(
        pytest.param(
            _form("grey_dilation", structure=structure, **_MIXED_PARITY), (_VOLUME,), id=row_id
        )
        for structure, row_id in (
            (None, "grey_dilation-mixed-parity"),
            (
                np.array([[0.2, -0.1, 0.3], [0.4, 0.05, -0.2]]),
                "grey_dilation-mixed-parity-structure",
            ),
        )
    ),
    # Weights, cval, and structure are live, serializable operands.
    *(
        pytest.param(
            lambda module, x, w, name=name: getattr(module, name)(
                x, w, mode="constant", cval=0.8, origin=(0, -1)
            ),
            (_FIELD, _WEIGHTS_2D),
            id=f"{name}-weights",
        )
        for name in ("convolve", "correlate")
    ),
    *(
        pytest.param(
            lambda module, x, w, name=name: getattr(module, name)(
                x, w, axis=1, mode="constant", cval=0.8, origin=1
            ),
            (_FIELD, _WEIGHTS_1D),
            id=f"{name}-weights",
        )
        for name in ("convolve1d", "correlate1d")
    ),
    pytest.param(
        lambda module, x, c: module.gaussian_filter(
            x, (0.8, 1.1), mode="constant", cval=c, radius=(2, 3)
        ),
        (_FIELD, np.array(0.7)),
        id="gaussian_filter-cval",
    ),
    pytest.param(
        lambda module, x, c: module.uniform_filter(
            x, (2, 3), mode="constant", cval=c, origin=(0, -1)
        ),
        (_FIELD, np.array(0.7)),
        id="uniform_filter-cval",
    ),
    pytest.param(
        lambda module, x, c: module.correlate(
            x, _WEIGHTS_2D, mode="constant", cval=c, origin=(0, -1)
        ),
        (_FIELD, np.array(0.7)),
        id="correlate-cval",
    ),
    pytest.param(
        lambda module, x, w, c: module.correlate(x, w, mode="constant", cval=c, origin=(0, -1)),
        (
            _FIELD + 1j * np.flip(_FIELD, axis=1),
            _WEIGHTS_2D + 1j * np.array([[0.11, -0.07], [0.23, -0.13]]),
            np.array(0.7 - 0.2j),
        ),
        id="correlate-complex",
    ),
    *(
        pytest.param(
            lambda module, x, s, c, name=name: getattr(module, name)(
                x, structure=s, mode="constant", cval=c
            ),
            (np.array([0.2, -0.4, 1.1, 0.3]), np.array([0.15, -0.2, 0.4]), np.array(cval)),
            id=f"{name}-structure-cval",
        )
        for name, cval in (("grey_dilation", 5.0), ("grey_erosion", -5.0))
    ),
)


@pytest.mark.parametrize(("form", "primals"), _DERIVATIVE_FORMS)
def test_ndimage_forms_differentiate_stage_and_serialize(
    form: Callable[..., object],
    primals: tuple[np.ndarray, ...],
) -> None:
    function = partial(form, ndimage)
    reference = partial(form, scipy_ndimage)
    argnums = tuple(range(len(primals)))
    rng = np.random.default_rng(0)
    tangents = tuple(_probe(rng, primal) for primal in primals)
    expected = reference(*primals)
    cotangent = _probe(rng, expected)
    step = 1e-6
    expected_directional = (
        reference(*(p + step * t for p, t in zip(primals, tangents, strict=True)))
        - reference(*(p - step * t for p, t in zip(primals, tangents, strict=True)))
    ) / (2 * step)

    value, directional = ad.jvp(function, argnums)(*primals, tangents=tangents)
    cotangents = ad.vjp(function, argnums)(*primals)[1](cotangent)
    program = ad.stage(function, *primals)
    pullback = ad.vjp_program(program, argnums)

    assert_array_equal(value, expected)
    assert_allclose(directional, expected_directional, rtol=2e-9, atol=2e-9)
    assert_allclose(
        np.real(np.vdot(directional, cotangent)),
        sum(np.real(np.vdot(t, c)) for t, c in zip(tangents, cotangents, strict=True)),
        rtol=2e-12,
        atol=2e-12,
    )
    for staged in (program, ad.StagedProgram.from_dict(program.to_dict())):
        assert_array_equal(staged(*primals), expected)
    for staged in (pullback, ad.StagedProgram.from_dict(pullback.to_dict())):
        for actual, desired in zip(staged(*primals, cotangent=cotangent), cotangents, strict=True):
            assert_allclose(actual, desired, rtol=2e-12, atol=2e-12)


_MODES = (
    "reflect",
    "constant",
    "nearest",
    "mirror",
    "wrap",
    "grid-mirror",
    "grid-constant",
    "grid-wrap",
)
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
type _LinearCase = tuple[str, np.ndarray, tuple[object, ...], dict[str, object], float]


@st.composite
def _linear_filters(draw: st.DrawFn) -> _LinearCase:
    """Draw a linear filter call: name, input, arguments, keywords, and cval."""
    name = draw(st.sampled_from(_LINEAR_NAMES))
    # Kernels may outgrow these axes, so boundary modes extend repeatedly.
    shapes = hnp.array_shapes(min_dims=1, max_dims=3, max_side=4)
    sample = draw(hnp.arrays(np.float64, shapes, elements=st.floats(-2, 2)))
    ndim = sample.ndim
    if name.endswith("1d") or name in {"sobel", "prewitt"}:
        kwargs: dict[str, object] = {"axis": draw(st.integers(-ndim, ndim - 1))}
        count = 1 if name.endswith("1d") else ndim
    else:
        order = draw(st.permutations(range(ndim)))
        axes = draw(st.none() | st.integers(1, ndim).map(lambda size: tuple(order[:size])))
        kwargs = {"axes": axes}
        count = ndim if axes is None else len(axes)

    def per_axis(strategy: st.SearchStrategy[object]) -> object:
        return draw(strategy | st.lists(strategy, min_size=count, max_size=count).map(tuple))

    def origin(size: int) -> int:
        return draw(st.integers(-(size // 2), (size - 1) // 2))

    mode, sigma, radius = (
        st.sampled_from(_MODES),
        st.floats(0.5, 1.5),
        st.none() | st.integers(1, 3),
    )
    args: tuple[object, ...] = ()
    if name in {"convolve", "correlate", "convolve1d", "correlate1d"}:
        weight_shapes = hnp.array_shapes(min_dims=count, max_dims=count, max_side=4)
        weights = draw(hnp.arrays(np.float64, weight_shapes, elements=st.floats(-1, 1)))
        # Unsorted axes map origins and weight axes differently in SciPy; an
        # origin valid for the smallest side is valid under either mapping.
        origins = tuple(origin(min(weights.shape)) for _ in weights.shape)
        args = (weights,)
        kwargs |= {"mode": draw(mode), "origin": origins[0] if count == 1 else origins}
    elif name.startswith("uniform"):
        sizes = draw(st.lists(st.integers(1, 4), min_size=count, max_size=count))
        origins = tuple(origin(size) for size in sizes)
        if name.endswith("1d"):
            args, kwargs["origin"], kwargs["mode"] = (sizes[0],), origins[0], draw(mode)
        else:
            kwargs |= {"size": tuple(sizes), "origin": origins, "mode": per_axis(mode)}
    elif name == "gaussian_filter1d":
        args = (draw(sigma),)
        kwargs |= {"order": draw(st.integers(0, 2)), "mode": draw(mode), "radius": draw(radius)}
    elif name == "gaussian_filter":
        args = (per_axis(sigma),)
        kwargs |= {"order": per_axis(st.integers(0, 2)), "mode": per_axis(mode)}
        kwargs["radius"] = per_axis(radius)
    elif name == "gaussian_laplace":
        # SciPy forwards radius to a full-rank gaussian_filter, so it stays scalar.
        args = (per_axis(sigma),)
        kwargs |= {"mode": per_axis(mode), "radius": draw(radius)}
    else:
        kwargs["mode"] = per_axis(mode)
    return name, sample, args, kwargs, draw(st.floats(-2, 2))


@settings(max_examples=max(10, settings.default.max_examples // 8), deadline=None)
@given(case=_linear_filters())
@example(
    (
        "gaussian_filter",
        np.arange(24.0).reshape(2, 3, 4) / 7,
        ((0.8, 1.1),),
        {"order": (1, 2), "mode": ("nearest", "mirror"), "radius": (2, 3), "axes": (2, 0)},
        1.7,
    )
)
# The fixed derivative forms cover the other boundary modes; pin SciPy's aliases.
@example(("gaussian_filter1d", np.linspace(-1, 2, 5), (0.9,), {"mode": "grid-mirror"}, 1.7))
@example(("gaussian_filter1d", np.linspace(-1, 2, 5), (0.9,), {"mode": "grid-wrap"}, 1.7))
@example(("gaussian_filter1d", np.linspace(-1, 2, 5), (0.9,), {"mode": "grid-constant"}, 1.7))
def test_linear_filters_are_exact_affine_maps_of_input_and_cval(case: _LinearCase) -> None:
    # Each filter is L0(x) + cval * offset, where L0 is SciPy's call with a zero
    # cval, so the JVP, the dense Jacobian's transpose, and the cval cotangent
    # all have exact SciPy references.
    name, sample, args, kwargs, cval = case

    def call(module: object, value: object, boundary: object) -> object:
        return getattr(module, name)(value, *args, cval=boundary, **kwargs)

    reference = partial(call, scipy_ndimage)
    rng = np.random.default_rng(0)
    tangent, cotangent = rng.normal(size=(2, *sample.shape))
    cval_tangent = rng.normal()
    offset = reference(np.zeros_like(sample), 1.0)
    basis = np.eye(sample.size).reshape(-1, *sample.shape)
    jacobian = np.stack([reference(unit, 0.0).ravel() for unit in basis], axis=1)
    operands = (sample, np.asarray(cval))

    value, directional = ad.jvp(partial(call, ndimage), (0, 1))(
        *operands, tangents=(tangent, np.asarray(cval_tangent))
    )
    input_cotangent, cval_cotangent = ad.vjp(partial(call, ndimage), (0, 1))(*operands)[1](
        cotangent
    )

    tolerance = {"rtol": 2e-13, "atol": 2e-13}
    assert_allclose(value, reference(sample, cval), **tolerance)
    assert_allclose(directional, reference(tangent, 0.0) + cval_tangent * offset, **tolerance)
    assert_allclose(input_cotangent.ravel(), jacobian.T @ cotangent.ravel(), **tolerance)
    assert_allclose(cval_cotangent, np.vdot(offset, cotangent), **tolerance)


@pytest.mark.parametrize("name", ["correlate1d", "convolve1d"])
@pytest.mark.parametrize("inner", [0, 1, 2], ids=("input", "weights", "cval"))
def test_constant_mode_filters_differentiate_reverse_over_reverse(name: str, inner: int) -> None:
    # A constant-mode pullback once returned a symbolic zero for an operand it
    # did not reach, and reverse-over-reverse then lost the other blocks.
    operands = (np.linspace(-1.2, 2.1, 6), _WEIGHTS_1D, np.asarray(0.7))

    def loss(value: object, weights: object, cval: object) -> object:
        filtered = getattr(ndimage, name)(value, weights, mode="constant", cval=cval)
        return np.sum(np.sin(filtered) ** 2)

    hessian = ad.hessian(loss, argnums=(0, 1, 2))(*operands)
    direction = np.linspace(-1.0, 1.0, operands[inner].size).reshape(operands[inner].shape)

    def directional_gradient(*values: object) -> object:
        return np.sum(ad.grad(loss, argnums=inner)(*values) * direction)

    for outer in range(3):
        actual = ad.grad(directional_gradient, argnums=outer)(*operands)
        expected = np.tensordot(direction, hessian[inner][outer], axes=direction.ndim)
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


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


def test_selection_derivatives_stay_adjoint_when_scipy_rounds_the_result() -> None:
    # SciPy returns [2, 1] here (float input gives [2.5, 1.5]), so no window
    # slot reproduces the result; both derivatives must reselect alike.
    sample = np.array([2, 1])
    structure = np.array([-0.5, -0.5])
    tangent = np.array([0.3, -0.7])
    cotangent = np.array([1.1, 0.4])

    def function(value: object) -> object:
        return ndimage.grey_erosion(
            sample, footprint=np.ones(2, dtype=bool), structure=value, output=np.float64
        )

    _value, directional = ad.jvp(function)(structure, tangents=tangent)
    _value, pullback = ad.vjp(function)(structure)

    assert_allclose(directional, [-(tangent[0] + tangent[1]) / 2, -tangent[1]], rtol=0, atol=1e-15)
    assert_allclose(
        pullback(cotangent),
        [-cotangent[0] / 2, -cotangent[0] / 2 - cotangent[1]],
        rtol=0,
        atol=1e-15,
    )


def test_selection_derivatives_keep_scipy_winners_beside_nan_outputs() -> None:
    # SciPy returns [2, 2, 0.4, 0.4, nan, -0.2, -0.2]: outputs 2 and 3 skip the
    # NaN, so only output 4 lacks a winning slot and must not reselect the rest.
    sample = np.array([2.0, -2.5, 0.4, np.nan, -0.5, -0.2, -2.0])
    tangent = np.array([0.3, -0.7, 0.5, 0.2, -0.1, 0.9, 0.6])
    cotangent = np.array([1.1, 0.4, -0.3, 0.8, 0.5, -0.6, 0.7])

    def function(value: object) -> object:
        return ndimage.maximum_filter1d(value, 3)

    with np.errstate(invalid="ignore"):
        _value, directional = ad.jvp(function)(sample, tangents=tangent)
    _value, pullback = ad.vjp(function)(sample)

    assert_allclose(
        directional,
        [tangent[0], tangent[0], tangent[2], tangent[2], np.nan, tangent[5], tangent[5]],
        rtol=0,
        atol=1e-15,
    )
    assert_allclose(
        pullback(cotangent),
        [cotangent[0] + cotangent[1], 0, cotangent[2] + cotangent[3], 0, 0, sum(cotangent[5:]), 0],
        rtol=0,
        atol=1e-15,
    )


_SELECTION_NAMES = (
    "maximum_filter",
    "minimum_filter",
    "maximum_filter1d",
    "minimum_filter1d",
    "median_filter",
    "rank_filter",
    "percentile_filter",
    "grey_dilation",
    "grey_erosion",
)


@dataclass(frozen=True)
class _SelectionCase:
    """One public selection call whose window slots SciPy can enumerate."""

    name: str
    sample: np.ndarray
    args: tuple[object, ...] = ()
    kwargs: Mapping[str, object] = field(default_factory=dict)
    cval: float = 0.0
    structure: np.ndarray | None = None

    def call(
        self,
        module: object,
        value: object,
        cval: object,
        structure: object = None,
    ) -> object:
        extra = {} if self.structure is None else {"structure": structure}
        return getattr(module, self.name)(value, *self.args, cval=cval, **self.kwargs, **extra)

    def slot_window(self) -> tuple[np.ndarray, tuple[int, ...]]:
        """Return the footprint and origins of the extremum or rank filter SciPy runs."""
        ndim = self.sample.ndim
        if self.name.endswith("1d"):
            axis = operator.index(self.kwargs.get("axis", -1)) % ndim
            footprint = np.ones(
                [self.args[0] if item == axis else 1 for item in range(ndim)], dtype=bool
            )
            origins = [self.kwargs.get("origin", 0) if item == axis else 0 for item in range(ndim)]
        else:
            if "footprint" in self.kwargs:
                footprint = np.asarray(self.kwargs["footprint"], dtype=bool)
            elif self.structure is not None:
                footprint = np.ones(self.structure.shape, dtype=bool)
            else:
                footprint = np.ones(np.broadcast_to(self.kwargs["size"], (ndim,)), dtype=bool)
            origins = np.broadcast_to(self.kwargs.get("origin", 0), (ndim,)).tolist()
        if self.name == "grey_dilation":
            # SciPy dilates with the mirrored footprint; even extents shift one slot.
            origins = [
                -origin - (1 - size % 2)
                for origin, size in zip(origins, footprint.shape, strict=True)
            ]
            footprint = np.flip(footprint)
        return footprint, tuple(int(origin) for origin in origins)


def _equal_share_oracle(
    case: _SelectionCase,
    tangents: tuple[np.ndarray, ...],
    cotangent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    """Share each output's derivative equally among its tied SciPy window slots."""
    footprint, origins = case.slot_window()
    index = np.arange(case.sample.size, dtype=np.float64).reshape(case.sample.shape)
    sources = np.stack(
        [
            scipy_ndimage.generic_filter(
                index,
                operator.itemgetter(slot),
                footprint=footprint,
                mode=case.kwargs.get("mode", "reflect"),
                cval=-1,
                origin=origins,
            )
            for slot in range(np.count_nonzero(footprint))
        ]
    ).astype(np.intp)
    inside = sources >= 0
    dilation = case.name == "grey_dilation"
    slot_shape = (-1, *((1,) * case.sample.ndim))

    def slot_values(
        value: np.ndarray,
        cval: object,
        structure: np.ndarray | None = None,
    ) -> np.ndarray:
        gathered = np.where(inside, value.ravel()[np.where(inside, sources, 0)], cval)
        if structure is None:
            return gathered
        offsets = np.flip(structure)[footprint] if dilation else -structure[footprint]
        return gathered + offsets.reshape(slot_shape)

    expected = case.call(scipy_ndimage, case.sample, case.cval, case.structure)
    winners = slot_values(case.sample, case.cval, case.structure) == expected
    assert np.all(np.any(winners, axis=0))
    share = winners / np.sum(winners, axis=0)
    directional = np.sum(share * slot_values(*tangents), axis=0)
    routed = share * cotangent
    cotangents: tuple[np.ndarray, ...] = (
        np.bincount(sources[inside], routed[inside], minlength=case.sample.size).reshape(
            case.sample.shape
        ),
        np.sum(routed[~inside]),
    )
    if case.structure is not None:
        structure_cotangent = np.zeros(footprint.shape)
        structure_cotangent[footprint] = np.sum(routed, axis=tuple(range(1, routed.ndim)))
        cotangents += (np.flip(structure_cotangent) if dilation else -structure_cotangent,)
    return expected, directional, cotangents


@st.composite
def _selection_cases(draw: st.DrawFn) -> _SelectionCase:
    name = draw(st.sampled_from(_SELECTION_NAMES))
    # Few levels force plateaus; sixteen levels mostly select unique winners.
    levels = draw(st.sampled_from((1, 2, 3, 16)))
    level = st.integers(0, levels - 1)
    # The oracle enumerates slots with generic_filter, which reads wrong sources
    # once a window far outgrows its axis (nine reflected slots over two
    # samples), so sides and windows stay at most four.
    shapes = hnp.array_shapes(min_dims=1, max_dims=2, max_side=4)
    sample = draw(hnp.arrays(np.float64, shapes, elements=level))
    kwargs: dict[str, object] = {"mode": draw(st.sampled_from(_MODES))}
    # SciPy 1.18's one-dimensional rank-filter fast path ignores false
    # footprint entries and misreads windows longer than the signal.
    full_window = sample.ndim == 1 and name in {"median_filter", "rank_filter", "percentile_filter"}
    extent = sample.size if full_window else 4

    def origin(size: int) -> int:
        return draw(st.integers(-(size // 2), (size - 1) // 2))

    args: tuple[object, ...] = ()
    if name.endswith("1d"):
        size = draw(st.integers(1, 4))
        args = (size,)
        kwargs |= {"axis": draw(st.integers(-sample.ndim, sample.ndim - 1)), "origin": origin(size)}
        shape: tuple[int, ...] = (size,)
    else:
        if full_window or draw(st.booleans()):
            shape = tuple(
                draw(st.lists(st.integers(1, extent), min_size=sample.ndim, max_size=sample.ndim))
            )
            kwargs["size"] = shape
            count = math.prod(shape)
        else:
            footprint = draw(
                hnp.arrays(
                    np.bool_,
                    hnp.array_shapes(min_dims=sample.ndim, max_dims=sample.ndim, max_side=4),
                )
            )
            footprint.flat[draw(st.integers(0, footprint.size - 1))] = True
            kwargs["footprint"] = footprint
            shape = footprint.shape
            count = np.count_nonzero(footprint)
        kwargs["origin"] = tuple(origin(size) for size in shape)
        if name == "rank_filter":
            args = (draw(st.integers(-count, count - 1)),)
        elif name == "percentile_filter":
            args = (float(draw(st.integers(-100, 100))),)
    structure = None
    if name.startswith("grey"):
        offsets = st.sampled_from((-0.5, 0.0, 0.5))
        structure = draw(st.none() | hnp.arrays(np.float64, shape, elements=offsets))
    return _SelectionCase(name, sample, args, kwargs, float(draw(level)), structure)


_DISTINCT_FIELD = np.array([[0.3, -1.2, 0.8], [2.1, -0.4, 1.5]])
_DISTINCT_GRID = (np.arange(100.0) * 67 % 100).reshape(10, 10) + 0.5


@settings(max_examples=max(20, settings.default.max_examples // 2), deadline=None)
@given(case=_selection_cases())
# Even plateau windows once counted dilation ties one slot off and routed
# reflect-mode cotangents through a box filter that is not its own adjoint.
@example(_SelectionCase("grey_dilation", _DISTINCT_FIELD, (), {"size": (2, 2), "mode": "nearest"}))
@example(_SelectionCase("maximum_filter", _DISTINCT_FIELD, (), {"size": 2, "mode": "reflect"}))
@example(_SelectionCase("median_filter", _DISTINCT_FIELD, (), {"size": (2, 2)}))
@example(
    _SelectionCase(
        "grey_dilation", np.array([[0.0, 1, 1], [1, 0, 1]]), (), {"size": 2, "origin": -1}
    )
)
@example(_SelectionCase("minimum_filter", np.zeros(3), (), {"size": 2, "mode": "reflect"}))
@example(_SelectionCase("maximum_filter1d", np.array([1.0, 1, 0, 2]), (2,), {"mode": "reflect"}))
@example(_SelectionCase("grey_dilation", np.array([1.0, 1, 0, 2]), (), {"size": 2, "mode": "wrap"}))
@example(
    _SelectionCase(
        "grey_dilation",
        np.array([1.0, 2, 0, 1, 2]),
        (),
        {"size": 4, "origin": -2, "mode": "nearest"},
    )
)
# Distinct values in windows over 25 slots route through one sort instead.
@example(_SelectionCase("minimum_filter", _DISTINCT_GRID, (), {"size": (5, 6), "mode": "constant"}))
@example(_SelectionCase("rank_filter", _DISTINCT_GRID, (-3,), {"size": (6, 5), "mode": "wrap"}))
# Rank filters share a constant field's tangent across every window slot.
@example(_SelectionCase("median_filter", np.ones(5), (), {"size": 3, "mode": "reflect"}))
@example(_SelectionCase("rank_filter", np.ones(5), (1,), {"size": 3, "mode": "reflect"}))
@example(_SelectionCase("percentile_filter", np.ones(5), (50.0,), {"size": 3, "mode": "reflect"}))
# A unit window is the identity; a long wrapped window ties each value four times.
@example(_SelectionCase("maximum_filter", np.arange(12.0).reshape(3, 4), (), {"size": 1}))
@example(
    _SelectionCase("rank_filter", np.tile(np.arange(12.0), 4), (16,), {"size": 33, "mode": "wrap"})
)
def test_selection_derivatives_share_ties_equally_across_window_slots(
    case: _SelectionCase,
) -> None:
    operands = (case.sample, np.asarray(case.cval))
    if case.structure is not None:
        operands += (case.structure,)
    argnums = tuple(range(len(operands)))
    rng = np.random.default_rng(0)
    tangents = tuple(rng.normal(size=np.shape(operand)) for operand in operands)
    cotangent = rng.normal(size=case.sample.shape)
    expected, expected_directional, expected_cotangents = _equal_share_oracle(
        case, tangents, cotangent
    )

    def function(*values: object) -> object:
        return case.call(ndimage, *values)

    def directional_derivative(*values: object) -> object:
        return ad.jvp(function, argnums=argnums)(
            *values[: len(operands)], tangents=values[len(operands) :]
        )[1]

    value, directional = ad.jvp(function, argnums=argnums)(*operands, tangents=tangents)
    _value, pullback = ad.vjp(function, argnums)(*operands)
    # Staged tracing exercises the traceable candidate path instead of the
    # concrete NumPy winner, plateau, and routing shortcuts.
    staged_directional = ad.stage(directional_derivative, *operands, *tangents)

    # Shares of at most 30 unit-scale slots, summed in a different order than
    # the oracle's, agree to a few ulps (the worst seen is about 1e-15).
    atol = 1e-14
    assert_array_equal(value, expected)
    assert_allclose(directional, expected_directional, rtol=0, atol=atol)
    assert_allclose(
        staged_directional(*operands, *tangents), expected_directional, rtol=0, atol=atol
    )
    for actual, desired in zip(pullback(cotangent), expected_cotangents, strict=True):
        assert_allclose(actual, desired, rtol=0, atol=atol)


@pytest.mark.parametrize(
    ("function", "sample"),
    [
        (lambda x: ndimage.maximum_filter(x, size=3), np.empty((0, 3))),
        # Structured morphology once padded the empty axis in its JVP.
        (lambda x: ndimage.grey_erosion(x, structure=np.zeros((3, 3))), np.empty((0, 3))),
        (lambda x: ndimage.maximum_filter(x, size=3), np.arange(6, dtype=np.int64).reshape(2, 3)),
    ],
    ids=["empty", "empty-grey_erosion-structure", "integer"],
)
def test_empty_and_integer_selections_have_zero_derivatives(
    function: Callable[[object], object],
    sample: np.ndarray,
) -> None:
    value, directional = ad.jvp(function)(sample, tangents=np.ones_like(sample))
    _value, pullback = ad.vjp(function)(sample)

    assert np.shape(value) == np.shape(directional) == sample.shape
    assert not np.any(directional)
    assert_array_equal(pullback(np.ones_like(sample)), np.zeros_like(sample))


@settings(max_examples=max(20, settings.default.max_examples // 2), deadline=None)
@given(
    case=_selection_cases(),
    position=st.integers(0, 15),
    poison=st.sampled_from((np.inf, -np.inf, np.nan)),
)
# Masking losing slots by multiplication once turned 0 * inf into NaN, and the
# plateau box sums spread a non-finite entry along its whole line.
@example(_SelectionCase("maximum_filter1d", np.array([0.0, 1, 1, 0, 1, 0, 0, 1]), (3,)), 3, np.inf)
# Plateaus outside reflect, constant and wrap modes sum through the padded box
# adjoint instead of a centered uniform filter.
@example(
    _SelectionCase(
        "maximum_filter",
        np.array([0.0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]),
        (),
        {"size": 3, "mode": "nearest"},
    ),
    1,
    np.inf,
)
@example(_SelectionCase("maximum_filter", _DISTINCT_GRID, (), {"size": 3}), 23, np.inf)
@example(_SelectionCase("median_filter", _DISTINCT_GRID, (), {"size": 3}), 47, np.nan)
@example(
    _SelectionCase(
        "grey_erosion",
        _DISTINCT_FIELD,
        (),
        {"size": (2, 2), "mode": "constant"},
        -0.5,
        np.array([[0.0, 0.5], [-0.5, 0.0]]),
    ),
    4,
    -np.inf,
)
# Distinct values in windows over 25 slots route through one sort instead.
@example(
    _SelectionCase("minimum_filter", _DISTINCT_GRID, (), {"size": (5, 6), "mode": "constant"}),
    55,
    np.inf,
)
# The plateau box sums leave an eps-sized residue where the unit perturbation
# never reaches (the corner window here), which must not count as reach.
@example(
    _SelectionCase(
        "maximum_filter", np.zeros((2, 4)), (), {"mode": "nearest", "size": 3, "origin": (0, 1)}
    ),
    0,
    np.inf,
)
def test_selection_derivatives_confine_nonfinite_entries_to_winning_slots(
    case: _SelectionCase,
    position: int,
    poison: float,
) -> None:
    # Each derivative is linear, so a non-finite entry must reach exactly the
    # entries its unit perturbation reaches and leave every other one finite.
    operands = (case.sample, np.asarray(case.cval))
    if case.structure is not None:
        operands += (case.structure,)
    argnums = tuple(range(len(operands)))
    rng = np.random.default_rng(0)
    tangents = tuple(rng.normal(size=np.shape(operand)) for operand in operands)
    index = np.unravel_index(position % case.sample.size, case.sample.shape)

    def function(*values: object) -> object:
        return case.call(ndimage, *values)

    def directional(sample_tangent: np.ndarray, others: tuple[np.ndarray, ...]) -> object:
        return ad.jvp(function, argnums=argnums)(*operands, tangents=(sample_tangent, *others))[1]

    def pullback(cotangent: np.ndarray) -> tuple[object, ...]:
        return ad.vjp(function, argnums)(*operands)[1](cotangent)

    def variants(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        finite, unit, poisoned = np.array(value), np.zeros_like(value), np.array(value)
        finite[index], unit[index], poisoned[index] = 0.0, 1.0, poison
        return finite, unit, poisoned

    def expected(reach: object, finite: object) -> np.ndarray:
        # A reached entry receives a tie share of at least 1/30; the plateau box
        # sums are only accurate to about eps per line, so smaller is no reach.
        signs = (np.greater(reach, 1e-12), np.less(reach, -1e-12))
        return np.select(signs, (poison, -poison), finite)

    finite, unit, poisoned = variants(tangents[0])
    assert_allclose(
        directional(poisoned, tangents[1:]),
        expected(
            directional(unit, tuple(np.zeros_like(item) for item in tangents[1:])),
            directional(finite, tangents[1:]),
        ),
        rtol=0,
        atol=1e-12,
    )
    finite, unit, poisoned = variants(rng.normal(size=case.sample.shape))
    for actual, reach, finite_cotangent in zip(
        pullback(poisoned), pullback(unit), pullback(finite), strict=True
    ):
        assert_allclose(actual, expected(reach, finite_cotangent), rtol=0, atol=1e-12)


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
def test_traced_output_destination_leaves_no_graph_trace(name: str) -> None:
    def function(value: object) -> object:
        return _call(name, ndimage, value, output=(3 * value).copy())

    artifact = ad.stage(function, _FIELD).to_dict()
    nodes = artifact["program"]["graph"]["nodes"]

    assert not {"array.multiply", "advect.copy"} & {node["op"] for node in nodes}
    assert all("has_destination" not in node["attrs"] for node in nodes)
    assert_array_equal(
        ad.StagedProgram.from_dict(artifact)(_FIELD), _call(name, scipy_ndimage, _FIELD)
    )


def test_selection_artifacts_store_only_source_configuration() -> None:
    program = ad.stage(
        lambda value: ndimage.grey_dilation(value, size=(2, 3), axes=(1, 0), origin=(0, -1)),
        _FIELD,
    )
    node = next(
        node
        for node in program.to_dict()["program"]["graph"]["nodes"]
        if node["op"] == "custom.scipy.ndimage.grey_dilation"
    )

    assert "has_footprint" not in node["attrs"]
    assert not any(name.startswith("neighborhood_") for name in node["attrs"])


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


def test_gaussian_scalar_and_empty_axes_are_not_special_case_footguns() -> None:
    scalar = np.array(2.0)
    field = np.array(_FIELD, copy=True)

    assert_array_equal(ndimage.gaussian_filter(scalar, 1.2), scalar)
    assert_array_equal(ndimage.gaussian_filter(field, (), axes=()), field)

    program = ad.stage(lambda value: ndimage.gaussian_filter(value, (), axes=()), field)
    restored = ad.StagedProgram.from_dict(program.to_dict())

    assert_array_equal(restored(field), field)
