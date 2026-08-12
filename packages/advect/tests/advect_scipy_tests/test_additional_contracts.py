"""Additional public SciPy frontend contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import ndimage as scipy_ndimage, special as scipy_special

import advect as ad
from advect.autodiff.api.implicit import ImplicitSolveError
from advect.scipy import ndimage, special
from advect.scipy.sparse.linalg import gmres_solver

if TYPE_CHECKING:
    from collections.abc import Callable


_FIELD = np.arange(12.0).reshape(3, 4)
_SPEC = (ad.ArraySpec(_FIELD.shape, _FIELD.dtype),)


@pytest.mark.parametrize(
    ("function", "error"),
    [
        (lambda x: ndimage.maximum_filter(x, size=None), RuntimeError),
        (
            lambda x: ndimage.maximum_filter(
                x,
                footprint=np.ones((3, 3), dtype=bool),
                axes=(0,),
            ),
            RuntimeError,
        ),
        (lambda x: ndimage.maximum_filter(x, size=3, axes=(0, "bad")), ValueError),
        (lambda x: ndimage.maximum_filter(x, size=3, axes=object()), ValueError),
        (lambda x: ndimage.maximum_filter(x, size=3, axes=(2,)), ValueError),
        (lambda x: ndimage.maximum_filter(x, size=3, axes=(0, 0)), ValueError),
        (lambda x: ndimage.maximum_filter(x, size=(3,)), RuntimeError),
        (lambda x: ndimage.maximum_filter1d(x, 3, axis=2), np.exceptions.AxisError),
        (lambda x: ndimage.maximum_filter(x, size=3, output=object()), TypeError),
        (
            lambda x: ndimage.maximum_filter(x, size=3, output=np.empty(3)),
            RuntimeError,
        ),
    ],
)
def test_staged_ndimage_rejects_invalid_public_configuration(
    function: Callable[[object], object],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        ad.stage(function, specs=_SPEC)


def test_staged_ndimage_accepts_scalar_axes_and_numpy_scalar_origin() -> None:
    function = lambda x: ndimage.maximum_filter(  # noqa: E731 - compact parity case
        x,
        size=3,
        axes=1,
        origin=np.int64(0),
    )
    program = ad.stage(function, specs=_SPEC)

    assert_array_equal(
        program(_FIELD),
        scipy_ndimage.maximum_filter(_FIELD, size=3, axes=1, origin=np.int64(0)),
    )


def test_ndimage_configuration_cannot_be_a_dynamic_operand() -> None:
    with pytest.raises(TypeError, match="configuration arguments must be concrete"):
        ad.stage(
            lambda x, origin: ndimage.maximum_filter(x, size=3, origin=origin),
            specs=(*_SPEC, ad.ArraySpec((), np.dtype(np.int64))),
        )


def test_ndimage_footprint_cannot_be_a_dynamic_operand() -> None:
    footprint = np.ones((3, 3), dtype=bool)

    with pytest.raises(TypeError, match=r"footprint.*must be concrete"):
        ad.stage(
            lambda x, live_footprint: ndimage.maximum_filter(x, footprint=live_footprint),
            specs=(*_SPEC, ad.ArraySpec(footprint.shape, footprint.dtype)),
        )


def test_staged_ndimage_output_must_be_owned_by_the_active_trace() -> None:
    destination = np.empty_like(_FIELD)

    with pytest.raises(TypeError, match=r"output=.*owned traced array"):
        ad.stage(
            lambda x: ndimage.maximum_filter(x, size=3, output=destination),
            specs=_SPEC,
        )


def test_unit_selection_filter_is_an_identity_with_an_identity_pullback() -> None:
    tangent = np.linspace(-0.4, 0.7, _FIELD.size).reshape(_FIELD.shape)
    function = lambda x: ndimage.maximum_filter(x, size=1)  # noqa: E731

    value, directional = ad.jvp(function)(_FIELD, tangents=tangent)
    _value, pullback = ad.vjp(function)(_FIELD)

    assert_array_equal(value, _FIELD)
    assert_array_equal(directional, tangent)
    assert_array_equal(pullback(tangent), tangent)


def test_empty_selection_filter_has_an_empty_pullback() -> None:
    sample = np.empty((0, 3))
    function = lambda x: ndimage.maximum_filter(x, size=3)  # noqa: E731

    value, pullback = ad.vjp(function)(sample)

    assert value.shape == sample.shape
    assert_array_equal(pullback(np.empty_like(value)), sample)


def test_integer_selection_filter_has_zero_derivative() -> None:
    sample = np.arange(6, dtype=np.int64).reshape(2, 3)

    _value, pullback = ad.vjp(lambda x: ndimage.maximum_filter(x, size=3))(sample)

    assert_array_equal(pullback(np.ones_like(sample)), np.zeros_like(sample))


def test_large_rank_filter_ties_have_a_real_adjoint() -> None:
    sample = np.tile(np.arange(12.0), 4)
    tangent = np.linspace(-0.5, 0.7, sample.size)
    cotangent = np.linspace(0.3, -0.2, sample.size)
    function = lambda x: ndimage.rank_filter(x, 16, size=33, mode="wrap")  # noqa: E731

    value, directional = ad.jvp(function)(sample, tangents=tangent)
    expected_value = scipy_ndimage.rank_filter(sample, 16, size=33, mode="wrap")
    _value, pullback = ad.vjp(function)(sample)

    assert_array_equal(value, expected_value)
    assert_allclose(np.vdot(directional, cotangent), np.vdot(tangent, pullback(cotangent)))


@pytest.mark.parametrize(
    "out",
    [(), (None, None)],
)
def test_staged_special_ufunc_requires_one_output(out: tuple[object, ...]) -> None:
    with pytest.raises(ValueError, match="exactly one entry"):
        ad.stage(lambda x: special.erf(x, out=out), specs=_SPEC)


@pytest.mark.parametrize(
    "signature",
    [np.str_("d->d"), np.bytes_(b"d->d")],
)
def test_staged_special_accepts_numpy_scalar_signatures(signature: object) -> None:
    program = ad.stage(lambda x: special.erf(x, signature=signature), specs=_SPEC)

    assert_allclose(program(_FIELD), scipy_special.erf(_FIELD, signature=signature))


def test_staged_special_accepts_numpy_scalar_text_options() -> None:
    kwargs = {"casting": np.str_("unsafe"), "order": np.bytes_(b"C")}
    program = ad.stage(lambda x: special.erf(x, **kwargs), specs=_SPEC)

    assert_allclose(program(_FIELD), scipy_special.erf(_FIELD, **kwargs))


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"unknown": True}, TypeError, "unexpected keyword"),
        ({"sig": "d->d", "signature": "d->d"}, TypeError, "both 'sig' and 'signature'"),
        ({"signature": None}, TypeError, "signature object"),
        ({"signature": 3}, TypeError, "signature object"),
        ({"subok": 1}, TypeError, "subok.*boolean"),
        ({"casting": 3}, TypeError, "casting must be str"),
        ({"order": 3}, TypeError, "order must be str"),
    ],
)
def test_staged_special_rejects_invalid_ufunc_options(
    kwargs: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        ad.stage(lambda x: special.erf(x, **kwargs), specs=_SPEC)


@pytest.mark.parametrize(
    "axis",
    [True, [0], (True,), ("bad",)],
)
def test_staged_softmax_requires_static_integer_axes(axis: object) -> None:
    with pytest.raises(TypeError, match="axis"):
        ad.stage(lambda x: special.softmax(x, axis=axis), specs=_SPEC)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rtol": -1.0},
        {"atol": -1.0},
        {"maxiter": 0},
    ],
)
def test_gmres_rejects_invalid_solver_configuration(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="GMRES"):
        gmres_solver(**kwargs)


def test_gmres_rejects_complex_operator_output_for_real_state() -> None:
    with pytest.raises(ImplicitSolveError, match="complex values for a real state"):
        gmres_solver()(lambda x: x + 1j, np.array([1.0, 2.0]))


def test_gmres_accepts_integer_rhs_and_returns_a_floating_solution() -> None:
    solution = gmres_solver(rtol=1e-12)(lambda x: 2 * x, np.array([2, 4]))

    assert_allclose(solution, np.array([1.0, 2.0]))
    assert np.issubdtype(np.asarray(solution).dtype, np.floating)
