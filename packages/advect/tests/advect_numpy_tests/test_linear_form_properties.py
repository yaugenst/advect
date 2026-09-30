"""Linear and affine NumPy forms differentiate as their own linear map.

Every form here is affine in its array operands once its static metadata is
fixed, so its JVP must equal ``f(d) - f(0)`` and its VJP must satisfy the
adjoint identity. Hypothesis draws the metadata (axes, widths, modes, index
sets with duplicates) where transposition and bookkeeping bugs hide.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hnp

import advect as ad
from advect.core._pytree import tree_map
from advect_numpy_tests._assertions import (
    assert_adjoint_identity,
    assert_tree_close,
    seeded_like,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class _LinearCase:
    """One affine public call: ``function(*primals)``."""

    function: Callable[..., Any]
    primals: tuple[np.ndarray[Any, Any], ...]


_ELEMENTS = st.floats(-10.0, 10.0, allow_nan=False, allow_subnormal=False)


def _array(
    draw: st.DrawFn,
    *,
    min_dims: int = 1,
    max_dims: int = 3,
    min_side: int = 1,
    dtype: type[np.inexact[Any]] = np.float64,
) -> np.ndarray[Any, Any]:
    shape = draw(
        hnp.array_shapes(min_dims=min_dims, max_dims=max_dims, min_side=min_side, max_side=4)
    )
    real = draw(hnp.arrays(np.float64, shape, elements=_ELEMENTS))
    if dtype is np.complex128:
        return real + 1j * draw(hnp.arrays(np.float64, shape, elements=_ELEMENTS))
    return real


def _axis(draw: st.DrawFn, ndim: int) -> int:
    return draw(st.integers(-ndim, ndim - 1))


def _updated(
    source: Any,
    values: Any,
    *,
    operation: Callable[..., None],
    args: tuple[Any, ...] = (),
    **options: Any,
) -> Any:
    result = source.copy()
    operation(result, *args, values, **options)
    return result


def _update_case(
    operation: Callable[..., None], source: Any, values: Any, *args: Any, **options: Any
) -> _LinearCase:
    return _LinearCase(
        partial(_updated, operation=operation, args=args, **options), (source, values)
    )


def _insert(value: Any, inserted: Any, *, obj: Any, axis: int | None) -> Any:
    return np.insert(value, obj, inserted, axis=axis)


def _kernel_first(value: Any, *, name: str, kernel: Any, mode: str) -> Any:
    return getattr(np, name)(kernel, value, mode=mode)


def _gradient(value: Any, *, spacing: float, axis: int | None, edge_order: int) -> Any:
    return np.gradient(value, spacing, axis=axis, edge_order=edge_order)


def _flat_diag(value: Any, *, k: int) -> Any:
    return np.diag(np.ravel(value), k=k)


def _pad_boundary(value: Any, boundary: Any, *, pad_width: Any, mode: str) -> Any:
    keyword = "constant_values" if mode == "constant" else "end_values"
    return np.pad(value, pad_width, mode=mode, **{keyword: boundary})


@st.composite
def _pad(draw: st.DrawFn) -> _LinearCase:
    value = _array(draw)
    widths = tuple(
        (draw(st.integers(0, 6)), draw(st.integers(0, 6))) for _axis_index in range(value.ndim)
    )
    mode = draw(
        st.sampled_from(("constant", "edge", "linear_ramp", "mean", "reflect", "symmetric", "wrap"))
    )
    options: dict[str, object] = {}
    if mode == "constant":
        options["constant_values"] = draw(_ELEMENTS)
    elif mode == "linear_ramp":
        options["end_values"] = draw(_ELEMENTS)
    elif mode == "mean":
        options["stat_length"] = draw(st.none() | st.integers(1, 4))
    elif mode in {"reflect", "symmetric"}:
        options["reflect_type"] = draw(st.sampled_from(("even", "odd")))
    return _LinearCase(partial(np.pad, pad_width=widths, mode=mode, **options), (value,))


@st.composite
def _shift_and_repeat(draw: st.DrawFn) -> _LinearCase:
    value = _array(draw)
    axis = draw(st.none() | st.integers(-value.ndim, value.ndim - 1))
    name = draw(st.sampled_from(("cumsum", "diff", "flip", "repeat", "roll")))
    options: dict[str, object] = {"axis": axis}
    if name == "diff":
        options = {"axis": _axis(draw, value.ndim), "n": draw(st.integers(0, 3))}
        if draw(st.booleans()):
            options["prepend"] = 0.5
    elif name == "repeat":
        options["repeats"] = draw(st.integers(0, 3))
    elif name == "roll":
        options["shift"] = draw(st.integers(-5, 5))
    return _LinearCase(partial(getattr(np, name), **options), (value,))


@st.composite
def _tile_and_axes(draw: st.DrawFn) -> _LinearCase:
    value = _array(draw)
    order = draw(st.permutations(range(value.ndim)))
    calls: tuple[Callable[..., Any], ...] = (
        partial(np.tile, reps=tuple(draw(st.lists(st.integers(0, 2), min_size=1, max_size=3)))),
        partial(np.transpose, axes=tuple(order)),
        partial(np.moveaxis, source=_axis(draw, value.ndim), destination=_axis(draw, value.ndim)),
        partial(np.reshape, shape=(-1,), order=draw(st.sampled_from(("C", "F")))),
    )
    return _LinearCase(draw(st.sampled_from(calls)), (value,))


@st.composite
def _selections(draw: st.DrawFn) -> _LinearCase:
    value = _array(draw)
    axis = _axis(draw, value.ndim)
    size = value.shape[axis]
    mode = draw(st.sampled_from(("raise", "wrap", "clip")))
    bound = size if mode == "raise" else 3 * size
    indices = draw(
        hnp.arrays(np.int64, draw(st.integers(0, 5)), elements=st.integers(-bound, bound - 1))
    )
    along_shape = list(value.shape)
    along_shape[axis] = draw(st.integers(0, 4))
    along = draw(hnp.arrays(np.int64, tuple(along_shape), elements=st.integers(0, size - 1)))
    obj = draw(
        st.integers(-size, size - 1)
        | st.lists(st.integers(-size, size - 1), max_size=3, unique=True)
        | st.builds(slice, st.none() | st.integers(0, size), st.none(), st.integers(1, 2))
    )
    calls: tuple[Callable[..., Any], ...] = (
        partial(np.take, indices=indices, axis=axis, mode=mode),
        partial(np.take_along_axis, indices=along, axis=axis),
        partial(np.delete, obj=obj, axis=axis),
    )
    return _LinearCase(draw(st.sampled_from(calls)), (value,))


@st.composite
def _splits(draw: st.DrawFn) -> _LinearCase:
    value = _array(draw)
    axis = _axis(draw, value.ndim)
    size = value.shape[axis]
    divisors = [count for count in range(1, size + 1) if size % count == 0]
    indices = st.lists(st.integers(0, size), max_size=3).map(sorted)
    calls = (
        partial(np.split, indices_or_sections=draw(st.sampled_from(divisors) | indices)),
        partial(np.array_split, indices_or_sections=draw(st.integers(1, size + 2) | indices)),
    )
    return _LinearCase(partial(draw(st.sampled_from(calls)), axis=axis), (value,))


@st.composite
def _diagonals(draw: st.DrawFn) -> _LinearCase:
    value = _array(draw, min_dims=2)
    first, second = draw(st.permutations(range(value.ndim)))[:2]
    offset = draw(st.integers(-3, 3))
    calls: tuple[Callable[..., Any], ...] = (
        partial(np.trace, offset=offset, axis1=first, axis2=second),
        partial(np.diagonal, offset=offset, axis1=first, axis2=second),
        partial(np.tril, k=offset),
        partial(np.triu, k=offset),
    )
    if value.ndim == 2:
        calls = (*calls, partial(np.diag, k=offset), partial(_flat_diag, k=offset))
    return _LinearCase(draw(st.sampled_from(calls)), (value,))


@st.composite
def _transforms(draw: st.DrawFn) -> _LinearCase:
    name = draw(st.sampled_from(("fft", "hfft", "ifft", "ihfft", "irfft", "rfft")))
    real = name in {"ihfft", "rfft"}
    value = _array(draw, dtype=np.float64 if real else np.complex128)
    axis = _axis(draw, value.ndim)
    size = draw(st.none() | st.integers(1, 6))
    if name in {"hfft", "irfft"} and size is None and value.shape[axis] == 1:
        size = 2
    norm = draw(st.sampled_from((None, "ortho", "forward")))
    return _LinearCase(partial(getattr(np.fft, name), n=size, axis=axis, norm=norm), (value,))


@st.composite
def _signals(draw: st.DrawFn) -> _LinearCase:
    name = draw(st.sampled_from(("convolve", "correlate")))
    # NumPy's convolve reads a rank-zero signal as one of length one.
    value = _array(draw, min_dims=0 if name == "convolve" else 1, max_dims=1)
    kernel = draw(hnp.arrays(np.float64, draw(st.integers(1, 4)), elements=_ELEMENTS))
    mode = draw(st.sampled_from(("full", "same", "valid")))
    if draw(st.booleans()):
        return _LinearCase(partial(getattr(np, name), v=kernel, mode=mode), (value,))
    return _LinearCase(partial(_kernel_first, name=name, kernel=kernel, mode=mode), (value,))


@st.composite
def _gradients(draw: st.DrawFn) -> _LinearCase:
    value = _array(draw, min_side=3)
    axis = draw(st.none() | st.integers(-value.ndim, value.ndim - 1))
    spacing = draw(st.floats(0.25, 4.0))
    order = draw(st.sampled_from((1, 2)))
    return _LinearCase(partial(_gradient, spacing=spacing, axis=axis, edge_order=order), (value,))


@st.composite
def _insertions(draw: st.DrawFn) -> _LinearCase:
    value = _array(draw)
    axis = draw(st.none() | st.integers(-value.ndim, value.ndim - 1))
    size = value.size if axis is None else value.shape[axis]
    obj = draw(
        st.integers(-size, size)
        | st.lists(st.integers(-size, size), max_size=3)
        | st.builds(slice, st.none() | st.integers(0, size), st.none(), st.integers(1, 2))
    )
    inserted = np.asarray(draw(_ELEMENTS))
    return _LinearCase(partial(_insert, obj=obj, axis=axis), (value, inserted))


@st.composite
def _mutations(draw: st.DrawFn) -> _LinearCase:
    source = _array(draw)
    flat = source.size
    replacement = draw(hnp.arrays(np.float64, draw(st.integers(1, 5)), elements=_ELEMENTS))
    mask = draw(hnp.arrays(np.bool_, source.shape))
    name = draw(
        st.sampled_from(("copyto", "fill_diagonal", "place", "put", "put_along_axis", "putmask"))
    )
    if name == "copyto":
        values = draw(
            hnp.arrays(
                np.float64, source.shape[draw(st.integers(0, source.ndim)) :], elements=_ELEMENTS
            )
        )
        return _update_case(np.copyto, source, values, where=mask)
    if name == "place":
        return _update_case(np.place, source, replacement, mask)
    if name == "putmask":
        return _update_case(np.putmask, source, replacement, mask)
    if name == "fill_diagonal":
        square = np.resize(source, (source.shape[0],) * max(source.ndim, 2))
        tall = source.reshape(source.shape[0], -1)
        target = draw(st.sampled_from((tall, square)))
        return _update_case(np.fill_diagonal, target, replacement, wrap=draw(st.booleans()))
    if name == "put":
        mode = draw(st.sampled_from(("raise", "wrap", "clip")))
        bound = flat if mode == "raise" else 3 * flat
        indices = draw(st.lists(st.integers(-bound, bound - 1), min_size=1, max_size=5))
        return _update_case(np.put, source, replacement, indices, mode=mode)
    axis = _axis(draw, source.ndim)
    shape = list(source.shape)
    shape[axis] = draw(st.integers(1, 3))
    indices = draw(
        hnp.arrays(np.int64, tuple(shape), elements=st.integers(0, source.shape[axis] - 1))
    )
    values = draw(hnp.arrays(np.float64, tuple(shape), elements=_ELEMENTS))
    return _update_case(np.put_along_axis, source, values, indices, axis=axis)


_FAMILIES = {
    "diagonals": _diagonals(),
    "gradients": _gradients(),
    "insertions": _insertions(),
    "mutations": _mutations(),
    "pad": _pad(),
    "selections": _selections(),
    "shift-and-repeat": _shift_and_repeat(),
    "signals": _signals(),
    "splits": _splits(),
    "tile-and-axes": _tile_and_axes(),
    "transforms": _transforms(),
}
# Former example tests, kept as explicit cases.
_EXAMPLES = {
    "put-wrap-last-write": _update_case(
        np.put, np.arange(4.0), np.array([10.0, 20.0]), np.array([-1, 6, 1]), mode="wrap"
    ),
    "put-clip-last-write": _update_case(
        np.put, np.arange(4.0), np.array([10.0, 20.0]), np.array([-1, 6, 1]), mode="clip"
    ),
    "put-repeated-index": _update_case(
        np.put, np.arange(4.0).reshape(2, 2), np.array([10.0, 20.0, 30.0]), [0, 3, 0]
    ),
    "put-along-axis-none": _update_case(
        np.put_along_axis,
        np.arange(6.0).reshape(2, 3),
        np.array([10.0, 20.0]),
        np.array([0, 3]),
        axis=None,
    ),
    "put-along-axis-broadcast": _update_case(
        np.put_along_axis,
        np.arange(24.0).reshape(2, 3, 4),
        np.array([[[100.0], [200.0], [300.0]]]),
        np.array([[[0], [2], [1]]]),
        axis=1,
    ),
    "putmask-flattened-mask": _update_case(
        np.putmask,
        np.arange(6.0).reshape(2, 3),
        np.array([10.0, 20.0]),
        np.array([True, False, True, False, True, False]),
    ),
    "fill-diagonal": _update_case(
        np.fill_diagonal, np.arange(4.0).reshape(2, 2), np.array([10.0, 20.0, 30.0])
    ),
    "fill-diagonal-wrap": _update_case(
        np.fill_diagonal, np.arange(8.0).reshape(4, 2), np.array([10.0, 20.0, 30.0]), wrap=True
    ),
    "copyto-where": _update_case(
        np.copyto,
        np.arange(4.0).reshape(2, 2),
        np.array([[10.0, 20.0], [30.0, 40.0]]),
        where=np.array([[True, False], [False, True]]),
    ),
    "place": _update_case(
        np.place,
        np.arange(4.0).reshape(2, 2),
        np.array([10.0, 20.0, 30.0]),
        np.array([[True, False], [True, True]]),
    ),
    "insert-negative-axis": _LinearCase(
        partial(_insert, obj=1, axis=-1), (np.arange(6.0).reshape(2, 3), np.array([10.0, 20.0]))
    ),
    "insert-into-empty": _LinearCase(partial(_insert, obj=0, axis=None), (np.empty(0), np.ones(2))),
    "insert-nothing": _LinearCase(partial(_insert, obj=[], axis=None), (np.ones(3), np.empty(0))),
    "pad-mean-whole-edge": _LinearCase(
        partial(np.pad, pad_width=((1, 2), (2, 1)), mode="mean"),
        (np.arange(1.0, 7.0).reshape(2, 3),),
    ),
    "pad-reflect-odd-periods": _LinearCase(
        partial(np.pad, pad_width=(18, 17), mode="reflect", reflect_type="odd"),
        (np.linspace(-0.7, 1.3, 3),),
    ),
    "diag-rectangular": _LinearCase(partial(np.diag, k=1), (np.arange(6.0).reshape(2, 3),)),
    # The transpose read the length of a rank-zero signal: "tuple index out of range".
    "convolve-rank-zero": _LinearCase(
        partial(np.convolve, v=np.array([0.8, 1.3]), mode="same"), (np.asarray(0.5),)
    ),
    "convolve-rank-zero-kernel": _LinearCase(
        partial(_kernel_first, name="convolve", kernel=np.array([0.8, 1.3]), mode="full"),
        (np.asarray(0.5),),
    ),
    "fftn-explicit-defaults": _LinearCase(
        partial(np.fft.fftn, s=None, axes=None), (np.arange(8.0).reshape(2, 4),)
    ),
    "fftshift-scalar-axis": _LinearCase(
        partial(np.fft.fftshift, axes=-1), (np.arange(8.0).reshape(2, 4),)
    ),
    "pad-constant-boundary": _LinearCase(
        partial(_pad_boundary, pad_width=(7, 5), mode="constant"),
        (np.array([0.2, 1.0, 2.5]), np.array([1.5, -0.5])),
    ),
    "pad-ramp-boundary": _LinearCase(
        partial(_pad_boundary, pad_width=(7, 5), mode="linear_ramp"),
        (np.array([0.2, 1.0, 2.5]), np.array([1.5, -0.5])),
    ),
    "pad-scalar-boundary": _LinearCase(
        partial(_pad_boundary, pad_width=((1, 0), (0, 2)), mode="constant"),
        (np.arange(6.0).reshape(2, 3), np.asarray(2.0)),
    ),
    "pad-per-axis-boundary": _LinearCase(
        partial(_pad_boundary, pad_width=((1, 2), (2, 1)), mode="constant"),
        (np.arange(6.0).reshape(2, 3), np.array([[1.0, 2.0], [3.0, 4.0]])),
    ),
    "pad-broadcast-width-boundary": _LinearCase(
        partial(_pad_boundary, pad_width=((1, 2),), mode="constant"),
        (np.arange(6.0).reshape(2, 3), np.asarray(2.0)),
    ),
}


def _assert_linear_part(case: _LinearCase) -> None:
    primals = case.primals
    snapshots = tuple(np.copy(value) for value in primals)
    directions = tuple(
        seeded_like(value, f"direction:{index}") for index, value in enumerate(primals)
    )
    argnums = tuple(range(len(primals)))

    primal, tangent = ad.jvp(case.function, argnums=argnums)(*primals, tangents=directions)

    assert_tree_close(primal, case.function(*primals), rtol=1e-12, atol=1e-12)
    zeros = tuple(np.zeros_like(value) for value in primals)
    linear_part = tree_map(np.subtract, case.function(*directions), case.function(*zeros))
    assert_tree_close(tangent, linear_part, rtol=1e-12, atol=1e-12)
    assert_adjoint_identity(case.function, primals, directions, tangent, argnums=argnums)
    for value, snapshot in zip(primals, snapshots, strict=True):
        np.testing.assert_array_equal(value, snapshot, strict=True)


@pytest.mark.parametrize("family", sorted(_FAMILIES))
@given(data=st.data())
def test_affine_forms_differentiate_as_their_linear_part(family: str, data: st.DataObject) -> None:
    _assert_linear_part(data.draw(_FAMILIES[family]))


@pytest.mark.parametrize("case", _EXAMPLES.values(), ids=_EXAMPLES.keys())
def test_affine_form_examples_differentiate_as_their_linear_part(case: _LinearCase) -> None:
    _assert_linear_part(case)


@pytest.mark.parametrize(
    ("transform", "dtype"),
    [
        (np.fft.fftn, np.complex128),
        (np.fft.ifftn, np.complex128),
        (np.fft.rfftn, np.float64),
        (np.fft.irfftn, np.complex128),
    ],
    ids=("fftn", "ifftn", "rfftn", "irfftn"),
)
def test_nd_fft_shape_defaults_to_the_trailing_axes(
    transform: Callable[..., Any],
    dtype: type[np.generic],
) -> None:
    # NumPy deprecates s= without axes=, so its reference names the trailing axes.
    value = np.arange(24.0).reshape(2, 3, 4).astype(dtype)
    direction = np.linspace(-1.0, 1.0, value.size).reshape(value.shape).astype(dtype)

    primal, tangent = ad.jvp(lambda x: transform(x, s=(3, 6)))(value, tangents=direction)

    np.testing.assert_allclose(primal, transform(value, s=(3, 6), axes=(-2, -1)))
    np.testing.assert_allclose(tangent, transform(direction, s=(3, 6), axes=(-2, -1)))
