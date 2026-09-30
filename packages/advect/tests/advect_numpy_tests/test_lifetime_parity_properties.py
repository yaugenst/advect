"""Dynamic, staged, and NumPy parity for forms lowered to explicit compositions.

Controlled reductions, ``average``, ``matrix_power``, ``include_initial``
scans, ``gradient``, scalar-branch selection, and ufunc methods are lowered to
several canonical operations instead of one.  Each property compares the
dynamic primal, the staged program, and NumPy itself in shape, dtype, and value.

Elements are small dyadic rationals, so every sum is exact and summation order
cannot make the comparison flaky.  The tolerances only absorb the single
division, square root, or inverse that follows those exact sums.
"""

from __future__ import annotations

import warnings
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from hypothesis import example, given, settings, strategies as st
from hypothesis.extra import numpy as hnp

import advect as ad
from advect.autodiff._ephemeral import trace_call
from advect.core._pytree import tree_leaves

if TYPE_CHECKING:
    from collections.abc import Callable

_DTYPES = st.sampled_from((np.float32, np.float64))
_DYADIC = st.integers(-64, 64).map(lambda value: value / 8)
_POSITIVE_DYADIC = st.integers(1, 64).map(lambda value: value / 8)
_AXES = st.sampled_from((None, 0, 1, -1, (0, 1)))
_SHAPES = st.tuples(st.integers(1, 3), st.integers(1, 4))

_SUMS = (np.sum, np.prod, np.nansum, np.nanprod)
_MEANS = (np.mean, np.nanmean)
_EXTREMA = (np.max, np.min, np.nanmax, np.nanmin)
_VARIANCES = (np.var, np.std, np.nanvar, np.nanstd)
# NumPy 2.1 added the scans that include their initial value.
_INITIAL_SCANS = tuple(
    getattr(np, name) for name in ("cumulative_sum", "cumulative_prod") if hasattr(np, name)
)


def _tolerance(dtype: object) -> float:
    return 2e-6 if np.dtype(dtype) == np.float32 else 1e-12


def _assert_lifetime_parity(
    function: Callable[..., Any],
    inputs: tuple[np.ndarray[Any, Any], ...],
    *,
    tolerance: float,
    staged: bool = True,
) -> None:
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = function(*inputs)
        trace = trace_call(
            function,
            args=inputs,
            kwargs={},
            argnums=tuple(range(len(inputs))),
            argnames=None,
        )
        try:
            lifetimes = [("dynamic", trace.output)]
        finally:
            trace.tape.release_payloads()
        if staged:
            # Example inputs stage at the newest revision the installed NumPy serves.
            lifetimes.append(("staged", ad.stage(function, *inputs)(*inputs)))
    expected_leaves = tree_leaves(expected)
    for lifetime, actual in lifetimes:
        actual_leaves = tree_leaves(actual)
        assert len(actual_leaves) == len(expected_leaves), lifetime
        for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
            # A full reduction returns a NumPy scalar, not a 0-d array.
            assert type(actual_leaf) is type(expected_leaf), lifetime
            actual_array = np.asarray(actual_leaf)
            expected_array = np.asarray(expected_leaf)
            assert actual_array.shape == expected_array.shape, lifetime
            assert actual_array.dtype == expected_array.dtype, lifetime
            # assert_allclose compares in float64, which merges wide integers.
            if tolerance == 0 or not np.issubdtype(expected_array.dtype, np.inexact):
                np.testing.assert_array_equal(actual_array, expected_array, err_msg=lifetime)
                continue
            np.testing.assert_allclose(
                actual_array,
                expected_array,
                rtol=tolerance,
                atol=tolerance,
                equal_nan=True,
                err_msg=lifetime,
            )


def _assert_lifetimes_raise(
    function: Callable[..., Any],
    inputs: tuple[np.ndarray[Any, Any], ...],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        function(*inputs)
    with pytest.raises(error, match=match):
        trace_call(
            function,
            args=inputs,
            kwargs={},
            argnums=tuple(range(len(inputs))),
            argnames=None,
        ).tape.release_payloads()
    with pytest.raises(error, match=match):
        ad.stage(function, *inputs)(*inputs)


@st.composite
def _values(
    draw: st.DrawFn,
    dtype: object,
    *,
    with_nan: bool = False,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray[Any, Any]:
    shape = draw(_SHAPES) if shape is None else shape
    values = draw(hnp.arrays(dtype, shape, elements=_DYADIC))
    if with_nan:
        values[draw(hnp.arrays(np.bool_, shape))] = np.nan
    return values


@st.composite
def _masks(draw: st.DrawFn, shape: tuple[int, ...]) -> np.ndarray[Any, Any]:
    mask_shape = draw(st.sampled_from((shape, (1, shape[-1]))))
    return draw(hnp.arrays(np.bool_, mask_shape))


def _reduced_count(shape: tuple[int, ...], axis: object) -> int:
    if axis is None:
        return int(np.prod(shape))
    axes = axis if isinstance(axis, tuple) else (axis,)
    return int(np.prod([shape[item] for item in axes]))


@settings(deadline=None)
@given(data=st.data(), function=st.sampled_from(_SUMS + _MEANS + _EXTREMA), dtype=_DTYPES)
def test_controlled_sums_means_and_extrema_agree_across_lifetimes(
    data: st.DataObject,
    function: Callable[..., Any],
    dtype: type[np.floating[Any]],
) -> None:
    values = data.draw(_values(dtype, with_nan=function.__name__.startswith("nan")))
    kwargs: dict[str, Any] = {"axis": data.draw(_AXES), "keepdims": data.draw(st.booleans())}
    if data.draw(st.booleans()):
        kwargs["where"] = data.draw(_masks(values.shape))
    if function not in _EXTREMA and data.draw(st.booleans()):
        kwargs["dtype"] = np.float64
    initial_forms = ("none", "python", "numpy", "traced")
    if function in _MEANS:
        initial_forms = ("none",)
    elif function in _EXTREMA and "where" in kwargs:
        initial_forms = initial_forms[1:]
    initial_form = data.draw(st.sampled_from(initial_forms))
    initial = data.draw(_DYADIC)
    # NumPy casts initial= to the result dtype, so it may be wider than the values.
    initial_dtype = data.draw(_DTYPES)
    if initial_form == "python":
        kwargs["initial"] = initial
    if initial_form == "numpy":
        kwargs["initial"] = np.dtype(initial_dtype).type(initial)
    if initial_form == "traced":
        _assert_lifetime_parity(
            lambda array, start: function(array, initial=start, **kwargs),
            (values, np.asarray(initial, dtype=initial_dtype)),
            tolerance=_tolerance(dtype),
        )
        return
    _assert_lifetime_parity(
        lambda array: function(array, **kwargs),
        (values,),
        tolerance=_tolerance(dtype),
    )


_INTEGERS = np.array([[3, -1, 4], [1, -5, 9]], dtype=np.int32)
_MASK = np.array([[True, False, True], [False, False, True]])


@pytest.mark.parametrize(
    ("function", "extra"),
    [
        pytest.param(lambda array: np.mean(array, where=_MASK), (), id="masked-mean"),
        pytest.param(lambda array: np.nanmean(array, (0, 1), where=_MASK), (), id="nanmean"),
        pytest.param(
            lambda array, start: np.nanmax(array, where=_MASK, initial=start),
            (np.float32(-2.5),),
            id="nanmax-traced-initial",
        ),
        pytest.param(
            lambda array: np.astype(np.sum(array), np.float64),
            (),
            id="astype-scalar",
            marks=pytest.mark.skipif(
                np.lib.NumpyVersion(np.__version__) < "2.1.0",
                reason="NumPy 2.0's astype rejects a NumPy scalar",
            ),
        ),
    ],
)
def test_full_float32_reductions_return_numpy_scalars(
    function: Callable[..., Any], extra: tuple[Any, ...]
) -> None:
    inputs = (_INTEGERS.astype(np.float32), *(np.asarray(value) for value in extra))
    _assert_lifetime_parity(function, inputs, tolerance=_tolerance(np.float32))


@pytest.mark.parametrize(
    ("function", "values", "initial"),
    [
        (lambda array: np.max(array, axis=1, where=_MASK, initial=np.float64(9)), _INTEGERS, ()),
        (lambda array: np.max(array, axis=1, where=_MASK, initial=2.5), _INTEGERS, ()),
        (lambda array, start: np.min(array, axis=1, initial=start), _INTEGERS, (2.5,)),
        (lambda array, start: np.prod(array, 1, where=_MASK, initial=start), _INTEGERS, (2.5,)),
        (lambda array, start: np.sum(array, initial=start), _INTEGERS.astype(np.float32), (1.5,)),
    ],
    ids=("numpy-scalar", "python-float", "traced-min", "traced-prod", "traced-sum"),
)
def test_controlled_reductions_cast_initial_to_the_result_dtype(
    function: Callable[..., Any],
    values: np.ndarray[Any, Any],
    initial: tuple[float, ...],
) -> None:
    inputs = (values, *(np.asarray(item) for item in initial))
    _assert_lifetime_parity(function, inputs, tolerance=_tolerance(np.float32))


@pytest.mark.parametrize("function", [np.sum, np.prod, np.max, np.nanmin])
@pytest.mark.parametrize("controls", [{}, {"where": _MASK}], ids=("plain", "where"))
@pytest.mark.parametrize(
    "initial",
    [np.array([1.0]), np.ones((1, 1)), [1.0]],
    ids=("vector", "matrix", "list"),
)
def test_controlled_reductions_reject_a_non_scalar_initial(
    function: Callable[..., Any],
    controls: dict[str, object],
    initial: object,
) -> None:
    _assert_lifetimes_raise(
        lambda array: function(array, axis=1, initial=initial, **controls),
        (np.array([[1.0, -2.0, 3.0], [4.0, 0.5, -6.0]]),),
        ValueError,
        "setting an array element with a sequence",
    )


@pytest.mark.parametrize("function", _EXTREMA)
@pytest.mark.parametrize(
    ("values", "initial", "error", "match"),
    [
        (_INTEGERS.astype(np.int8), 1000, OverflowError, "1000 out of bounds for int8"),
        (_INTEGERS.astype(np.int8), np.int64(1000), OverflowError, "1000 out of bounds for int8"),
        (_INTEGERS.astype(np.int8), 1000.0, OverflowError, "1000 out of bounds for int8"),
        (np.abs(_INTEGERS).astype(np.uint8), -1, OverflowError, "-1 out of bounds for uint8"),
        (_INTEGERS.astype(np.float64), 1 + 2j, TypeError, "not 'complex'"),
    ],
    ids=("python-int", "numpy-int", "python-float", "negative-unsigned", "complex"),
)
def test_controlled_extrema_reject_an_initial_numpy_cannot_convert(
    function: Callable[..., Any],
    values: np.ndarray[Any, Any],
    initial: object,
    error: type[Exception],
    match: str,
) -> None:
    _assert_lifetimes_raise(
        lambda array: function(array, axis=1, where=_MASK, initial=initial),
        (values,),
        error,
        match,
    )


@pytest.mark.parametrize("function", _EXTREMA)
@pytest.mark.parametrize(
    "initial",
    [np.uint64(2**63), np.array(2**63, dtype=np.uint64), 2**63],
    ids=("numpy-scalar", "0d-array", "python-int"),
)
def test_controlled_extrema_keep_an_initial_only_the_data_dtype_holds(
    function: Callable[..., Any],
    initial: object,
) -> None:
    values = np.array([[1, 2**64 - 1, 2**63 + 7], [2**63 + 5, 4, 6]], dtype=np.uint64)

    _assert_lifetime_parity(
        lambda array: function(array, axis=1, where=_MASK, initial=initial),
        (values,),
        tolerance=0,
    )


@pytest.mark.parametrize("function", [np.nanmax, np.nanmin])
@pytest.mark.parametrize("controls", [{}, {"where": _MASK}], ids=("plain", "where"))
@pytest.mark.parametrize("form", ["static", "traced"])
def test_nan_extrema_let_selected_numbers_win_over_a_nan_initial(
    function: Callable[..., Any],
    controls: dict[str, object],
    form: str,
) -> None:
    values = np.array([[3.0, np.nan, 4.0], [np.nan, np.nan, np.nan]])
    if form == "static":
        _assert_lifetime_parity(
            lambda array: function(array, axis=1, initial=np.nan, **controls),
            (values,),
            tolerance=0,
        )
        return
    _assert_lifetime_parity(
        lambda array, start: function(array, axis=1, initial=start, **controls),
        (values, np.asarray(np.nan)),
        tolerance=0,
    )


@pytest.mark.parametrize("controls", [{}, {"where": _MASK}], ids=("plain", "where"))
def test_variance_accepts_a_zero_ddof_beside_correction(controls: dict[str, object]) -> None:
    _assert_lifetime_parity(
        lambda array: np.var(array, axis=1, ddof=0, correction=1, **controls),
        (_INTEGERS.astype(np.float64),),
        tolerance=_tolerance(np.float64),
    )


@pytest.mark.parametrize("controls", [{}, {"where": _MASK}], ids=("plain", "where"))
def test_variance_rejects_a_nonzero_ddof_beside_correction(controls: dict[str, object]) -> None:
    _assert_lifetimes_raise(
        lambda array: np.var(array, axis=1, ddof=1, correction=1, **controls),
        (_INTEGERS.astype(np.float64),),
        ValueError,
        "ddof and correction can't be provided simultaneously",
    )


@pytest.mark.parametrize("controls", [{}, {"where": _MASK}], ids=("plain", "where"))
@pytest.mark.parametrize(
    ("function", "control"),
    [
        (np.sum, {"initial": Fraction(1, 2)}),
        (np.max, {"initial": Fraction(1, 2)}),
        (np.var, {"ddof": Fraction(1, 2)}),
        (np.var, {"correction": Fraction(1, 2)}),
        (np.std, {"ddof": Fraction(1, 2)}),
        (np.std, {"correction": Fraction(1, 2)}),
    ],
    ids=("sum-initial", "max-initial", "var-ddof", "var-correction", "std-ddof", "std-correction"),
)
def test_controlled_reductions_take_any_python_number_as_a_static_control(
    function: Callable[..., Any],
    control: dict[str, object],
    controls: dict[str, object],
) -> None:
    values = _INTEGERS.astype(np.float64)

    def reduce(array: Any) -> Any:
        return function(array, axis=1, **control, **controls)

    # The where= extrema and variances fold the control into the lowering.  Staging
    # records any other control as a graph attribute, which admits built-in scalars.
    lowered = "where" in controls and function is not np.sum
    _assert_lifetime_parity(reduce, (values,), tolerance=_tolerance(np.float64), staged=lowered)
    if not lowered:
        with pytest.raises(TypeError, match="unsupported graph attribute"):
            ad.stage(reduce, values)(values)


@settings(deadline=None)
@given(data=st.data(), function=st.sampled_from(_VARIANCES), dtype=_DTYPES)
def test_controlled_variances_agree_across_lifetimes(
    data: st.DataObject,
    function: Callable[..., Any],
    dtype: type[np.floating[Any]],
) -> None:
    values = data.draw(_values(dtype, with_nan=function.__name__.startswith("nan")))
    axis = data.draw(_AXES)
    kwargs: dict[str, Any] = {"axis": axis, "keepdims": data.draw(st.booleans())}
    if data.draw(st.booleans()):
        kwargs["where"] = data.draw(_masks(values.shape))
    if data.draw(st.booleans()):
        kwargs["dtype"] = np.float64
    count = _reduced_count(values.shape, axis)
    correction = data.draw(st.integers(0, count + 1))
    correction_form = data.draw(st.sampled_from(("default", "ddof", "correction", "traced")))
    if correction_form in {"ddof", "correction"}:
        kwargs[correction_form] = correction
    inputs: tuple[np.ndarray[Any, Any], ...] = (values,)
    if correction_form == "traced":
        inputs = (*inputs, np.asarray(correction, dtype=dtype))
    if data.draw(st.booleans()):
        mean_shape = np.mean(values, axis=axis, keepdims=True).shape
        inputs = (*inputs, data.draw(hnp.arrays(dtype, mean_shape, elements=_DYADIC)))

    def reduce(array: Any, *controls: Any) -> Any:
        options = dict(kwargs)
        remaining = list(controls)
        if correction_form == "traced":
            options["correction"] = remaining.pop(0)
        if remaining:
            options["mean"] = remaining.pop(0)
        return function(array, **options)

    _assert_lifetime_parity(reduce, inputs, tolerance=_tolerance(dtype))


@settings(deadline=None)
@given(
    data=st.data(),
    dtype=_DTYPES,
    form=st.sampled_from(("where-x", "where-y", "select", "concatenate", "dot", "stack")),
)
def test_python_scalar_operands_agree_across_lifetimes(
    data: st.DataObject,
    dtype: type[np.floating[Any]],
    form: str,
) -> None:
    values = data.draw(_values(dtype))
    mask = data.draw(hnp.arrays(np.bool_, values.shape))
    scalar = data.draw(st.one_of(_DYADIC, st.integers(-3, 3)))
    # where and concatenate promote a Python scalar weakly; dot and stack coerce it first.
    functions: dict[str, Callable[[Any], Any]] = {
        "where-x": lambda array: np.where(mask, array, scalar),
        "where-y": lambda array: np.where(array > 0, scalar, array),
        "select": lambda array: np.select([array > 0, mask], [array, scalar], default=scalar),
        "concatenate": lambda array: np.concatenate((array, scalar), axis=None),
        "dot": lambda array: np.dot(array, scalar),
        "stack": lambda array: np.stack((array[0, 0], scalar)),
    }
    # np.select is dynamic-only; np.where is the staged spelling.
    _assert_lifetime_parity(
        functions[form],
        (values,),
        tolerance=_tolerance(dtype),
        staged=form != "select",
    )


@settings(deadline=None)
@given(data=st.data(), dtype=_DTYPES)
def test_average_agrees_across_lifetimes(
    data: st.DataObject,
    dtype: type[np.floating[Any]],
) -> None:
    values = data.draw(_values(dtype))
    axis = data.draw(_AXES)
    kwargs: dict[str, Any] = {
        "axis": axis,
        "keepdims": data.draw(st.booleans()),
        "returned": data.draw(st.booleans()),
    }
    weight_shapes = [None, values.shape]
    if axis is not None:
        axes = axis if isinstance(axis, tuple) else (axis,)
        weight_shapes.append(tuple(values.shape[item] for item in axes))
    weight_shape = data.draw(st.sampled_from(weight_shapes))
    if weight_shape is None:
        _assert_lifetime_parity(
            lambda array: np.average(array, **kwargs),
            (values,),
            tolerance=_tolerance(dtype),
        )
        return
    weights = data.draw(hnp.arrays(dtype, weight_shape, elements=_POSITIVE_DYADIC))
    _assert_lifetime_parity(
        lambda array, weight: np.average(array, weights=weight, **kwargs),
        (values, weights),
        tolerance=_tolerance(dtype),
    )


@st.composite
def _invertible_matrices(draw: st.DrawFn) -> np.ndarray[Any, Any]:
    dtype = draw(_DTYPES)
    size = draw(st.integers(1, 3))
    matrix = draw(hnp.arrays(dtype, (size, size), elements=_DYADIC.map(lambda v: v / 8)))
    signs = draw(hnp.arrays(dtype, (size,), elements=st.sampled_from((-1.0, 1.0))))
    # A dominant diagonal keeps every negative power well conditioned.
    matrix[np.diag_indices(size)] = signs * (size + 1)
    return matrix


_DOMINANT = np.array([[3.0, 0.125], [-0.25, -3.0]])


@settings(deadline=None)
@given(
    matrix=_invertible_matrices(),
    exponent=st.one_of(
        st.integers(-3, 3),
        st.booleans(),
        st.integers(-3, 3).map(np.int64),
        st.integers(-3, 3).map(np.array),
    ),
)
# NumPy reads the exponent with operator.index; staging once rejected the
# first two spellings, and the dynamic lifetime a 0-d integer array.
@example(matrix=_DOMINANT, exponent=False)
@example(matrix=_DOMINANT.astype(np.float32), exponent=np.int64(-2))
@example(matrix=_DOMINANT, exponent=np.array(3))
def test_matrix_power_agrees_across_lifetimes(
    matrix: np.ndarray[Any, Any],
    exponent: object,
) -> None:
    tolerance = 1e-5 if matrix.dtype == np.float32 else 1e-12
    _assert_lifetime_parity(
        lambda array: np.linalg.matrix_power(array, exponent),
        (matrix,),
        tolerance=tolerance,
    )


@settings(deadline=None)
@given(
    data=st.data(),
    dtype=_DTYPES,
    function=st.sampled_from((np.cumsum, np.cumprod, *_INITIAL_SCANS)),
)
def test_cumulative_scans_agree_across_lifetimes(
    data: st.DataObject,
    dtype: type[np.floating[Any]],
    function: Callable[..., Any],
) -> None:
    # A 0-d input scans as a vector, and cumsum and cumprod flatten for axis=None.
    rank = data.draw(st.integers(0, 2))
    shape = data.draw(_SHAPES)[2 - rank :]
    values = data.draw(_values(dtype, shape=shape))
    flattens = function in {np.cumsum, np.cumprod}
    kwargs: dict[str, Any] = {} if flattens else {"include_initial": data.draw(st.booleans())}
    if (rank == 2 and not flattens) or data.draw(st.booleans()):
        kwargs["axis"] = data.draw(st.integers(-max(rank, 1), max(rank, 1) - 1))
    if data.draw(st.booleans()):
        kwargs["dtype"] = np.float64
    _assert_lifetime_parity(
        lambda array: function(array, **kwargs),
        (values,),
        tolerance=_tolerance(dtype),
    )


@pytest.mark.parametrize("function", _INITIAL_SCANS)
@pytest.mark.parametrize("axis", [None, 0, -1])
def test_cumulative_scans_include_the_initial_of_a_0d_input(
    function: Callable[..., Any],
    axis: int | None,
) -> None:
    # NumPy scans a 0-d input as a one-element vector.
    _assert_lifetime_parity(
        lambda array: function(array, axis=axis, include_initial=True),
        (np.asarray(1.5),),
        tolerance=0,
    )


@pytest.mark.parametrize("function", _INITIAL_SCANS)
def test_cumulative_initial_requires_an_axis_beyond_one_dimension(
    function: Callable[..., Any],
) -> None:
    _assert_lifetimes_raise(
        lambda array: function(array, include_initial=True),
        (_INTEGERS.astype(np.float64),),
        ValueError,
        "For arrays which have more than one dimension ``axis`` argument is required",
    )


@settings(deadline=None)
@given(data=st.data(), dtype=_DTYPES, edge_order=st.sampled_from((1, 2)))
def test_gradient_agrees_across_lifetimes(
    data: st.DataObject,
    dtype: type[np.floating[Any]],
    edge_order: int,
) -> None:
    shape = data.draw(st.tuples(st.integers(3, 4), st.integers(3, 5)))
    values = data.draw(_values(dtype, shape=shape))
    axis = data.draw(st.sampled_from((None, 0, 1, -1, (0, 1), (1, 0))))
    axes = (0, 1) if axis is None else axis if isinstance(axis, tuple) else (axis,)
    spacing_form = data.draw(
        st.sampled_from(("default", "python", "numpy", "coordinates", "sequences"))
    )
    # Spacings may be wider than the values; NumPy keeps the values' dtype.
    spacing_dtype = data.draw(_DTYPES)
    scalar = data.draw(_POSITIVE_DYADIC)
    coordinates = tuple(
        np.cumsum(data.draw(hnp.arrays(spacing_dtype, (shape[item],), elements=_POSITIVE_DYADIC)))
        for item in axes
    )
    inputs = (values, *coordinates) if spacing_form == "coordinates" else (values,)

    def gradient(array: Any, *traced: Any) -> Any:
        spacings = {
            "default": (),
            "python": (scalar,),
            "numpy": (np.dtype(spacing_dtype).type(scalar),),
            "coordinates": traced,
            "sequences": tuple(coordinate.tolist() for coordinate in coordinates),
        }[spacing_form]
        return np.gradient(array, *spacings, axis=axis, edge_order=edge_order)

    # Traced coordinates always take the coordinate formula, in the coordinates'
    # own precision, where NumPy takes its uniform one for evenly spaced concrete
    # coordinates.  The two round differently by a few ulp of the largest formula
    # term, which reaches 2 / h * max|f| = 16 * max|f| for h >= 1/8.
    traced = spacing_form == "coordinates"
    precision = spacing_dtype if traced else dtype
    tolerance = max(_tolerance(dtype), _tolerance(precision)) * (16 if traced else 1)
    _assert_lifetime_parity(gradient, inputs, tolerance=tolerance)


def test_staged_float32_values_promote_beside_a_numpy_float64_scalar() -> None:
    _assert_lifetime_parity(
        lambda array: array * np.float64(0.1),
        (np.array([1.0, 2.5], dtype=np.float32),),
        tolerance=_tolerance(np.float64),
    )


@pytest.mark.parametrize(
    "spacing",
    [np.float64(0.625), [0.0, 1.0, 3.0, 4.0], np.array([0, 1, 3, 4]), range(0, 8, 2)],
    ids=("numpy-scalar", "sequence", "integer-coordinates", "range"),
)
def test_gradient_keeps_the_values_dtype_for_wider_spacings(spacing: object) -> None:
    values = np.array([[1.0, 2.5, 2.0, 4.0], [0.5, 0.25, 3.0, 1.0]], dtype=np.float32)

    _assert_lifetime_parity(
        lambda array: np.gradient(array, spacing, axis=1, edge_order=2),
        (values,),
        tolerance=_tolerance(values.dtype),
    )


@settings(deadline=None)
@given(
    data=st.data(),
    dtype=_DTYPES,
    ufunc=st.sampled_from((np.add, np.multiply)),
    method=st.sampled_from(("reduce", "accumulate", "outer")),
)
def test_ufunc_methods_agree_across_lifetimes(
    data: st.DataObject,
    dtype: type[np.floating[Any]],
    ufunc: np.ufunc,
    method: str,
) -> None:
    values = data.draw(_values(dtype))
    if method == "outer":
        other = data.draw(_values(dtype))
        _assert_lifetime_parity(
            ufunc.outer,
            (values, other),
            tolerance=_tolerance(dtype),
        )
        return
    kwargs: dict[str, Any] = {}
    if data.draw(st.booleans()):
        kwargs["axis"] = data.draw(st.integers(-2, 1))
    if data.draw(st.booleans()):
        kwargs["dtype"] = np.float64
    _assert_lifetime_parity(
        lambda array: getattr(ufunc, method)(array, **kwargs),
        (values,),
        tolerance=_tolerance(dtype),
    )
