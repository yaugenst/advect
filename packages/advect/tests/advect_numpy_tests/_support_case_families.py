"""Executable NumPy function cases grouped by reusable call shape."""

from __future__ import annotations

from typing import TYPE_CHECKING

from advect_numpy_tests._support_cases import (
    _COMPLEX,
    _INDEX,
    _MATRIX,
    _POSITIVE,
    _REAL,
    _RECTANGULAR,
    _RIGHT,
    ArrayInput,
    DType,
    Function,
    Input,
    _binary,
    _function,
    _unary,
    base_cases,
)

if TYPE_CHECKING:
    from advect_numpy_tests._support_cases import (
        NumpySupportCase,
    )

_VECTOR = ArrayInput([1.0, 2.0, 4.0], "float64")
_SHORT = ArrayInput([0.5, 1.5], "float64")
_SCALAR = ArrayInput(0.5, "float64")
_OTHER_SCALAR = ArrayInput(2.0, "float64")
_COMPLEX_MATRIX = ArrayInput(
    [[1.5 + 0.2j, -0.2 + 0.3j], [0.4 - 0.1j, 0.8 + 0.5j]],
    "complex128",
)
_BOOL = ArrayInput([True, False, True, False], "bool")
_NONNEGATIVE_INDEX = ArrayInput([0, 1, 0, 2], "int64")


def _algorithm_cases() -> tuple[NumpySupportCase, ...]:
    samples = ArrayInput([0.1, 0.4, 0.8, 0.2], "float64")
    y_samples = ArrayInput([0.2, 0.7, 0.9, 0.6], "float64")
    weights = ArrayInput([1.0, 2.0, 3.0, 0.5], "float64")
    edges = [0.0, 0.5, 1.0]
    edge_rows = ArrayInput([edges, [0.0, 0.25, 1.0]], "float64")
    return (
        _function("apply_along_axis", (_MATRIX,), (Function("sum"), 1, Input(0))),
        _function("apply_over_axes", (_MATRIX,), (Function("sum"), Input(0), (0,))),
        _function(
            "ravel_multi_index",
            (
                ArrayInput([0, 1], "int64"),
                ArrayInput([1, 2], "int64"),
            ),
            ((Input(0), Input(1)), (2, 3)),
        ),
        _function("unravel_index", (ArrayInput([1, 5], "int64"),), (Input(0), (2, 3))),
        _function("arange", (_REAL,), (4,), (("dtype", DType("float64")), ("like", Input(0)))),
        _function("block", (_SHORT, _SHORT), ([Input(0), Input(1)],)),
        _function("logspace", (_SCALAR, _OTHER_SCALAR), (Input(0), Input(1), 5)),
        _function("geomspace", (_SCALAR, _OTHER_SCALAR), (Input(0), Input(1), 5)),
        _unary("unstack", _MATRIX),
        _function("lib.stride_tricks.sliding_window_view", (_REAL,), (Input(0), 2)),
        _function(
            "lib.stride_tricks.sliding_window_view",
            (_MATRIX,),
            (Input(0), (2, 2)),
            (("axis", (0, 1)),),
            variant="axis-tuple",
        ),
        _unary("sort_complex"),
        _unary("unwrap", ArrayInput([0.0, 2.8, -2.8, 0.2], "float64")),
        _unary("real_if_close", ArrayInput([1.0 + 1e-15j, 2.0 - 1e-15j], "complex128")),
        _unary("i0"),
        _function(
            "bincount",
            (_NONNEGATIVE_INDEX, weights),
            (Input(0),),
            (("weights", Input(1)), ("minlength", 4)),
        ),
        _function("insert", (_REAL, _SHORT), (Input(0), [1, 3], Input(1))),
        _function(
            "insert",
            (_MATRIX, ArrayInput([9.0, 10.0], "float64")),
            (Input(0), 1, Input(1)),
            (("axis", 0),),
            variant="axis-row",
        ),
        _function(
            "histogram",
            (samples, weights),
            (Input(0),),
            (("bins", edges), ("weights", Input(1)), ("density", True)),
            derivative_argnums=((1,),),
        ),
        _function("histogram_bin_edges", (samples,), (Input(0),), (("bins", 3),)),
        _function(
            "histogram",
            (samples, _SCALAR, _OTHER_SCALAR),
            (Input(0),),
            (("bins", 3), ("range", (Input(1), Input(2)))),
            derivative_argnums=((0,), (0, 1), (0, 2), (0, 1, 2)),
            variant="integer-bins-traced-range",
        ),
        _function(
            "histogram_bin_edges",
            (samples, _SCALAR, _OTHER_SCALAR),
            (Input(0),),
            (("bins", 3), ("range", (Input(1), Input(2)))),
            derivative_argnums=((0,), (0, 1), (0, 2), (0, 1, 2)),
            variant="traced-range",
        ),
        _function(
            "histogram2d",
            (samples, y_samples, weights),
            (Input(0), Input(1)),
            (("bins", (edges, edges)), ("weights", Input(2)), ("density", True)),
            derivative_argnums=((2,),),
        ),
        _function(
            "histogram2d",
            (samples, y_samples),
            (Input(0), Input(1)),
            (("bins", 3),),
            variant="unweighted-integer-bins",
        ),
        _function(
            "histogram2d",
            (samples, y_samples, edge_rows),
            (Input(0), Input(1)),
            (("bins", Input(2)),),
            derivative_argnums=((2,),),
            variant="ndarray-edge-rows",
        ),
        _function(
            "histogramdd",
            (ArrayInput([[0.1, 0.2], [0.4, 0.7], [0.8, 0.9], [0.2, 0.6]], "float64"), weights),
            (Input(0),),
            (("bins", (edges, edges)), ("weights", Input(1)), ("density", True)),
            derivative_argnums=((1,),),
        ),
        _function(
            "histogramdd",
            (samples, y_samples),
            ((Input(0), Input(1)),),
            (("bins", 3), ("range", ((0.0, 1.0), (0.0, 1.0)))),
            variant="column-sequence",
        ),
    )


def _creation_and_alias_cases() -> tuple[NumpySupportCase, ...]:
    constructors = tuple(
        _function(name, (_REAL,), (Input(0),), (("like", Input(0)),))
        for name in ("array", "asarray", "asanyarray")
    )
    return (
        *constructors,
        _function("identity", (_REAL,), (3,), (("dtype", DType("float64")), ("like", Input(0)))),
        _function("tri", (_REAL,), (3, 4), (("dtype", DType("float64")), ("like", Input(0)))),
        _unary("cumulative_sum", _REAL, (("axis", 0),)),
        _unary("cumulative_prod", _POSITIVE, (("axis", 0),)),
        _binary(
            "linalg.cross",
            ArrayInput([[1.0, 2.0, 3.0]], "float64"),
            ArrayInput([[3.0, 1.0, 2.0]], "float64"),
        ),
        _binary("linalg.matmul", _MATRIX, _MATRIX),
        _binary("linalg.outer"),
        _function("linalg.tensordot", (_MATRIX, _MATRIX), (Input(0), Input(1)), (("axes", 1),)),
        _unary("linalg.matrix_norm", _MATRIX),
        _unary("linalg.vector_norm"),
        _unary("linalg.matrix_transpose", _MATRIX),
        _unary("matrix_transpose", _MATRIX),
    )


def _shape_and_stack_cases() -> tuple[NumpySupportCase, ...]:
    split_kwargs = (("axis", 0),)
    return (
        _unary("atleast_1d", _SCALAR),
        _unary("atleast_2d", _SHORT),
        _unary("atleast_3d", _MATRIX),
        _function("hstack", (_SHORT, _SHORT), ((Input(0), Input(1)),)),
        _function("vstack", (_SHORT, _SHORT), ((Input(0), Input(1)),)),
        _function(
            "row_stack",
            (_SHORT, _SHORT),
            ((Input(0), Input(1)),),
            expected_deprecation=r"`row_stack` alias is deprecated",
        ),
        _function("dstack", (_SHORT, _SHORT), ((Input(0), Input(1)),)),
        _function("column_stack", (_SHORT, _SHORT), ((Input(0), Input(1)),)),
        _function("append", (_REAL, _SHORT), (Input(0), Input(1))),
        _function("delete", (_REAL,), (Input(0), [1, 3])),
        _function("diagflat", (_REAL,), (Input(0),)),
        _function("ediff1d", (_REAL,), (Input(0),)),
        _function("resize", (_REAL,), (Input(0), (2, 3))),
        _function("meshgrid", (_SHORT, _VECTOR), (Input(0), Input(1))),
        _function("broadcast_arrays", (_MATRIX, _SHORT), (Input(0), Input(1))),
        _function("array_split", (_MATRIX,), (Input(0), 2), split_kwargs),
        _function("split", (_MATRIX,), (Input(0), 2), split_kwargs),
        _function("hsplit", (_MATRIX,), (Input(0), 2)),
        _function("vsplit", (_MATRIX,), (Input(0), 2)),
        _function(
            "dsplit",
            (ArrayInput([[[1.0, 2.0], [3.0, 4.0]]], "float64"),),
            (Input(0), 2),
        ),
        _unary("rollaxis", _MATRIX, (("axis", 1),)),
        _unary("repeat", _REAL, (("repeats", 2),)),
        _function("tile", (_MATRIX,), (Input(0), (2, 1))),
        _unary("triu", _MATRIX),
        _unary("diag", _MATRIX),
        _unary("diagonal", _MATRIX),
        _unary("imag", _COMPLEX),
        _function("empty_like", (_MATRIX,), (Input(0),), compare_values=False),
    )


def _composite_cases() -> tuple[NumpySupportCase, ...]:
    condition = ArrayInput([True, False, True, False], "bool")
    complement = [False, True, False, True]
    nan_values = ArrayInput([1.0, float("nan"), 3.0, 5.0], "float64")
    nan_matrix = ArrayInput(
        [[1.0, float("nan"), 2.0], [3.0, 4.0, float("nan")]],
        "float64",
    )
    covariance_matrix = ArrayInput(
        [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [5.0, 3.0]],
        "float64",
    )
    covariance_other = ArrayInput([[0.5], [1.5], [2.5], [4.0]], "float64")
    frequency_weights = ArrayInput([1, 2, 1, 2], "int64")
    analytic_weights = ArrayInput([1.0, 0.5, 2.0, 1.5], "float64")
    return (
        _function("average", (_REAL, _POSITIVE), (Input(0),), (("weights", Input(1)),)),
        _function(
            "average",
            (_MATRIX,),
            (Input(0),),
            (("axis", (0, 1)), ("keepdims", True), ("returned", True)),
            variant="unweighted-returned",
        ),
        _function(
            "average",
            (_MATRIX, _SHORT),
            (Input(0),),
            (
                ("axis", 1),
                ("weights", Input(1)),
                ("keepdims", True),
                ("returned", True),
            ),
            variant="axis-weights-returned",
        ),
        _unary("ptp"),
        _function("trapezoid", (_REAL, _POSITIVE), (Input(0),), (("x", Input(1)),)),
        _function(
            "trapezoid",
            (_MATRIX, _SHORT),
            (Input(0),),
            (("x", Input(1)), ("axis", 1)),
            variant="matrix-vector-coordinates",
        ),
        _function(
            "trapezoid",
            (_MATRIX, _SCALAR),
            (Input(0),),
            (("dx", Input(1)), ("axis", 0)),
            variant="traced-spacing",
        ),
        _unary("nancumsum", nan_values),
        _unary("nancumprod", ArrayInput([1.0, float("nan"), 2.0, 3.0], "float64")),
        _unary(
            "nancumsum",
            nan_matrix,
            (("axis", 1), ("dtype", DType("float32"))),
            variant="axis-dtype",
        ),
        _unary("nancumprod", nan_matrix, (("axis", 0),), variant="axis"),
        _unary("round", stages=True),
        _unary("round", _REAL, (("decimals", 1),), variant="decimals"),
        _unary("around", _REAL, (("decimals", 1),)),
        _function("fix", (_REAL,), (Input(0),), expected_deprecation=r"numpy\.fix is deprecated"),
        _binary("vdot", _COMPLEX, _COMPLEX),
        _function(
            "linalg.multi_dot", (_MATRIX, _MATRIX, _MATRIX), ((Input(0), Input(1), Input(2)),)
        ),
        _function(
            "select",
            (_REAL, _RIGHT),
            (
                (condition.data, complement),
                (Input(0), Input(1)),
                0.0,
            ),
        ),
        _function("piecewise", (_REAL, condition), (Input(0), (Input(1),), (2.0, -1.0))),
        _function(
            "piecewise",
            (_REAL, condition),
            (Input(0), [Input(1)], [Function("negative"), 2.0]),
            variant="callable-branch",
        ),
        _function(
            "choose", (_INDEX, _REAL, _RIGHT), (Input(0), (Input(1), Input(2))), (("mode", "clip"),)
        ),
        _function(
            "choose",
            (ArrayInput([-1, 0, 3, 1], "int64"), _REAL),
            (Input(0), (Input(1), 4.0, -2.0)),
            (("mode", "wrap"),),
            derivative_argnums=((1,),),
            variant="wrap-mixed-choices",
        ),
        _function("compress", (_REAL,), (condition.data, Input(0))),
        _function("extract", (condition, _REAL), (Input(0), Input(1))),
        _function("vander", (_REAL,), (Input(0),), (("N", 4),)),
        _unary("cov", _MATRIX),
        _function(
            "cov",
            (covariance_matrix, frequency_weights),
            (Input(0),),
            (("rowvar", False), ("fweights", Input(1))),
            variant="frequency-weighted",
        ),
        _function(
            "cov",
            (covariance_matrix, covariance_other, analytic_weights, frequency_weights),
            (Input(0),),
            (
                ("y", Input(1)),
                ("rowvar", False),
                ("ddof", 1),
                ("fweights", Input(3)),
                ("aweights", Input(2)),
                ("dtype", DType("float64")),
            ),
            variant="combined-weights",
        ),
        _unary("corrcoef", _MATRIX),
        _function(
            "corrcoef",
            (covariance_matrix, covariance_other),
            (Input(0),),
            (("y", Input(1)), ("rowvar", False), ("dtype", DType("float64"))),
            variant="additional-columns",
        ),
        _unary(
            "corrcoef",
            ArrayInput([[2.0, 1.0 + 0.2j], [1.0 - 0.2j, 3.0]], "complex128"),
            variant="complex",
        ),
    )


def _ordering_and_predicate_cases() -> tuple[NumpySupportCase, ...]:
    membership_values = ArrayInput([0.5, 2.0], "float64")
    return (
        _function("argpartition", (_REAL,), (Input(0), 2)),
        _unary("argwhere"),
        _unary("flatnonzero"),
        _unary("nonzero"),
        _function("digitize", (_REAL, _POSITIVE), (Input(0), Input(1))),
        _function("lexsort", (_REAL, _RIGHT), ((Input(0), Input(1)),)),
        _function("lexsort", (_MATRIX,), (Input(0),), variant="key-matrix"),
        _function("isin", (_REAL, membership_values), (Input(0), Input(1))),
        _function("ix_", (_SHORT, _VECTOR), (Input(0), Input(1))),
        _binary("setdiff1d"),
        _binary("intersect1d"),
        _binary("setxor1d"),
        _binary("union1d"),
        _unary("trim_zeros", ArrayInput([0.0, 1.0, 2.0, 0.0], "float64")),
        _unary("diag_indices_from", _MATRIX),
        _unary("tril_indices_from", _MATRIX),
        _unary("triu_indices_from", _MATRIX),
        _function("nanargmin", (ArrayInput([2.0, float("nan"), -1.0], "float64"),), (Input(0),)),
        _function("nanargmax", (ArrayInput([2.0, float("nan"), -1.0], "float64"),), (Input(0),)),
        _binary("isclose"),
        _binary("allclose"),
        _binary("array_equal"),
        _binary("array_equiv"),
        _unary("iscomplex", _COMPLEX),
        _unary("isreal", _COMPLEX),
        _unary("isposinf", ArrayInput([1.0, float("inf"), -2.0], "float64")),
        _unary("isneginf", ArrayInput([1.0, float("-inf"), -2.0], "float64")),
    )


def _polynomial_cases() -> tuple[NumpySupportCase, ...]:
    coefficients = ArrayInput([1.0, -2.0, 0.5], "float64")
    other = ArrayInput([0.5, 1.0], "float64")
    coordinates = ArrayInput([0.0, 1.0, 2.0, 3.0], "float64")
    observations = ArrayInput([0.2, 1.1, 3.8, 9.2], "float64")
    weights = ArrayInput([1.0, 0.5, 2.0, 1.5], "float64")
    return (
        _unary("poly", _VECTOR),
        _binary("polyadd", coefficients, other),
        _binary("polysub", coefficients, other),
        _binary("polymul", coefficients, other),
        _binary("polydiv", coefficients, other),
        _function("polyfit", (coordinates, observations), (Input(0), Input(1), 2)),
        _function(
            "polyfit",
            (coordinates, observations, weights),
            (Input(0), Input(1), 2),
            (("w", Input(2)), ("cov", True)),
            variant="weighted-covariance",
        ),
        _function(
            "polyfit",
            (coordinates, observations, weights),
            (Input(0), Input(1), 2),
            (("w", Input(2)), ("cov", "unscaled")),
            variant="weighted-unscaled-covariance",
        ),
        _function("polyder", (coefficients,), (Input(0),), (("m", 1),)),
        _function("polyint", (coefficients,), (Input(0),), (("m", 1),)),
        _function("polyval", (coefficients, _REAL), (Input(0), Input(1))),
        _unary("roots", coefficients),
        _unary("roots", ArrayInput([1.0, 0.25, 1.0], "float64"), variant="complex-output"),
    )


def _statistics_and_unique_cases() -> tuple[NumpySupportCase, ...]:
    quantiles = ArrayInput([0.25, 0.75], "float64")
    quantile_weights = ArrayInput([1.0, 4.0, 2.0, 1.0], "float64")
    nan_values = ArrayInput([0.0, float("nan"), 2.0, 5.0], "float64")
    quantile_cases = tuple(
        _function(name, (_REAL, quantiles), (Input(0), Input(1)), (("method", "linear"),))
        for name in ("quantile", "nanquantile")
    )
    percentile_coordinates = ArrayInput([25.0, 75.0], "float64")
    percentile_weights = ArrayInput([[1.0, 2.0], [3.0, 4.0]], "float64")
    percentile_cases = (
        _function(
            "percentile",
            (_REAL, percentile_coordinates),
            (Input(0), Input(1)),
            (("method", "linear"),),
        ),
        _function(
            "nanpercentile",
            (nan_values, percentile_coordinates),
            (Input(0), Input(1)),
            (("method", "linear"),),
        ),
    )
    unique_values = ArrayInput([2.0, 1.0, 2.0, 3.0], "float64")
    return (
        *quantile_cases,
        *percentile_cases,
        _function(
            "quantile",
            (_REAL, quantiles, quantile_weights),
            (Input(0), Input(1)),
            (("method", "inverted_cdf"), ("weights", Input(2))),
            variant="weighted",
        ),
        _function(
            "percentile",
            (_MATRIX, percentile_coordinates, percentile_weights),
            (Input(0), Input(1)),
            (
                ("axis", 1),
                ("keepdims", True),
                ("method", "inverted_cdf"),
                ("weights", Input(2)),
            ),
            variant="weighted-axis-keepdims",
        ),
        _unary("median"),
        _unary("nanmedian", nan_values),
        _unary("unique", unique_values),
        _unary("unique_values", unique_values),
        _unary("unique_all", unique_values),
        _unary("unique_counts", unique_values),
        _unary("unique_inverse", unique_values),
    )


def _scientific_cases() -> tuple[NumpySupportCase, ...]:
    fft_real = ArrayInput([[0.0, 1.0], [2.0, 3.0]], "float64")
    fft_complex = ArrayInput([[0.0 + 0.5j, 1.0 - 0.25j], [2.0 + 1.0j, 3.0 - 0.5j]], "complex128")
    half_spectrum = ArrayInput(
        [[1.0 + 0.0j, 0.5 - 0.25j], [2.0 + 0.0j, -0.5 + 0.75j]],
        "complex128",
    )
    return (
        _unary("angle", _COMPLEX),
        _unary("sinc"),
        _unary("amax"),
        _unary("amin"),
        _binary("convolve"),
        _binary("correlate"),
        _function("einsum", (_MATRIX, _MATRIX), ("ij,ij->", Input(0), Input(1))),
        _function(
            "einsum",
            (_MATRIX, _MATRIX),
            (Input(0), [0, 1], Input(1), [0, 1], []),
            variant="sublist",
        ),
        _unary("fft.fft2", fft_complex),
        _unary("fft.ifft2", fft_complex),
        _unary("fft.rfft2", fft_real),
        _unary("fft.irfft2", half_spectrum),
        _function(
            "linspace",
            (_SCALAR, _OTHER_SCALAR),
            (Input(0), Input(1), 6),
            (("dtype", DType("float64")),),
        ),
        _function(
            "linspace",
            (_SCALAR, _OTHER_SCALAR),
            (Input(0), Input(1), 6),
            (("dtype", DType("float64")), ("endpoint", False), ("retstep", True)),
            variant="retstep",
        ),
        _function("interp", (_SHORT, _VECTOR, _VECTOR), (Input(0), Input(1), Input(2))),
        _function(
            "interp",
            (_SHORT, _VECTOR, _VECTOR, _SCALAR, _OTHER_SCALAR),
            (Input(0), Input(1), Input(2)),
            (("left", Input(3)), ("right", Input(4))),
            derivative_argnums=((0,), (1,), (2,), (2, 3), (2, 4), (2, 3, 4), (0, 1, 2, 3, 4)),
            variant="fill-values",
        ),
        _function(
            "pad",
            (_REAL,),
            (Input(0), (2, 1)),
            (("mode", "constant"), ("constant_values", (1.5, -0.5))),
        ),
        _function(
            "pad",
            (_REAL,),
            (Input(0), (2, 1)),
            (("mode", "reflect"), ("reflect_type", "odd")),
            variant="reflect-odd",
        ),
    )


def _linalg_cases() -> tuple[NumpySupportCase, ...]:
    tensor = ArrayInput(
        [
            [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]],
            [[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]],
        ],
        "float64",
    )
    return (
        _unary("linalg.pinv", _MATRIX, (("hermitian", True),), variant="hermitian"),
        _unary("linalg.cond", _MATRIX),
        _unary("linalg.matrix_rank", _RECTANGULAR),
        _function(
            "linalg.lstsq",
            (_RECTANGULAR, ArrayInput([1.0, 2.0, 3.0], "float64")),
            (Input(0), Input(1)),
            (("rcond", None),),
        ),
        _unary("linalg.qr", _RECTANGULAR, result_adapter="fields"),
        _unary("linalg.qr", _RECTANGULAR, (("mode", "r"),), variant="mode-r"),
        _unary("linalg.slogdet", _MATRIX, result_adapter="fields"),
        _unary("linalg.eig", _COMPLEX_MATRIX, stages=True, result_adapter="fields"),
        _unary("linalg.eigvals", _COMPLEX_MATRIX, stages=True),
        _unary("linalg.eig", _MATRIX, variant="real-dynamic", result_adapter="fields"),
        _unary("linalg.eigvals", _MATRIX, variant="real-dynamic"),
        _function("linalg.tensorinv", (tensor,), (Input(0),), (("ind", 2),)),
        _function("linalg.tensorsolve", (tensor, _MATRIX), (Input(0), Input(1))),
    )


def _scimath_cases() -> tuple[NumpySupportCase, ...]:
    negative = ArrayInput([-4.0, -0.5, 0.25, 2.0], "float64")
    unary = tuple(
        _unary(f"lib.scimath.{name}", negative)
        for name in ("sqrt", "log", "log10", "log2", "arcsin", "arccos", "arctanh")
    )
    return (
        *unary,
        _function("lib.scimath.logn", (_POSITIVE, negative), (Input(0), Input(1))),
        _binary("lib.scimath.power", negative, _RIGHT),
    )


def _composite_spelling_cases() -> tuple[NumpySupportCase, ...]:
    """Qualify composite spellings whose lowering takes a distinct branch."""
    grid = ArrayInput([[1.0, 2.0, 4.0, 7.0], [3.0, 5.0, 6.0, 9.0], [2.0, 8.0, 3.0, 4.0]], "float64")
    wide = ArrayInput([[0.0, 1.0, 3.0], [2.0, 5.0, 4.0]], "float64")
    edges = ArrayInput([[10.0], [20.0]], "float64")
    return (
        *(
            _function("linalg.matrix_power", (_MATRIX,), (Input(0), exponent), variant=variant)
            for exponent, variant in ((-2, "negative"), (0, "zero"))
        ),
        _unary("size", _MATRIX, (("axis", 1),), variant="axis", result_adapter="array"),
        *(
            case
            for name, value in (("cumulative_sum", _REAL), ("cumulative_prod", _POSITIVE))
            for case in (
                _unary(
                    name,
                    ArrayInput(value.data, "float32"),
                    (("dtype", DType("float64")), ("include_initial", True)),
                    variant="include-initial",
                ),
                _unary(
                    name, wide, (("axis", -1), ("include_initial", True)), variant="initial-axis"
                ),
            )
        ),
        _function(
            "average", (_MATRIX, _SHORT), (Input(0), 1, Input(1), True), variant="positional"
        ),
        _function(
            "var",
            (wide, ArrayInput([[1.0], [3.0]], "float64"), ArrayInput(1.0, "float64")),
            (Input(0),),
            (("axis", 1), ("mean", Input(1)), ("correction", Input(2)), ("keepdims", True)),
            variant="live-mean-correction",
            # NumPy dispatches var on its data and mean, never on correction alone.
            derivative_argnums=((0,), (1,), (0, 1), (0, 2), (0, 1, 2)),
        ),
        _function(
            "gradient",
            (grid,),
            (Input(0), [0.0, 1.0, 3.0], [0.0, 0.5, 2.0, 5.0]),
            (("axis", (0, 1)), ("edge_order", 2)),
            variant="coordinates",
        ),
        _function(
            "gradient",
            (grid, ArrayInput([0.0, 0.5, 2.0, 5.0], "float64")),
            (Input(0), Input(1)),
            (("axis", 1), ("edge_order", 2)),
            variant="traced-coordinates",
        ),
        _function(
            "gradient",
            (grid,),
            (Input(0), 2.0),
            (("axis", (0, 1)), ("edge_order", 2)),
            variant="scalar-spacing",
        ),
        _function(
            "diff",
            (wide, edges, edges),
            (Input(0),),
            (("n", 2), ("axis", 1), ("prepend", Input(1)), ("append", Input(2))),
            variant="live-boundaries",
        ),
        _function(
            "compress", (_MATRIX,), ([True, False], Input(0)), (("axis", 0),), variant="axis"
        ),
        _function("compress", (_MATRIX,), ([True, False], Input(0), 0, None), variant="positional"),
        _function(
            "compress",
            (_MATRIX,),
            ([True, False], Input(0)),
            (("axis", 0), ("out", None)),
            variant="out-none",
        ),
    )


def support_cases() -> tuple[NumpySupportCase, ...]:
    """Return the complete executable NumPy lifetime contract."""
    cases = (
        *base_cases(),
        *_algorithm_cases(),
        *_creation_and_alias_cases(),
        *_shape_and_stack_cases(),
        *_composite_cases(),
        *_ordering_and_predicate_cases(),
        *_polynomial_cases(),
        *_statistics_and_unique_cases(),
        *_scientific_cases(),
        *_linalg_cases(),
        *_scimath_cases(),
        *_composite_spelling_cases(),
    )
    identifiers = [case.identifier for case in cases]
    if len(identifiers) != len(set(identifiers)):
        duplicates = sorted(
            identifier for identifier in set(identifiers) if identifiers.count(identifier) > 1
        )
        message = f"duplicate NumPy support cases: {duplicates}"
        raise RuntimeError(message)
    return tuple(sorted(cases, key=lambda case: case.identifier))


__all__ = ["support_cases"]
