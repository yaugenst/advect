"""Executable NumPy function cases grouped by reusable call shape."""

from __future__ import annotations

from typing import Literal

from advect_numpy_tests._support_cases import (
    ArrayInput,
    DType,
    Function,
    Input,
    NumpySupportCase,
)

_ALL_MODES = ("dynamic", "staged", "serialized")
_DYNAMIC_ONLY = ("dynamic",)

_REAL = ArrayInput([-1.5, -0.25, 0.5, 2.0], "float64")
_RIGHT = ArrayInput([0.75, 1.5, 2.0, 0.5], "float64")
_POSITIVE = ArrayInput([0.25, 0.5, 1.5, 3.0], "float64")
_VECTOR = ArrayInput([1.0, 2.0, 4.0], "float64")
_SHORT = ArrayInput([0.5, 1.5], "float64")
_SCALAR = ArrayInput(0.5, "float64")
_OTHER_SCALAR = ArrayInput(2.0, "float64")
_MATRIX = ArrayInput([[4.0, 1.0], [1.0, 3.0]], "float64")
_RECTANGULAR = ArrayInput([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]], "float64")
_COMPLEX = ArrayInput([1.0 + 0.5j, -2.0 + 1.0j, 0.25 - 0.75j], "complex128")
_COMPLEX_MATRIX = ArrayInput(
    [[1.5 + 0.2j, -0.2 + 0.3j], [0.4 - 0.1j, 0.8 + 0.5j]],
    "complex128",
)
_BOOL = ArrayInput([True, False, True, False], "bool")
_INDEX = ArrayInput([2, 0, 2, 1], "int64")
_NONNEGATIVE_INDEX = ArrayInput([0, 1, 0, 2], "int64")


def _case(
    path: str,
    inputs: tuple[ArrayInput, ...],
    args: tuple[object, ...],
    derivative_argnums: tuple[tuple[int, ...], ...] | None,
    kwargs: tuple[tuple[str, object], ...] = (),
    *,
    modes: tuple[str, ...] = _DYNAMIC_ONLY,
    variant: str = "baseline",
    compare_values: bool = True,
    trace_argnums: tuple[int, ...] | None = None,
    result_adapter: Literal["identity", "array", "dtype_num", "tuple"] = "identity",
    expected_deprecation: str | None = None,
) -> NumpySupportCase:
    return NumpySupportCase(
        callable=f"numpy.{path}",
        kind="function",
        inputs=inputs,
        args=args,
        derivative_argnums=derivative_argnums,
        kwargs=kwargs,
        modes=modes,
        variant=variant,
        compare_values=compare_values,
        trace_argnums=trace_argnums,
        result_adapter=result_adapter,
        expected_deprecation=expected_deprecation,
    )


def _unary(
    path: str,
    value: ArrayInput = _REAL,
    *,
    derivative: bool = True,
    modes: tuple[str, ...] = _DYNAMIC_ONLY,
    kwargs: tuple[tuple[str, object], ...] = (),
    variant: str = "baseline",
    result_adapter: Literal["identity", "array", "dtype_num", "tuple"] = "identity",
) -> NumpySupportCase:
    return _case(
        path,
        (value,),
        (Input(0),),
        ((0,),) if derivative else None,
        kwargs,
        modes=modes,
        variant=variant,
        result_adapter=result_adapter,
    )


def _binary(
    path: str,
    left: ArrayInput = _REAL,
    right: ArrayInput = _RIGHT,
    *,
    derivative: bool = True,
    modes: tuple[str, ...] = _DYNAMIC_ONLY,
    kwargs: tuple[tuple[str, object], ...] = (),
    variant: str = "baseline",
    result_adapter: Literal["identity", "array", "dtype_num", "tuple"] = "identity",
) -> NumpySupportCase:
    groups = ((0,), (1,), (0, 1)) if derivative else None
    return _case(
        path,
        (left, right),
        (Input(0), Input(1)),
        groups,
        kwargs,
        modes=modes,
        variant=variant,
        result_adapter=result_adapter,
    )


def _algorithm_cases() -> tuple[NumpySupportCase, ...]:
    samples = ArrayInput([0.1, 0.4, 0.8, 0.2], "float64")
    y_samples = ArrayInput([0.2, 0.7, 0.9, 0.6], "float64")
    weights = ArrayInput([1.0, 2.0, 3.0, 0.5], "float64")
    edges = [0.0, 0.5, 1.0]
    return (
        _case(
            "apply_along_axis",
            (_MATRIX,),
            (Function("sum"), 1, Input(0)),
            ((0,),),
        ),
        _case(
            "apply_over_axes",
            (_MATRIX,),
            (Function("sum"), Input(0), (0,)),
            ((0,),),
        ),
        _case(
            "ravel_multi_index",
            (
                ArrayInput([0, 1], "int64"),
                ArrayInput([1, 2], "int64"),
            ),
            ((Input(0), Input(1)), (2, 3)),
            None,
        ),
        _case(
            "unravel_index",
            (ArrayInput([1, 5], "int64"),),
            (Input(0), (2, 3)),
            None,
        ),
        _case(
            "arange",
            (_REAL,),
            (4,),
            None,
            (("dtype", DType("float64")), ("like", Input(0))),
            modes=_ALL_MODES,
        ),
        _case("block", (_SHORT, _SHORT), ([Input(0), Input(1)],), ((0,), (1,), (0, 1))),
        _case("logspace", (_SCALAR, _OTHER_SCALAR), (Input(0), Input(1), 5), ((0,), (1,), (0, 1))),
        _case("geomspace", (_SCALAR, _OTHER_SCALAR), (Input(0), Input(1), 5), ((0,), (1,), (0, 1))),
        _unary("unstack", _MATRIX),
        _case(
            "lib.stride_tricks.sliding_window_view",
            (_REAL,),
            (Input(0), 2),
            ((0,),),
        ),
        _unary("sort_complex"),
        _unary("unwrap", ArrayInput([0.0, 2.8, -2.8, 0.2], "float64")),
        _unary("real_if_close", ArrayInput([1.0 + 1e-15j, 2.0 - 1e-15j], "complex128")),
        _unary("i0"),
        _case(
            "bincount",
            (_NONNEGATIVE_INDEX, weights),
            (Input(0),),
            ((1,),),
            (("weights", Input(1)), ("minlength", 4)),
        ),
        _case(
            "insert",
            (_REAL, _SHORT),
            (Input(0), [1, 3], Input(1)),
            ((0,), (1,), (0, 1)),
        ),
        _case(
            "histogram",
            (samples, weights),
            (Input(0),),
            ((1,),),
            (("bins", edges), ("weights", Input(1)), ("density", True)),
        ),
        _case(
            "histogram_bin_edges",
            (samples,),
            (Input(0),),
            ((0,),),
            (("bins", 3),),
        ),
        _case(
            "histogram2d",
            (samples, y_samples, weights),
            (Input(0), Input(1)),
            ((2,),),
            (("bins", (edges, edges)), ("weights", Input(2)), ("density", True)),
        ),
        _case(
            "histogramdd",
            (ArrayInput([[0.1, 0.2], [0.4, 0.7], [0.8, 0.9], [0.2, 0.6]], "float64"), weights),
            (Input(0),),
            ((1,),),
            (("bins", (edges, edges)), ("weights", Input(1)), ("density", True)),
        ),
    )


def _creation_and_alias_cases() -> tuple[NumpySupportCase, ...]:
    constructors = tuple(
        _case(
            name,
            (_REAL,),
            (Input(0),),
            ((0,),),
            (("like", Input(0)),),
            modes=_ALL_MODES,
        )
        for name in ("array", "asarray", "asanyarray")
    )
    return (
        *constructors,
        _case(
            "identity",
            (_REAL,),
            (3,),
            None,
            (("dtype", DType("float64")), ("like", Input(0))),
        ),
        _case(
            "tri",
            (_REAL,),
            (3, 4),
            None,
            (("dtype", DType("float64")), ("like", Input(0))),
        ),
        _unary("cumulative_sum", modes=_ALL_MODES, kwargs=(("axis", 0),)),
        _unary("cumulative_prod", _POSITIVE, modes=_ALL_MODES, kwargs=(("axis", 0),)),
        _binary(
            "linalg.cross",
            ArrayInput([[1.0, 2.0, 3.0]], "float64"),
            ArrayInput([[3.0, 1.0, 2.0]], "float64"),
            modes=_ALL_MODES,
        ),
        _binary("linalg.matmul", _MATRIX, _MATRIX, modes=_ALL_MODES),
        _binary("linalg.outer", modes=_ALL_MODES),
        _case(
            "linalg.tensordot",
            (_MATRIX, _MATRIX),
            (Input(0), Input(1)),
            ((0,), (1,), (0, 1)),
            (("axes", 1),),
            modes=_ALL_MODES,
        ),
        _unary("linalg.matrix_norm", _MATRIX, modes=_ALL_MODES),
        _unary("linalg.vector_norm", modes=_ALL_MODES),
        _unary("linalg.matrix_transpose", _MATRIX, modes=_ALL_MODES),
        _unary("matrix_transpose", _MATRIX, modes=_ALL_MODES),
    )


def _shape_and_stack_cases() -> tuple[NumpySupportCase, ...]:
    split_kwargs = (("axis", 0),)
    return (
        _unary("atleast_1d", _SCALAR),
        _unary("atleast_2d", _SHORT),
        _unary("atleast_3d", _MATRIX),
        _case("hstack", (_SHORT, _SHORT), ((Input(0), Input(1)),), ((0,), (1,), (0, 1))),
        _case("vstack", (_SHORT, _SHORT), ((Input(0), Input(1)),), ((0,), (1,), (0, 1))),
        _case(
            "row_stack",
            (_SHORT, _SHORT),
            ((Input(0), Input(1)),),
            ((0,), (1,), (0, 1)),
            expected_deprecation=r"`row_stack` alias is deprecated",
        ),
        _case("dstack", (_SHORT, _SHORT), ((Input(0), Input(1)),), ((0,), (1,), (0, 1))),
        _case("column_stack", (_SHORT, _SHORT), ((Input(0), Input(1)),), ((0,), (1,), (0, 1))),
        _case("append", (_REAL, _SHORT), (Input(0), Input(1)), ((0,), (1,), (0, 1))),
        _case("delete", (_REAL,), (Input(0), [1, 3]), ((0,),)),
        _case("diagflat", (_REAL,), (Input(0),), ((0,),)),
        _case("ediff1d", (_REAL,), (Input(0),), ((0,),)),
        _case("resize", (_REAL,), (Input(0), (2, 3)), ((0,),)),
        _case("meshgrid", (_SHORT, _VECTOR), (Input(0), Input(1)), ((0,), (1,), (0, 1))),
        _case("broadcast_arrays", (_MATRIX, _SHORT), (Input(0), Input(1)), ((0,), (1,), (0, 1))),
        _case(
            "array_split",
            (_MATRIX,),
            (Input(0), 2),
            ((0,),),
            split_kwargs,
            result_adapter="tuple",
        ),
        _case(
            "split",
            (_MATRIX,),
            (Input(0), 2),
            ((0,),),
            split_kwargs,
            result_adapter="tuple",
        ),
        _case("hsplit", (_MATRIX,), (Input(0), 2), ((0,),), result_adapter="tuple"),
        _case("vsplit", (_MATRIX,), (Input(0), 2), ((0,),), result_adapter="tuple"),
        _case(
            "dsplit",
            (ArrayInput([[[1.0, 2.0], [3.0, 4.0]]], "float64"),),
            (Input(0), 2),
            ((0,),),
            result_adapter="tuple",
        ),
        _unary("rollaxis", _MATRIX, kwargs=(("axis", 1),)),
        _unary("repeat", kwargs=(("repeats", 2),), modes=_ALL_MODES),
        _case("tile", (_MATRIX,), (Input(0), (2, 1)), ((0,),), modes=_ALL_MODES),
        _unary("triu", _MATRIX, modes=_ALL_MODES),
        _unary("diag", _MATRIX),
        _unary("diagonal", _MATRIX, modes=_ALL_MODES),
        _unary("imag", _COMPLEX, modes=_ALL_MODES),
        _case(
            "empty_like",
            (_MATRIX,),
            (Input(0),),
            None,
            modes=_ALL_MODES,
            compare_values=False,
        ),
    )


def _composite_cases() -> tuple[NumpySupportCase, ...]:
    condition = ArrayInput([True, False, True, False], "bool")
    complement = [False, True, False, True]
    nan_values = ArrayInput([1.0, float("nan"), 3.0, 5.0], "float64")
    return (
        _case(
            "average",
            (_REAL, _POSITIVE),
            (Input(0),),
            ((0,), (1,), (0, 1)),
            (("weights", Input(1)),),
            modes=_ALL_MODES,
        ),
        _unary("ptp"),
        _case(
            "trapezoid", (_REAL, _POSITIVE), (Input(0),), ((0,), (1,), (0, 1)), (("x", Input(1)),)
        ),
        _unary("nancumsum", nan_values),
        _unary("nancumprod", ArrayInput([1.0, float("nan"), 2.0, 3.0], "float64")),
        _unary("round", modes=_ALL_MODES),
        _unary("round", kwargs=(("decimals", 1),), variant="decimals"),
        _unary("around", kwargs=(("decimals", 1),)),
        _case(
            "fix",
            (_REAL,),
            (Input(0),),
            ((0,),),
            expected_deprecation=r"numpy\.fix is deprecated",
        ),
        _binary("vdot", _COMPLEX, _COMPLEX),
        _case(
            "linalg.multi_dot",
            (_MATRIX, _MATRIX, _MATRIX),
            ((Input(0), Input(1), Input(2)),),
            ((0,), (1,), (2,), (0, 1, 2)),
        ),
        _case(
            "select",
            (_REAL, _RIGHT),
            (
                (condition.data, complement),
                (Input(0), Input(1)),
                0.0,
            ),
            ((0,), (1,), (0, 1)),
        ),
        _case("piecewise", (_REAL, condition), (Input(0), (Input(1),), (2.0, -1.0)), ((0,),)),
        _case(
            "piecewise",
            (_REAL, condition),
            (Input(0), [Input(1)], [Function("negative"), 2.0]),
            ((0,),),
            variant="callable-branch",
        ),
        _case(
            "choose",
            (_INDEX, _REAL, _RIGHT),
            (Input(0), (Input(1), Input(2))),
            ((1,), (2,), (1, 2)),
            (("mode", "clip"),),
        ),
        _case(
            "compress",
            (_REAL,),
            (condition.data, Input(0)),
            ((0,),),
            modes=_ALL_MODES,
        ),
        _case("extract", (condition, _REAL), (Input(0), Input(1)), ((1,),)),
        _case("vander", (_REAL,), (Input(0),), ((0,),), (("N", 4),)),
        _unary("cov", _MATRIX),
        _unary("corrcoef", _MATRIX),
    )


def _ordering_and_predicate_cases() -> tuple[NumpySupportCase, ...]:
    membership_values = ArrayInput([0.5, 2.0], "float64")
    return (
        _case("argpartition", (_REAL,), (Input(0), 2), None),
        _unary("argwhere", derivative=False),
        _unary("flatnonzero", derivative=False),
        _unary("nonzero", derivative=False),
        _case("digitize", (_REAL, _POSITIVE), (Input(0), Input(1)), None),
        _case("lexsort", (_REAL, _RIGHT), ((Input(0), Input(1)),), None),
        _case("isin", (_REAL, membership_values), (Input(0), Input(1)), None),
        _case("ix_", (_SHORT, _VECTOR), (Input(0), Input(1)), None),
        _binary("setdiff1d"),
        _binary("intersect1d"),
        _binary("setxor1d"),
        _binary("union1d"),
        _unary("trim_zeros", ArrayInput([0.0, 1.0, 2.0, 0.0], "float64")),
        _unary("diag_indices_from", _MATRIX, derivative=False),
        _unary("tril_indices_from", _MATRIX, derivative=False),
        _unary("triu_indices_from", _MATRIX, derivative=False),
        _case("nanargmin", (ArrayInput([2.0, float("nan"), -1.0], "float64"),), (Input(0),), None),
        _case("nanargmax", (ArrayInput([2.0, float("nan"), -1.0], "float64"),), (Input(0),), None),
        _binary("isclose", derivative=False),
        _binary("allclose", derivative=False),
        _binary("array_equal", derivative=False),
        _binary("array_equiv", derivative=False),
        _unary("iscomplex", _COMPLEX, derivative=False),
        _unary("isreal", _COMPLEX, derivative=False),
        _unary("isposinf", ArrayInput([1.0, float("inf"), -2.0], "float64"), derivative=False),
        _unary("isneginf", ArrayInput([1.0, float("-inf"), -2.0], "float64"), derivative=False),
    )


def _polynomial_cases() -> tuple[NumpySupportCase, ...]:
    coefficients = ArrayInput([1.0, -2.0, 0.5], "float64")
    other = ArrayInput([0.5, 1.0], "float64")
    coordinates = ArrayInput([0.0, 1.0, 2.0, 3.0], "float64")
    observations = ArrayInput([0.2, 1.1, 3.8, 9.2], "float64")
    return (
        _unary("poly", _VECTOR),
        _binary("polyadd", coefficients, other),
        _binary("polysub", coefficients, other),
        _binary("polymul", coefficients, other),
        _binary("polydiv", coefficients, other),
        _case(
            "polyfit", (coordinates, observations), (Input(0), Input(1), 2), ((0,), (1,), (0, 1))
        ),
        _case("polyder", (coefficients,), (Input(0),), ((0,),), (("m", 1),)),
        _case("polyint", (coefficients,), (Input(0),), ((0,),), (("m", 1),)),
        _case("polyval", (coefficients, _REAL), (Input(0), Input(1)), ((0,), (1,), (0, 1))),
        _unary("roots", coefficients),
        _unary(
            "roots",
            ArrayInput([1.0, 0.25, 1.0], "float64"),
            variant="complex-output",
        ),
    )


def _statistics_and_unique_cases() -> tuple[NumpySupportCase, ...]:
    quantiles = ArrayInput([0.25, 0.75], "float64")
    nan_values = ArrayInput([0.0, float("nan"), 2.0, 5.0], "float64")
    quantile_cases = tuple(
        _case(
            name,
            (_REAL, quantiles),
            (Input(0), Input(1)),
            ((0,), (1,), (0, 1)),
            (("method", "linear"),),
        )
        for name in ("quantile", "nanquantile")
    )
    percentile_coordinates = ArrayInput([25.0, 75.0], "float64")
    percentile_cases = (
        _case(
            "percentile",
            (_REAL, percentile_coordinates),
            (Input(0), Input(1)),
            ((0,), (1,), (0, 1)),
            (("method", "linear"),),
        ),
        _case(
            "nanpercentile",
            (nan_values, percentile_coordinates),
            (Input(0), Input(1)),
            ((0,), (1,), (0, 1)),
            (("method", "linear"),),
        ),
    )
    unique_values = ArrayInput([2.0, 1.0, 2.0, 3.0], "float64")
    return (
        *quantile_cases,
        *percentile_cases,
        _unary("median"),
        _unary("nanmedian", nan_values),
        _unary("unique", unique_values, derivative=False),
        _unary("unique_values", unique_values),
        _unary("unique_all", unique_values),
        _unary("unique_counts", unique_values, derivative=False),
        _unary("unique_inverse", unique_values, derivative=False),
    )


def _scientific_cases() -> tuple[NumpySupportCase, ...]:
    fft_real = ArrayInput([[0.0, 1.0], [2.0, 3.0]], "float64")
    fft_complex = ArrayInput([[0.0 + 0.5j, 1.0 - 0.25j], [2.0 + 1.0j, 3.0 - 0.5j]], "complex128")
    half_spectrum = ArrayInput(
        [[1.0 + 0.0j, 0.5 - 0.25j], [2.0 + 0.0j, -0.5 + 0.75j]],
        "complex128",
    )
    return (
        _unary("angle", _COMPLEX, modes=_ALL_MODES),
        _unary("sinc"),
        _unary("amax"),
        _unary("amin"),
        _binary("convolve", modes=_ALL_MODES),
        _binary("correlate", modes=_ALL_MODES),
        _case("einsum", (_MATRIX, _MATRIX), ("ij,ij->", Input(0), Input(1)), ((0,), (1,), (0, 1))),
        _unary("fft.fft2", fft_complex, modes=_ALL_MODES),
        _unary("fft.ifft2", fft_complex, modes=_ALL_MODES),
        _unary("fft.rfft2", fft_real, modes=_ALL_MODES),
        _unary("fft.irfft2", half_spectrum, modes=_ALL_MODES),
        _case(
            "linspace",
            (_SCALAR, _OTHER_SCALAR),
            (Input(0), Input(1), 6),
            ((0,), (1,), (0, 1)),
            (("dtype", DType("float64")),),
        ),
        _case(
            "linspace",
            (_SCALAR, _OTHER_SCALAR),
            (Input(0), Input(1), 6),
            ((0,), (1,), (0, 1)),
            (("dtype", DType("float64")), ("endpoint", False), ("retstep", True)),
            variant="retstep",
        ),
        _case(
            "interp",
            (_SHORT, _VECTOR, _VECTOR),
            (Input(0), Input(1), Input(2)),
            ((0,), (1,), (2,), (0, 1, 2)),
        ),
        _case(
            "interp",
            (_SHORT, _VECTOR, _VECTOR, _SCALAR, _OTHER_SCALAR),
            (Input(0), Input(1), Input(2)),
            ((0,), (1,), (2,), (2, 3), (2, 4), (2, 3, 4), (0, 1, 2, 3, 4)),
            (("left", Input(3)), ("right", Input(4))),
            variant="fill-values",
        ),
        _case(
            "pad",
            (_REAL,),
            (Input(0), (2, 1)),
            ((0,),),
            (("mode", "constant"), ("constant_values", (1.5, -0.5))),
        ),
        _case(
            "pad",
            (_REAL,),
            (Input(0), (2, 1)),
            ((0,),),
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
        _unary("linalg.cond", _MATRIX),
        _unary("linalg.matrix_rank", _RECTANGULAR, derivative=False),
        _case(
            "linalg.lstsq",
            (_RECTANGULAR, ArrayInput([1.0, 2.0, 3.0], "float64")),
            (Input(0), Input(1)),
            ((0,), (1,), (0, 1)),
            (("rcond", None),),
        ),
        _unary(
            "linalg.qr",
            _RECTANGULAR,
            modes=_ALL_MODES,
            result_adapter="tuple",
        ),
        _unary(
            "linalg.qr",
            _RECTANGULAR,
            modes=_ALL_MODES,
            kwargs=(("mode", "r"),),
            variant="mode-r",
        ),
        _unary(
            "linalg.slogdet",
            _MATRIX,
            modes=_ALL_MODES,
            result_adapter="tuple",
        ),
        _unary(
            "linalg.eig",
            _COMPLEX_MATRIX,
            modes=_ALL_MODES,
            result_adapter="tuple",
        ),
        _unary("linalg.eigvals", _COMPLEX_MATRIX, modes=_ALL_MODES),
        _unary(
            "linalg.eig",
            _MATRIX,
            variant="real-dynamic",
            result_adapter="tuple",
        ),
        _unary("linalg.eigvals", _MATRIX, variant="real-dynamic"),
        _case("linalg.tensorinv", (tensor,), (Input(0),), ((0,),), (("ind", 2),)),
        _case("linalg.tensorsolve", (tensor, _MATRIX), (Input(0), Input(1)), ((0,), (1,), (0, 1))),
    )


def _scimath_cases() -> tuple[NumpySupportCase, ...]:
    negative = ArrayInput([-4.0, -0.5, 0.25, 2.0], "float64")
    unary = tuple(
        _unary(f"lib.scimath.{name}", negative)
        for name in ("sqrt", "log", "log10", "log2", "arcsin", "arccos", "arctanh")
    )
    return (
        *unary,
        _case(
            "lib.scimath.logn", (_POSITIVE, negative), (Input(0), Input(1)), ((0,), (1,), (0, 1))
        ),
        _binary("lib.scimath.power", negative, _RIGHT),
    )


_EXTENDED_FUNCTION_CASES = (
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
)


def _validate_cases() -> None:
    identifiers = [case.identifier for case in _EXTENDED_FUNCTION_CASES]
    duplicate_identifiers = sorted(
        identifier for identifier in set(identifiers) if identifiers.count(identifier) > 1
    )
    if duplicate_identifiers:
        msg = f"duplicate extended NumPy case identifiers: {duplicate_identifiers}"
        raise RuntimeError(msg)


_validate_cases()


def function_family_cases() -> tuple[NumpySupportCase, ...]:
    """Return executable cases for grouped NumPy function families."""
    return tuple(sorted(_EXTENDED_FUNCTION_CASES, key=lambda case: case.identifier))


__all__ = ["function_family_cases"]
