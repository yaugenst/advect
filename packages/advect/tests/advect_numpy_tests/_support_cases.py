"""Executable NumPy support cases used only by qualification tests.

Each case names one foreign NumPy call and contains only portable Python data.
Runtime declarations live in :mod:`advect.numpy._support_contract` and supply
each form's lifetimes and derivative availability. These sample inputs and
invocation recipes prove those declarations without shipping test specimens in
the installed package.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from advect.numpy._support_contract import numpy_support_declarations

type NumpyCallableKind = Literal["array_method", "function", "ufunc_call", "ufunc_method"]
type DerivativeArgnums = tuple[tuple[int, ...], ...]
type ResultAdapter = Literal["identity", "array", "dtype_num", "fields"]


@dataclass(frozen=True, slots=True)
class ArrayInput:
    """One NumPy array constructed by the qualification runner."""

    data: object
    dtype: str


@dataclass(frozen=True, slots=True)
class Input:
    """Reference a materialized input inside nested call arguments."""

    index: int


@dataclass(frozen=True, slots=True)
class DType:
    """Resolve a dtype name against NumPy."""

    name: str


@dataclass(frozen=True, slots=True)
class Function:
    """Resolve a portable NumPy callable used as a static argument."""

    path: str


@dataclass(frozen=True, slots=True)
class NumpySupportCase:
    """One executable public spelling and input-role contract.

    The form's declaration supplies its lifetimes and whether it has
    derivatives. A differentiable case differentiates each floating input alone
    and all of them together unless ``derivative_argnums`` names the exact
    independently active input-role groups. ``stages`` marks a documented
    spelling that also stages although its form is declared dynamic-only.
    """

    callable: str
    kind: NumpyCallableKind
    inputs: tuple[ArrayInput, ...]
    args: tuple[object, ...]
    kwargs: tuple[tuple[str, object], ...] = ()
    variant: str = "baseline"
    derivative_argnums: DerivativeArgnums | None = None
    stages: bool = False
    compare_values: bool = True
    return_input: int | None = None
    result_adapter: ResultAdapter = "identity"
    expected_deprecation: str | None = None

    def __post_init__(self) -> None:
        if self.return_input is not None and not 0 <= self.return_input < len(self.inputs):
            message = f"{self.identifier}: invalid returned input index: {self.return_input}"
            raise ValueError(message)
        groups = self.derivative_argnums
        if groups is None:
            return
        malformed = [
            group
            for group in groups
            if not group
            or group != tuple(sorted(set(group)))
            or any(index < 0 or index >= len(self.inputs) for index in group)
            or any(not self.inputs[index].dtype.startswith(("float", "complex")) for index in group)
        ]
        if not groups or len(groups) != len(set(groups)) or malformed:
            message = f"{self.identifier}: invalid derivative argument groups: {groups}"
            raise ValueError(message)

    @property
    def identifier(self) -> str:
        """Return the stable qualification-case identifier."""
        if self.variant == "baseline":
            return f"{self.kind}:{self.callable}"
        return f"{self.kind}:{self.callable}[{self.variant}]"


_REAL = ArrayInput([-1.4, -0.3, 0.6, 1.7], "float64")
_RIGHT = ArrayInput([0.8, 1.3, 2.2, 0.45], "float64")
_POSITIVE = ArrayInput([0.3, 0.7, 1.6, 3.1], "float64")
_UNIT = ArrayInput([-0.75, -0.25, 0.25, 0.75], "float64")
_NONZERO = ArrayInput([-1.3, -0.45, 0.55, 2.1], "float64")
_COMPLEX = ArrayInput([1.0 + 0.5j, -2.0 + 1.0j, 0.25 - 0.75j], "complex128")
_MATRIX = ArrayInput([[4.0, 1.0], [1.0, 3.0]], "float64")
_RECTANGULAR = ArrayInput([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]], "float64")
_VECTOR = ArrayInput([1.0, 2.0], "float64")
_INDEX = ArrayInput([2, 0, 2, 1], "int64")
_BOOL = ArrayInput([[True, False], [False, True]], "bool")
_INT_LEFT = ArrayInput([[1, 2], [3, 4]], "int64")
_INT_RIGHT = ArrayInput([[4, 1], [2, 3]], "int64")
# A mutation must trace its destination, so replacement values join it.
_UPDATE_GROUPS: DerivativeArgnums = ((0,), (0, 1))
_MASKED_UPDATE_GROUPS: DerivativeArgnums = ((0,), (0, 2))


def _function(
    path: str,
    inputs: tuple[ArrayInput, ...],
    args: tuple[object, ...],
    kwargs: tuple[tuple[str, object], ...] = (),
    **options: object,
) -> NumpySupportCase:
    return NumpySupportCase(f"numpy.{path}", "function", inputs, args, kwargs, **options)


def _unary(
    path: str,
    value: ArrayInput = _REAL,
    kwargs: tuple[tuple[str, object], ...] = (),
    **options: object,
) -> NumpySupportCase:
    return _function(path, (value,), (Input(0),), kwargs, **options)


def _binary(
    path: str,
    left: ArrayInput = _REAL,
    right: ArrayInput = _RIGHT,
    kwargs: tuple[tuple[str, object], ...] = (),
    **options: object,
) -> NumpySupportCase:
    return _function(path, (left, right), (Input(0), Input(1)), kwargs, **options)


def _method(
    path: str,
    inputs: tuple[ArrayInput, ...],
    args: tuple[object, ...] = (),
    kwargs: tuple[tuple[str, object], ...] = (),
    **options: object,
) -> NumpySupportCase:
    kind = "array_method" if path.startswith("ndarray.") else "ufunc_method"
    return NumpySupportCase(f"numpy.{path}", kind, inputs, args, kwargs, **options)


def _ufunc(name: str, *inputs: ArrayInput, **options: object) -> NumpySupportCase:
    arguments = tuple(Input(index) for index in range(len(inputs)))
    return NumpySupportCase(f"numpy.{name}", "ufunc_call", inputs, arguments, **options)


def _ufunc_cases() -> tuple[NumpySupportCase, ...]:
    unary_domains = {
        "absolute": _NONZERO,
        "arccos": _UNIT,
        "arccosh": ArrayInput([1.25, 1.5, 2.0, 3.0], "float64"),
        "arcsin": _UNIT,
        "arcsinh": _REAL,
        "arctan": _REAL,
        "arctanh": _UNIT,
        "cbrt": _NONZERO,
        "ceil": _REAL,
        "conjugate": _COMPLEX,
        "cos": _REAL,
        "cosh": _REAL,
        "deg2rad": _REAL,
        "degrees": _REAL,
        "exp": _REAL,
        "exp2": _REAL,
        "expm1": _REAL,
        "fabs": _NONZERO,
        "floor": _REAL,
        "frexp": _POSITIVE,
        "invert": _INT_LEFT,
        "isfinite": _REAL,
        "isinf": _REAL,
        "isnan": _REAL,
        "log": _POSITIVE,
        "log10": _POSITIVE,
        "log1p": _UNIT,
        "log2": _POSITIVE,
        "logical_not": _BOOL,
        "modf": _REAL,
        "negative": _REAL,
        "positive": _REAL,
        "rad2deg": _REAL,
        "radians": _REAL,
        "reciprocal": _NONZERO,
        "rint": _REAL,
        "sign": _NONZERO,
        "signbit": _REAL,
        "sin": _REAL,
        "sinh": _REAL,
        "spacing": _POSITIVE,
        "sqrt": _POSITIVE,
        "square": _REAL,
        "tan": _UNIT,
        "tanh": _REAL,
        "trunc": _REAL,
    }
    binary_domains = {
        "add": (_REAL, _RIGHT),
        "arctan2": (_NONZERO, _NONZERO),
        "bitwise_and": (_INT_LEFT, _INT_RIGHT),
        "bitwise_or": (_INT_LEFT, _INT_RIGHT),
        "bitwise_xor": (_INT_LEFT, _INT_RIGHT),
        "copysign": (_NONZERO, _NONZERO),
        "divide": (_REAL, _NONZERO),
        "divmod": (_POSITIVE, _NONZERO),
        "equal": (_REAL, _RIGHT),
        "float_power": (_POSITIVE, _RIGHT),
        "floor_divide": (_POSITIVE, _NONZERO),
        "fmax": (_REAL, _RIGHT),
        "fmin": (_REAL, _RIGHT),
        "fmod": (_POSITIVE, _NONZERO),
        "greater": (_REAL, _RIGHT),
        "greater_equal": (_REAL, _RIGHT),
        "heaviside": (_NONZERO, _RIGHT),
        "hypot": (_NONZERO, _NONZERO),
        "ldexp": (_REAL, ArrayInput([1, 2, -1, 0], "int32")),
        "left_shift": (_INT_LEFT, _INT_RIGHT),
        "less": (_REAL, _RIGHT),
        "less_equal": (_REAL, _RIGHT),
        "logaddexp": (_REAL, _RIGHT),
        "logaddexp2": (_REAL, _RIGHT),
        "logical_and": (_BOOL, _BOOL),
        "logical_or": (_BOOL, _BOOL),
        "logical_xor": (_BOOL, _BOOL),
        "matmul": (_MATRIX, _MATRIX),
        "matvec": (_MATRIX, _VECTOR),
        "maximum": (_REAL, _RIGHT),
        "minimum": (_REAL, _RIGHT),
        "multiply": (_REAL, _RIGHT),
        "nextafter": (_REAL, _RIGHT),
        "not_equal": (_REAL, _RIGHT),
        "power": (_POSITIVE, _RIGHT),
        "remainder": (_POSITIVE, _NONZERO),
        "right_shift": (_INT_LEFT, _INT_RIGHT),
        "subtract": (_REAL, _RIGHT),
        "vecdot": (_VECTOR, _VECTOR),
        "vecmat": (_VECTOR, _MATRIX),
    }
    return (
        *(_ufunc(name, domain) for name, domain in unary_domains.items()),
        *(
            _ufunc(name, *domains, derivative_argnums=((0,),) if name == "copysign" else None)
            for name, domains in binary_domains.items()
        ),
    )


def _outer_cases(calls: tuple[NumpySupportCase, ...]) -> tuple[NumpySupportCase, ...]:
    """Qualify each declared ``ufunc.outer`` form on its binary call's inputs."""
    declared = {
        declaration.callable
        for declaration in numpy_support_declarations()
        if declaration.kind == "ufunc_method"
    }
    return tuple(
        replace(case, kind="ufunc_method", callable=f"{case.callable}.outer", args=(Input(1),))
        for case in calls
        if f"{case.callable}.outer" in declared
    )


def _function_cases() -> tuple[NumpySupportCase, ...]:
    reductions = [
        "max",
        "mean",
        "min",
        "nanmax",
        "nanmean",
        "nanmin",
        "nanprod",
        "nanstd",
        "nansum",
        "nanvar",
        "prod",
        "std",
        "sum",
        "var",
    ]
    return (
        *(
            _unary(
                name,
                _POSITIVE if name in {"nanprod", "prod"} else _REAL,
                (("axis", 0), ("keepdims", True)),
            )
            for name in reductions
        ),
        *(
            _function(
                name,
                (_MATRIX, _BOOL),
                (Input(0),),
                (("axis", 0), ("keepdims", True), ("initial", initial), ("where", Input(1))),
                variant="where-initial",
            )
            for name, initial in (("max", -10.0), ("min", 10.0), ("sum", 2.0))
        ),
        _unary("cumprod", _POSITIVE, (("axis", 0),)),
        _unary("cumsum", _REAL, (("axis", 0),)),
        _unary("all", _MATRIX, (("axis", 0),)),
        _unary("any", _MATRIX, (("axis", 1),)),
        _unary("argsort"),
        _unary("count_nonzero", _REAL, (("axis", 0),)),
        _binary("searchsorted", ArrayInput([1.0, 3.0, 5.0, 7.0], "float64")),
        _function("reshape", (_REAL,), (Input(0), (2, 2))),
        _unary("transpose", _MATRIX),
        _function("moveaxis", (_MATRIX,), (Input(0), 0, 1)),
        _function("swapaxes", (_MATRIX,), (Input(0), 0, 1)),
        _unary("ravel", _MATRIX),
        _unary("flip", _MATRIX, (("axis", 0),)),
        _unary("fliplr", _MATRIX),
        _unary("flipud", _MATRIX),
        _function("roll", (_REAL,), (Input(0), 1)),
        _unary("rot90", _MATRIX),
        _unary("squeeze", ArrayInput([[[1.0, 2.0]]], "float64")),
        _function("expand_dims", (_REAL,), (Input(0), 0)),
        _function("broadcast_to", (_VECTOR,), (Input(0), (2, 2))),
        _function("concatenate", (_REAL, _RIGHT), ((Input(0), Input(1)),)),
        _function("stack", (_REAL, _RIGHT), ((Input(0), Input(1)),), (("axis", 0),)),
        _unary("diff", _REAL, (("n", 1),)),
        _unary("gradient"),
        _unary("nan_to_num"),
        _binary("dot", _MATRIX, _MATRIX),
        _binary("inner"),
        _binary("outer"),
        _binary("kron", _MATRIX, _MATRIX),
        _binary("cross", *(ArrayInput([[1.0, 2.0, 3.0]], "float64"),) * 2),
        _binary("tensordot", _MATRIX, _MATRIX, (("axes", 1),)),
        _function("where", (_BOOL, _MATRIX, _MATRIX), (Input(0), Input(1), Input(2))),
        _function("clip", (_REAL,), (Input(0), -0.5, 1.5)),
        _unary("sort"),
        _function("partition", (_REAL,), (Input(0), 2)),
        _binary("take", _REAL, _INDEX),
        _binary("take_along_axis", _REAL, _INDEX, (("axis", 0),)),
        _unary("copy", _MATRIX),
        _function("full", (ArrayInput(2.5, "float32"),), ((2, 3), Input(0)), (("like", Input(0)),)),
        _function("eye", (_REAL,), (3,), (("like", Input(0)),)),
        *(
            _function(name, (_REAL,), ((2, 3),), (("dtype", DType("float32")), ("like", Input(0))))
            for name in ("zeros", "ones")
        ),
        _function(
            "empty",
            (_REAL,),
            ((2, 3),),
            (("dtype", DType("float32")), ("like", Input(0))),
            compare_values=False,
        ),
        _unary("zeros_like", _MATRIX),
        _unary("ones_like", _MATRIX),
        # The template anchors NumPy dispatch while the fill carries its value derivative.
        _binary("full_like", _MATRIX, ArrayInput(2.5, "float64"), derivative_argnums=((0, 1),)),
    )


def _linalg_cases() -> tuple[NumpySupportCase, ...]:
    return (
        *(
            _unary(f"linalg.{name}", _MATRIX)
            for name in ("cholesky", "det", "eigvalsh", "inv", "norm", "diagonal", "trace")
        ),
        *(_unary(f"linalg.{name}", _RECTANGULAR) for name in ("pinv", "svdvals")),
        _binary("linalg.solve", _MATRIX, _VECTOR),
        _function("linalg.matrix_power", (_MATRIX,), (Input(0), 3)),
        _unary("linalg.eigh", _MATRIX, result_adapter="fields"),
        _unary(
            "linalg.svd",
            _RECTANGULAR,
            (("full_matrices", False),),
            result_adapter="fields",
        ),
        _binary("linalg.vecdot", _RECTANGULAR, _RECTANGULAR, (("axis", -1),)),
    )


def _fft_cases() -> tuple[NumpySupportCase, ...]:
    real = ArrayInput([0.0, 1.0, 2.0, 3.0], "float64")
    complex_input = ArrayInput([0.0 + 0.5j, 1.0 - 0.25j, 2.0 + 1.0j, 3.0 - 0.5j], "complex128")
    half_spectrum = ArrayInput([1.0 + 0.0j, 0.5 - 0.25j, 2.0 + 0.0j], "complex128")
    inputs = {
        "fft": complex_input,
        "fftn": complex_input,
        "fftshift": complex_input,
        "hfft": half_spectrum,
        "ifft": complex_input,
        "ifftn": complex_input,
        "ifftshift": complex_input,
        "ihfft": real,
        "irfft": half_spectrum,
        "irfftn": half_spectrum,
        "rfft": real,
        "rfftn": real,
    }
    return tuple(_unary(f"fft.{name}", value) for name, value in inputs.items())


def _method_cases() -> tuple[NumpySupportCase, ...]:
    return (
        _method("ndarray.astype", (_REAL,), (DType("float32"),)),
        _method("ndarray.copy", (_REAL,)),
        _method("ndarray.item", (ArrayInput([2.0], "float64"),)),
        _method("ndarray.reshape", (_REAL,), ((2, 2),)),
        _method("ndarray.sum", (_REAL,)),
        _method("ndarray.transpose", (_MATRIX,)),
        _method("add.reduce", (_REAL,)),
        _method("multiply.reduce", (_POSITIVE,)),
        _method("add.accumulate", (_REAL,)),
        _method("multiply.accumulate", (_POSITIVE,)),
    )


def _additional_existing_function_cases() -> tuple[NumpySupportCase, ...]:
    """Qualify forms whose concrete runtime implementation was never deleted."""
    metadata_cases = (
        _function("can_cast", (_REAL,), (Input(0), DType("complex128")), result_adapter="array"),
        _binary("common_type", _REAL, _INDEX, result_adapter="dtype_num"),
        *(
            _unary(name, value, result_adapter="array")
            for name, value in (
                ("iscomplexobj", _COMPLEX),
                ("isrealobj", _REAL),
                ("ndim", _MATRIX),
                ("shape", _MATRIX),
                ("size", _MATRIX),
            )
        ),
        _function(
            "result_type", (_REAL,), (Input(0), DType("float32")), result_adapter="dtype_num"
        ),
    )
    replacement = ArrayInput([[5.0, 6.0], [7.0, 8.0]], "float64")
    mutation_cases = (
        _binary("copyto", _MATRIX, replacement, return_input=0, derivative_argnums=_UPDATE_GROUPS),
        _binary(
            "fill_diagonal", _MATRIX, _VECTOR, return_input=0, derivative_argnums=_UPDATE_GROUPS
        ),
        *(
            _function(
                name,
                (_MATRIX, _BOOL, _VECTOR),
                (Input(0), Input(1), Input(2)),
                return_input=0,
                derivative_argnums=_MASKED_UPDATE_GROUPS,
            )
            for name in ("place", "putmask")
        ),
        _function(
            "put",
            (_REAL, _INDEX, _RIGHT),
            (Input(0), Input(1), Input(2)),
            return_input=0,
            derivative_argnums=_MASKED_UPDATE_GROUPS,
        ),
        _function(
            "put_along_axis",
            (_MATRIX, ArrayInput([[1, 0], [0, 1]], "int64"), _MATRIX),
            (Input(0), Input(1), Input(2), 1),
            return_input=0,
            derivative_argnums=_MASKED_UPDATE_GROUPS,
        ),
    )
    return (
        _unary("argmax", _MATRIX, (("axis", 1),)),
        _unary("argmin", _MATRIX, (("axis", 1),)),
        _function("astype", (_REAL,), (Input(0), DType("float32"))),
        _unary("real", _COMPLEX),
        _unary("trace", _MATRIX),
        _unary("tril", _MATRIX),
        *metadata_cases,
        *mutation_cases,
    )


def _version_conditional_function_cases() -> tuple[NumpySupportCase, ...]:
    """Qualify aliases only on the NumPy minors that still expose them."""
    return (
        _binary("in1d", expected_deprecation=r"`in1d` is deprecated"),
        _unary("trapz", expected_deprecation=r"`trapz` is deprecated"),
    )


def _material_variant_cases() -> tuple[NumpySupportCase, ...]:
    """Qualify materially distinct controls on otherwise covered public forms."""
    nan_matrix = ArrayInput([[1.0, float("nan")], [3.0, 4.0]], "float64")
    vector_mask = ArrayInput([True, False, True, True], "bool")
    tall_matrix = ArrayInput(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        "float64",
    )
    return (
        _function(
            "nanmean",
            (nan_matrix, _BOOL),
            (Input(0),),
            (("axis", 0), ("keepdims", True), ("where", Input(1))),
            variant="where",
        ),
        _function(
            "nanmax",
            (nan_matrix, _BOOL),
            (Input(0),),
            (("axis", 0), ("initial", -10.0), ("keepdims", True), ("where", Input(1))),
            variant="where-initial",
        ),
        _function(
            "nanvar",
            (nan_matrix, _BOOL),
            (Input(0),),
            (
                ("axis", 0),
                ("correction", 0.5),
                ("keepdims", True),
                ("mean", ((2.0, 4.0),)),
                ("where", Input(1)),
            ),
            variant="controlled",
        ),
        _function(
            "prod",
            (_POSITIVE, vector_mask),
            (Input(0),),
            (("axis", 0), ("initial", 2.0), ("where", Input(1))),
            variant="where-initial",
        ),
        _function(
            "concatenate",
            (_MATRIX, _MATRIX),
            ((Input(0), Input(1)),),
            (("axis", 1),),
            variant="axis-one",
        ),
        _function(
            "full",
            (ArrayInput(2.5, "float32"),),
            ((2, 3), Input(0)),
            (
                ("device", "cpu"),
                ("dtype", DType("float32")),
                ("like", Input(0)),
                ("order", "F"),
            ),
            variant="metadata",
        ),
        _function(
            "zeros",
            (_REAL,),
            ((2, 3),),
            (
                ("device", "cpu"),
                ("dtype", DType("float32")),
                ("like", Input(0)),
                ("order", "F"),
            ),
            variant="metadata",
        ),
        _unary(
            "zeros_like",
            _MATRIX,
            (("device", "cpu"), ("order", "C"), ("shape", (4,)), ("subok", False)),
            variant="metadata",
        ),
        _binary(
            "full_like",
            _MATRIX,
            ArrayInput(2.5, "float64"),
            (
                ("device", "cpu"),
                ("dtype", DType("float32")),
                ("order", "C"),
                ("shape", (4,)),
                ("subok", False),
            ),
            variant="metadata",
            derivative_argnums=((0, 1),),
        ),
        _function(
            "eye",
            (_REAL,),
            (3,),
            (
                ("M", 4),
                ("device", "cpu"),
                ("dtype", DType("float32")),
                ("k", 1),
                ("like", Input(0)),
                ("order", "C"),
            ),
            variant="metadata",
        ),
        _function("reshape", (_REAL,), (Input(0), (2, 2)), (("order", "F"),), variant="order"),
        _unary("transpose", _MATRIX, (("axes", (1, 0)),), variant="axes"),
        _function(
            "roll", (_MATRIX,), (Input(0), (1, -1)), (("axis", (0, 1)),), variant="paired-axes"
        ),
        _unary("diagonal", _MATRIX, (("offset", 1),), variant="offset"),
        _unary("trace", _MATRIX, (("offset", 1),), variant="offset"),
        _function("diagonal", (_MATRIX,), (Input(0), 1, 1, 0), variant="positional"),
        _function(
            "trace", (_MATRIX,), (Input(0), -1, 1, 0, DType("float32")), variant="positional"
        ),
        _unary("repeat", _MATRIX, (("axis", 1), ("repeats", 2)), variant="axis"),
        _unary(
            "diff", _REAL, (("append", (4.0,)), ("axis", 0), ("prepend", 0.0)), variant="boundaries"
        ),
        _function(
            "copyto",
            (_MATRIX, _MATRIX, _BOOL),
            (Input(0), Input(1)),
            (("casting", "unsafe"), ("where", Input(2))),
            derivative_argnums=_UPDATE_GROUPS,
            return_input=0,
            variant="where-casting",
        ),
        _binary(
            "fill_diagonal",
            tall_matrix,
            _VECTOR,
            (("wrap", True),),
            derivative_argnums=_UPDATE_GROUPS,
            return_input=0,
            variant="wrap",
        ),
        _function(
            "put",
            (_REAL, ArrayInput([-1, 4, 5], "int64"), _RIGHT),
            (Input(0), Input(1), Input(2)),
            (("mode", "wrap"),),
            derivative_argnums=_MASKED_UPDATE_GROUPS,
            return_input=0,
            variant="wrap",
        ),
        _function(
            "take",
            (_MATRIX, ArrayInput([-1, 3, 4], "int64")),
            (Input(0), Input(1), 1),
            (("mode", "wrap"),),
            variant="wrap",
        ),
        _function(
            "take",
            (_MATRIX, ArrayInput([-1, 3, 4], "int64")),
            (Input(0), Input(1)),
            (("mode", "clip"),),
            variant="clip",
        ),
        _function(
            "put_along_axis",
            (_REAL, _INDEX, _RIGHT),
            (Input(0), Input(1), Input(2), None),
            derivative_argnums=_MASKED_UPDATE_GROUPS,
            return_input=0,
            variant="axis-none",
        ),
        _unary("argsort", _MATRIX, (("axis", 0), ("stable", True)), variant="stable-axis"),
        _binary(
            "searchsorted",
            ArrayInput([1.0, 3.0, 5.0, 7.0], "float64"),
            _RIGHT,
            (("side", "right"), ("sorter", (0, 1, 2, 3))),
            variant="side-sorter",
        ),
        _binary(
            "isin",
            kwargs=(("assume_unique", True), ("invert", True), ("kind", "sort")),
            variant="options",
        ),
        _binary(
            "intersect1d",
            kwargs=(("assume_unique", True), ("return_indices", True)),
            variant="indices",
        ),
        _method(
            "ndarray.astype",
            (_REAL,),
            (DType("float32"),),
            (("casting", "unsafe"), ("copy", True), ("order", "F"), ("subok", False)),
            variant="controls",
        ),
        _method(
            "ndarray.sum",
            (_MATRIX, _BOOL),
            (),
            (
                ("axis", 0),
                ("dtype", DType("float32")),
                ("initial", 1.0),
                ("keepdims", True),
                ("where", Input(1)),
            ),
            variant="controls",
        ),
        _method("ndarray.sum", (_MATRIX,), (0,), variant="positional-axis"),
        _method(
            "add.reduce",
            (_MATRIX, _BOOL),
            (),
            (
                ("axis", 1),
                ("dtype", DType("float32")),
                ("initial", 1.0),
                ("keepdims", True),
                ("where", Input(1)),
            ),
            variant="controls",
        ),
        _method("multiply.accumulate", (_MATRIX,), (), (("axis", 1),), variant="axis-one"),
    )


def base_cases() -> tuple[NumpySupportCase, ...]:
    """Return the ufunc, method, and core function cases."""
    calls = _ufunc_cases()
    return (
        *calls,
        *_outer_cases(calls),
        *_function_cases(),
        *_linalg_cases(),
        *_fft_cases(),
        *_method_cases(),
        *_additional_existing_function_cases(),
        *_version_conditional_function_cases(),
        *_material_variant_cases(),
    )


__all__ = [
    "ArrayInput",
    "DType",
    "Function",
    "Input",
    "NumpySupportCase",
    "base_cases",
]
