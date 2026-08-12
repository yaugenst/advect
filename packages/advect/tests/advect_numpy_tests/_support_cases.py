"""Executable NumPy support cases used only by qualification tests.

Each case names one foreign NumPy call and contains only portable Python data.
Runtime declarations live in :mod:`advect.numpy._support_contract`.  These
sample inputs and invocation recipes prove those declarations without shipping
test specimens in the installed package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type NumpyCallableKind = Literal["array_method", "function", "ufunc_call", "ufunc_method"]
type DerivativeArgnums = tuple[tuple[int, ...], ...]

_ALL_MODES = ("dynamic", "staged", "serialized")
_DYNAMIC_ONLY = ("dynamic",)


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
    """One executable public spelling, input-role, and lifetime contract."""

    callable: str
    kind: NumpyCallableKind
    inputs: tuple[ArrayInput, ...]
    args: tuple[object, ...]
    derivative_argnums: DerivativeArgnums | None
    kwargs: tuple[tuple[str, object], ...] = ()
    modes: tuple[str, ...] = _ALL_MODES
    variant: str = "baseline"
    compare_values: bool = True
    trace_argnums: tuple[int, ...] | None = None
    return_input: int | None = None
    result_adapter: Literal["identity", "array", "dtype_num", "tuple"] = "identity"
    expected_deprecation: str | None = None

    def __post_init__(self) -> None:
        if self.trace_argnums is not None and (
            not self.trace_argnums
            or self.trace_argnums != tuple(sorted(set(self.trace_argnums)))
            or any(index < 0 or index >= len(self.inputs) for index in self.trace_argnums)
        ):
            message = f"{self.identifier}: invalid trace argument indices: {self.trace_argnums}"
            raise ValueError(message)
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


_REAL = ArrayInput([-1.5, -0.25, 0.5, 2.0], "float64")
_RIGHT = ArrayInput([0.75, 1.5, 2.0, 0.5], "float64")
_POSITIVE = ArrayInput([0.25, 0.5, 1.5, 3.0], "float64")
_UNIT = ArrayInput([-0.75, -0.25, 0.25, 0.75], "float64")
_NONZERO = ArrayInput([-1.5, -0.5, 0.5, 2.0], "float64")
_COMPLEX = ArrayInput([1.0 + 0.5j, -2.0 + 1.0j, 0.25 - 0.75j], "complex128")
_MATRIX = ArrayInput([[4.0, 1.0], [1.0, 3.0]], "float64")
_RECTANGULAR = ArrayInput([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]], "float64")
_VECTOR = ArrayInput([1.0, 2.0], "float64")
_INDEX = ArrayInput([2, 0, 2, 1], "int64")
_BOOL = ArrayInput([[True, False], [False, True]], "bool")
_INT_LEFT = ArrayInput([[1, 2], [3, 4]], "int64")
_INT_RIGHT = ArrayInput([[4, 1], [2, 3]], "int64")

_NOT_APPLICABLE_DERIVATIVE_FORMS = frozenset(
    {
        "numpy.all",
        "numpy.any",
        "numpy.argmax",
        "numpy.argmin",
        "numpy.bitwise_and",
        "numpy.bitwise_or",
        "numpy.bitwise_xor",
        "numpy.argsort",
        "numpy.count_nonzero",
        "numpy.equal",
        "numpy.greater",
        "numpy.greater_equal",
        "numpy.invert",
        "numpy.in1d",
        "numpy.isfinite",
        "numpy.isinf",
        "numpy.isnan",
        "numpy.less",
        "numpy.less_equal",
        "numpy.logical_and",
        "numpy.logical_not",
        "numpy.logical_or",
        "numpy.logical_xor",
        "numpy.not_equal",
        "numpy.searchsorted",
        "numpy.signbit",
    }
)
_UNSUPPORTED_DERIVATIVE_FORMS = frozenset(
    {
        "numpy.empty",
        "numpy.eye",
        "numpy.ones",
        "numpy.zeros",
    }
)
_DERIVATIVE_ARGNUM_OVERRIDES: dict[str, DerivativeArgnums] = {
    "numpy.copysign": ((0,),),
    # The template anchors NumPy dispatch while the fill carries its value derivative.
    "numpy.full_like": ((0, 1),),
}


def _derivative_argnums(
    callable_name: str,
    inputs: tuple[ArrayInput, ...],
) -> DerivativeArgnums | None:
    if callable_name in _NOT_APPLICABLE_DERIVATIVE_FORMS | _UNSUPPORTED_DERIVATIVE_FORMS:
        return None
    indices = tuple(
        index for index, value in enumerate(inputs) if value.dtype.startswith(("float", "complex"))
    )
    if not indices:
        return None
    individual = tuple((index,) for index in indices)
    default = (*individual, indices) if len(indices) > 1 else individual
    return _DERIVATIVE_ARGNUM_OVERRIDES.get(callable_name, default)


def _ufunc(
    name: str,
    *inputs: ArrayInput,
    modes: tuple[str, ...] = _ALL_MODES,
) -> NumpySupportCase:
    callable_name = f"numpy.{name}"
    return NumpySupportCase(
        callable=callable_name,
        kind="ufunc_call",
        inputs=inputs,
        args=tuple(Input(index) for index in range(len(inputs))),
        derivative_argnums=_derivative_argnums(callable_name, inputs),
        modes=modes,
    )


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
        "isfinite": _REAL,
        "isinf": _REAL,
        "isnan": _REAL,
        "log": _POSITIVE,
        "log10": _POSITIVE,
        "log1p": _UNIT,
        "log2": _POSITIVE,
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
    dynamic_only = {
        "cbrt",
        "deg2rad",
        "degrees",
        "exp2",
        "fabs",
        "frexp",
        "modf",
        "rad2deg",
        "radians",
    }
    cases = [
        _ufunc(name, domain, modes=_DYNAMIC_ONLY if name in dynamic_only else _ALL_MODES)
        for name, domain in unary_domains.items()
    ]
    binary_domains = {
        "add": (_REAL, _RIGHT),
        "arctan2": (_NONZERO, _NONZERO),
        "copysign": (_NONZERO, _NONZERO),
        "divide": (_REAL, _NONZERO),
        "divmod": (_POSITIVE, _NONZERO),
        "float_power": (_POSITIVE, _RIGHT),
        "floor_divide": (_POSITIVE, _NONZERO),
        "fmax": (_REAL, _RIGHT),
        "fmin": (_REAL, _RIGHT),
        "fmod": (_POSITIVE, _NONZERO),
        "heaviside": (_NONZERO, _RIGHT),
        "hypot": (_NONZERO, _NONZERO),
        "logaddexp": (_REAL, _RIGHT),
        "logaddexp2": (_REAL, _RIGHT),
        "maximum": (_REAL, _RIGHT),
        "minimum": (_REAL, _RIGHT),
        "multiply": (_REAL, _RIGHT),
        "nextafter": (_REAL, _RIGHT),
        "power": (_POSITIVE, _RIGHT),
        "remainder": (_POSITIVE, _NONZERO),
        "subtract": (_REAL, _RIGHT),
    }
    dynamic_only.update({"divmod", "float_power", "fmax", "fmin", "fmod", "logaddexp2"})
    cases.extend(
        _ufunc(name, *domains, modes=_DYNAMIC_ONLY if name in dynamic_only else _ALL_MODES)
        for name, domains in binary_domains.items()
    )
    cases.extend(
        (
            _ufunc("ldexp", _REAL, ArrayInput([1, 2, -1, 0], "int32")),
            _ufunc("matmul", _MATRIX, _MATRIX),
            _ufunc("matvec", _MATRIX, _VECTOR),
            _ufunc("vecdot", _VECTOR, _VECTOR),
            _ufunc("vecmat", _VECTOR, _MATRIX),
        )
    )
    cases.extend(
        (
            _ufunc("bitwise_and", _INT_LEFT, _INT_RIGHT),
            _ufunc("bitwise_or", _INT_LEFT, _INT_RIGHT),
            _ufunc("bitwise_xor", _INT_LEFT, _INT_RIGHT),
            _ufunc("equal", _REAL, _RIGHT),
            _ufunc("greater", _REAL, _RIGHT),
            _ufunc("greater_equal", _REAL, _RIGHT),
            _ufunc("invert", _INT_LEFT),
            _ufunc("less", _REAL, _RIGHT),
            _ufunc("less_equal", _REAL, _RIGHT),
            _ufunc("logical_and", _BOOL, _BOOL),
            _ufunc("logical_not", _BOOL),
            _ufunc("logical_or", _BOOL, _BOOL),
            _ufunc("logical_xor", _BOOL, _BOOL),
            _ufunc("not_equal", _REAL, _RIGHT),
        )
    )
    return tuple(cases)


def _function(
    path: str,
    inputs: tuple[ArrayInput, ...],
    args: tuple[object, ...],
    kwargs: tuple[tuple[str, object], ...] = (),
    *,
    modes: tuple[str, ...] = _ALL_MODES,
    variant: str = "baseline",
    compare_values: bool = True,
    derivative_argnums: DerivativeArgnums | Literal["auto"] | None = "auto",
    trace_argnums: tuple[int, ...] | None = None,
    return_input: int | None = None,
    result_adapter: Literal["identity", "array", "dtype_num", "tuple"] = "identity",
    expected_deprecation: str | None = None,
) -> NumpySupportCase:
    callable_name = f"numpy.{path}"
    resolved_derivative_argnums = (
        _derivative_argnums(callable_name, inputs)
        if derivative_argnums == "auto"
        else derivative_argnums
    )
    return NumpySupportCase(
        callable=callable_name,
        kind="function",
        inputs=inputs,
        args=args,
        derivative_argnums=resolved_derivative_argnums,
        kwargs=kwargs,
        modes=modes,
        variant=variant,
        compare_values=compare_values,
        trace_argnums=trace_argnums,
        return_input=return_input,
        result_adapter=result_adapter,
        expected_deprecation=expected_deprecation,
    )


def _function_cases() -> tuple[NumpySupportCase, ...]:
    cases: list[NumpySupportCase] = []
    for name in (
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
    ):
        domain = _POSITIVE if name in {"nanprod", "prod"} else _REAL
        cases.append(
            _function(
                name,
                (domain,),
                (Input(0),),
                (("axis", 0), ("keepdims", True)),
            )
        )
    for name, initial in (("max", -10.0), ("min", 10.0), ("sum", 2.0)):
        cases.append(
            _function(
                name,
                (_MATRIX, _BOOL),
                (Input(0),),
                (("axis", 0), ("keepdims", True), ("initial", initial), ("where", Input(1))),
                variant="where-initial",
            )
        )
    cases.extend(
        _function(
            name,
            (_POSITIVE if name == "cumprod" else _REAL,),
            (Input(0),),
            (("axis", 0),),
        )
        for name in ("cumprod", "cumsum")
    )
    cases.extend(
        (
            _function("all", (_MATRIX,), (Input(0),), (("axis", 0),)),
            _function("any", (_MATRIX,), (Input(0),), (("axis", 1),)),
            _function("argsort", (_REAL,), (Input(0),)),
            _function("count_nonzero", (_REAL,), (Input(0),), (("axis", 0),)),
            _function(
                "searchsorted",
                (ArrayInput([1.0, 3.0, 5.0, 7.0], "float64"), _RIGHT),
                (Input(0), Input(1)),
            ),
            _function("reshape", (_REAL,), (Input(0), (2, 2))),
            _function("transpose", (_MATRIX,), (Input(0),)),
            _function("moveaxis", (_MATRIX,), (Input(0), 0, 1)),
            _function("swapaxes", (_MATRIX,), (Input(0), 0, 1), modes=_DYNAMIC_ONLY),
            _function("ravel", (_MATRIX,), (Input(0),), modes=_DYNAMIC_ONLY),
            _function("flip", (_MATRIX,), (Input(0),), (("axis", 0),)),
            _function("fliplr", (_MATRIX,), (Input(0),), modes=_DYNAMIC_ONLY),
            _function("flipud", (_MATRIX,), (Input(0),), modes=_DYNAMIC_ONLY),
            _function("roll", (_REAL,), (Input(0), 1)),
            _function("rot90", (_MATRIX,), (Input(0),), modes=_DYNAMIC_ONLY),
            _function("squeeze", (ArrayInput([[[1.0, 2.0]]], "float64"),), (Input(0),)),
            _function("expand_dims", (_REAL,), (Input(0), 0)),
            _function("broadcast_to", (_VECTOR,), (Input(0), (2, 2))),
            _function("concatenate", (_REAL, _RIGHT), ((Input(0), Input(1)),)),
            _function("stack", (_REAL, _RIGHT), ((Input(0), Input(1)),), (("axis", 0),)),
            _function("diff", (_REAL,), (Input(0),), (("n", 1),)),
            _function("gradient", (_REAL,), (Input(0),)),
            _function("nan_to_num", (_REAL,), (Input(0),), modes=_DYNAMIC_ONLY),
            _function("dot", (_MATRIX, _MATRIX), (Input(0), Input(1))),
            _function("inner", (_REAL, _RIGHT), (Input(0), Input(1)), modes=_DYNAMIC_ONLY),
            _function("outer", (_REAL, _RIGHT), (Input(0), Input(1))),
            _function("kron", (_MATRIX, _MATRIX), (Input(0), Input(1)), modes=_DYNAMIC_ONLY),
            _function(
                "cross", (ArrayInput([[1.0, 2.0, 3.0]], "float64"),) * 2, (Input(0), Input(1))
            ),
            _function("tensordot", (_MATRIX, _MATRIX), (Input(0), Input(1)), (("axes", 1),)),
            _function("where", (_BOOL, _MATRIX, _MATRIX), (Input(0), Input(1), Input(2))),
            _function("clip", (_REAL,), (Input(0), -0.5, 1.5)),
            _function("sort", (_REAL,), (Input(0),)),
            _function("partition", (_REAL,), (Input(0), 2), modes=_DYNAMIC_ONLY),
            _function("take", (_REAL, _INDEX), (Input(0), Input(1))),
            _function(
                "take_along_axis",
                (_REAL, _INDEX),
                (Input(0), Input(1)),
                (("axis", 0),),
            ),
            _function("copy", (_MATRIX,), (Input(0),)),
            _function(
                "full", (ArrayInput(2.5, "float32"),), ((2, 3), Input(0)), (("like", Input(0)),)
            ),
            _function("eye", (_REAL,), (3,), (("like", Input(0)),)),
            _function(
                "zeros",
                (_REAL,),
                ((2, 3),),
                (("dtype", DType("float32")), ("like", Input(0))),
            ),
            _function(
                "ones",
                (_REAL,),
                ((2, 3),),
                (("dtype", DType("float32")), ("like", Input(0))),
            ),
            _function(
                "empty",
                (_REAL,),
                ((2, 3),),
                (("dtype", DType("float32")), ("like", Input(0))),
                compare_values=False,
            ),
            _function("zeros_like", (_MATRIX,), (Input(0),)),
            _function("ones_like", (_MATRIX,), (Input(0),)),
            _function("full_like", (_MATRIX, ArrayInput(2.5, "float64")), (Input(0), Input(1))),
        )
    )
    return tuple(cases)


def _linalg_cases() -> tuple[NumpySupportCase, ...]:
    unary = (
        "cholesky",
        "det",
        "eigvalsh",
        "inv",
        "norm",
        "pinv",
        "svdvals",
    )
    cases = [
        _function(
            f"linalg.{name}",
            (_RECTANGULAR if name in {"pinv", "qr", "svd", "svdvals"} else _MATRIX,),
            (Input(0),),
        )
        for name in unary
    ]
    cases.extend(
        (
            _function("linalg.solve", (_MATRIX, _VECTOR), (Input(0), Input(1))),
            _function("linalg.matrix_power", (_MATRIX,), (Input(0), 3)),
            _function("linalg.diagonal", (_MATRIX,), (Input(0),)),
            _function("linalg.trace", (_MATRIX,), (Input(0),)),
            _function(
                "linalg.eigh",
                (_MATRIX,),
                (Input(0),),
                result_adapter="tuple",
            ),
            _function(
                "linalg.svd",
                (_RECTANGULAR,),
                (Input(0),),
                (("full_matrices", False),),
                result_adapter="tuple",
            ),
            _function(
                "linalg.vecdot",
                (_RECTANGULAR, _RECTANGULAR),
                (Input(0), Input(1)),
                (("axis", -1),),
            ),
        )
    )
    return tuple(cases)


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
    return tuple(_function(f"fft.{name}", (value,), (Input(0),)) for name, value in inputs.items())


def _method_cases() -> tuple[NumpySupportCase, ...]:
    array_methods = (
        NumpySupportCase(
            "numpy.ndarray.astype", "array_method", (_REAL,), (DType("float32"),), ((0,),)
        ),
        NumpySupportCase("numpy.ndarray.copy", "array_method", (_REAL,), (), ((0,),)),
        NumpySupportCase(
            "numpy.ndarray.item",
            "array_method",
            (ArrayInput([2.0], "float64"),),
            (),
            ((0,),),
        ),
        NumpySupportCase("numpy.ndarray.reshape", "array_method", (_REAL,), ((2, 2),), ((0,),)),
        NumpySupportCase("numpy.ndarray.sum", "array_method", (_REAL,), (), ((0,),)),
        NumpySupportCase(
            "numpy.ndarray.transpose",
            "array_method",
            (_MATRIX,),
            (),
            ((0,),),
            modes=_DYNAMIC_ONLY,
        ),
    )
    methods = [
        NumpySupportCase("numpy.add.reduce", "ufunc_method", (_REAL,), (), ((0,),)),
        NumpySupportCase("numpy.multiply.reduce", "ufunc_method", (_POSITIVE,), (), ((0,),)),
        NumpySupportCase("numpy.add.accumulate", "ufunc_method", (_REAL,), (), ((0,),)),
        NumpySupportCase("numpy.multiply.accumulate", "ufunc_method", (_POSITIVE,), (), ((0,),)),
    ]
    for name in (
        "add",
        "arctan2",
        "divide",
        "floor_divide",
        "hypot",
        "logaddexp",
        "maximum",
        "minimum",
        "multiply",
        "power",
        "remainder",
        "subtract",
    ):
        left, right = (
            (_POSITIVE, _NONZERO)
            if name in {"divide", "floor_divide", "remainder"}
            else (_POSITIVE, _RIGHT)
        )
        methods.append(
            NumpySupportCase(
                f"numpy.{name}.outer",
                "ufunc_method",
                (left, right),
                (Input(1),),
                ((0,), (1,), (0, 1)),
            )
        )
    return (*array_methods, *methods)


def _additional_existing_function_cases() -> tuple[NumpySupportCase, ...]:
    """Qualify forms whose concrete runtime implementation was never deleted."""
    metadata_cases = (
        _function(
            "can_cast",
            (_REAL,),
            (Input(0), DType("complex128")),
            modes=_DYNAMIC_ONLY,
            derivative_argnums=None,
            result_adapter="array",
        ),
        _function(
            "common_type",
            (_REAL, _INDEX),
            (Input(0), Input(1)),
            modes=_DYNAMIC_ONLY,
            compare_values=False,
            derivative_argnums=None,
            result_adapter="dtype_num",
        ),
        _function(
            "iscomplexobj",
            (_COMPLEX,),
            (Input(0),),
            modes=_DYNAMIC_ONLY,
            derivative_argnums=None,
            result_adapter="array",
        ),
        _function(
            "isrealobj",
            (_REAL,),
            (Input(0),),
            modes=_DYNAMIC_ONLY,
            derivative_argnums=None,
            result_adapter="array",
        ),
        _function(
            "ndim",
            (_MATRIX,),
            (Input(0),),
            modes=_DYNAMIC_ONLY,
            derivative_argnums=None,
            result_adapter="array",
        ),
        _function(
            "result_type",
            (_REAL,),
            (Input(0), DType("float32")),
            modes=_DYNAMIC_ONLY,
            compare_values=False,
            derivative_argnums=None,
            result_adapter="dtype_num",
        ),
        _function(
            "shape",
            (_MATRIX,),
            (Input(0),),
            modes=_DYNAMIC_ONLY,
            derivative_argnums=None,
            result_adapter="array",
        ),
        _function(
            "size",
            (_MATRIX,),
            (Input(0),),
            derivative_argnums=None,
            result_adapter="array",
        ),
    )
    mutation_cases = (
        _function(
            "copyto",
            (_MATRIX, ArrayInput([[5.0, 6.0], [7.0, 8.0]], "float64")),
            (Input(0), Input(1)),
            modes=_DYNAMIC_ONLY,
            return_input=0,
            derivative_argnums=((0,), (0, 1)),
        ),
        _function(
            "fill_diagonal",
            (_MATRIX, _VECTOR),
            (Input(0), Input(1)),
            modes=_DYNAMIC_ONLY,
            return_input=0,
            derivative_argnums=((0,), (0, 1)),
        ),
        _function(
            "place",
            (_MATRIX, _BOOL, _VECTOR),
            (Input(0), Input(1), Input(2)),
            modes=_DYNAMIC_ONLY,
            return_input=0,
            derivative_argnums=((0,), (0, 2)),
        ),
        _function(
            "put",
            (_REAL, _INDEX, _RIGHT),
            (Input(0), Input(1), Input(2)),
            modes=_DYNAMIC_ONLY,
            return_input=0,
            derivative_argnums=((0,), (0, 2)),
        ),
        _function(
            "put_along_axis",
            (
                _MATRIX,
                ArrayInput([[1, 0], [0, 1]], "int64"),
                _MATRIX,
            ),
            (Input(0), Input(1), Input(2), 1),
            modes=_DYNAMIC_ONLY,
            return_input=0,
            derivative_argnums=((0,), (0, 2)),
        ),
        _function(
            "putmask",
            (_MATRIX, _BOOL, _VECTOR),
            (Input(0), Input(1), Input(2)),
            modes=_DYNAMIC_ONLY,
            return_input=0,
            derivative_argnums=((0,), (0, 2)),
        ),
    )
    return (
        _function(
            "argmax",
            (_MATRIX,),
            (Input(0),),
            (("axis", 1),),
            derivative_argnums=None,
        ),
        _function(
            "argmin",
            (_MATRIX,),
            (Input(0),),
            (("axis", 1),),
            derivative_argnums=None,
        ),
        _function("astype", (_REAL,), (Input(0), DType("float32"))),
        _function("real", (_COMPLEX,), (Input(0),)),
        _function("trace", (_MATRIX,), (Input(0),)),
        _function("tril", (_MATRIX,), (Input(0),)),
        *metadata_cases,
        *mutation_cases,
    )


def _version_conditional_function_cases() -> tuple[NumpySupportCase, ...]:
    """Qualify aliases only on the NumPy minors that still expose them."""
    return (
        _function(
            "in1d",
            (_REAL, _RIGHT),
            (Input(0), Input(1)),
            modes=_DYNAMIC_ONLY,
            derivative_argnums=None,
            expected_deprecation=r"`in1d` is deprecated",
        ),
        _function(
            "trapz",
            (_REAL,),
            (Input(0),),
            modes=_DYNAMIC_ONLY,
            expected_deprecation=r"`trapz` is deprecated",
        ),
    )


def _additional_outer_cases() -> tuple[NumpySupportCase, ...]:
    """Qualify omitted ordinary ufunc outer spellings."""

    def outer(
        name: str,
        left: ArrayInput,
        right: ArrayInput,
        *,
        modes: tuple[str, ...] = _ALL_MODES,
        derivative_argnums: DerivativeArgnums | Literal["auto"] | None = "auto",
    ) -> NumpySupportCase:
        callable_name = f"numpy.{name}.outer"
        groups = (
            _derivative_argnums(callable_name, (left, right))
            if derivative_argnums == "auto"
            else derivative_argnums
        )
        return NumpySupportCase(
            callable=callable_name,
            kind="ufunc_method",
            inputs=(left, right),
            args=(Input(1),),
            derivative_argnums=groups,
            modes=modes,
        )

    nondifferentiable = (
        outer("bitwise_and", _INT_LEFT, _INT_RIGHT, derivative_argnums=None),
        outer("bitwise_or", _INT_LEFT, _INT_RIGHT, derivative_argnums=None),
        outer("bitwise_xor", _INT_LEFT, _INT_RIGHT, derivative_argnums=None),
        outer("equal", _REAL, _RIGHT, derivative_argnums=None),
        outer("greater", _REAL, _RIGHT, derivative_argnums=None),
        outer("greater_equal", _REAL, _RIGHT, derivative_argnums=None),
        outer("less", _REAL, _RIGHT, derivative_argnums=None),
        outer("less_equal", _REAL, _RIGHT, derivative_argnums=None),
        outer("logical_and", _BOOL, _BOOL, derivative_argnums=None),
        outer("logical_or", _BOOL, _BOOL, derivative_argnums=None),
        outer("logical_xor", _BOOL, _BOOL, derivative_argnums=None),
        outer("not_equal", _REAL, _RIGHT, derivative_argnums=None),
    )
    return (
        *nondifferentiable,
        outer("copysign", _NONZERO, _NONZERO, derivative_argnums=((0,),)),
        outer("float_power", _POSITIVE, _RIGHT, modes=_DYNAMIC_ONLY),
        outer("fmax", _REAL, _RIGHT, modes=_DYNAMIC_ONLY),
        outer("fmin", _REAL, _RIGHT, modes=_DYNAMIC_ONLY),
        outer("fmod", _POSITIVE, _NONZERO, modes=_DYNAMIC_ONLY),
        outer("heaviside", _NONZERO, _RIGHT),
        outer("ldexp", _REAL, ArrayInput([1, 2, -1, 0], "int32")),
        outer("logaddexp2", _REAL, _RIGHT, modes=_DYNAMIC_ONLY),
        outer("nextafter", _REAL, _RIGHT),
    )


def support_cases() -> tuple[NumpySupportCase, ...]:
    """Return the complete executable NumPy lifetime contract."""
    from advect_numpy_tests._support_case_families import (  # noqa: PLC0415
        function_family_cases,
    )

    cases = (
        *_ufunc_cases(),
        *_function_cases(),
        *_linalg_cases(),
        *_fft_cases(),
        *_method_cases(),
        *_additional_existing_function_cases(),
        *_version_conditional_function_cases(),
        *_additional_outer_cases(),
        *function_family_cases(),
    )
    identifiers = [case.identifier for case in cases]
    if len(identifiers) != len(set(identifiers)):
        duplicates = sorted(
            identifier for identifier in set(identifiers) if identifiers.count(identifier) > 1
        )
        message = f"duplicate NumPy support cases: {duplicates}"
        raise RuntimeError(message)
    return tuple(sorted(cases, key=lambda case: case.identifier))


__all__ = [
    "ArrayInput",
    "DType",
    "Function",
    "Input",
    "NumpySupportCase",
    "support_cases",
]
