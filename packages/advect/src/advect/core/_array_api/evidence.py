"""Executable evidence for Advect's declared Array API support surface.

The catalog is deliberately data-only.  A case describes provider inputs and
one namespace call without importing an Advect runtime.  Qualification runners
can therefore execute the same case directly, under a concrete trace, through
an abstractly staged program, and after staged-program serialization.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from advect.core._array_api.profiles import (
    LATEST_ARRAY_API_VERSION,
    materialize_array_api_profile,
)
from advect.core._array_api.signatures import official_parameter_names, official_signatures

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "ArrayInput",
    "DType",
    "Device",
    "Input",
    "MetadataCase",
    "OperationCase",
    "case_parameter_values",
    "input_indices",
    "metadata_cases",
    "operation_cases",
    "operation_evidence_cases",
    "operation_modes",
    "static_variant",
    "static_variant_requirements",
]


@dataclass(frozen=True, slots=True)
class ArrayInput:
    """One provider array constructed from portable Python data."""

    data: object
    dtype: str


@dataclass(frozen=True, slots=True)
class Input:
    """Reference an array input by position inside call arguments."""

    index: int


@dataclass(frozen=True, slots=True)
class DType:
    """Resolve a dtype name against the active provider namespace."""

    name: str


@dataclass(frozen=True, slots=True)
class Device:
    """Resolve the device of one materialized provider input."""

    input_index: int = 0


@dataclass(frozen=True, slots=True)
class OperationCase:
    """One executable callable/parameter/static-variant contract."""

    path: str
    inputs: tuple[ArrayInput, ...]
    args: tuple[object, ...]
    kwargs: tuple[tuple[str, object], ...] = ()
    compare_values: bool = True
    portable: bool = False
    variant: str = "baseline"
    modes: tuple[str, ...] = ("dynamic", "staged", "serialized")

    @property
    def identifier(self) -> str:
        """Return the stable case identifier."""
        if self.variant == "baseline":
            return self.path
        return f"{self.path}[{self.variant}]"


@dataclass(frozen=True, slots=True)
class MetadataCase:
    """One compile-time metadata invocation embedded in an array program."""

    path: str
    data: object
    dtype: str
    parameters: tuple[str, ...]
    modes: tuple[str, ...] = ("dynamic", "staged", "serialized")

    @property
    def identifier(self) -> str:
        """Return the stable case identifier."""
        return self.path


@dataclass(frozen=True, slots=True)
class _SignatureParameter:
    name: str
    positional: bool
    has_default: bool
    default: object = None
    variadic: bool = False


def _signature_parameters(
    path: str,
    version: str = LATEST_ARRAY_API_VERSION,
) -> tuple[_SignatureParameter, ...]:
    parsed = ast.parse(f"def operation{official_signatures(version)[path]}:\n    pass\n")
    function = parsed.body[0]
    if not isinstance(function, ast.FunctionDef):
        message = f"Could not parse official Array API signature for {path!r}"
        raise TypeError(message)
    arguments = function.args
    positional_nodes = (*arguments.posonlyargs, *arguments.args)
    positional_defaults = (None,) * (len(positional_nodes) - len(arguments.defaults)) + tuple(
        arguments.defaults
    )
    parameters = [
        _SignatureParameter(
            name=node.arg,
            positional=True,
            has_default=default is not None,
            default=ast.literal_eval(default) if default is not None else None,
        )
        for node, default in zip(positional_nodes, positional_defaults, strict=True)
    ]
    if arguments.vararg is not None:
        parameters.append(
            _SignatureParameter(
                name=arguments.vararg.arg,
                positional=True,
                has_default=False,
                variadic=True,
            )
        )
    parameters.extend(
        _SignatureParameter(
            name=node.arg,
            positional=False,
            has_default=default is not None,
            default=ast.literal_eval(default) if default is not None else None,
        )
        for node, default in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True)
    )
    return tuple(parameters)


def case_parameter_values(
    case: OperationCase,
    version: str = LATEST_ARRAY_API_VERSION,
) -> Mapping[str, object]:
    """Bind one evidence case to its frozen official signature."""
    parameters = _signature_parameters(case.path, version)
    positional = [
        parameter for parameter in parameters if parameter.positional and not parameter.variadic
    ]
    variadic = next((parameter for parameter in parameters if parameter.variadic), None)
    if variadic is None and len(case.args) > len(positional):
        message = f"{case.identifier} has too many positional arguments"
        raise ValueError(message)
    values = {
        parameter.name: value for parameter, value in zip(positional, case.args, strict=False)
    }
    if variadic is not None:
        values[variadic.name] = tuple(case.args[len(positional) :])
    for name, value in case.kwargs:
        if name in values:
            message = f"{case.identifier} binds {name!r} more than once"
            raise ValueError(message)
        values[name] = value
    for parameter in parameters:
        if parameter.name not in values:
            if not parameter.has_default:
                message = f"{case.identifier} does not bind required parameter {parameter.name!r}"
                raise ValueError(message)
            values[parameter.name] = parameter.default
    unknown = sorted(set(values).difference(parameter.name for parameter in parameters))
    if unknown:
        message = f"{case.identifier} binds unknown parameters {unknown!r}"
        raise ValueError(message)
    return values


def input_indices(value: object) -> frozenset[int]:
    """Return every runtime input referenced by a nested evidence value."""
    if isinstance(value, Input):
        return frozenset({value.index})
    if isinstance(value, (tuple, list)):
        return frozenset().union(*(input_indices(item) for item in value))
    if isinstance(value, dict):
        return frozenset().union(*(input_indices(item) for item in value.values()))
    return frozenset()


def static_variant_requirements(
    path: str,
    name: str,
    version: str = LATEST_ARRAY_API_VERSION,
) -> frozenset[str]:
    """Return the semantic variants required for one static parameter."""
    parameter = next(
        parameter for parameter in _signature_parameters(path, version) if parameter.name == name
    )
    return frozenset({"default", "explicit"} if parameter.has_default else {"explicit"})


def static_variant(
    case: OperationCase,
    name: str,
    version: str = LATEST_ARRAY_API_VERSION,
) -> str:
    """Classify one bound static value as the default or explicit variant."""
    parameter = next(
        parameter
        for parameter in _signature_parameters(case.path, version)
        if parameter.name == name
    )
    value = case_parameter_values(case, version)[name]
    if parameter.has_default and value == parameter.default:
        return "default"
    return "explicit"


_SENTINEL = ArrayInput([0.0], "float32")
_REAL = ArrayInput([[-1.5, -0.25, 0.5], [1.25, 2.0, 3.5]], "float64")
_POSITIVE = ArrayInput([[0.25, 0.5, 1.0], [1.5, 2.0, 4.0]], "float64")
_UNIT = ArrayInput([[-0.8, -0.25, 0.0], [0.25, 0.5, 0.8]], "float64")
_COMPLEX = ArrayInput(
    [[1.0 + 0.5j, -2.0 + 1.0j], [0.25 - 0.75j, 1.5 + 2.0j]],
    "complex128",
)
_INTEGER = ArrayInput([[1, 2, 3], [4, 5, 6]], "int32")
_INTEGER_RIGHT = ArrayInput([[3, 1, 7], [2, 6, 4]], "int32")
_BOOL = ArrayInput([[True, False, True], [False, False, True]], "bool")
_BOOL_RIGHT = ArrayInput([[False, False, True], [True, False, True]], "bool")
_RIGHT = ArrayInput([[0.75, 1.5, 2.0], [2.5, 1.25, 0.5]], "float64")
_DYNAMIC_ONLY_PATHS = frozenset(
    {
        "arange",
        "astype",
        "empty",
        "eye",
        "fft.fftfreq",
        "fft.rfftfreq",
        "linspace",
        "nonzero",
        "ones",
        "unique_all",
        "unique_counts",
        "unique_inverse",
        "unique_values",
        "zeros",
    }
)


def operation_modes(path: str) -> tuple[str, ...]:
    """Return the lifetimes every evidence case for ``path`` must execute."""
    if path in _DYNAMIC_ONLY_PATHS:
        return ("dynamic",)
    return ("dynamic", "staged", "serialized")


def _alternate_static_value(path: str, name: str, default: object) -> object:
    if name == "device":
        return Device()
    if name == "dtype":
        return DType("float32")
    if name == "copy":
        return default is not True
    if name == "mode" and path == "linalg.qr":
        return "complete"
    if name == "axis" and path == "linalg.cross":
        return -2
    if name == "axis" and path in {"linalg.vecdot", "vecdot"}:
        return -2
    if name == "indexing":
        return "ij"
    alternatives = {
        "axis": 1,
        "descending": True,
        "include_initial": True,
        "keepdims": True,
        "n": 3,
        "norm": "ortho",
        "ord": 1,
        "s": (2, 4),
        "stable": False,
    }
    if name in alternatives:
        return alternatives[name]
    message = (
        f"{path}.{name} has a default-valued static parameter but no "
        "executable alternate evidence value"
    )
    raise ValueError(message)


def _replace_parameter(
    case: OperationCase,
    *,
    name: str,
    value: object,
    semantic_variant: str,
    version: str = LATEST_ARRAY_API_VERSION,
) -> OperationCase:
    values = dict(case_parameter_values(case, version))
    values[name] = value
    if name == "axes" and value is None and case.path in {"fft.irfftn", "fft.rfftn"}:
        values["s"] = None
    parameters = _signature_parameters(case.path, version)
    inputs = case.inputs
    if name == "axis" and semantic_variant == "default":
        if case.path == "cumulative_prod":
            inputs = (ArrayInput([0.25, 0.5, 1.0, 1.5], "float64"),)
        elif case.path == "cumulative_sum":
            inputs = (ArrayInput([-1.5, -0.25, 0.5, 1.25], "float64"),)
        elif case.path == "take":
            inputs = (
                ArrayInput([1.0, 2.0, 3.0, 4.0], "float32"),
                case.inputs[1],
            )
    elif name == "axis" and case.path == "linalg.cross":
        inputs = (
            ArrayInput([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], "float64"),
            ArrayInput([[0.5, 1.5], [-1.0, 0.25], [2.0, -0.75]], "float64"),
        )
    elif (
        name == "axes"
        and semantic_variant == "default"
        and case.path in {"linalg.tensordot", "tensordot"}
    ):
        inputs = (
            ArrayInput(
                [
                    [
                        [1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 12.0],
                    ],
                    [
                        [13.0, 14.0, 15.0, 16.0],
                        [17.0, 18.0, 19.0, 20.0],
                        [21.0, 22.0, 23.0, 24.0],
                    ],
                ],
                "float64",
            ),
            ArrayInput(
                [
                    [[1.0, 0.5], [0.0, 1.0], [1.5, -0.5], [2.0, 0.25]],
                    [[0.5, 1.0], [1.0, 0.0], [-1.0, 2.0], [0.25, 0.75]],
                    [[1.0, -1.0], [0.5, 0.5], [2.0, 1.0], [-0.5, 1.5]],
                ],
                "float64",
            ),
        )
    positional_args: list[object] = []
    for parameter in parameters:
        if not parameter.positional:
            continue
        value = values[parameter.name]
        if parameter.variadic:
            if not isinstance(value, tuple):
                message = f"{parameter.name} must contain variadic positional values"
                raise TypeError(message)
            positional_args.extend(value)
        else:
            positional_args.append(value)
    return replace(
        case,
        inputs=inputs,
        args=tuple(positional_args),
        kwargs=tuple(
            (parameter.name, values[parameter.name])
            for parameter in parameters
            if not parameter.positional
        ),
        portable=False,
        variant=f"{name}={semantic_variant}",
    )


def _unary_cases() -> list[OperationCase]:
    by_input = {
        "abs": _REAL,
        "acos": _UNIT,
        "acosh": ArrayInput([[1.1, 1.25, 2.0], [3.0, 4.0, 8.0]], "float64"),
        "asin": _UNIT,
        "asinh": _REAL,
        "atan": _REAL,
        "atanh": _UNIT,
        "ceil": _REAL,
        "cos": _REAL,
        "cosh": _REAL,
        "exp": _REAL,
        "expm1": _REAL,
        "floor": _REAL,
        "log": _POSITIVE,
        "log10": _POSITIVE,
        "log1p": _UNIT,
        "log2": _POSITIVE,
        "negative": _REAL,
        "positive": _REAL,
        "reciprocal": _POSITIVE,
        "round": _REAL,
        "sign": _REAL,
        "sin": _REAL,
        "sinh": _REAL,
        "sqrt": _POSITIVE,
        "square": _REAL,
        "tan": _UNIT,
        "tanh": _REAL,
        "trunc": _REAL,
    }
    portable = {
        "abs",
        "cos",
        "exp",
        "log",
        "negative",
        "sin",
        "sqrt",
        "square",
        "tanh",
    }
    cases = [
        OperationCase(
            path=path,
            inputs=(input_value,),
            args=(Input(0),),
            portable=path in portable,
        )
        for path, input_value in by_input.items()
    ]
    cases.extend(
        [
            OperationCase("bitwise_invert", (_INTEGER,), (Input(0),)),
            OperationCase("conj", (_COMPLEX,), (Input(0),)),
            OperationCase("imag", (_COMPLEX,), (Input(0),)),
            OperationCase("real", (_COMPLEX,), (Input(0),)),
            OperationCase("logical_not", (_BOOL,), (Input(0),)),
            OperationCase(
                "isfinite",
                (ArrayInput([0.0, float("inf"), float("nan")], "float64"),),
                (Input(0),),
            ),
            OperationCase(
                "isinf",
                (ArrayInput([0.0, float("inf"), float("nan")], "float64"),),
                (Input(0),),
            ),
            OperationCase(
                "isnan",
                (ArrayInput([0.0, float("inf"), float("nan")], "float64"),),
                (Input(0),),
            ),
            OperationCase("signbit", (_REAL,), (Input(0),)),
        ]
    )
    return cases


def _binary_cases() -> list[OperationCase]:
    real_paths = (
        "add",
        "atan2",
        "copysign",
        "divide",
        "equal",
        "floor_divide",
        "greater",
        "greater_equal",
        "hypot",
        "less",
        "less_equal",
        "logaddexp",
        "maximum",
        "minimum",
        "multiply",
        "nextafter",
        "not_equal",
        "remainder",
        "subtract",
    )
    portable = {"add", "atan2", "divide", "equal", "multiply", "subtract"}
    cases = [
        OperationCase(
            path=path,
            inputs=(_POSITIVE, _RIGHT),
            args=(Input(0), Input(1)),
            portable=path in portable,
        )
        for path in real_paths
    ]
    cases.append(
        OperationCase(
            "pow",
            (_POSITIVE, ArrayInput([[2.0, 0.5, 1.5], [1.0, 2.0, 0.5]], "float64")),
            (Input(0), Input(1)),
        )
    )
    cases.extend(
        OperationCase(
            path,
            (_INTEGER, _INTEGER_RIGHT),
            (Input(0), Input(1)),
        )
        for path in ("bitwise_and", "bitwise_or", "bitwise_xor")
    )
    shifts = ArrayInput([[0, 1, 2], [1, 0, 2]], "int32")
    cases.extend(
        OperationCase(path, (_INTEGER, shifts), (Input(0), Input(1)))
        for path in ("bitwise_left_shift", "bitwise_right_shift")
    )
    cases.extend(
        OperationCase(
            path,
            (_BOOL, _BOOL_RIGHT),
            (Input(0), Input(1)),
        )
        for path in ("logical_and", "logical_or", "logical_xor")
    )
    return cases


def _reduction_cases() -> list[OperationCase]:
    return [
        OperationCase(
            "all",
            (_BOOL,),
            (Input(0),),
            (("axis", 1), ("keepdims", True)),
        ),
        OperationCase(
            "any",
            (_BOOL,),
            (Input(0),),
            (("axis", 0), ("keepdims", True)),
        ),
        OperationCase("argmax", (_REAL,), (Input(0),), (("axis", 1),)),
        OperationCase("argmin", (_REAL,), (Input(0),), (("axis", 0),)),
        OperationCase(
            "count_nonzero",
            (_INTEGER,),
            (Input(0),),
            (("axis", 1), ("keepdims", True)),
        ),
        OperationCase("cumulative_prod", (_POSITIVE,), (Input(0),), (("axis", 1),)),
        OperationCase("cumulative_sum", (_REAL,), (Input(0),), (("axis", 1),)),
        OperationCase("max", (_REAL,), (Input(0),), (("axis", 1),), portable=True),
        OperationCase(
            "mean",
            (_REAL,),
            (Input(0),),
            (("axis", 0), ("keepdims", True)),
            portable=True,
        ),
        OperationCase("min", (_REAL,), (Input(0),), (("axis", 0),)),
        OperationCase("prod", (_POSITIVE,), (Input(0),), (("axis", 1),)),
        OperationCase(
            "std",
            (_REAL,),
            (Input(0),),
            (("axis", 1), ("correction", 1.0), ("keepdims", True)),
        ),
        OperationCase(
            "sum",
            (_REAL,),
            (Input(0),),
            (("axis", 1), ("keepdims", True)),
            portable=True,
        ),
        OperationCase(
            "var",
            (_REAL,),
            (Input(0),),
            (("axis", 0), ("correction", 1.0)),
        ),
    ]


def _shape_cases() -> list[OperationCase]:
    matrix = ArrayInput([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "float32")
    second = ArrayInput([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], "float32")
    return [
        OperationCase(
            "asarray",
            (matrix,),
            (Input(0),),
            (("dtype", DType("float64")), ("copy", True)),
        ),
        OperationCase(
            "astype",
            (matrix,),
            (Input(0), DType("float64")),
        ),
        OperationCase(
            "broadcast_to",
            (ArrayInput([[1.0, 2.0, 3.0]], "float32"),),
            (Input(0), (2, 3)),
        ),
        OperationCase(
            "broadcast_arrays",
            (
                ArrayInput([1.0, 2.0, 3.0], "float32"),
                ArrayInput([[4.0], [5.0]], "float32"),
            ),
            (Input(0), Input(1)),
        ),
        OperationCase(
            "concat",
            (matrix, second),
            ((Input(0), Input(1)),),
            (("axis", 0),),
            portable=True,
        ),
        OperationCase(
            "diff",
            (
                matrix,
                ArrayInput([[0.0], [1.0]], "float32"),
                ArrayInput([[7.0], [8.0]], "float32"),
            ),
            (Input(0),),
            (
                ("axis", 1),
                ("n", 2),
                ("prepend", Input(1)),
                ("append", Input(2)),
            ),
        ),
        OperationCase("expand_dims", (matrix,), (Input(0),), (("axis", 1),)),
        OperationCase("flip", (matrix,), (Input(0),), (("axis", 1),)),
        OperationCase("matrix_transpose", (matrix,), (Input(0),)),
        OperationCase(
            "meshgrid",
            (
                ArrayInput([1.0, 2.0, 3.0], "float32"),
                ArrayInput([4.0, 5.0], "float32"),
            ),
            (Input(0), Input(1)),
        ),
        OperationCase(
            "moveaxis",
            (ArrayInput([[[1.0, 2.0]], [[3.0, 4.0]]], "float32"),),
            (Input(0), 0, 2),
        ),
        OperationCase("permute_dims", (matrix,), (Input(0), (1, 0)), portable=True),
        OperationCase(
            "nonzero",
            (ArrayInput([[0, 2, 0], [4, 0, 6]], "int32"),),
            (Input(0),),
        ),
        OperationCase("repeat", (matrix,), (Input(0), 2), (("axis", 1),)),
        OperationCase("reshape", (matrix,), (Input(0), (3, 2)), portable=True),
        OperationCase(
            "roll",
            (matrix,),
            (Input(0), 1),
            (("axis", 1),),
        ),
        OperationCase(
            "squeeze",
            (ArrayInput([[[1.0, 2.0, 3.0]]], "float32"),),
            (Input(0),),
            (("axis", 1),),
        ),
        OperationCase(
            "argsort",
            (ArrayInput([[3.0, 1.0, 2.0], [0.0, -1.0, 4.0]], "float64"),),
            (Input(0),),
            (("axis", 1), ("descending", True), ("stable", True)),
        ),
        OperationCase(
            "searchsorted",
            (
                ArrayInput([1.0, 3.0, 5.0, 7.0], "float64"),
                ArrayInput([0.0, 3.0, 4.0, 9.0], "float64"),
                ArrayInput([1, 0, 3, 2], "int64"),
            ),
            (Input(0), Input(1)),
            (("side", "right"), ("sorter", Input(2))),
        ),
        OperationCase(
            "sort",
            (ArrayInput([[3.0, 1.0, 2.0], [0.0, -1.0, 4.0]], "float64"),),
            (Input(0),),
            (("axis", 1),),
            portable=True,
        ),
        OperationCase(
            "stack",
            (matrix, second),
            ((Input(0), Input(1)),),
            (("axis", 1),),
            portable=True,
        ),
        OperationCase(
            "take",
            (
                matrix,
                ArrayInput([2, 0, 2, 1], "int64"),
            ),
            (Input(0), Input(1)),
            (("axis", 1),),
        ),
        OperationCase(
            "take_along_axis",
            (
                second,
                ArrayInput([[2, 0], [1, 1]], "int64"),
            ),
            (Input(0), Input(1)),
            (("axis", 1),),
        ),
        OperationCase("tile", (matrix,), (Input(0), (2, 1))),
        OperationCase("tril", (matrix,), (Input(0),), (("k", -1),)),
        OperationCase("triu", (matrix,), (Input(0),), (("k", 1),)),
        OperationCase(
            "unique_all",
            (ArrayInput([[2.0, 1.0], [2.0, 3.0]], "float64"),),
            (Input(0),),
        ),
        OperationCase(
            "unique_counts",
            (ArrayInput([[2.0, 1.0], [2.0, 3.0]], "float64"),),
            (Input(0),),
        ),
        OperationCase(
            "unique_inverse",
            (ArrayInput([[2.0, 1.0], [2.0, 3.0]], "float64"),),
            (Input(0),),
        ),
        OperationCase(
            "unique_values",
            (ArrayInput([[2.0, 1.0], [2.0, 3.0]], "float64"),),
            (Input(0),),
        ),
        OperationCase("unstack", (matrix,), (Input(0),), (("axis", 1),)),
    ]


def _creation_cases() -> list[OperationCase]:
    matrix = ArrayInput([[1.0, 2.0], [3.0, 4.0]], "float32")
    return [
        OperationCase(
            "arange",
            (_SENTINEL,),
            (1, 8, 2),
            (("dtype", DType("float32")),),
        ),
        OperationCase(
            "empty",
            (_SENTINEL,),
            ((2, 3),),
            (("dtype", DType("float32")),),
            compare_values=False,
        ),
        OperationCase("empty_like", (matrix,), (Input(0),), compare_values=False),
        OperationCase(
            "eye",
            (_SENTINEL,),
            (3, 4),
            (("k", 1), ("dtype", DType("float64"))),
        ),
        OperationCase(
            "full",
            (ArrayInput(2.5, "float32"),),
            ((2, 3), Input(0)),
            (("dtype", DType("float32")),),
        ),
        OperationCase(
            "full_like",
            (matrix, ArrayInput(2.5, "float32")),
            (Input(0), Input(1)),
            (("dtype", DType("float64")),),
        ),
        OperationCase(
            "linspace",
            (_SENTINEL,),
            (-1.0, 1.0, 5),
            (("dtype", DType("float32")), ("endpoint", False)),
            portable=True,
        ),
        OperationCase(
            "ones",
            (_SENTINEL,),
            ((2, 3),),
            (("dtype", DType("float32")),),
        ),
        OperationCase("ones_like", (matrix,), (Input(0),)),
        OperationCase(
            "zeros",
            (_SENTINEL,),
            ((2, 3),),
            (("dtype", DType("float32")),),
        ),
        OperationCase("zeros_like", (matrix,), (Input(0),)),
    ]


def _fft_cases() -> list[OperationCase]:
    real = ArrayInput([[0.0, 1.0, 2.0, 3.0], [1.0, -1.0, 0.5, 2.0]], "float64")
    complex_input = ArrayInput(
        [
            [0.0 + 0.5j, 1.0 - 0.25j, 2.0 + 1.0j, 3.0 - 0.5j],
            [1.0 + 0.0j, -1.0 + 0.5j, 0.5 - 1.0j, 2.0 + 0.25j],
        ],
        "complex128",
    )
    half_spectrum = ArrayInput(
        [[1.0 + 0.0j, 0.5 - 0.25j, 2.0 + 0.0j], [0.5 + 0.0j, -1.0 + 0.5j, 1.0 + 0.0j]],
        "complex128",
    )
    return [
        OperationCase("fft.fft", (complex_input,), (Input(0),), (("axis", 1),), portable=True),
        OperationCase("fft.ifft", (complex_input,), (Input(0),), (("axis", 1),)),
        OperationCase(
            "fft.fftn",
            (complex_input,),
            (Input(0),),
            (("axes", (0, 1)), ("norm", "ortho")),
        ),
        OperationCase(
            "fft.fftfreq",
            (_SENTINEL,),
            (6,),
            (("d", 0.25), ("dtype", DType("float64"))),
        ),
        OperationCase(
            "fft.hfft",
            (half_spectrum,),
            (Input(0),),
            (("n", 4), ("axis", 1)),
        ),
        OperationCase(
            "fft.ifftn",
            (complex_input,),
            (Input(0),),
            (("axes", (0, 1)), ("norm", "ortho")),
        ),
        OperationCase(
            "fft.ihfft",
            (real,),
            (Input(0),),
            (("n", 4), ("axis", 1)),
        ),
        OperationCase("fft.rfft", (real,), (Input(0),), (("axis", 1),), portable=True),
        OperationCase(
            "fft.rfftfreq",
            (_SENTINEL,),
            (6,),
            (("d", 0.25), ("dtype", DType("float64"))),
        ),
        OperationCase("fft.irfft", (half_spectrum,), (Input(0),), (("n", 4), ("axis", 1))),
        OperationCase(
            "fft.rfftn",
            (real,),
            (Input(0),),
            (("s", (2, 4)), ("axes", (0, 1))),
        ),
        OperationCase(
            "fft.irfftn",
            (half_spectrum,),
            (Input(0),),
            (("s", (2, 4)), ("axes", (0, 1))),
        ),
        OperationCase("fft.fftshift", (real,), (Input(0),), (("axes", (0, 1)),)),
        OperationCase("fft.ifftshift", (real,), (Input(0),), (("axes", (0, 1)),)),
    ]


def _linalg_cases() -> list[OperationCase]:
    matrix = ArrayInput([[4.0, 1.0], [1.0, 3.0]], "float64")
    rectangular = ArrayInput([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]], "float64")
    left = ArrayInput([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "float64")
    right = ArrayInput([[1.0, 2.0], [0.5, 1.0], [2.0, -1.0]], "float64")
    return [
        OperationCase(
            "linalg.cholesky",
            (matrix,),
            (Input(0),),
            (("upper", True),),
        ),
        OperationCase(
            "linalg.cross",
            (
                left,
                ArrayInput([[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]], "float64"),
            ),
            (Input(0), Input(1)),
        ),
        OperationCase(
            "linalg.diagonal",
            (ArrayInput([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "float64"),),
            (Input(0),),
            (("offset", 1),),
        ),
        OperationCase("linalg.det", (matrix,), (Input(0),), portable=True),
        OperationCase("linalg.eigh", (matrix,), (Input(0),)),
        OperationCase("linalg.eigvalsh", (matrix,), (Input(0),), portable=True),
        OperationCase("linalg.inv", (matrix,), (Input(0),), portable=True),
        OperationCase("linalg.matrix_power", (matrix,), (Input(0), 3)),
        OperationCase(
            "linalg.matrix_rank",
            (
                ArrayInput(
                    [
                        [[1.0, 0.0], [0.0, 0.1]],
                        [[1.0, 0.0], [0.0, 0.1]],
                    ],
                    "float64",
                ),
                ArrayInput([0.05, 0.2], "float64"),
            ),
            (Input(0),),
            (("rtol", Input(1)),),
        ),
        OperationCase(
            "linalg.matmul",
            (left, right),
            (Input(0), Input(1)),
        ),
        OperationCase("linalg.matrix_transpose", (left,), (Input(0),)),
        OperationCase(
            "linalg.matrix_norm",
            (rectangular,),
            (Input(0),),
            (("keepdims", True), ("ord", "fro")),
        ),
        OperationCase(
            "linalg.outer",
            (
                ArrayInput([1.0, 2.0, 3.0], "float64"),
                ArrayInput([0.5, -1.0], "float64"),
            ),
            (Input(0), Input(1)),
        ),
        OperationCase(
            "linalg.pinv",
            (rectangular,),
            (Input(0),),
            portable=True,
        ),
        OperationCase("linalg.qr", (rectangular,), (Input(0),), (("mode", "reduced"),)),
        OperationCase("linalg.slogdet", (matrix,), (Input(0),)),
        OperationCase(
            "linalg.solve",
            (matrix, ArrayInput([1.0, 2.0], "float64")),
            (Input(0), Input(1)),
            portable=True,
        ),
        OperationCase(
            "linalg.svd",
            (rectangular,),
            (Input(0),),
            (("full_matrices", False),),
        ),
        OperationCase("linalg.svdvals", (rectangular,), (Input(0),), portable=True),
        OperationCase(
            "linalg.tensordot",
            (
                ArrayInput([[[1.0, 2.0], [3.0, 4.0]]], "float64"),
                ArrayInput([[1.0, 0.5, -1.0], [2.0, 1.0, 0.25]], "float64"),
            ),
            (Input(0), Input(1)),
            (("axes", 1),),
        ),
        OperationCase(
            "linalg.trace",
            (ArrayInput([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "float64"),),
            (Input(0),),
            (("offset", 1),),
        ),
        OperationCase(
            "linalg.vecdot",
            (left, ArrayInput([[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]], "float64")),
            (Input(0), Input(1)),
            (("axis", -1),),
        ),
        OperationCase(
            "linalg.vector_norm",
            (left,),
            (Input(0),),
            (("axis", 1), ("keepdims", True), ("ord", 2.0)),
        ),
        OperationCase("matmul", (left, right), (Input(0), Input(1)), portable=True),
        OperationCase(
            "tensordot",
            (
                ArrayInput([[[1.0, 2.0], [3.0, 4.0]]], "float64"),
                ArrayInput([[1.0, 0.5, -1.0], [2.0, 1.0, 0.25]], "float64"),
            ),
            (Input(0), Input(1)),
            (("axes", 1),),
        ),
        OperationCase(
            "vecdot",
            (left, ArrayInput([[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]], "float64")),
            (Input(0), Input(1)),
            (("axis", -1),),
        ),
    ]


def operation_cases(
    version: str = LATEST_ARRAY_API_VERSION,
) -> tuple[OperationCase, ...]:
    """Return one baseline executable case per array-returning callable."""
    cases = [
        *_unary_cases(),
        *_binary_cases(),
        *_reduction_cases(),
        *_shape_cases(),
        *_creation_cases(),
        *_fft_cases(),
        *_linalg_cases(),
        OperationCase(
            "clip",
            (
                _REAL,
                ArrayInput(-0.5, "float64"),
                ArrayInput(1.5, "float64"),
            ),
            (Input(0),),
            (("min", Input(1)), ("max", Input(2))),
            portable=True,
        ),
        OperationCase(
            "where",
            (_BOOL, _REAL, _RIGHT),
            (Input(0), Input(1), Input(2)),
            portable=True,
        ),
    ]
    cases.sort(key=_operation_case_identifier)
    identifiers = [case.identifier for case in cases]
    if len(identifiers) != len(set(identifiers)):
        duplicates = sorted(
            identifier for identifier in set(identifiers) if identifiers.count(identifier) > 1
        )
        msg = f"duplicate Array API qualification cases: {duplicates}"
        raise RuntimeError(msg)
    profile = materialize_array_api_profile(version)
    return tuple(
        replace(
            case,
            kwargs=tuple(
                (name, value)
                for name, value in case.kwargs
                if name in official_parameter_names(case.path, version)
            ),
            modes=operation_modes(case.path),
        )
        for case in cases
        if profile.admits(case.path)
    )


def _operation_case_identifier(case: OperationCase) -> str:
    return case.identifier


def operation_evidence_cases(
    static_parameters: Mapping[str, tuple[str, ...]],
    version: str = LATEST_ARRAY_API_VERSION,
) -> tuple[OperationCase, ...]:
    """Expand baselines into default and explicit static-parameter evidence."""
    evidence: list[OperationCase] = []
    for case in operation_cases(version):
        evidence.append(case)
        if case.path == "linalg.pinv":
            evidence.append(
                replace(
                    case,
                    inputs=(*case.inputs, ArrayInput(1e-7, "float64")),
                    kwargs=(("rtol", Input(1)),),
                    portable=False,
                    variant="rtol=live",
                )
            )
        if case.path == "linalg.matrix_power":
            evidence.extend(
                (
                    replace(
                        case,
                        args=(Input(0), 0),
                        portable=False,
                        variant="n=zero",
                    ),
                    replace(
                        case,
                        args=(Input(0), -2),
                        portable=False,
                        variant="n=negative",
                    ),
                )
            )
        if case.path == "linalg.matrix_rank":
            evidence.append(
                replace(
                    case,
                    inputs=case.inputs[:1],
                    kwargs=(),
                    portable=False,
                    variant="rtol=default",
                )
            )
        if case.path == "unstack":
            evidence.append(
                replace(
                    case,
                    inputs=(ArrayInput([], "float64"),),
                    kwargs=(),
                    variant="empty-axis",
                )
            )
        parameter_values = case_parameter_values(case, version)
        signature_parameters = {
            parameter.name: parameter for parameter in _signature_parameters(case.path, version)
        }
        for name in static_parameters.get(case.path, ()):
            parameter = signature_parameters[name]
            value = parameter_values[name]
            if not parameter.has_default:
                continue
            if value == parameter.default:
                alternate = _alternate_static_value(case.path, name, parameter.default)
                evidence.append(
                    _replace_parameter(
                        case,
                        name=name,
                        value=alternate,
                        semantic_variant="explicit",
                        version=version,
                    )
                )
            else:
                evidence.append(
                    _replace_parameter(
                        case,
                        name=name,
                        value=parameter.default,
                        semantic_variant="default",
                        version=version,
                    )
                )
    identifiers = [case.identifier for case in evidence]
    if len(identifiers) != len(set(identifiers)):
        duplicates = sorted(
            identifier for identifier in set(identifiers) if identifiers.count(identifier) > 1
        )
        message = f"duplicate Array API evidence cases: {duplicates}"
        raise RuntimeError(message)
    return tuple(evidence)


def metadata_cases() -> tuple[MetadataCase, ...]:
    """Return executable evidence for the five compile-time metadata callables."""
    return (
        MetadataCase("can_cast", [1.0, 2.0], "float32", ("from_", "to")),
        MetadataCase("finfo", [1.0, 2.0], "float32", ("type",)),
        MetadataCase("iinfo", [1, 2], "int32", ("type",)),
        MetadataCase("isdtype", [1.0, 2.0], "float32", ("dtype", "kind")),
        MetadataCase(
            "result_type",
            [1.0, 2.0],
            "float32",
            ("arrays_and_dtypes",),
        ),
    )
