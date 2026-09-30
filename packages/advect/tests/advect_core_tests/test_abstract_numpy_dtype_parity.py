"""Staged dtype declarations agree with NumPy 2 across every staged dtype."""

from __future__ import annotations

import itertools
import math
import operator
import warnings
from typing import TYPE_CHECKING, Any, cast

import array_api_strict as strict
import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import example, given

import advect as ad
from advect.core._abstract_domains import elementwise, operation_semantics, reductions
from advect.core._abstract_helpers import (
    can_cast_dtype,
    coerced_dtype,
    discovered_dtype,
    dtype_kind_bits,
    dtype_name,
    promote_dtype,
    safely_casts,
)
from advect.core._array_api.profiles import LATEST_ARRAY_API_VERSION

if TYPE_CHECKING:
    from collections.abc import Callable

    from hypothesis.strategies import DataObject

_DTYPES = (
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
)
_DTYPE_KINDS = (
    "bool",
    "signed integer",
    "unsigned integer",
    "integral",
    "real floating",
    "complex floating",
    "numeric",
)
_FINFO_FIELDS = ("bits", "eps", "max", "min", "smallest_normal")
_SEMANTICS = {name: (schema, evaluator) for name, schema, evaluator in operation_semantics()}
_PYTHON_SCALARS: tuple[object, ...] = (True, 1, 1.0, 1j)
_WEAK_SPECS = {
    type(scalar): ad.ArraySpec((), np.result_type(scalar).name, weak=True)
    for scalar in _PYTHON_SCALARS
}
# NumPy reads a Python int in a sequence by value: int64, else uint64, else object.
_PYTHON_INT_BOUNDARIES = (-(2**63) - 1, -(2**63), -1, 0, 2**63 - 1, 2**63, 2**64 - 1, 2**64)
_ROW = [0.1, 0.2]
_NUMPY_VECTOR = np.asarray([3.0, 4.0])
_STRICT_VECTOR = strict.asarray([3.0, 4.0])


def _numpy_result_dtype(
    function: Callable[..., object],
    operands: tuple[object, ...],
    **kwargs: object,
) -> str | None:
    """Return NumPy's eager result dtype, or ``None`` when NumPy has no loop.

    A string operand names an array dtype; any other operand is a Python scalar.
    """
    values = [np.ones(2, dtype=item) if isinstance(item, str) else item for item in operands]
    try:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(function(*values, **kwargs)).dtype.name
    except TypeError:
        return None


def _operand_spec(operand: object) -> ad.ArraySpec:
    if isinstance(operand, str):
        return ad.ArraySpec((2,), operand)
    return _WEAK_SPECS[type(operand)]


def _assert_staged_matches_eager(
    operation: Callable[..., object],
    *values: np.ndarray,
) -> None:
    expected = operation(*values)
    program = ad.stage(
        operation,
        specs=tuple(ad.ArraySpec(value.shape, value.dtype.name) for value in values),
    )
    for staged in (program, ad.StagedProgram.from_dict(program.to_dict())):
        actual = staged(*values)
        for actual_output, expected_output in zip(
            actual if isinstance(actual, tuple) else (actual,),
            expected if isinstance(expected, tuple) else (expected,),
            strict=True,
        ):
            actual_array, expected_array = np.asarray(actual_output), np.asarray(expected_output)
            assert (actual_array.shape, actual_array.dtype) == (
                expected_array.shape,
                expected_array.dtype,
            )
            np.testing.assert_array_equal(actual_array, expected_array)


def _dtype_sets(max_size: int) -> list[tuple[str, ...]]:
    return [
        dtypes for size in range(max_size + 1) for dtypes in itertools.combinations(_DTYPES, size)
    ]


def test_promotion_matches_numpy_result_type_for_every_strong_dtype_set() -> None:
    mismatches = [
        (dtypes, actual)
        for dtypes in _dtype_sets(len(_DTYPES))[1:]
        if (actual := promote_dtype([ad.ArraySpec((), dtype) for dtype in dtypes]))
        != np.result_type(*dtypes).name
    ]
    assert mismatches == []


def test_promotion_matches_nep50_for_python_scalar_operands() -> None:
    mismatches = []
    for dtypes in _dtype_sets(2):
        strong = [ad.ArraySpec((2,), dtype) for dtype in dtypes]
        for count in (1, 2, 3):
            for scalars in itertools.product(_PYTHON_SCALARS, repeat=count):
                specs = [*strong, *(_WEAK_SPECS[type(scalar)] for scalar in scalars)]
                expected = np.result_type(*dtypes, *scalars).name
                mismatches.extend(
                    (dtypes, scalars, actual)
                    for ordering in (specs, specs[::-1])
                    if (actual := promote_dtype(ordering)) != expected
                )
    assert mismatches == []


def test_sequence_dtype_discovery_matches_numpy_coercion_for_every_ordered_triple() -> None:
    mismatches = [
        (dtypes, actual)
        for dtypes in itertools.product(_DTYPES, repeat=3)
        if (actual := coerced_dtype(dtypes))
        != np.asarray([np.dtype(dtype).type(0) for dtype in dtypes]).dtype.name
    ]
    assert mismatches == []


def test_leaf_dtype_discovery_matches_numpy_coercion() -> None:
    leaves = (
        *_PYTHON_INT_BOUNDARIES,
        *_PYTHON_SCALARS,
        *(np.dtype(dtype).type(0) for dtype in _DTYPES),
    )
    mismatches = [
        (leaf, actual)
        for leaf in leaves
        if (actual := dtype_name(discovered_dtype(leaf))) != np.asarray([leaf]).dtype.name
    ]
    assert mismatches == []


def test_safe_casting_matches_numpy_for_every_dtype_pair() -> None:
    mismatches = [
        (source, target)
        for source, target in itertools.product(_DTYPES, repeat=2)
        if safely_casts(source, target) != np.can_cast(source, target, casting="safe")
    ]
    assert mismatches == []


@pytest.mark.parametrize("dtype", [dtype for dtype in _DTYPES if dtype != "float16"])
def test_dtype_categories_resolve_numpy_and_strict_dtypes_side_by_side(dtype: str) -> None:
    # A strict dtype hashes like its NumPy twin but warns when compared with
    # it, so memoized dtype names must never compare across providers.
    for provider_dtype in (np.dtype(dtype), getattr(strict, dtype), np.dtype(dtype)):
        assert dtype_kind_bits(provider_dtype) == dtype_kind_bits(dtype)


def test_can_cast_is_numpy_safe_casting_within_an_array_api_category() -> None:
    def category(dtype: str) -> str:
        return {"bool": "bool", "int": "integral", "uint": "integral"}.get(
            dtype_kind_bits(dtype)[0], "inexact"
        )

    for source, target in itertools.product(_DTYPES, repeat=2):
        expected = np.can_cast(source, target, "safe") and category(source) == category(target)
        assert can_cast_dtype(source, target) is expected, (source, target)


def _staged_namespace_answer(dtype: str, query: Callable[[Any, Any], object]) -> object:
    """Answer one query on the abstract namespace of a value inside a stage() trace."""
    answers = []

    def probe(value: Any) -> Any:
        answers.append(query(value.__array_namespace__(), value))
        return value

    ad.stage(probe, specs=(ad.ArraySpec((2,), dtype),))
    (answer,) = answers
    return answer


def _dtype_metadata(namespace: Any, value: Any) -> dict[tuple[str, ...], object]:
    def info(query: Callable[[], Any], fields: tuple[str, ...]) -> object:
        try:
            result = query()
        except (TypeError, ValueError):
            return "rejected"
        return (*(getattr(result, field) for field in fields), str(result.dtype).split(".")[-1])

    dtypes = {name: getattr(namespace, name) for name in _DTYPES if name != "float16"}
    return {
        **{("kind", kind): namespace.isdtype(value.dtype, kind) for kind in _DTYPE_KINDS},
        **{
            ("isdtype", name): namespace.isdtype(value.dtype, dtype)
            for name, dtype in dtypes.items()
        },
        **{("can_cast", name): namespace.can_cast(value, dtype) for name, dtype in dtypes.items()},
        ("finfo", "array"): info(lambda: namespace.finfo(value), _FINFO_FIELDS),
        ("finfo", "dtype"): info(lambda: namespace.finfo(value.dtype), _FINFO_FIELDS),
        ("iinfo", "array"): info(lambda: namespace.iinfo(value), ("bits", "max", "min")),
        ("iinfo", "dtype"): info(lambda: namespace.iinfo(value.dtype), ("bits", "max", "min")),
    }


@pytest.mark.parametrize("dtype", [dtype for dtype in _DTYPES if dtype != "float16"])
def test_staged_dtype_metadata_matches_array_api_strict(dtype: str) -> None:
    expected = _dtype_metadata(strict, strict.ones(2, dtype=getattr(strict, dtype)))

    assert _staged_namespace_answer(dtype, _dtype_metadata) == expected


@pytest.mark.parametrize("dtype", _DTYPES)
def test_staged_result_type_matches_numpy_promotion(dtype: str) -> None:
    others = (*_DTYPES, *_PYTHON_SCALARS)

    def result_types(namespace: Any, value: Any) -> list[str]:
        return [
            namespace.result_type(value, getattr(namespace, other) if type(other) is str else other)
            for other in others
        ]

    expected = [np.result_type(np.ones(2, dtype=dtype), other).name for other in others]
    assert _staged_namespace_answer(dtype, result_types) == expected


def test_dtype_metadata_variants_drive_staged_control_flow() -> None:
    def promote_if_metadata_matches(value: Any) -> Any:
        namespace = value.__array_namespace__()
        result_dtype = namespace.result_type(value, namespace.float64, 1)
        float_info = namespace.finfo(namespace.complex64)
        int_info = namespace.iinfo(namespace.uint8)
        matches = (
            namespace.isdtype(value.dtype, ("real floating", namespace.int32))
            and not namespace.can_cast(namespace.float64, value.dtype)
            and float_info.bits == 32
            and int_info.max == 255
        )
        return namespace.astype(value if matches else -value, result_dtype)

    value = strict.asarray([1.0, -2.0], dtype=strict.float32)
    program = ad.stage(
        promote_if_metadata_matches,
        specs=(ad.ArraySpec(value.shape, value.dtype),),
    )

    for staged in (program, ad.StagedProgram.from_dict(program.to_dict())):
        actual = staged(value)
        assert actual.dtype == strict.float64
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(value))


def test_metadata_and_array_methods_drive_staged_computation() -> None:
    def compute(value: Any) -> Any:
        namespace = value.__array_namespace__()
        info = namespace.__array_namespace_info__()
        metadata_matches = (
            info is namespace
            and namespace.result_type(np.asarray([1], dtype=np.int16)) == "int16"
            and namespace.isdtype(value.dtype, np.dtype("complex64"))
        )
        return value.real + value.item(2) + value.mean() if metadata_matches else -value.real

    value = np.asarray([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64)
    program = ad.stage(compute, specs=(ad.ArraySpec(value.shape, "complex64"),))
    restored = ad.StagedProgram.from_dict(program.to_dict())

    expected = value.real + value.item(2) + value.mean()
    np.testing.assert_allclose(program(value), expected)
    np.testing.assert_allclose(restored(value), expected)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    "declare",
    [
        pytest.param(lambda value: ((value,), None), id="example"),
        pytest.param(lambda value: ((), (ad.ArraySpec((2,), value.dtype.name),)), id="name"),
        pytest.param(lambda value: ((), (ad.ArraySpec((2,), value.dtype.type),)), id="type"),
        pytest.param(lambda value: ((), (ad.ArraySpec((2,), value.dtype),)), id="dtype"),
    ],
)
def test_staged_numpy_code_sees_the_dtype_objects_eager_code_sees(
    dtype: str,
    declare: Callable[[np.ndarray], tuple[tuple[object, ...], tuple[ad.ArraySpec, ...] | None]],
) -> None:
    value = np.ones(2, dtype=dtype)
    seen: list[tuple[object, ...]] = []

    def observe(array: Any) -> Any:
        view, compared = array[::-1], array != 0
        seen.append(
            (
                array.dtype,
                type(array.dtype),
                array.dtype == value.dtype.type,
                {value.dtype: "hashes alike"}.get(array.dtype),
                (array.dtype.kind, array.dtype.itemsize, array.dtype.name),
                view.dtype,
                compared.dtype,
                compared.dtype == np.bool_,
            )
        )
        return compared

    observe(value)
    examples, specs = declare(value)
    ad.stage(
        observe,
        *examples,
        specs=specs,
        array_api_version=min(np.__array_api_version__, LATEST_ARRAY_API_VERSION),
    )

    assert seen[1] == seen[0]


def test_staged_gradients_take_the_dtype_branch_dynamic_code_takes() -> None:
    value = np.asarray([0.5, 1.0, 2.0], dtype=np.float32)

    def loss(array: Any) -> Any:
        return np.sum(np.sin(array) if array.dtype == np.float32 else np.cos(array))

    expected = ad.grad(loss)(value)
    np.testing.assert_allclose(expected, np.cos(value))
    for program in (ad.stage(ad.grad(loss), value), ad.grad(ad.stage(loss, value))):
        actual = program(value)
        assert actual.dtype == expected.dtype
        np.testing.assert_allclose(actual, expected)


def test_the_public_program_constructor_stages_against_numpy_dtypes() -> None:
    seen: list[object] = []

    def scale(array: Any) -> Any:
        seen.append(array.dtype)
        return array * 2 if array.dtype == np.float32 else array

    program = ad.StagedProgram(
        scale,
        specs=(ad.ArraySpec((2,), "float32"),),
        kw_specs={},
        array_api_version=min(np.__array_api_version__, LATEST_ARRAY_API_VERSION),
    )

    assert seen == [np.dtype(np.float32)]
    np.testing.assert_array_equal(program(np.asarray([1.0, 2.0], dtype=np.float32)), [2.0, 4.0])


@pytest.mark.parametrize("dtype", ["float128", "longdouble", "bfloat16", "object"])
def test_unsupported_dtypes_are_rejected_while_staging(dtype: str) -> None:
    with pytest.raises(TypeError, match="Unsupported staged dtype"):
        ad.stage(lambda value: value + 1, specs=(ad.ArraySpec((2,), dtype),))


@pytest.mark.parametrize(
    ("operation", "dtype"),
    [
        pytest.param(lambda value: value.astype(object), "object", id="astype-object"),
        pytest.param(lambda value: value.astype("S3"), "S3", id="astype-bytes"),
        pytest.param(lambda value: np.sum(value, dtype="S3"), "S3", id="reduction-bytes"),
    ],
)
def test_numpy_results_of_unsupported_dtypes_are_rejected_while_staging(
    operation: Callable[[Any], Any],
    dtype: str,
) -> None:
    with pytest.raises(TypeError) as caught:
        ad.stage(operation, np.ones(2))

    assert str(caught.value) == (
        f"Unsupported staged dtype {dtype!r}; staged programs support only the canonical "
        "bool and numeric dtypes: bool, uint8, int8, uint16, int16, uint32, int32, uint64, "
        "int64, float16, float32, float64, complex64, complex128"
    )


@pytest.mark.parametrize(
    "spelling", ["Float32", "FLOAT32", np.float32, np.dtype(np.float32), strict.float32]
)
def test_every_dtype_spelling_declares_the_canonical_call_contract(spelling: object) -> None:
    value = np.asarray([1.0, 2.0], dtype=np.float32)
    program = cast(
        "ad.StagedProgram",
        ad.stage(
            lambda x: x * 2,
            specs=(ad.ArraySpec((2,), spelling),),
            array_api_version=min(np.__array_api_version__, LATEST_ARRAY_API_VERSION),
        ),
    )

    assert program.signature == ((ad.ArraySpec((2,), "float32"),), {})
    np.testing.assert_array_equal(program(value), value * 2)


@pytest.mark.parametrize("dtype", ["float32", "int16"])
def test_module_qualified_dtype_names_stage_as_their_dtype(dtype: str) -> None:
    value = np.arange(3, dtype=dtype)
    expected = value * 2 + 1
    program = ad.stage(lambda array: array * 2 + 1, specs=(ad.ArraySpec((3,), f"numpy.{dtype}"),))

    actual = np.asarray(program(value))

    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("operation", "dtypes"),
    [
        pytest.param(lambda left, right: left + right, ("float16", "float16"), id="float16"),
        pytest.param(lambda left, right: left + right, ("int32", "float32"), id="int32-float32"),
        pytest.param(lambda left, right: left * right, ("uint32", "complex64"), id="uint-complex"),
        pytest.param(lambda left, _right: left * 1.0, ("int8", "int8"), id="int-weak-float"),
        pytest.param(lambda left, _right: left * 2.0, ("float16", "int8"), id="float16-weak"),
        # Guard: the base promotion already staged this case correctly.
        pytest.param(
            lambda left, _right: left + 1j, ("float16", "int8"), id="float16-weak-complex"
        ),
        pytest.param(lambda left, _right: left + 1, ("bool", "bool"), id="bool-weak-int"),
        pytest.param(
            lambda left, right: np.where(left > 0, left, right), ("int32", "float32"), id="where"
        ),
        # Guard: the base promotion already staged this case correctly.
        pytest.param(
            lambda left, right: np.concatenate((left, right)),
            ("int16", "float16"),
            id="concatenate",
        ),
        pytest.param(np.ldexp, ("float32", "int64"), id="ldexp-keeps-mantissa-dtype"),
        pytest.param(np.power, ("bool", "bool"), id="bool-power"),
        pytest.param(lambda left, _right: np.square(left), ("bool", "bool"), id="bool-square"),
        pytest.param(lambda left, right: abs(left - right), ("int8", "int8"), id="builtin-abs"),
    ],
)
def test_staged_mixed_dtype_promotion_matches_eager_numpy(
    operation: Callable[[object, object], object],
    dtypes: tuple[str, str],
) -> None:
    _assert_staged_matches_eager(
        operation,
        np.asarray([1, 0, 3], dtype=dtypes[0]),
        np.asarray([2, 5, 1], dtype=dtypes[1]),
    )


def test_mixed_integer_promotion_and_scalar_dot_round_trip() -> None:
    _assert_staged_matches_eager(
        lambda left, right: (left + right, np.dot(left[0], right)),
        np.asarray([2], dtype=np.int64),
        np.asarray([1, 3], dtype=np.uint64),
    )


def _shift_through_an_alias(value: Any) -> Any:
    result = value.copy()
    alias = result
    result <<= np.int32(1)
    return alias


def _shift_through_a_view(value: Any) -> Any:
    result = value.copy()
    view = result[1:]
    view >>= 1
    return result


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda x: x << 1, id="left-shift"),
        pytest.param(lambda x: 64 >> x, id="reflected-right-shift"),
        pytest.param(lambda x: ~x, id="invert"),
        pytest.param(_shift_through_an_alias, id="in-place-alias"),
        pytest.param(_shift_through_a_view, id="in-place-view"),
    ],
)
def test_staged_shift_and_invert_operators_follow_eager_numpy(
    operation: Callable[[Any], object],
) -> None:
    _assert_staged_matches_eager(operation, np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int32))


@pytest.mark.parametrize(
    "op",
    # astype and clip take their result dtype or bounds from attributes.
    sorted(set(elementwise.RULES) - {"array.astype", "array.clip"}),
)
def test_elementwise_result_dtype_matches_numpy_for_every_dtype(op: str) -> None:
    schema, evaluator = _SEMANTICS[op]
    function = getattr(np, op.rsplit(".", 1)[1])
    trailing = (*_DTYPES, *_PYTHON_SCALARS)
    mismatches = []
    for operands in itertools.product(_DTYPES, *(trailing,) * (schema.operands - 1)):
        expected = _numpy_result_dtype(function, operands)
        if expected is None:
            continue
        (actual,) = evaluator([_operand_spec(operand) for operand in operands], {})
        if actual.dtype != expected:
            mismatches.append((operands, actual.dtype, expected))
    assert mismatches == []


@pytest.mark.parametrize("op", sorted(reductions.RULES))
def test_reduction_result_dtype_matches_numpy_for_every_dtype(op: str) -> None:
    _schema, evaluator = _SEMANTICS[op]
    function = getattr(np, op.rsplit(".", 1)[1])
    mismatches = []
    for dtype in _DTYPES:
        expected = _numpy_result_dtype(function, (dtype,), axis=0)
        if expected is None:
            continue
        (actual,) = evaluator([ad.ArraySpec((2,), dtype)], {"axis": 0})
        if actual.dtype != expected:
            mismatches.append((dtype, actual.dtype, expected))
    assert mismatches == []


def test_staged_logical_operations_return_bool_for_numeric_operands() -> None:
    def logical(left: object, right: object) -> tuple[object, object, object]:
        return np.logical_and(left, right), np.logical_or(left, right), np.logical_xor(left, right)

    left = np.asarray([0.0, 1.5, -2.0])
    right = np.asarray([3, 0, 0], dtype=np.int32)
    program = ad.stage(
        logical,
        specs=(ad.ArraySpec(left.shape, "float64"), ad.ArraySpec(right.shape, "int32")),
    )

    for staged in (program, ad.StagedProgram.from_dict(program.to_dict())):
        for actual, expected in zip(staged(left, right), logical(left, right), strict=True):
            assert actual.dtype == np.bool_
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda value: value.astype(float), id="astype-float"),
        pytest.param(lambda value: value.astype(complex), id="astype-complex"),
        pytest.param(lambda value: np.sum(value, axis=1, dtype=float), id="sum-float"),
        pytest.param(lambda value: np.cumsum(value, axis=1, dtype=float) + 1, id="cumsum-float"),
        pytest.param(lambda value: np.full_like(value, 2, dtype=int), id="full-like-int"),
    ],
)
def test_python_scalar_types_stage_as_numpy_default_dtypes(
    operation: Callable[[np.ndarray], object],
) -> None:
    _assert_staged_matches_eager(operation, np.asarray([[1, 0, 3], [2, 5, 1]], dtype=np.int32))


@pytest.mark.parametrize(
    ("operation", "dtypes"),
    [
        pytest.param(np.sqrt, ("int32",), id="sqrt"),
        pytest.param(np.exp, ("bool",), id="exp"),
        pytest.param(np.arctan2, ("int32", "int32"), id="arctan2"),
        pytest.param(np.hypot, ("int64", "uint8"), id="hypot"),
        pytest.param(lambda value: np.hypot(value, 2.0), ("int8",), id="hypot-weak-float"),
        pytest.param(np.ldexp, ("int8", "int32"), id="ldexp"),
        pytest.param(np.angle, ("int16",), id="angle"),
        pytest.param(np.mean, ("int32",), id="mean"),
        pytest.param(lambda value: np.std(value, axis=1), ("uint16",), id="std"),
        pytest.param(np.nanvar, ("bool",), id="nanvar"),
    ],
)
def test_staged_inexact_operations_on_exact_dtypes_match_eager_numpy(
    operation: Callable[..., object],
    dtypes: tuple[str, ...],
) -> None:
    values = ([[1, 0, 3], [2, 5, 1]], [[2, 1, 0], [1, 3, 4]])
    _assert_staged_matches_eager(
        operation,
        *(np.asarray(value, dtype=dtype) for value, dtype in zip(values, dtypes, strict=False)),
    )


@pytest.mark.parametrize(
    ("operation", "dtypes"),
    [
        pytest.param(np.linalg.inv, ("int32",), id="inv"),
        pytest.param(np.linalg.det, ("bool",), id="det"),
        pytest.param(np.linalg.solve, ("float32", "int8"), id="solve"),
        pytest.param(np.linalg.svd, ("uint8",), id="svd"),
        pytest.param(np.linalg.eigvalsh, ("int64",), id="eigvalsh"),
        pytest.param(np.linalg.norm, ("int16",), id="norm"),
        pytest.param(np.fft.fft, ("int32",), id="fft"),
        pytest.param(np.fft.rfft2, ("uint16",), id="rfft2"),
        pytest.param(np.fft.irfft, ("bool",), id="irfft"),
        # Only the last axis is inverted from a real input; the others turn complex first.
        pytest.param(np.fft.irfft2, ("float16",), id="irfft2-float16"),
    ],
)
def test_staged_linalg_and_fft_dtypes_match_eager_numpy(
    operation: Callable[..., object],
    dtypes: tuple[str, ...],
) -> None:
    values = ([[2, 1], [1, 3]], [[1, 0], [2, 1]])
    _assert_staged_matches_eager(
        operation,
        *(np.asarray(value, dtype=dtype) for value, dtype in zip(values, dtypes, strict=False)),
    )


@pytest.mark.parametrize(
    ("operation", "dtype"),
    [
        pytest.param(lambda value: value * np.sqrt(np.float64(2.0)), "float32", id="float64"),
        pytest.param(lambda value: value + np.complex128(1j), "complex64", id="complex128"),
        pytest.param(
            lambda value: value.__array_namespace__().asarray([np.float64(1.0), value[0]]),
            "float32",
            id="nested-asarray",
        ),
        pytest.param(
            lambda value: value.__array_namespace__().asarray([np.float32(1.0), 0.1, 2.0]),
            "float32",
            id="sequence-asarray",
        ),
        pytest.param(
            lambda value: value + [np.float32(1.0), 0.1, 2.0],  # noqa: RUF005 - array addition
            "float32",
            id="sequence",
        ),
        pytest.param(lambda value: value * 2.0, "float32", id="python-float-stays-weak"),
    ],
)
def test_numpy_scalar_operands_stage_as_strong_rank_zero_arrays(
    operation: Callable[[np.ndarray], object],
    dtype: str,
) -> None:
    _assert_staged_matches_eager(operation, np.asarray([1.0, 2.0, 3.0], dtype=dtype))


_PYTHON_SCALAR_STEPS: dict[str, Callable[[Any], Any]] = {
    "negate": operator.neg,
    "absolute": abs,
    "square": lambda scalar: scalar * scalar,
    "shift": lambda scalar: scalar + 1,
    "complement": lambda scalar: 1 - scalar,
    "power": lambda scalar: scalar**2,
    "halve": lambda scalar: scalar / 2,
    "real": lambda scalar: scalar.real,
    "compare": lambda scalar: scalar.real > 0,
}


@given(
    dtype=st.sampled_from(("float16", "float32", "complex64", "int32")),
    scalar=st.sampled_from((True, 3, 0.75, 0.5 - 1.5j)),
    steps=st.lists(st.sampled_from(sorted(_PYTHON_SCALAR_STEPS)), min_size=1, max_size=3),
    combine=st.sampled_from((operator.add, operator.mul)),
)
@example(dtype="float32", scalar=0.75, steps=["square"], combine=operator.add)
def test_python_scalar_arithmetic_stays_weak_until_it_meets_an_array(
    dtype: str,
    scalar: complex,
    steps: list[str],
    combine: Callable[[Any, Any], Any],
) -> None:
    """A Python-scalar argument keeps NumPy's weak promotion through scalar arithmetic."""

    def operation(array: Any, value: Any) -> Any:
        for step in steps:
            value = _PYTHON_SCALAR_STEPS[step](value)
        return combine(array, value)

    array = np.asarray([0.5, -1.5], dtype=dtype)
    expected = operation(array, scalar)
    program = cast("ad.StagedProgram", ad.stage(operation, array, scalar))
    for staged in (program, ad.StagedProgram.from_dict(program.to_dict())):
        actual = staged(array, scalar)
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)


_PYTHON_INTEGER_STEPS: dict[str, Callable[[Any, Any], Any]] = {
    "left-shift": lambda scalar, _xp: scalar << 1,
    "right-shift": lambda scalar, _xp: scalar >> 1,
    "invert": lambda scalar, _xp: ~scalar,
    "and": lambda scalar, _xp: scalar & 6,
    # An Array API function of a Python int returns a strong int64 value.
    "xp.bitwise_left_shift": lambda scalar, xp: xp.bitwise_left_shift(scalar, 1),
    "xp.bitwise_right_shift": lambda scalar, xp: xp.bitwise_right_shift(scalar, 1),
    "xp.bitwise_invert": lambda scalar, xp: xp.bitwise_invert(scalar),
}


@given(
    dtype=st.sampled_from(("int8", "int16", "int32")),
    scalar=st.integers(min_value=0, max_value=5),
    step=st.sampled_from(sorted(_PYTHON_INTEGER_STEPS)),
)
@example(dtype="int8", scalar=3, step="left-shift")
@example(dtype="int8", scalar=3, step="xp.bitwise_left_shift")
def test_python_integer_bitwise_steps_keep_their_eager_category(
    dtype: str,
    scalar: int,
    step: str,
) -> None:
    """Python's shifts and inversion of a Python int are weak; Array API functions are strong."""

    def operation(array: Any, value: Any) -> Any:
        return array * _PYTHON_INTEGER_STEPS[step](value, array.__array_namespace__())

    array = np.asarray([1, -2, 3], dtype=dtype)
    expected = operation(array, scalar)
    program = cast("ad.StagedProgram", ad.stage(operation, array, scalar))
    nested = ad.stage(
        lambda array, value: program(array, value),  # noqa: PLW0108 - explicit trace boundary
        array,
        scalar,
    )
    for staged in (program, ad.StagedProgram.from_dict(program.to_dict()), nested):
        actual = staged(array, scalar)
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)


# NumPy calls on a Python scalar return a strong NumPy scalar.
_NUMPY_SCALAR_STEPS: dict[str, Callable[[Any], Any]] = {
    "np.multiply": lambda scalar: np.multiply(scalar, scalar),
    "np.power": lambda scalar: np.power(scalar, 2),
    "np.absolute": np.abs,
    "np.subtract": lambda scalar: np.subtract(1, scalar),
    "np.sin": np.sin,
}
_REAL_SCALAR_STEPS = {
    **{name: step for name, step in _PYTHON_SCALAR_STEPS.items() if name != "compare"},
    **_NUMPY_SCALAR_STEPS,
}


@given(
    dtype=st.sampled_from(("float16", "float32", "float64")),
    scalar=st.sampled_from((0.75, -2.5)),
    steps=st.lists(st.sampled_from(sorted(_REAL_SCALAR_STEPS)), min_size=1, max_size=3),
    combine=st.sampled_from((operator.add, operator.mul, np.multiply, np.maximum)),
)
@example(dtype="float32", scalar=0.75, steps=["np.multiply"], combine=operator.mul)
@example(dtype="float32", scalar=0.75, steps=["square"], combine=np.maximum)
def test_staged_program_transforms_keep_each_scalar_category(
    dtype: str,
    scalar: float,
    steps: list[str],
    combine: Callable[[Any, Any], Any],
) -> None:
    """Compiled transforms replay a program's weak and strong scalars as it recorded them."""

    def loss(array: Any, value: Any) -> Any:
        for step in steps:
            value = _REAL_SCALAR_STEPS[step](value)
        return np.sum(combine(array, value))

    array = np.asarray([0.5, -1.5], dtype=dtype)
    program = cast("ad.StagedProgram", ad.stage(loss, array, scalar))
    primal = loss(array, scalar)
    staged_primal = program(array, scalar)
    assert staged_primal.dtype == primal.dtype
    np.testing.assert_array_equal(staged_primal, primal)

    expected = ad.grad(loss)(array, scalar)
    cotangent = primal.dtype.type(1)
    for actual in (
        ad.grad(program)(array, scalar),
        ad.vjp_program(program)(array, scalar, cotangent=cotangent),
        ad.stage(ad.grad(program), array, scalar)(array, scalar),
    ):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)


def _replace_with_a_traced_sequence(value: Any) -> object:
    result = value.copy()
    result[:] = [value[1], 2.0]
    return result


@pytest.mark.parametrize(
    ("operation", "dtype"),
    [
        pytest.param(
            lambda value: (
                value.__array_namespace__().asarray(_ROW, dtype=np.float32),
                value + _ROW,
            ),
            "float16",
            id="one-list-at-two-dtypes",
        ),
        pytest.param(
            lambda value: value.__array_namespace__().asarray([[value], [_STRICT_VECTOR]]),
            "float64",
            id="strict-vector-in-concrete-row",
        ),
        pytest.param(
            lambda value: value.__array_namespace__().asarray([[value], [_NUMPY_VECTOR]]),
            "float32",
            id="numpy-vector-in-concrete-row",
        ),
        pytest.param(
            lambda value: value.__array_namespace__().asarray([_NUMPY_VECTOR, _STRICT_VECTOR]),
            "float64",
            id="vectors-without-tracers",
        ),
        # NumPy's coercion promotes leaf by leaf, so the order of mixed leaves matters.
        pytest.param(
            lambda value: value.__array_namespace__().asarray(
                [np.int8(1), np.uint8(2), np.float16(3)]
            ),
            "float16",
            id="coercion-order",
        ),
        pytest.param(
            lambda value: value.__array_namespace__().asarray(
                [np.float16(3), np.uint8(2), np.int8(1)]
            ),
            "float16",
            id="coercion-order-reversed",
        ),
        pytest.param(
            lambda value: value.__array_namespace__().asarray(
                [[value[0], value[1]], [np.int8(1), np.uint8(2)]]
            ),
            "float16",
            id="coercion-across-rows",
        ),
        pytest.param(
            lambda value: value[0] + [np.int8(1), np.uint8(2), np.float16(3)],
            "float16",
            id="coercion-of-an-operand",
        ),
        # A Python int beyond int64 is a uint64 leaf.
        pytest.param(
            lambda value: value.__array_namespace__().asarray([np.uint64(1), 2**63]),
            "uint64",
            id="python-uint64",
        ),
        pytest.param(
            lambda value: value.__array_namespace__().asarray([value[0], 2**64 - 1]),
            "uint64",
            id="python-uint64-beside-a-tracer",
        ),
        pytest.param(
            lambda value: value.__array_namespace__().asarray([value[0], 2**63]),
            "int64",
            id="python-uint64-beside-int64",
        ),
        pytest.param(
            lambda value: value + [np.uint64(1), 2**63],  # noqa: RUF005 - array addition
            "uint64",
            id="python-uint64-in-an-operand",
        ),
        pytest.param(
            lambda value: value.__array_namespace__().asarray(2**63),
            "uint64",
            id="python-uint64-scalar",
        ),
        # An operand sequence is assembled like asarray's.
        pytest.param(
            lambda value: value + [value[0], 2.0],  # noqa: RUF005 - array addition
            "float32",
            id="tracer-in-an-operand",
        ),
        pytest.param(
            lambda value: [value[1], 2.0] + value,  # noqa: RUF005 - array addition
            "float64",
            id="tracer-in-a-reflected-operand",
        ),
        pytest.param(
            lambda value: value + [_NUMPY_VECTOR, _STRICT_VECTOR],  # noqa: RUF005
            "float32",
            id="arrays-in-an-operand",
        ),
        pytest.param(_replace_with_a_traced_sequence, "float32", id="tracer-in-a-replacement"),
        # An explicit dtype casts a NumPy scalar or rank-zero array leaf like astype.
        pytest.param(
            lambda value: value.__array_namespace__().asarray(
                [[value[0]], [np.array(300)]], dtype="int8"
            ),
            "int8",
            id="rank-zero-array-narrowed",
        ),
        pytest.param(
            lambda value: value.__array_namespace__().asarray([np.array(300)], dtype="int8"),
            "int8",
            id="rank-zero-array-narrowed-without-tracers",
        ),
        pytest.param(
            lambda value: value.__array_namespace__().asarray(
                [[value[0]], [np.int8(-1)]], dtype="uint8"
            ),
            "int8",
            id="numpy-scalar-narrowed-to-unsigned",
        ),
        pytest.param(
            lambda value: value.__array_namespace__().asarray(
                [[value[0]], [np.float64(1e6)]], dtype="float16"
            ),
            "float16",
            id="numpy-scalar-overflowing-float16",
            marks=pytest.mark.filterwarnings("ignore:overflow encountered in cast"),
        ),
        pytest.param(
            lambda value: value.__array_namespace__().asarray(np.float64(1e6), dtype="float16"),
            "float16",
            id="bare-numpy-scalar-overflowing-float16",
            marks=pytest.mark.filterwarnings("ignore:overflow encountered in cast"),
        ),
    ],
)
def test_staged_sequence_construction_matches_eager_numpy(
    operation: Callable[[np.ndarray], object],
    dtype: str,
) -> None:
    _assert_staged_matches_eager(operation, np.asarray([1.0, 2.0], dtype=dtype))


@pytest.mark.parametrize(
    "value",
    [np.asarray([2.0], dtype=np.float32), strict.asarray([2.0], dtype=strict.float32)],
)
@pytest.mark.parametrize("scalar", [0.0, np.float32(0.0)])
@pytest.mark.parametrize("dtype", [None, "float32"])
def test_nested_asarray_promotes_python_scalars_before_stacking(
    value: Any,
    scalar: object,
    dtype: str | None,
) -> None:
    def assemble(value: Any) -> object:
        namespace = value.__array_namespace__()
        options = {} if dtype is None else {"dtype": getattr(namespace, dtype)}
        return namespace.asarray([[value[0], scalar]], **options)

    program = ad.stage(assemble, specs=(ad.ArraySpec(value.shape, "float32"),))
    restored = ad.StagedProgram.from_dict(program.to_dict())

    expected_dtype = np.float64 if dtype is None and type(scalar) is float else np.float32
    for actual in (program(value), restored(value)):
        assert str(actual.dtype).endswith(np.dtype(expected_dtype).name)
        np.testing.assert_array_equal(np.asarray(actual), [[2.0, 0.0]])


@pytest.mark.parametrize(
    ("leaves", "dtype"),
    [
        pytest.param([np.float64(0.1), np.float64(-3e38)], "float32", id="float64-to-float32"),
        pytest.param([np.float64(65504.0), np.float64(-0.5)], "float16", id="float64-to-float16"),
        pytest.param([np.int64(127), np.int64(-128)], "int8", id="int64-to-int8"),
        pytest.param([np.int64(0), np.int64(255)], "uint8", id="int64-to-uint8"),
        pytest.param([np.float64(-2.7), np.float64(3.9)], "int16", id="float64-to-int16"),
        pytest.param([np.complex128(0.5 - 2j), np.complex128(1j)], "complex64", id="to-complex64"),
        pytest.param([np.float64(np.nan), np.float64(0.0)], "bool", id="float64-to-bool"),
    ],
)
def test_in_range_narrowing_leaves_fold_into_one_sequence_constant(
    leaves: list[Any],
    dtype: str,
) -> None:
    values = leaves * 25

    def operation(value: Any) -> Any:
        return value.__array_namespace__().asarray(values, dtype=dtype)

    _assert_staged_matches_eager(operation, np.zeros(2, dtype=dtype))
    program = cast(
        "ad.StagedProgram",
        ad.stage(operation, specs=(ad.ArraySpec((2,), dtype),)),
    )
    ops = [program.graph.get_node(node_id).op for node_id in program.graph.node_ids()]
    assert ops.count("advect.const") == 1
    assert "array.stack" not in ops


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda xp: xp.asarray([2**64]), id="beyond-uint64"),
        pytest.param(lambda xp: xp.asarray([-(2**63) - 1]), id="below-int64"),
        pytest.param(lambda xp: xp.asarray([np.float32(1.0), 2**70]), id="beside-a-float"),
        pytest.param(lambda xp: xp.asarray(2**64), id="scalar"),
    ],
)
def test_python_ints_that_numpy_stores_as_objects_are_rejected_while_staging(
    operation: Callable[[Any], object],
) -> None:
    with pytest.raises(TypeError, match="Unsupported staged dtype 'object'"):
        ad.stage(
            lambda value: operation(value.__array_namespace__()),
            specs=(ad.ArraySpec((2,), "float64"),),
        )


def _converts_exactly(value: int, dtype: str | None) -> bool:
    """Whether NumPy converts the Python int *value* to *dtype*, or a discovered one."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            return np.asarray([value], dtype=dtype).dtype != object
        except (OverflowError, RuntimeWarning):
            return False


_REAL_DTYPES = tuple(dtype for dtype in _DTYPES if not dtype.startswith("complex"))
# The Python ints each target, or discovery (None), converts without overflow.
_PYTHON_INTS = {
    target: tuple(
        value for value in (*_PYTHON_INT_BOUNDARIES, 1, 2) if _converts_exactly(value, target)
    )
    for target in (None, *_DTYPES)
}
# Leaf dtypes and Python scalar types of one drawn sequence. Integral leaves keep
# a Python int beyond int64 visible, and coercion order matters when mixed-sign
# integers precede a narrow float.
_LEAF_FAMILIES = (
    (_DTYPES, (bool, int, float, complex)),
    (tuple(dtype for dtype in _REAL_DTYPES if not dtype.startswith("float")), (int,)),
    (("int8", "uint8", "float16"), ()),
)


def _draw_family(data: DataObject, target: str | None) -> tuple[tuple[str, ...], tuple[type, ...]]:
    """Draw the leaf family of one sequence, without complex leaves for a real *target*."""
    dtypes, python_types = data.draw(st.sampled_from(_LEAF_FAMILIES), label="family")
    if target in _REAL_DTYPES:
        # NumPy rejects or discards an imaginary part cast to a real dtype.
        dtypes = tuple(dtype for dtype in dtypes if dtype in _REAL_DTYPES)
        python_types = tuple(kind for kind in python_types if kind is not complex)
    return dtypes, python_types


def _draw_scalar(
    data: DataObject,
    family: tuple[tuple[str, ...], tuple[type, ...]],
    target: str | None,
) -> object:
    """Draw a Python or NumPy scalar of *family* that converts exactly to *target*."""
    dtypes, python_types = family
    kind = data.draw(st.sampled_from((*python_types, "numpy")), label="scalar")
    if kind is int:
        return data.draw(st.sampled_from(_PYTHON_INTS[target]), label="value")
    value = data.draw(st.integers(0, 2), label="value")
    if kind == "numpy":
        return np.dtype(data.draw(st.sampled_from(dtypes), label="dtype")).type(value)
    return kind(value)


def _draw_leaf(
    data: DataObject,
    shape: tuple[int, ...],
    family: tuple[tuple[str, ...], tuple[type, ...]],
    target: str | None,
) -> tuple[bool, object]:
    """Draw one sequence leaf as (traced, value); a vector leaf may be a list of scalars."""
    size = math.prod(shape)
    kind = data.draw(st.sampled_from(("tracer", "numpy", "strict", "scalars")), label="leaf")
    if kind == "scalars":
        scalars = [_draw_scalar(data, family, target) for _ in range(size)]
        return False, scalars if shape else scalars[0]
    dtype = data.draw(st.sampled_from(family[0]), label="dtype")
    values = data.draw(st.lists(st.integers(0, 2), min_size=size, max_size=size), label="values")
    array = np.asarray(values, dtype=dtype).reshape(shape)
    # array_api_strict has no float16, so that leaf stays a NumPy array.
    if kind == "strict" and dtype != "float16":
        return False, strict.asarray(array)
    return kind == "tracer", array


def _nest(leaves: list[object], shape: tuple[int, ...], container: type) -> object:
    if len(shape) == 1:
        return container(leaves)
    size = len(leaves) // shape[0]
    return container(
        _nest(leaves[start : start + size], shape[1:], container)
        for start in range(0, len(leaves), size)
    )


@given(data=st.data())
def test_staged_nested_sequence_construction_matches_numpy(data: DataObject) -> None:
    """Staged sequences assemble tracers, concrete arrays and scalars like NumPy's coercion.

    A sequence is built by asarray, with or without an explicit dtype, or lifted as
    an operand of a False anchor, which leaves its dtype and values unchanged.
    """
    construction = data.draw(st.sampled_from(("asarray", "operand")), label="construction")
    target = (
        None
        if construction == "operand"
        else data.draw(st.none() | st.sampled_from(_DTYPES), label="dtype")
    )
    nesting = data.draw(hnp.array_shapes(min_dims=1, max_dims=2, max_side=3), label="nesting")
    leaf_shape = data.draw(st.sampled_from(((), (2,))), label="leaf shape")
    family = _draw_family(data, target)
    leaves = [_draw_leaf(data, leaf_shape, family, target) for _ in range(math.prod(nesting))]
    container = data.draw(st.sampled_from((list, tuple)), label="container")

    def construct(anchor: Any, *traced: Any) -> object:
        values = iter(traced)
        flat = [next(values) if is_traced else value for is_traced, value in leaves]
        nested = _nest(flat, nesting, container)
        if construction == "operand":
            return anchor + nested
        return anchor.__array_namespace__().asarray(nested, dtype=target)

    inputs = [value for is_traced, value in leaves if is_traced]
    _assert_staged_matches_eager(construct, np.zeros(1, dtype=bool), *inputs)
