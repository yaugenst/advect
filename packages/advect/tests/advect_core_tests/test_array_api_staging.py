"""Abstract staging of Array API forms beyond the qualification evidence.

``test_array_api_operation_qualification.py`` runs every declared evidence
case through its dynamic, staged and serialized lifetimes. This module owns
the staged forms that evidence does not: creation functions whose evidence is
dynamic-only, literal and empty sequences, single-precision FFTs, narrow
integer accumulation, batched and tuple-axis variants, include_initial and
difference boundaries, revision-dependent accumulation, and the fixed-arity
linear-algebra result containers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import array_api_strict as strict
import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad

if TYPE_CHECKING:
    from collections.abc import Callable

_SENTINEL = strict.asarray([0.0], dtype=strict.float32)
_CUBE = strict.asarray(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
_VECTOR = strict.ones((4,), dtype=strict.complex64)
_MATRIX = strict.ones((2, 4), dtype=strict.complex64)
_REAL_VECTOR = strict.ones((4,), dtype=strict.float32)
_REAL_MATRIX = strict.ones((2, 4), dtype=strict.float32)


def _assert_staged_matches_eager(
    operation: Callable[..., Any], *inputs: Any, tolerance: float
) -> ad.StagedProgram:
    """Compare the live and JSON-restored programs with eager strict results."""

    def function(*values: Any) -> Any:
        return operation(values[0].__array_namespace__(), *values)

    expected = function(*inputs)
    program = ad.stage(function, specs=tuple(ad.ArraySpec(v.shape, v.dtype) for v in inputs))
    restored = ad.StagedProgram.from_dict(program.to_dict())
    references = expected if isinstance(expected, tuple) else (expected,)
    for result in (program(*inputs), restored(*inputs)):
        results = result if isinstance(result, tuple) else (result,)
        for actual, reference in zip(results, references, strict=True):
            assert (actual.shape, actual.dtype) == (reference.shape, reference.dtype)
            assert_allclose(
                np.asarray(actual), np.asarray(reference), rtol=tolerance, atol=tolerance
            )
    return program


@pytest.mark.parametrize(
    ("operation", "value", "tolerance"),
    [
        pytest.param(
            lambda xp, _: xp.arange(1, 8, 2, dtype=xp.float32), _SENTINEL, 1e-7, id="arange"
        ),
        pytest.param(lambda xp, _: xp.eye(3, 4, k=1, dtype=xp.float64), _SENTINEL, 1e-7, id="eye"),
        pytest.param(
            lambda xp, _: xp.full((2, 3), 2.5, dtype=xp.float32), _SENTINEL, 1e-7, id="full"
        ),
        pytest.param(
            lambda xp, _: xp.linspace(-1.0, 1.0, 5, dtype=xp.float32),
            _SENTINEL,
            1e-7,
            id="linspace",
        ),
        pytest.param(
            lambda xp, _: xp.fft.fftfreq(6, d=0.25, dtype=xp.float64), _SENTINEL, 1e-7, id="fftfreq"
        ),
        pytest.param(
            lambda xp, _: xp.fft.rfftfreq(6, d=0.25, dtype=xp.float32),
            _SENTINEL,
            1e-7,
            id="rfftfreq",
        ),
        pytest.param(
            lambda xp, _: xp.asarray([[1, 2], [3, 4]], dtype=xp.float32),
            _SENTINEL,
            1e-7,
            id="asarray-sequence",
        ),
        pytest.param(lambda xp, _: xp.asarray([]), _SENTINEL, 1e-7, id="asarray-empty"),
        pytest.param(
            lambda xp, _: xp.asarray([[], []]), _SENTINEL, 1e-7, id="asarray-nested-empty"
        ),
        pytest.param(
            lambda xp, x: xp.asarray([[x[0], x[1]], [x[1], x[0]]], dtype=xp.float64, copy=True),
            strict.asarray([1.0, 2.0], dtype=strict.float32),
            0.0,
            id="asarray-nested-tracers-copy-and-cast",
        ),
        pytest.param(
            lambda xp, x: x + xp.asarray(-0.31, dtype=x.dtype),
            strict.asarray([0.0], dtype=strict.float64),
            0.0,
            id="typed-scalar-constant",
        ),
        pytest.param(
            lambda _xp, x: abs(x), strict.asarray([-1.5, 0.0, 2.0]), 0.0, id="builtin-abs"
        ),
        pytest.param(
            lambda xp, x: xp.moveaxis(x, (0, 2), (2, 0)), _CUBE, 1e-7, id="moveaxis-tuple"
        ),
        pytest.param(
            lambda xp, x: xp.cumulative_sum(x, axis=1),
            strict.asarray([[1, 2, 3], [4, 5, 6]], dtype=strict.int8),
            1e-7,
            id="cumulative-sum-int8",
        ),
        pytest.param(
            lambda xp, x: xp.cumulative_sum(x, include_initial=True),
            strict.asarray([1.0, 2.0, 3.0], dtype=strict.float32),
            0.0,
            id="cumulative-sum-include-initial",
        ),
        pytest.param(
            lambda xp, x: xp.cumulative_prod(x, axis=1, include_initial=True),
            strict.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=strict.float32),
            0.0,
            id="cumulative-prod-include-initial-axis",
        ),
        pytest.param(
            lambda xp, x: xp.linalg.diagonal(x, offset=1), _CUBE, 1e-7, id="batched-diagonal"
        ),
        pytest.param(lambda xp, x: xp.fft.fft(x, n=6), _VECTOR, 1e-5, id="fft"),
        pytest.param(lambda xp, x: xp.fft.ifft(x, n=6), _VECTOR, 1e-5, id="ifft"),
        pytest.param(lambda xp, x: xp.fft.fftn(x, s=(3, 6), axes=(0, 1)), _MATRIX, 1e-5, id="fftn"),
        pytest.param(
            lambda xp, x: xp.fft.ifftn(x, s=(3, 6), axes=(0, 1)), _MATRIX, 1e-5, id="ifftn"
        ),
        pytest.param(lambda xp, x: xp.fft.rfft(x, n=6), _REAL_VECTOR, 1e-5, id="rfft"),
        pytest.param(lambda xp, x: xp.fft.irfft(x, n=6), _VECTOR, 1e-5, id="irfft"),
        pytest.param(
            lambda xp, x: xp.fft.rfftn(x, s=(3, 6), axes=(0, 1)), _REAL_MATRIX, 1e-5, id="rfftn"
        ),
        pytest.param(
            lambda xp, x: xp.fft.irfftn(x, s=(3, 6), axes=(0, 1)), _MATRIX, 1e-5, id="irfftn"
        ),
        pytest.param(lambda xp, x: xp.fft.fftshift(x), _MATRIX, 1e-5, id="fftshift"),
        pytest.param(lambda xp, x: xp.fft.ifftshift(x), _MATRIX, 1e-5, id="ifftshift"),
    ],
)
def test_staged_array_api_forms_match_strict(
    operation: Callable[..., Any], value: Any, tolerance: float
) -> None:
    _assert_staged_matches_eager(operation, value, tolerance=tolerance)


def test_asarray_records_a_requested_device_on_its_cast() -> None:
    program = ad.stage(
        lambda item: item.__array_namespace__().asarray(item, device="cuda:0", copy=True),
        specs=(ad.ArraySpec((2,), "float32"),),
    )
    cast_node = next(
        program.graph.get_node(node_id)
        for node_id in program.graph.node_ids()
        if program.graph.get_node(node_id).op == "array.astype"
    )
    assert cast_node.attrs["_advect_device"] == "cuda:0"


def test_diff_stages_boundaries_and_the_zero_order_identity() -> None:
    def differences(value: Any) -> tuple[Any, Any]:
        namespace = value.__array_namespace__()
        return (
            namespace.diff(value, axis=1, prepend=0.0, append=9.0),
            namespace.diff(value, n=0, axis=1),
        )

    value = strict.reshape(strict.arange(6, dtype=strict.float32), (2, 3))
    program = ad.stage(differences, specs=(ad.ArraySpec(value.shape, value.dtype),))
    expected = np.asarray(value)
    for staged in (program, ad.StagedProgram.from_dict(program.to_dict())):
        scalar_boundaries, identity = staged(value)
        np.testing.assert_array_equal(
            np.asarray(scalar_boundaries),
            np.diff(expected, axis=1, prepend=0.0, append=9.0),
        )
        np.testing.assert_array_equal(np.asarray(identity), expected)


@pytest.mark.parametrize(
    ("input_dtype", "expected_dtype"),
    [
        (strict.int8, strict.int64),
        (strict.uint8, strict.uint64),
        (strict.float32, strict.float32),
    ],
)
def test_reduction_abstract_spec_uses_array_api_accumulation_dtype(
    input_dtype: object,
    expected_dtype: object,
) -> None:
    value = strict.asarray([1, 2, 3], dtype=input_dtype)

    def total(argument: object) -> object:
        return argument.__array_namespace__().sum(argument)

    result = ad.stage(
        total,
        specs=(ad.ArraySpec(value.shape, value.dtype),),
    )(value)

    assert result.dtype == expected_dtype


@pytest.mark.parametrize(
    ("array_api_version", "input_dtype", "expected_dtype"),
    [
        ("2022.12", "float32", "float64"),
        ("2022.12", "complex64", "complex128"),
        ("2023.12", "float32", "float32"),
        ("2024.12", "complex64", "complex64"),
    ],
)
def test_array_api_revision_controls_staged_accumulation_dtype(
    array_api_version: str,
    input_dtype: str,
    expected_dtype: str,
) -> None:
    program = ad.stage(
        lambda value: value.__array_namespace__().sum(value),
        specs=(ad.ArraySpec((3,), input_dtype),),
        array_api_version=array_api_version,
    )

    assert program.to_dict()["program"]["output_specs"][0]["dtype"] == expected_dtype


@pytest.mark.parametrize(
    "value",
    [
        np.asarray([1.0, 2.0], dtype=np.float32),
        strict.asarray([1.0, 2.0], dtype=strict.float32),
    ],
    ids=["numpy", "array-api-strict"],
)
def test_2022_staged_replay_normalizes_accumulations_and_weak_scalars(
    value: object,
) -> None:
    def transform(argument: Any) -> tuple[object, object, object]:
        namespace = argument.__array_namespace__()
        square = namespace.linalg.outer(argument, argument)
        return 0.25 * namespace.sum(argument), 1j * argument, namespace.linalg.trace(square)

    try:
        total, rotated, trace = ad.stage(
            transform,
            specs=(ad.ArraySpec(value.shape, value.dtype),),
            array_api_version="2022.12",
        )(value)

        assert str(total.dtype).endswith("float64")
        assert str(rotated.dtype).endswith("complex64")
        assert str(trace.dtype).endswith("float64")
    finally:
        strict.set_array_api_strict_flags(api_version="2024.12")


@pytest.mark.parametrize(
    "loss",
    [
        pytest.param(lambda x: np.sum(x * x), id="sum"),
        pytest.param(lambda x: (x * x).sum(), id="method-sum"),
        pytest.param(np.prod, id="prod"),
        pytest.param(lambda x: np.trace(x * x), id="trace"),
        pytest.param(lambda x: np.cumsum(x * x, axis=1)[0, -1], id="cumsum"),
    ],
)
def test_2022_numpy_program_transforms_keep_numpy_accumulation_dtypes(
    loss: Callable[[Any], Any],
) -> None:
    """Replaying a NumPy-authored reduction under 2022.12 keeps NumPy's dtype.

    NumPy 2.0 reports that revision, so its default programs replay this way.
    """
    value = np.asarray([[0.5, 1.5], [-1.0, 2.0]], dtype=np.float32)
    program = ad.stage(loss, value, array_api_version="2022.12")
    expected = ad.grad(loss)(value)

    gradient = ad.grad(program)(value)
    pulled = ad.vjp_program(program)(value, cotangent=np.float32(1.0))

    for actual in (gradient, pulled):
        assert actual.dtype == np.float32
        assert_allclose(actual, expected)


def test_python_scalar_arithmetic_stays_weak_before_it_meets_a_strict_array() -> None:
    value = strict.asarray([0.5, 1.5], dtype=strict.float32)

    def blend(argument: Any, scale: float) -> Any:
        return (1 - scale) * argument + scale * scale

    program = ad.stage(blend, value, 0.75)

    for staged in (program, ad.StagedProgram.from_dict(program.to_dict())):
        result = staged(value, 0.75)
        assert result.dtype == strict.float32
        np.testing.assert_array_equal(np.asarray(result), np.asarray(blend(value, 0.75)))


@pytest.mark.parametrize(
    "scalar",
    [
        pytest.param(lambda scale: scale, id="argument"),
        pytest.param(lambda scale: scale * scale, id="arithmetic"),
    ],
)
def test_program_gradients_promote_a_traced_weak_scalar(
    scalar: Callable[[Any], Any],
) -> None:
    """A compiled gradient replays ``maximum(weak scalar, float32)`` as float32."""
    value = strict.asarray([0.5, 1.5], dtype=strict.float32)

    def loss(argument: Any, scale: float) -> Any:
        xp = argument.__array_namespace__()
        return xp.sum(xp.maximum(scalar(scale), argument))

    program = ad.stage(loss, value, 0.75, array_api_version="2024.12")
    expected = ad.grad(loss)(value, 0.75)

    for gradient in (
        ad.grad(program)(value, 0.75),
        ad.stage(ad.grad(program), value, 0.75)(value, 0.75),
    ):
        assert gradient.dtype == strict.float32
        np.testing.assert_array_equal(np.asarray(gradient), np.asarray(expected))


def test_staged_code_sees_the_dtype_objects_of_its_example_provider() -> None:
    value = strict.asarray([1.0, 2.0], dtype=strict.float32)
    seen: list[tuple[object, ...]] = []

    def observe(argument: Any) -> Any:
        xp = argument.__array_namespace__()
        seen.append(
            (
                argument.dtype,
                (argument > 0).dtype,
                xp.float32,
                xp.result_type(argument, xp.float64),
                xp.finfo(argument).dtype,
                xp.iinfo(xp.int8).dtype,
            )
        )
        return xp.sin(argument) if argument.dtype == strict.float32 else xp.cos(argument)

    expected = observe(value)
    program = ad.stage(observe, value)

    assert seen[1] == seen[0]
    assert_allclose(np.asarray(program(value)), np.asarray(expected))


def test_specs_alone_present_numpy_dtypes_to_staged_code() -> None:
    seen: list[tuple[object, ...]] = []

    def observe(argument: Any) -> Any:
        seen.append((argument.dtype, argument.__array_namespace__().float32))
        return argument

    ad.stage(observe, specs=(ad.ArraySpec((2,), strict.float32),))

    assert seen == [(np.dtype(np.float32), np.dtype(np.float32))]


def test_a_staged_dtype_its_provider_lacks_fails_clearly() -> None:
    value = strict.asarray([1.0, 2.0], dtype=strict.float32)

    def to_half(argument: Any) -> Any:
        xp = argument.__array_namespace__()
        assert not hasattr(xp, "float16")
        return xp.astype(argument, np.float16).dtype

    with pytest.raises(TypeError, match="'array_api_strict' has no float16 dtype"):
        ad.stage(to_half, value)


def test_program_derivations_present_numpy_dtypes_to_the_rules_they_run() -> None:
    """A program records no provider, so its derivations stage against NumPy."""
    seen: list[object] = []

    @ad.primitive(name="tests.array_api_staging.dtype_probe")
    def cube(x: Any) -> Any:
        return x * x * x

    @cube.def_abstract
    def _cube_abstract(x: Any) -> Any:
        return x.spec

    @cube.def_jvp
    def _cube_jvp(output: Any, primals: Any, tangents: Any) -> Any:
        del output
        (x,), (tangent,) = primals, tangents
        seen.append(x.dtype)
        return 3 * x * x * tangent

    value = strict.asarray([1.0, 2.0], dtype=strict.float64)

    def loss(x: Any) -> Any:
        return x.__array_namespace__().sum(cube(x))

    ad.stage(ad.grad(loss), value)
    program = ad.stage(loss, value)
    for derived in (program, ad.StagedProgram.from_dict(program.to_dict())):
        gradient = ad.grad(derived)
        assert_allclose(np.asarray(gradient(value)), 3 * np.asarray(value) ** 2)

    assert seen == [strict.float64, np.dtype(np.float64), np.dtype(np.float64)]


@pytest.mark.parametrize(
    ("nest", "expected"),
    [
        pytest.param(lambda inner: lambda x: ad.stage(inner, x)(x), [2.0, -4.0], id="stage"),
        pytest.param(
            lambda inner: ad.grad(lambda x: x.__array_namespace__().sum(ad.stage(inner, x)(x))),
            [2.0, 2.0],
            id="grad",
        ),
        pytest.param(
            lambda inner: lambda x: ad.jvp(lambda y: ad.stage(inner, y)(y))(x, tangents=x)[1],
            [2.0, -4.0],
            id="jvp",
        ),
    ],
)
def test_a_nested_stage_presents_its_enclosing_provider_without_recording_it(
    nest: Callable[[Callable[[Any], Any]], Callable[[Any], Any]], expected: list[float]
) -> None:
    # The inner stage's examples are the enclosing trace's values, possibly
    # wrapped by a transform; probing their namespace would record into its graph.
    value = strict.asarray([1.0, -2.0], dtype=strict.float64)
    seen: list[object] = []

    def inner(array: Any) -> Any:
        seen.append(array.dtype)
        return array * 2

    program = ad.stage(nest(inner), value)

    assert seen == [strict.float64]
    assert program.trace is not None
    assert "array.empty" not in {node.op for node in program.trace.nodes}
    assert_allclose(np.asarray(program(value)), expected)


def test_dynamic_trace_materializes_scalars_for_a_2022_provider() -> None:
    strict.set_array_api_strict_flags(api_version="2022.12")
    try:
        value = strict.arange(4, dtype=strict.float32)

        def loss(argument: object) -> object:
            namespace = argument.__array_namespace__()
            scaled = 2 * argument
            return namespace.sum(scaled * scaled)

        gradient = ad.grad(loss)(value)

        assert gradient.shape == value.shape
        assert gradient.dtype == strict.float32
        assert_allclose(np.asarray(gradient), np.asarray([0.0, 8.0, 16.0, 24.0]))
    finally:
        strict.set_array_api_strict_flags(api_version="2024.12")


def test_mixed_signed_unsigned_outer_uses_array_api_promotion() -> None:
    program = ad.stage(
        lambda left, right: left.__array_namespace__().linalg.outer(left, right),
        specs=(ad.ArraySpec((2,), "uint8"), ad.ArraySpec((3,), "int8")),
        array_api_version="2022.12",
    )

    output_spec = program.to_dict()["program"]["output_specs"][0]
    assert output_spec["shape"] == [2, 3]
    assert output_spec["dtype"] == "int16"


def test_concat_axis_none_abstract_spec_flattens_inputs() -> None:
    _assert_staged_matches_eager(
        lambda xp, first, second: xp.concat((first, second), axis=None),
        strict.ones((2, 2), dtype=strict.float32),
        strict.ones((3,), dtype=strict.float32),
        tolerance=0.0,
    )


@pytest.mark.parametrize(
    ("operation", "input_value", "op_name", "expected_fields"),
    [
        (
            lambda xp, x: xp.linalg.eigh(x),
            strict.asarray([[3.0, 0.5], [0.5, 1.0]], dtype=strict.float32),
            "array_ext.linalg.eigh",
            ("eigenvalues", "eigenvectors"),
        ),
        (
            lambda xp, x: xp.linalg.qr(x, mode="complete"),
            strict.asarray([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]], dtype=strict.float32),
            "array_ext.linalg.qr",
            ("Q", "R"),
        ),
        (
            lambda xp, x: xp.linalg.slogdet(x),
            strict.asarray([[3.0, 0.5], [0.5, 1.0]], dtype=strict.float32),
            "array_ext.linalg.slogdet",
            ("sign", "logabsdet"),
        ),
        (
            lambda xp, x: xp.linalg.svd(x, full_matrices=False),
            strict.asarray([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]], dtype=strict.float32),
            "array_ext.linalg.svd",
            ("U", "S", "Vh"),
        ),
    ],
    ids=["eigh", "qr-complete", "slogdet", "svd-reduced"],
)
def test_fixed_arity_linalg_results_stage_with_standard_fields(
    operation: Callable[[object, object], object],
    input_value: Any,
    op_name: str,
    expected_fields: tuple[str, ...],
) -> None:
    program = _assert_staged_matches_eager(operation, input_value, tolerance=1e-5)

    for staged in (program, ad.StagedProgram.from_dict(program.to_dict())):
        assert staged(input_value)._fields == expected_fields
    expected = tuple(operation(strict, input_value))
    parent = next(
        program.graph.get_node(node_id)
        for node_id in program.graph.node_ids()
        if program.graph.get_node(node_id).op == op_name
    )
    assert parent.num_outputs == len(expected_fields)
    assert parent.output_shapes == [list(output.shape) for output in expected]
    assert parent.output_dtypes is not None
    assert [str(dtype).rsplit(".", 1)[-1] for dtype in parent.output_dtypes] == [
        str(output.dtype).rsplit(".", 1)[-1] for output in expected
    ]
