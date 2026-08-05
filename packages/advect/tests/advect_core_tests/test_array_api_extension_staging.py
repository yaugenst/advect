"""Abstract staging coverage for admitted Array API extension operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import array_api_strict as strict
import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize(
    ("operation", "input_value"),
    [
        (lambda xp, x: xp.fft.fft(x, n=6), strict.ones((4,), dtype=strict.complex64)),
        (lambda xp, x: xp.fft.ifft(x, n=6), strict.ones((4,), dtype=strict.complex64)),
        (
            lambda xp, x: xp.fft.fftn(x, s=(3, 6), axes=(0, 1)),
            strict.ones((2, 4), dtype=strict.complex64),
        ),
        (
            lambda xp, x: xp.fft.ifftn(x, s=(3, 6), axes=(0, 1)),
            strict.ones((2, 4), dtype=strict.complex64),
        ),
        (lambda xp, x: xp.fft.rfft(x, n=6), strict.ones((4,), dtype=strict.float32)),
        (lambda xp, x: xp.fft.irfft(x, n=6), strict.ones((4,), dtype=strict.complex64)),
        (
            lambda xp, x: xp.fft.rfftn(x, s=(3, 6), axes=(0, 1)),
            strict.ones((2, 4), dtype=strict.float32),
        ),
        (
            lambda xp, x: xp.fft.irfftn(x, s=(3, 6), axes=(0, 1)),
            strict.ones((2, 4), dtype=strict.complex64),
        ),
        (lambda xp, x: xp.fft.fftshift(x), strict.ones((2, 4), dtype=strict.complex64)),
        (lambda xp, x: xp.fft.ifftshift(x), strict.ones((2, 4), dtype=strict.complex64)),
    ],
    ids=[
        "fft",
        "ifft",
        "fftn",
        "ifftn",
        "rfft",
        "irfft",
        "rfftn",
        "irfftn",
        "fftshift",
        "ifftshift",
    ],
)
def test_fft_extension_abstract_specs_match_strict_provider(
    operation: Callable[[object, object], object],
    input_value: object,
) -> None:
    def function(value: object) -> object:
        return operation(value.__array_namespace__(), value)

    expected = function(input_value)
    program = ad.stage(
        function,
        specs=(ad.ArraySpec(input_value.shape, input_value.dtype),),
    )
    actual = program(input_value)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("function_name", ["rfft", "rfftn"])
def test_real_fft_preserves_double_precision_in_abstract_result(
    function_name: str,
) -> None:
    value = strict.asarray([1.0, 2.0, 3.0, 4.0], dtype=strict.float64)

    def transform(argument: object) -> object:
        namespace = argument.__array_namespace__()
        return getattr(namespace.fft, function_name)(argument)

    program = ad.stage(
        transform,
        specs=(ad.ArraySpec(value.shape, value.dtype),),
    )
    result = program(value)

    assert result.dtype == strict.complex128


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
    def transform(argument: object) -> tuple[object, object]:
        namespace = argument.__array_namespace__()
        return 0.25 * namespace.sum(argument), 1j * argument

    try:
        total, rotated = ad.stage(
            transform,
            specs=(ad.ArraySpec(value.shape, value.dtype),),
            array_api_version="2022.12",
        )(value)

        assert str(total.dtype).endswith("float64")
        assert str(rotated.dtype).endswith("complex64")
    finally:
        strict.set_array_api_strict_flags(api_version="2024.12")


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
    left = strict.ones((2, 2), dtype=strict.float32)
    right = strict.ones((3,), dtype=strict.float32)

    def flatten_join(first: object, second: object) -> object:
        return first.__array_namespace__().concat((first, second), axis=None)

    result = ad.stage(
        flatten_join,
        specs=(
            ad.ArraySpec(left.shape, left.dtype),
            ad.ArraySpec(right.shape, right.dtype),
        ),
    )(left, right)

    assert result.shape == (7,)
    assert result.dtype == strict.float32


def test_solve_extension_abstract_spec_matches_strict_provider() -> None:
    matrix = strict.asarray([[3.0, 1.0], [1.0, 2.0]], dtype=strict.float32)
    right = strict.asarray([1.0, 2.0], dtype=strict.float32)

    def solve(a: object, b: object) -> object:
        return a.__array_namespace__().linalg.solve(a, b)

    expected = solve(matrix, right)
    program = ad.stage(
        solve,
        specs=(
            ad.ArraySpec(matrix.shape, matrix.dtype),
            ad.ArraySpec(right.shape, right.dtype),
        ),
    )
    actual = program(matrix, right)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6, atol=1e-6)


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
    input_value: object,
    op_name: str,
    expected_fields: tuple[str, ...],
) -> None:
    def decompose(value: object) -> object:
        return operation(value.__array_namespace__(), value)

    expected = tuple(operation(strict, input_value))
    program = ad.stage(
        decompose,
        specs=(ad.ArraySpec(input_value.shape, input_value.dtype),),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())

    for actual in (program(input_value), restored(input_value)):
        assert actual._fields == expected_fields
        assert len(actual) == len(expected_fields)
        for field in expected_fields:
            assert getattr(actual, field) is not None
        for output, expected_output in zip(actual, expected, strict=True):
            assert output.shape == expected_output.shape
            assert output.dtype == expected_output.dtype
            assert_allclose(
                np.asarray(output),
                np.asarray(expected_output),
                rtol=1e-5,
                atol=1e-5,
            )

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
