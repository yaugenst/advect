# ruff: noqa: PLW0108
"""Public validation and metadata contracts for abstract staging domains."""

from __future__ import annotations

from typing import TYPE_CHECKING

import array_api_strict as strict
import numpy as np
import pytest

import advect as ad

if TYPE_CHECKING:
    from collections.abc import Callable


def _array_spec(shape: tuple[int, ...], dtype: str = "float32") -> ad.ArraySpec:
    return ad.ArraySpec(shape, dtype)


@pytest.mark.parametrize(
    ("operation", "specs", "error", "match"),
    [
        pytest.param(
            lambda x: np.linalg.norm(x),
            (_array_spec((2, 3, 4)),),
            ValueError,
            "requires axis=",
            id="norm-rank",
        ),
        pytest.param(
            lambda x: np.linalg.norm(x, axis=(0, 1, 2)),
            (_array_spec((2, 3, 4)),),
            ValueError,
            "one or two axes",
            id="norm-axis-count",
        ),
        pytest.param(
            lambda x: np.linalg.cholesky(x),
            (_array_spec((3,)),),
            ValueError,
            "at least two dimensions",
            id="cholesky-rank",
        ),
        pytest.param(
            lambda x: np.linalg.det(x),
            (_array_spec((2, 3)),),
            ValueError,
            "square matrix",
            id="det-square",
        ),
        pytest.param(
            lambda x: np.linalg.cholesky(x, upper=1),
            (_array_spec((2, 2)),),
            TypeError,
            "upper must be a bool",
            id="cholesky-upper",
        ),
        pytest.param(
            lambda x: np.linalg.eigvals(x),
            (_array_spec((2, 2)),),
            TypeError,
            "requires a complex input",
            id="eigvals-real",
        ),
        pytest.param(
            lambda x: np.linalg.eigvalsh(x, UPLO="X"),
            (_array_spec((2, 2)),),
            ValueError,
            "UPLO must be 'L' or 'U'",
            id="eigvalsh-uplo",
        ),
        pytest.param(
            lambda x: np.linalg.matrix_norm(x, keepdims=1),
            (_array_spec((2, 2)),),
            TypeError,
            "keepdims must be a bool",
            id="matrix-norm-keepdims",
        ),
        pytest.param(
            lambda x, tolerance: np.linalg.pinv(x, rtol=tolerance),
            (_array_spec((2, 3, 2)), _array_spec((1, 2))),
            ValueError,
            "tolerance must broadcast",
            id="pinv-tolerance-broadcast",
        ),
        pytest.param(
            lambda x: np.linalg.vector_norm(x, keepdims=1),
            (_array_spec((2, 3)),),
            TypeError,
            "keepdims must be a bool",
            id="vector-norm-keepdims",
        ),
        pytest.param(
            lambda matrix, right: np.linalg.solve(matrix, right),
            (_array_spec((2, 3)), _array_spec((3,))),
            ValueError,
            "coefficient input.*square matrix",
            id="solve-square",
        ),
        pytest.param(
            lambda matrix, right: np.linalg.solve(matrix, right),
            (_array_spec((3, 3)), _array_spec((2,))),
            ValueError,
            "right-hand side.*core dimension",
            id="solve-vector-core",
        ),
        pytest.param(
            lambda matrix, right: np.linalg.solve(matrix, right),
            (_array_spec((3, 3)), _array_spec((2, 4))),
            ValueError,
            "right-hand side.*core dimension",
            id="solve-matrix-core",
        ),
        pytest.param(
            lambda x: np.linalg.eig(x),
            (_array_spec((3,), "complex64"),),
            ValueError,
            "at least two dimensions",
            id="eig-rank",
        ),
        pytest.param(
            lambda x: np.linalg.eigh(x),
            (_array_spec((2, 3)),),
            ValueError,
            "square matrix",
            id="eigh-square",
        ),
        pytest.param(
            lambda x: np.linalg.eig(x),
            (_array_spec((2, 2)),),
            TypeError,
            "requires a complex input",
            id="eig-real",
        ),
        pytest.param(
            lambda x: np.linalg.qr(x, mode="raw"),
            (_array_spec((2, 2)),),
            ValueError,
            "mode must be 'reduced' or 'complete'",
            id="qr-mode",
        ),
        pytest.param(
            lambda x: np.linalg.svd(x, compute_uv=0),
            (_array_spec((2, 2)),),
            ValueError,
            "compute_uv must be true",
            id="svd-compute-uv",
        ),
        pytest.param(
            lambda x: np.linalg.svd(x, hermitian=1),
            (_array_spec((2, 2)),),
            TypeError,
            "hermitian must be a bool",
            id="svd-hermitian",
        ),
        pytest.param(
            lambda x: np.linalg.svd(x, full_matrices=1),
            (_array_spec((2, 2)),),
            TypeError,
            "full_matrices must be a bool",
            id="svd-full-matrices",
        ),
        pytest.param(
            lambda left, right: np.linalg.outer(left, right),
            (_array_spec((2, 2)), _array_spec((2,))),
            ValueError,
            "one-dimensional",
            id="outer-rank",
        ),
        pytest.param(
            lambda left, right: np.linalg.cross(left, right),
            (_array_spec((2, 2)), _array_spec((2, 2))),
            ValueError,
            "three-component vectors",
            id="cross-components",
        ),
        pytest.param(
            lambda left, right: np.linalg.vecdot(left, right),
            (_array_spec((2, 3)), _array_spec((2, 4))),
            ValueError,
            "equal length",
            id="vecdot-length",
        ),
        pytest.param(
            lambda left, right: np.dot(left, right),
            (_array_spec((2, 3)), _array_spec((4, 2))),
            ValueError,
            "contracted dimensions.*equal lengths",
            id="dot-core",
        ),
        pytest.param(
            lambda x: np.transpose(x, axes=(0,)),
            (_array_spec((2, 3)),),
            ValueError,
            "every input axis exactly once",
            id="transpose-axes",
        ),
        pytest.param(
            lambda x: np.broadcast_to(x, (1, 3)),
            (_array_spec((2, 3)),),
            ValueError,
            "Cannot broadcast shape",
            id="broadcast-target",
        ),
        pytest.param(
            lambda x: np.expand_dims(x, axis=(0, 0)),
            (_array_spec((2, 3)),),
            ValueError,
            "Repeated expansion axis",
            id="expand-dims-repeat",
        ),
        pytest.param(
            lambda x: np.squeeze(x, axis=0),
            (_array_spec((2, 3)),),
            ValueError,
            "Cannot squeeze non-unit axes",
            id="squeeze-non-unit",
        ),
        pytest.param(
            lambda x: np.repeat(x, -1),
            (_array_spec((2, 3)),),
            NotImplementedError,
            "requires one non-negative integer",
            id="repeat-negative",
        ),
        pytest.param(
            lambda x: np.tile(x, (2, -1)),
            (_array_spec((2, 3)),),
            ValueError,
            "repetitions must be non-negative",
            id="tile-negative",
        ),
        pytest.param(
            lambda left, right: np.concatenate((left, right), axis=0),
            (_array_spec((2, 3)), _array_spec((4,))),
            ValueError,
            "equal rank",
            id="concatenate-rank",
        ),
        pytest.param(
            lambda left, right: np.concatenate((left, right), axis=0),
            (_array_spec((2, 3)), _array_spec((4, 4))),
            ValueError,
            "disagree outside the joined axis",
            id="concatenate-shape",
        ),
        pytest.param(
            lambda left, right: np.stack((left, right), axis=0),
            (_array_spec((2, 3)), _array_spec((2, 4))),
            ValueError,
            "identical shapes",
            id="stack-shape",
        ),
        pytest.param(
            lambda values, queries: np.searchsorted(values, queries),
            (_array_spec((2, 3)), _array_spec((2,))),
            ValueError,
            "sorted input must be one-dimensional",
            id="searchsorted-rank",
        ),
        pytest.param(
            lambda values, queries: np.searchsorted(values, queries, side="middle"),
            (_array_spec((3,)), _array_spec((2,))),
            ValueError,
            "side must be 'left' or 'right'",
            id="searchsorted-side",
        ),
        pytest.param(
            lambda values, indices: np.take_along_axis(values, indices, axis=1),
            (_array_spec((2, 3)), _array_spec((3,), "int64")),
            ValueError,
            "same rank",
            id="take-along-axis-rank",
        ),
        pytest.param(
            lambda x: np.diagonal(x),
            (_array_spec((3,)),),
            ValueError,
            "at least two dimensions",
            id="diagonal-rank",
        ),
        pytest.param(
            lambda x: np.diagonal(x, offset=1.5),
            (_array_spec((2, 3)),),
            TypeError,
            "offset must be an integer",
            id="diagonal-offset",
        ),
        pytest.param(
            lambda x: np.diagonal(x, axis1=0, axis2=0),
            (_array_spec((2, 3)),),
            ValueError,
            "axes must be distinct",
            id="diagonal-axes",
        ),
        pytest.param(
            lambda x: np.trace(x),
            (_array_spec((3,)),),
            ValueError,
            "at least two dimensions",
            id="trace-rank",
        ),
        pytest.param(
            lambda x: np.trace(x, axis1=0, axis2=0),
            (_array_spec((2, 3)),),
            ValueError,
            "axes must be distinct",
            id="trace-axes",
        ),
        pytest.param(
            lambda x: x.__array_namespace__().eye(-1, dtype=x.dtype),
            (_array_spec((1,)),),
            ValueError,
            "dimensions must be non-negative integers",
            id="eye-dimensions",
        ),
        pytest.param(
            lambda x: x.__array_namespace__().linspace(0.0, 1.0, -1, dtype=x.dtype),
            (_array_spec((1,)),),
            ValueError,
            "num must be a non-negative integer",
            id="linspace-num",
        ),
        pytest.param(
            lambda x: x.__array_namespace__().linspace(
                0.0,
                1.0,
                3,
                endpoint=1,
                dtype=x.dtype,
            ),
            (_array_spec((1,)),),
            TypeError,
            "endpoint must be a bool",
            id="linspace-endpoint",
        ),
        pytest.param(
            lambda left, right: np.convolve(left, right),
            (_array_spec((2, 3)), _array_spec((2,))),
            ValueError,
            "inputs must be one-dimensional",
            id="convolve-rank",
        ),
        pytest.param(
            lambda left, right: np.convolve(left, right),
            (_array_spec((0,)), _array_spec((2,))),
            ValueError,
            "inputs cannot be empty",
            id="convolve-empty",
        ),
        pytest.param(
            lambda left, right: np.convolve(left, right, mode="other"),
            (_array_spec((2,)), _array_spec((2,))),
            ValueError,
            "mode must be full, same, or valid",
            id="convolve-mode",
        ),
        pytest.param(
            lambda x: x.__array_namespace__().fft.fftfreq(0, dtype=x.dtype),
            (_array_spec((1,)),),
            ValueError,
            "n must be a positive integer",
            id="fftfreq-size",
        ),
        pytest.param(
            lambda x: x.__array_namespace__().fft.rfftfreq(4, d=0, dtype=x.dtype),
            (_array_spec((1,)),),
            ValueError,
            "d must be a nonzero real scalar",
            id="rfftfreq-spacing",
        ),
        pytest.param(
            lambda x: x.__array_namespace__().cumulative_sum(x),
            (_array_spec((2, 3)),),
            ValueError,
            "require axis=",
            id="cumulative-axis",
        ),
        pytest.param(
            lambda x: x.__array_namespace__().sum(x, axis=True),
            (_array_spec((2, 3)),),
            TypeError,
            "Axis must be an integer",
            id="axis-bool",
        ),
        pytest.param(
            lambda x: x.__array_namespace__().sum(x, axis=1.5),
            (_array_spec((2, 3)),),
            TypeError,
            "integer or iterable",
            id="axis-type",
        ),
        pytest.param(
            lambda x: x.__array_namespace__().sum(x, axis=(0, 0)),
            (_array_spec((2, 3)),),
            ValueError,
            "Repeated axis",
            id="axis-repeat",
        ),
        pytest.param(
            lambda left, right: left @ right,
            (_array_spec(()), _array_spec((2,))),
            ValueError,
            "at least one dimension",
            id="matmul-scalar",
        ),
        pytest.param(
            lambda left, right: left @ right,
            (_array_spec((2, 3)), _array_spec((4, 2))),
            ValueError,
            "core dimensions disagree",
            id="matmul-core",
        ),
        pytest.param(
            lambda x: x.__array_namespace__().reshape(x, object()),
            (_array_spec((6,)),),
            TypeError,
            "Shape must be an integer or iterable",
            id="shape-type",
        ),
        pytest.param(
            lambda x: x.__array_namespace__().reshape(x, (True, 6)),
            (_array_spec((6,)),),
            TypeError,
            "Shape must contain integers",
            id="shape-component",
        ),
        pytest.param(
            lambda x: x.__array_namespace__().fft.fft(x, n="4"),
            (_array_spec((4,)),),
            TypeError,
            "length must be an integer or None",
            id="fft-length-type",
        ),
        pytest.param(
            lambda x: x.__array_namespace__().fft.fft(x, n=0),
            (_array_spec((4,)),),
            ValueError,
            "positive integer",
            id="fft-length-value",
        ),
        pytest.param(
            lambda x: x.__array_namespace__().fft.fft(x),
            (_array_spec((4,), "int32"),),
            TypeError,
            "FFT input must be floating-point or complex",
            id="fft-dtype",
        ),
        pytest.param(
            lambda x: x.__array_namespace__().fft.fftn(x, axes=()),
            (_array_spec((2, 3)),),
            ValueError,
            "axes must be non-empty",
            id="fftn-empty-axes",
        ),
        pytest.param(
            lambda x: x.__array_namespace__().fft.fftn(x, s=(2,), axes=(0, 1)),
            (_array_spec((2, 3)),),
            ValueError,
            "sizes and axes must have equal length",
            id="fftn-size-count",
        ),
        pytest.param(
            lambda x: x.__array_namespace__().reshape(x, (-1, -1)),
            (_array_spec((6,)),),
            ValueError,
            "Invalid reshape target",
            id="reshape-unknown-count",
        ),
        pytest.param(
            lambda x: x.__array_namespace__().reshape(x, (0, -1)),
            (_array_spec((6,)),),
            ValueError,
            "reshape changes element count",
            id="reshape-zero-known-size",
        ),
        pytest.param(
            lambda x: x.__array_namespace__().reshape(x, (4, 2)),
            (_array_spec((6,)),),
            ValueError,
            "reshape changes element count",
            id="reshape-element-count",
        ),
        pytest.param(
            lambda x: x.__array_namespace__().moveaxis(x, (0, 1), (2,)),
            (_array_spec((2, 3, 4)),),
            ValueError,
            "source and destination must have equal length",
            id="moveaxis-axis-count",
        ),
        pytest.param(
            lambda left, right: left.__array_namespace__().linalg.tensordot(
                left,
                right,
                axes=True,
            ),
            (_array_spec((2, 3)), _array_spec((3, 2))),
            TypeError,
            "axes must be an integer or a pair",
            id="tensordot-bool",
        ),
        pytest.param(
            lambda left, right: left.__array_namespace__().linalg.tensordot(
                left,
                right,
                axes=3,
            ),
            (_array_spec((2, 3)), _array_spec((3, 2))),
            ValueError,
            "Invalid tensordot axes count",
            id="tensordot-axis-count",
        ),
        pytest.param(
            lambda left, right: left.__array_namespace__().linalg.tensordot(
                left,
                right,
                axes=1.5,
            ),
            (_array_spec((2, 3)), _array_spec((3, 2))),
            TypeError,
            "axes must be an integer or a pair",
            id="tensordot-axis-type",
        ),
        pytest.param(
            lambda left, right: left.__array_namespace__().linalg.tensordot(
                left,
                right,
                axes=((0,),),
            ),
            (_array_spec((2, 3)), _array_spec((3, 2))),
            ValueError,
            "must contain two axis sequences",
            id="tensordot-axis-pair",
        ),
        pytest.param(
            lambda left, right: left.__array_namespace__().linalg.tensordot(
                left,
                right,
                axes=((0, 1), (0,)),
            ),
            (_array_spec((2, 3)), _array_spec((2, 3))),
            ValueError,
            "axis lists must have equal length",
            id="tensordot-axis-list-count",
        ),
        pytest.param(
            lambda left, right: left.__array_namespace__().linalg.tensordot(
                left,
                right,
                axes=((1,), (0,)),
            ),
            (_array_spec((2, 3)), _array_spec((4, 2))),
            ValueError,
            "contraction dimensions disagree",
            id="tensordot-core",
        ),
        pytest.param(
            lambda x: x.__array_namespace__().arange(bool(1), 3, dtype=x.dtype),
            (_array_spec((1,)),),
            TypeError,
            "concrete real scalars",
            id="arange-start",
        ),
        pytest.param(
            lambda x: x.__array_namespace__().arange(0, object(), dtype=x.dtype),
            (_array_spec((1,)),),
            TypeError,
            "concrete real scalars",
            id="arange-stop",
        ),
        pytest.param(
            lambda x: x.__array_namespace__().arange(0, 3, object(), dtype=x.dtype),
            (_array_spec((1,)),),
            TypeError,
            "concrete real scalars",
            id="arange-step-type",
        ),
        pytest.param(
            lambda x: x.__array_namespace__().arange(0, 3, 0, dtype=x.dtype),
            (_array_spec((1,)),),
            ValueError,
            "step must be nonzero",
            id="arange-step-zero",
        ),
        pytest.param(
            lambda x: x.__array_namespace__(api_version="2023.12").sum(x),
            (_array_spec((2,)),),
            ValueError,
            "requested.*targets",
            id="namespace-version",
        ),
        pytest.param(
            lambda x: len(x),
            (_array_spec((2,)),),
            ad.TracingError,
            r"len\(\).*not allowed",
            id="len-data-dependence",
        ),
        pytest.param(
            lambda x: x.item(),
            (_array_spec((2,)),),
            ValueError,
            "array of size 1",
            id="item-size",
        ),
        pytest.param(
            lambda x: x + object(),
            (_array_spec((2,)),),
            TypeError,
            "Cannot stage concrete operand",
            id="concrete-operand-type",
        ),
        pytest.param(
            lambda x: x.__array_namespace__().asarray(),
            (_array_spec((2,)),),
            TypeError,
            "asarray.*requires an input",
            id="asarray-missing-input",
        ),
        pytest.param(
            lambda x: x.__array_namespace__().cumulative_sum(include_initial=True),
            (_array_spec((2,)),),
            TypeError,
            "expects an array and optional axis",
            id="cumulative-missing-input",
        ),
        pytest.param(
            lambda x: x.__array_namespace__().diff(),
            (_array_spec((2,)),),
            TypeError,
            "diff.*expects",
            id="diff-missing-input",
        ),
        pytest.param(
            lambda x: x.__array_namespace__().searchsorted(x, sorter=x),
            (_array_spec((2,)),),
            TypeError,
            "expects two positional array arguments",
            id="searchsorted-missing-query",
        ),
    ],
)
def test_invalid_abstract_domain_contracts_fail_while_staging(
    operation: Callable[..., object],
    specs: tuple[ad.ArraySpec, ...],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        ad.stage(operation, specs=specs)


@pytest.mark.parametrize("operation", ["concat", "stack"])
def test_empty_array_api_sequences_report_the_contract(operation: str) -> None:
    def combine(value: object) -> object:
        namespace = value.__array_namespace__()
        return getattr(namespace, operation)(())

    with pytest.raises(ValueError, match="requires a non-empty list or tuple of arrays"):
        ad.stage(combine, specs=(_array_spec((1,)),))


@pytest.mark.parametrize("operation", [np.concatenate, np.stack])
def test_empty_numpy_sequences_preserve_provider_errors(
    operation: Callable[[object], object],
) -> None:
    with pytest.raises(ValueError, match="need at least one array"):
        ad.stage(lambda _value: operation(()), specs=(_array_spec((1,)),))


def test_metadata_and_array_methods_drive_staged_computation() -> None:
    def compute(value: object) -> object:
        namespace = value.__array_namespace__()
        info = namespace.__array_namespace_info__()
        metadata_matches = (
            info is namespace
            and namespace.result_type(np.asarray([1], dtype=np.int16)) == "int16"
            and namespace.isdtype(value.dtype, np.dtype("complex64"))
        )
        return value.real + value.item(2) + value.mean() if metadata_matches else -value.real

    value = np.asarray([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64)
    program = ad.stage(compute, specs=(_array_spec(value.shape, "complex64"),))
    restored = ad.StagedProgram.from_dict(program.to_dict())

    expected = value.real + value.item(2) + value.mean()
    np.testing.assert_allclose(program(value), expected)
    np.testing.assert_allclose(restored(value), expected)


@pytest.mark.parametrize(
    "value",
    [np.asarray([2.0], dtype=np.float32), strict.asarray([2.0], dtype=strict.float32)],
)
@pytest.mark.parametrize("scalar", [0.0, np.float32(0.0)])
@pytest.mark.parametrize("dtype", [None, "float32"])
def test_nested_asarray_promotes_python_scalars_before_stacking(
    value: object,
    scalar: object,
    dtype: str | None,
) -> None:
    def assemble(value: object) -> object:
        namespace = value.__array_namespace__()
        options = {} if dtype is None else {"dtype": getattr(namespace, dtype)}
        return namespace.asarray([[value[0], scalar]], **options)

    program = ad.stage(assemble, specs=(_array_spec(value.shape),))
    restored = ad.StagedProgram.from_dict(program.to_dict())

    expected_dtype = np.float64 if dtype is None and type(scalar) is float else np.float32
    for actual in (program(value), restored(value)):
        assert str(actual.dtype).endswith(np.dtype(expected_dtype).name)
        np.testing.assert_array_equal(np.asarray(actual), [[2.0, 0.0]])


def test_nested_staged_scalar_preserves_weak_metadata() -> None:
    scalar_spec = ad.ArraySpec((), "float64", weak=True)
    inner = ad.stage(lambda value: value + 1, specs=(scalar_spec,))
    outer = ad.stage(lambda value: 2 * inner(value), specs=(scalar_spec,))

    assert outer.signature[0] == (scalar_spec,)
    assert outer(3.0) == 8.0
    assert ad.StagedProgram.from_dict(outer.to_dict())(3.0) == 8.0


def test_mixed_integer_promotion_and_scalar_dot_round_trip() -> None:
    def compute(left: object, right: object) -> tuple[object, object]:
        return left + right, np.dot(left[0], right)

    left = np.asarray([2], dtype=np.int64)
    right = np.asarray([1, 3], dtype=np.uint64)
    program = ad.stage(
        compute,
        specs=(_array_spec(left.shape, "int64"), _array_spec(right.shape, "uint64")),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())

    for promoted, dotted in (program(left, right), restored(left, right)):
        assert promoted.dtype == np.dtype("float64")
        np.testing.assert_array_equal(promoted, left + right)
        np.testing.assert_array_equal(dotted, np.dot(left[0], right))


@pytest.mark.parametrize("shape", [(-1,), (1, -2)])
def test_array_spec_rejects_negative_dimensions(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ad.ArraySpec(shape, "float32")


@pytest.mark.parametrize(
    ("operation", "error", "match"),
    [
        pytest.param(
            lambda x: _mutate_input_view(x),
            ad.MutationError,
            "staged input",
            id="input-view",
        ),
        pytest.param(
            lambda x: _mutate_copied_view(x),
            ad.MutationError,
            "through a staged view",
            id="copied-view",
        ),
        pytest.param(
            lambda x: _mutate_reshaped_view(x),
            ad.MutationError,
            "nested or reshaped staged view",
            id="reshaped-view",
        ),
        pytest.param(
            lambda x: _change_indexed_dtype(x),
            ad.MutationError,
            "would change shape or dtype",
            id="indexed-dtype",
        ),
        pytest.param(
            lambda x: _change_array_dtype(x),
            ad.MutationError,
            "would change shape or dtype",
            id="array-dtype",
        ),
        pytest.param(
            lambda x: _assign_wrong_shape(x),
            ValueError,
            "Cannot assign shape",
            id="assignment-shape",
        ),
    ],
)
def test_staged_mutation_rejects_alias_and_metadata_changes(
    operation: Callable[[object], object],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        ad.stage(operation, specs=(_array_spec((2, 2)),))


def _mutate_input_view(value: object) -> object:
    value[0] += 1
    return value


def _mutate_copied_view(value: object) -> object:
    result = value.copy()
    view = result[1:]
    view[0] = 2
    return result


def _mutate_reshaped_view(value: object) -> object:
    result = value.copy()
    view = result.reshape((4,))
    view += 1
    return result


def _change_indexed_dtype(value: object) -> object:
    result = value.copy()
    result[0] += 1j
    return result


def _change_array_dtype(value: object) -> object:
    result = value.copy()
    result += 1j
    return result


def _assign_wrong_shape(value: object) -> object:
    result = value.copy()
    result[0] = value
    return result
