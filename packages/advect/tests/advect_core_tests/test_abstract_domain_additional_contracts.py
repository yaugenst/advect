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


def _case(
    operation: Callable[..., object],
    shapes: tuple[tuple[int, ...], ...],
    match: str,
    error: type[Exception] = ValueError,
    dtypes: tuple[str, ...] = (),
) -> tuple[Callable[..., object], tuple[ad.ArraySpec, ...], type[Exception], str]:
    resolved_dtypes = dtypes or ("float32",) * len(shapes)
    specs = tuple(
        _array_spec(shape, dtype) for shape, dtype in zip(shapes, resolved_dtypes, strict=True)
    )
    return operation, specs, error, match


_INVALID_STAGING_CASES = {
    "norm-rank": _case(lambda x: np.linalg.norm(x), ((2, 3, 4),), "requires axis="),
    "norm-axis-count": _case(
        lambda x: np.linalg.norm(x, axis=(0, 1, 2)), ((2, 3, 4),), "one or two axes"
    ),
    "cholesky-rank": _case(lambda x: np.linalg.cholesky(x), ((3,),), "at least two dimensions"),
    "det-square": _case(lambda x: np.linalg.det(x), ((2, 3),), "square matrix"),
    "cholesky-upper": _case(
        lambda x: np.linalg.cholesky(x, upper=1), ((2, 2),), "upper must be a bool", TypeError
    ),
    "eigvals-real": _case(
        lambda x: np.linalg.eigvals(x), ((2, 2),), "requires a complex input", TypeError
    ),
    "eigvalsh-uplo": _case(
        lambda x: np.linalg.eigvalsh(x, UPLO="X"), ((2, 2),), "UPLO must be 'L' or 'U'"
    ),
    "matrix-norm-keepdims": _case(
        lambda x: np.linalg.matrix_norm(x, keepdims=1),
        ((2, 2),),
        "keepdims must be a bool",
        TypeError,
    ),
    "pinv-tolerance-broadcast": _case(
        lambda x, tolerance: np.linalg.pinv(x, rtol=tolerance),
        ((2, 3, 2), (1, 2)),
        "tolerance must broadcast",
    ),
    "vector-norm-keepdims": _case(
        lambda x: np.linalg.vector_norm(x, keepdims=1),
        ((2, 3),),
        "keepdims must be a bool",
        TypeError,
    ),
    "solve-square": _case(
        lambda matrix, right: np.linalg.solve(matrix, right),
        ((2, 3), (3,)),
        "coefficient input.*square matrix",
    ),
    "solve-vector-core": _case(
        lambda matrix, right: np.linalg.solve(matrix, right),
        ((3, 3), (2,)),
        "right-hand side.*core dimension",
    ),
    "solve-matrix-core": _case(
        lambda matrix, right: np.linalg.solve(matrix, right),
        ((3, 3), (2, 4)),
        "right-hand side.*core dimension",
    ),
    "eig-rank": _case(
        lambda x: np.linalg.eig(x), ((3,),), "at least two dimensions", ValueError, ("complex64",)
    ),
    "eigh-square": _case(lambda x: np.linalg.eigh(x), ((2, 3),), "square matrix"),
    "eig-real": _case(lambda x: np.linalg.eig(x), ((2, 2),), "requires a complex input", TypeError),
    "qr-mode": _case(
        lambda x: np.linalg.qr(x, mode="raw"), ((2, 2),), "mode must be 'reduced' or 'complete'"
    ),
    "svd-compute-uv": _case(
        lambda x: np.linalg.svd(x, compute_uv=0), ((2, 2),), "compute_uv must be true"
    ),
    "svd-hermitian": _case(
        lambda x: np.linalg.svd(x, hermitian=1), ((2, 2),), "hermitian must be a bool", TypeError
    ),
    "svd-full-matrices": _case(
        lambda x: np.linalg.svd(x, full_matrices=1),
        ((2, 2),),
        "full_matrices must be a bool",
        TypeError,
    ),
    "outer-rank": _case(
        lambda left, right: np.linalg.outer(left, right), ((2, 2), (2,)), "one-dimensional"
    ),
    "cross-components": _case(
        lambda left, right: np.linalg.cross(left, right),
        ((2, 2), (2, 2)),
        "three-component vectors",
    ),
    "vecdot-length": _case(
        lambda left, right: np.linalg.vecdot(left, right), ((2, 3), (2, 4)), "equal length"
    ),
    "dot-core": _case(
        lambda left, right: np.dot(left, right),
        ((2, 3), (4, 2)),
        "contracted dimensions.*equal lengths",
    ),
    "transpose-axes": _case(
        lambda x: np.transpose(x, axes=(0,)), ((2, 3),), "every input axis exactly once"
    ),
    "broadcast-target": _case(
        lambda x: np.broadcast_to(x, (1, 3)), ((2, 3),), "Cannot broadcast shape"
    ),
    "expand-dims-repeat": _case(
        lambda x: np.expand_dims(x, axis=(0, 0)), ((2, 3),), "Repeated expansion axis"
    ),
    "squeeze-non-unit": _case(
        lambda x: np.squeeze(x, axis=0), ((2, 3),), "Cannot squeeze non-unit axes"
    ),
    "repeat-negative": _case(
        lambda x: np.repeat(x, -1),
        ((2, 3),),
        "requires one non-negative integer",
        NotImplementedError,
    ),
    "tile-negative": _case(
        lambda x: np.tile(x, (2, -1)), ((2, 3),), "repetitions must be non-negative"
    ),
    "concatenate-rank": _case(
        lambda left, right: np.concatenate((left, right), axis=0), ((2, 3), (4,)), "equal rank"
    ),
    "concatenate-shape": _case(
        lambda left, right: np.concatenate((left, right), axis=0),
        ((2, 3), (4, 4)),
        "disagree outside the joined axis",
    ),
    "stack-shape": _case(
        lambda left, right: np.stack((left, right), axis=0), ((2, 3), (2, 4)), "identical shapes"
    ),
    "searchsorted-rank": _case(
        lambda values, queries: np.searchsorted(values, queries),
        ((2, 3), (2,)),
        "sorted input must be one-dimensional",
    ),
    "searchsorted-side": _case(
        lambda values, queries: np.searchsorted(values, queries, side="middle"),
        ((3,), (2,)),
        "side must be 'left' or 'right'",
    ),
    "take-along-axis-rank": _case(
        lambda values, indices: np.take_along_axis(values, indices, axis=1),
        ((2, 3), (3,)),
        "same rank",
        ValueError,
        ("float32", "int64"),
    ),
    "diagonal-rank": _case(lambda x: np.diagonal(x), ((3,),), "at least two dimensions"),
    "diagonal-offset": _case(
        lambda x: np.diagonal(x, offset=1.5), ((2, 3),), "offset must be an integer", TypeError
    ),
    "diagonal-axes": _case(
        lambda x: np.diagonal(x, axis1=0, axis2=0), ((2, 3),), "axes must be distinct"
    ),
    "trace-rank": _case(lambda x: np.trace(x), ((3,),), "at least two dimensions"),
    "trace-axes": _case(
        lambda x: np.trace(x, axis1=0, axis2=0), ((2, 3),), "axes must be distinct"
    ),
    "eye-dimensions": _case(
        lambda x: x.__array_namespace__().eye(-1, dtype=x.dtype),
        ((1,),),
        "dimensions must be non-negative integers",
    ),
    "linspace-num": _case(
        lambda x: x.__array_namespace__().linspace(0.0, 1.0, -1, dtype=x.dtype),
        ((1,),),
        "num must be a non-negative integer",
    ),
    "linspace-endpoint": _case(
        lambda x: x.__array_namespace__().linspace(0.0, 1.0, 3, endpoint=1, dtype=x.dtype),
        ((1,),),
        "endpoint must be a bool",
        TypeError,
    ),
    "convolve-rank": _case(
        lambda left, right: np.convolve(left, right),
        ((2, 3), (2,)),
        "inputs must be one-dimensional",
    ),
    "convolve-empty": _case(
        lambda left, right: np.convolve(left, right), ((0,), (2,)), "inputs cannot be empty"
    ),
    "convolve-mode": _case(
        lambda left, right: np.convolve(left, right, mode="other"),
        ((2,), (2,)),
        "mode must be full, same, or valid",
    ),
    "fftfreq-size": _case(
        lambda x: x.__array_namespace__().fft.fftfreq(0, dtype=x.dtype),
        ((1,),),
        "n must be a positive integer",
    ),
    "rfftfreq-spacing": _case(
        lambda x: x.__array_namespace__().fft.rfftfreq(4, d=0, dtype=x.dtype),
        ((1,),),
        "d must be a nonzero real scalar",
    ),
    "cumulative-axis": _case(
        lambda x: x.__array_namespace__().cumulative_sum(x), ((2, 3),), "require axis="
    ),
    "axis-bool": _case(
        lambda x: x.__array_namespace__().sum(x, axis=True),
        ((2, 3),),
        "Axis must be an integer",
        TypeError,
    ),
    "axis-type": _case(
        lambda x: x.__array_namespace__().sum(x, axis=1.5),
        ((2, 3),),
        "integer or iterable",
        TypeError,
    ),
    "axis-repeat": _case(
        lambda x: x.__array_namespace__().sum(x, axis=(0, 0)), ((2, 3),), "Repeated axis"
    ),
    "matmul-scalar": _case(lambda left, right: left @ right, ((), (2,)), "at least one dimension"),
    "matmul-core": _case(
        lambda left, right: left @ right, ((2, 3), (4, 2)), "core dimensions disagree"
    ),
    "shape-type": _case(
        lambda x: x.__array_namespace__().reshape(x, object()),
        ((6,),),
        "Shape must be an integer or iterable",
        TypeError,
    ),
    "shape-component": _case(
        lambda x: x.__array_namespace__().reshape(x, (True, 6)),
        ((6,),),
        "Shape must contain integers",
        TypeError,
    ),
    "fft-length-type": _case(
        lambda x: x.__array_namespace__().fft.fft(x, n="4"),
        ((4,),),
        "length must be an integer or None",
        TypeError,
    ),
    "fft-length-value": _case(
        lambda x: x.__array_namespace__().fft.fft(x, n=0), ((4,),), "positive integer"
    ),
    "fft-dtype": _case(
        lambda x: x.__array_namespace__().fft.fft(x),
        ((4,),),
        "FFT input must be floating-point or complex",
        TypeError,
        ("int32",),
    ),
    "fftn-empty-axes": _case(
        lambda x: x.__array_namespace__().fft.fftn(x, axes=()), ((2, 3),), "axes must be non-empty"
    ),
    "fftn-size-count": _case(
        lambda x: x.__array_namespace__().fft.fftn(x, s=(2,), axes=(0, 1)),
        ((2, 3),),
        "sizes and axes must have equal length",
    ),
    "reshape-unknown-count": _case(
        lambda x: x.__array_namespace__().reshape(x, (-1, -1)), ((6,),), "Invalid reshape target"
    ),
    "reshape-zero-known-size": _case(
        lambda x: x.__array_namespace__().reshape(x, (0, -1)),
        ((6,),),
        "reshape changes element count",
    ),
    "reshape-element-count": _case(
        lambda x: x.__array_namespace__().reshape(x, (4, 2)),
        ((6,),),
        "reshape changes element count",
    ),
    "moveaxis-axis-count": _case(
        lambda x: x.__array_namespace__().moveaxis(x, (0, 1), (2,)),
        ((2, 3, 4),),
        "source and destination must have equal length",
    ),
    "tensordot-bool": _case(
        lambda left, right: left.__array_namespace__().linalg.tensordot(left, right, axes=True),
        ((2, 3), (3, 2)),
        "axes must be an integer or a pair",
        TypeError,
    ),
    "tensordot-axis-count": _case(
        lambda left, right: left.__array_namespace__().linalg.tensordot(left, right, axes=3),
        ((2, 3), (3, 2)),
        "Invalid tensordot axes count",
    ),
    "tensordot-axis-type": _case(
        lambda left, right: left.__array_namespace__().linalg.tensordot(left, right, axes=1.5),
        ((2, 3), (3, 2)),
        "axes must be an integer or a pair",
        TypeError,
    ),
    "tensordot-axis-pair": _case(
        lambda left, right: left.__array_namespace__().linalg.tensordot(left, right, axes=((0,),)),
        ((2, 3), (3, 2)),
        "must contain two axis sequences",
    ),
    "tensordot-axis-list-count": _case(
        lambda left, right: left.__array_namespace__().linalg.tensordot(
            left, right, axes=((0, 1), (0,))
        ),
        ((2, 3), (2, 3)),
        "axis lists must have equal length",
    ),
    "tensordot-core": _case(
        lambda left, right: left.__array_namespace__().linalg.tensordot(
            left, right, axes=((1,), (0,))
        ),
        ((2, 3), (4, 2)),
        "contraction dimensions disagree",
    ),
    "arange-start": _case(
        lambda x: x.__array_namespace__().arange(bool(1), 3, dtype=x.dtype),
        ((1,),),
        "concrete real scalars",
        TypeError,
    ),
    "arange-stop": _case(
        lambda x: x.__array_namespace__().arange(0, object(), dtype=x.dtype),
        ((1,),),
        "concrete real scalars",
        TypeError,
    ),
    "arange-step-type": _case(
        lambda x: x.__array_namespace__().arange(0, 3, object(), dtype=x.dtype),
        ((1,),),
        "concrete real scalars",
        TypeError,
    ),
    "arange-step-zero": _case(
        lambda x: x.__array_namespace__().arange(0, 3, 0, dtype=x.dtype),
        ((1,),),
        "step must be nonzero",
    ),
    "namespace-version": _case(
        lambda x: x.__array_namespace__(api_version="2023.12").sum(x), ((2,),), "requested.*targets"
    ),
    "len-data-dependence": _case(
        lambda x: len(x), ((2,),), "len\\(\\).*not allowed", ad.TracingError
    ),
    "item-size": _case(lambda x: x.item(), ((2,),), "array of size 1"),
    "concrete-operand-type": _case(
        lambda x: x + object(), ((2,),), "Cannot stage concrete operand", TypeError
    ),
    "asarray-missing-input": _case(
        lambda x: x.__array_namespace__().asarray(),
        ((2,),),
        "asarray.*requires an input",
        TypeError,
    ),
    "cumulative-missing-input": _case(
        lambda x: x.__array_namespace__().cumulative_sum(include_initial=True),
        ((2,),),
        "expects an array and optional axis",
        TypeError,
    ),
    "diff-missing-input": _case(
        lambda x: x.__array_namespace__().diff(), ((2,),), "diff.*expects", TypeError
    ),
    "searchsorted-missing-query": _case(
        lambda x: x.__array_namespace__().searchsorted(x, sorter=x),
        ((2,),),
        "expects two positional array arguments",
        TypeError,
    ),
}


@pytest.mark.parametrize(
    ("operation", "specs", "error", "match"),
    _INVALID_STAGING_CASES.values(),
    ids=_INVALID_STAGING_CASES,
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
