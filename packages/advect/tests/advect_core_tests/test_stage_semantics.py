# ruff: noqa: PLW0108
"""Focused contracts for conservative abstract staging semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import array_api_strict as strict
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

import advect as ad
from advect.core._abstract_helpers import broadcast_shape

if TYPE_CHECKING:
    from collections.abc import Callable

_SHAPES = st.lists(
    st.integers(min_value=0, max_value=4),
    min_size=0,
    max_size=4,
).map(tuple)
# array_api_strict inherits NumPy's leniency for axis=0/-1 on rank-0 arrays,
# which the Array API does not define, so axis-taking calls draw ranked shapes.
_RANKED = st.lists(st.integers(min_value=0, max_value=3), min_size=1, max_size=4).map(tuple)
_AXIS = st.integers(min_value=-5, max_value=4)
_AXES = st.one_of(_AXIS, st.lists(_AXIS, max_size=3).map(tuple))
_EXTENTS = st.lists(st.integers(min_value=-1, max_value=6), max_size=4).map(tuple)
_NO_ARGS: st.SearchStrategy[tuple[object, ...]] = st.just(())


@given(st.lists(_SHAPES, min_size=1, max_size=4))
def test_abstract_broadcast_shape_matches_numpy(shapes: list[tuple[int, ...]]) -> None:
    try:
        expected = np.broadcast_shapes(*shapes)
    except ValueError:
        with pytest.raises(ValueError, match="Shapes are not broadcast-compatible"):
            broadcast_shape(*shapes)
    else:
        assert broadcast_shape(*shapes) == expected


def _call(
    name: str,
    *shapes: st.SearchStrategy[tuple[int, ...]],
    args: st.SearchStrategy[tuple[object, ...]] = _NO_ARGS,
    sequence: bool = False,
    **kwargs: st.SearchStrategy[object],
) -> st.SearchStrategy[
    tuple[str, tuple[tuple[int, ...], ...], tuple[object, ...], dict[str, object], bool]
]:
    return st.tuples(
        st.just(name), st.tuples(*shapes), args, st.fixed_dictionaries(kwargs), st.just(sequence)
    )


_SHAPE_CALLS = st.one_of(
    _call("matmul", _SHAPES, _SHAPES),
    _call("tensordot", _SHAPES, _SHAPES, axes=st.integers(0, 3)),
    _call("reshape", _SHAPES, shape=_EXTENTS),
    _call("broadcast_to", _SHAPES, shape=_SHAPES),
    _call("expand_dims", _SHAPES, axis=_AXES),
    _call("squeeze", _RANKED, axis=_AXES),
    _call("permute_dims", _SHAPES, axes=st.lists(st.integers(-4, 4), max_size=4).map(tuple)),
    _call("moveaxis", _RANKED, args=st.tuples(_AXES, _AXES)),
    _call("tile", _SHAPES, args=st.tuples(st.lists(st.integers(0, 3), max_size=4).map(tuple))),
    _call("repeat", _RANKED, args=st.tuples(st.integers(0, 2)), axis=st.none() | _AXIS),
    _call("concat", _SHAPES, _SHAPES, sequence=True, axis=st.none() | _AXIS),
    _call("stack", _SHAPES, _SHAPES, sequence=True, axis=_AXIS),
    _call("sum", _RANKED, axis=st.none() | _AXES, keepdims=st.booleans()),
)


def _output_shapes(
    function: Any, shapes: tuple[tuple[int, ...], ...], *, staged: bool
) -> list[tuple[int, ...]] | str:
    try:
        if not staged:
            return [
                function(*(strict.zeros(shape, dtype=strict.float32) for shape in shapes)).shape
            ]
        specs = tuple(ad.ArraySpec(shape, "float32") for shape in shapes)
        program = cast("ad.StagedProgram", ad.stage(function, specs=specs))
    except (TypeError, ValueError, IndexError):
        return "rejected"
    return [tuple(spec["shape"]) for spec in program.to_dict()["program"]["output_specs"]]


@given(_SHAPE_CALLS)
@settings(deadline=None)
def test_abstract_shapes_match_array_api_strict(
    call: tuple[str, tuple[tuple[int, ...], ...], tuple[object, ...], dict[str, object], bool],
) -> None:
    """Staging accepts exactly the calls the reference provider accepts, with its shapes."""
    name, shapes, args, kwargs, sequence = call

    def function(*values: Any) -> Any:
        operands = (values,) if sequence else values
        return getattr(values[0].__array_namespace__(), name)(*operands, *args, **kwargs)

    assert _output_shapes(function, shapes, staged=True) == _output_shapes(
        function, shapes, staged=False
    )


def _case(
    operation: Callable[..., object],
    shapes: tuple[tuple[int, ...], ...],
    match: str,
    error: type[Exception] = ValueError,
    dtypes: tuple[str, ...] = (),
) -> tuple[Callable[..., object], tuple[ad.ArraySpec, ...], type[Exception], str]:
    resolved_dtypes = dtypes or ("float32",) * len(shapes)
    specs = tuple(
        ad.ArraySpec(shape, dtype) for shape, dtype in zip(shapes, resolved_dtypes, strict=True)
    )
    return operation, specs, error, match


def _add_to_input(value: Any) -> Any:
    value += 1
    return value


def _assign_to_input(value: Any) -> Any:
    value[0] = 1
    return value


def _add_to_input_view(value: Any) -> Any:
    value[0] += 1
    return value


def _assign_through_copied_view(value: Any) -> Any:
    result = value.copy()
    view = result[1:]
    view[0] = 2
    return result


def _add_through_reshaped_view(value: Any) -> Any:
    result = value.copy()
    view = result.reshape((4,))
    view += 1
    return result


def _add_complex_to_row(value: Any) -> Any:
    result = value.copy()
    result[0] += 1j
    return result


def _add_complex(value: Any) -> Any:
    result = value.copy()
    result += 1j
    return result


def _assign_wrong_shape(value: Any) -> Any:
    result = value.copy()
    result[0] = value
    return result


_INVALID_STAGING_CASES = {
    # linear algebra
    "norm-rank": _case(lambda x: np.linalg.norm(x, ord=2), ((2, 3, 4),), "requires axis="),
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
    "matmul-scalar": _case(lambda left, right: left @ right, ((), (2,)), "at least one dimension"),
    "matmul-core": _case(
        lambda left, right: left @ right, ((2, 3), (4, 2)), "core dimensions disagree"
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
    # shapes, axes and sequences
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
    "concatenate-empty": _case(lambda _x: np.concatenate(()), ((1,),), "need at least one array"),
    "stack-empty": _case(lambda _x: np.stack(()), ((1,),), "need at least one array"),
    "concat-empty": _case(
        lambda x: x.__array_namespace__().concat(()),
        ((1,),),
        "requires a non-empty list or tuple of arrays",
    ),
    "array-api-stack-empty": _case(
        lambda x: x.__array_namespace__().stack(()),
        ((1,),),
        "requires a non-empty list or tuple of arrays",
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
    # creation
    "eye-dimensions": _case(
        lambda x: x.__array_namespace__().eye(-1, dtype=x.dtype),
        ((1,),),
        "dimensions must be non-negative integers",
    ),
    "creation-default-dtype": _case(
        lambda x: x.__array_namespace__().zeros((2,), dtype=None),
        ((1,),),
        r"requires \('dtype',\)",
        TypeError,
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
    "asarray-missing-input": _case(
        lambda x: x.__array_namespace__().asarray(),
        ((2,),),
        "asarray.*requires an input",
        TypeError,
    ),
    "asarray-sequence-no-copy": _case(
        lambda x: x.__array_namespace__().asarray([1, 2], copy=False),
        ((2,),),
        r"copy=False.*sequence",
    ),
    "asarray-float-no-copy": _case(
        lambda x: x.__array_namespace__().asarray(1.0, copy=False),
        ((2,),),
        r"copy=False.*Python scalar",
    ),
    "asarray-int-no-copy": _case(
        lambda x: x.__array_namespace__().asarray(1, copy=False),
        ((2,),),
        r"copy=False.*Python scalar",
    ),
    "asarray-bool-no-copy": _case(
        lambda x: x.__array_namespace__().asarray(True, copy=False),  # noqa: FBT003
        ((2,),),
        r"copy=False.*Python scalar",
    ),
    "asarray-invalid-copy": _case(
        lambda x: x.__array_namespace__().asarray(x, copy="yes"),
        ((2,),),
        "copy must be a bool or None",
        TypeError,
    ),
    "asarray-ragged-sequence": _case(
        lambda x: x.__array_namespace__().asarray([[1, 2], [3]]),
        ((2,),),
        "rectangular nested sequence",
    ),
    "asarray-unknown-option": _case(
        lambda x: x.__array_namespace__().asarray(x, order="C"),
        ((2,),),
        "supports only asarray",
        TypeError,
    ),
    # scans and differences
    "cumulative-axis": _case(
        lambda x: x.__array_namespace__().cumulative_sum(x), ((2, 3),), "require axis="
    ),
    "cumulative-include-initial-axis": _case(
        lambda x: x.__array_namespace__().cumulative_sum(x, include_initial=True),
        ((2, 3),),
        "require axis=",
    ),
    "cumulative-positional-axis": _case(
        lambda x: x.__array_namespace__().cumulative_prod(x, 1, axis=0, include_initial=True),
        ((2, 3),),
        "expects one positional array argument",
        TypeError,
    ),
    "cumulative-missing-input": _case(
        lambda x: x.__array_namespace__().cumulative_sum(include_initial=True),
        ((2,),),
        "expects one positional array argument",
        TypeError,
    ),
    "diff-missing-input": _case(
        lambda x: x.__array_namespace__().diff(),
        ((2,),),
        "diff.*missing required argument 'x'",
        TypeError,
    ),
    "diff-positional-n": _case(
        lambda x: x.__array_namespace__().diff(x, 1, n=2),
        ((2, 3),),
        "takes 1 positional arguments but 2 were given",
        TypeError,
    ),
    "diff-unknown-option": _case(
        lambda x: x.__array_namespace__().diff(x, period=2),
        ((2, 3),),
        "does not support.*period",
        TypeError,
    ),
    "diff-boolean-n": _case(
        lambda x: x.__array_namespace__().diff(x, n=True), ((2, 3),), "non-negative integer"
    ),
    "diff-invalid-axis": _case(
        lambda x: x.__array_namespace__().diff(x, axis=2), ((2, 3),), "out of bounds"
    ),
    "searchsorted-missing-query": _case(
        lambda x: x.__array_namespace__().searchsorted(x, sorter=x),
        ((2,),),
        "expects two positional array arguments",
        TypeError,
    ),
    # FFT
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
    "fft-length-type": _case(
        lambda x: x.__array_namespace__().fft.fft(x, n="4"),
        ((4,),),
        "length must be an integer or None",
        TypeError,
    ),
    "fft-length-value": _case(
        lambda x: x.__array_namespace__().fft.fft(x, n=0), ((4,),), "positive integer"
    ),
    "fftn-empty-axes": _case(
        lambda x: x.__array_namespace__().fft.fftn(x, axes=()), ((2, 3),), "axes must be non-empty"
    ),
    "fftn-size-count": _case(
        lambda x: x.__array_namespace__().fft.fftn(x, s=(2,), axes=(0, 1)),
        ((2, 3),),
        "sizes and axes must have equal length",
    ),
    # dtype metadata
    "result-type-empty": _case(
        lambda x: x.__array_namespace__().result_type(),
        ((2,),),
        "requires at least one argument",
        TypeError,
    ),
    "isdtype-invalid": _case(
        lambda x: x.__array_namespace__().isdtype("object", "numeric"),
        ((2,),),
        "Unsupported staged dtype",
        TypeError,
    ),
    "can-cast-invalid": _case(
        lambda x: x.__array_namespace__().can_cast("object", x.__array_namespace__().float32),
        ((2,),),
        "Unsupported dtype pair",
        TypeError,
    ),
    "finfo-int": _case(
        lambda x: x.__array_namespace__().finfo(x.__array_namespace__().int32),
        ((2,),),
        "requires a floating-point dtype",
        TypeError,
    ),
    "iinfo-float": _case(
        lambda x: x.__array_namespace__().iinfo(x.__array_namespace__().float32),
        ((2,),),
        "requires an integer dtype",
        TypeError,
    ),
    # data dependence, operands and revisions
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
    "unknown-operation": _case(
        lambda x: np.partition(x, 1), ((3,),), "no abstract staging rule", NotImplementedError
    ),
    # mutation
    "input-in-place": _case(_add_to_input, ((2,),), "staged input", ad.MutationError),
    "input-assignment": _case(_assign_to_input, ((2,),), "staged input", ad.MutationError),
    "input-out": _case(lambda x: np.add(x, 1, out=x), ((2,),), "staged input", ad.MutationError),
    "input-view": _case(_add_to_input_view, ((2, 2),), "staged input", ad.MutationError),
    "copied-view": _case(
        _assign_through_copied_view, ((2, 2),), "through a staged view", ad.MutationError
    ),
    "reshaped-view": _case(
        _add_through_reshaped_view,
        ((2, 2),),
        "nested or reshaped staged view",
        ad.MutationError,
    ),
    "indexed-dtype": _case(
        _add_complex_to_row, ((2, 2),), "would change shape or dtype", ad.MutationError
    ),
    "array-dtype": _case(_add_complex, ((2, 2),), "would change shape or dtype", ad.MutationError),
    "assignment-shape": _case(_assign_wrong_shape, ((2, 2),), "Cannot assign shape"),
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


@pytest.mark.parametrize(
    ("transform", "dtype"),
    [
        (np.fft.fft2, "complex128"),
        (np.fft.ifft2, "complex128"),
        (np.fft.rfft2, "float64"),
        (np.fft.irfft2, "complex128"),
    ],
    ids=["fft2", "ifft2", "rfft2", "irfft2"],
)
def test_staged_two_dimensional_fft_sizes_apply_to_the_last_two_axes(
    transform: Callable[..., np.ndarray],
    dtype: str,
) -> None:
    def operation(value: object) -> object:
        return transform(value, s=(3, 6))

    rng = np.random.default_rng(0)
    value = rng.normal(size=(2, 3, 4)).astype(dtype)
    if dtype.startswith("complex"):
        value += 1j * rng.normal(size=value.shape)
    expected, dynamic_pullback = ad.vjp(operation)(value)
    cotangent = np.conj(expected) + 1.0
    program = ad.stage(operation, specs=(ad.ArraySpec(value.shape, dtype),))

    for staged in (program, ad.StagedProgram.from_dict(program.to_dict())):
        np.testing.assert_allclose(staged(value), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        ad.vjp_program(program)(value, cotangent=cotangent),
        dynamic_pullback(cotangent),
        rtol=1e-12,
        atol=1e-12,
    )


def _graph_nodes(program: ad.StagedProgram) -> list[Any]:
    graph = program.graph
    return [graph.get_node(node_id) for node_id in graph.node_ids()]


class _FutureNamespace:
    __name__ = "future_array"
    __array_api_version__ = "2099.12"

    @staticmethod
    def __array_namespace_info__() -> object:
        return object()

    @staticmethod
    def asarray(value: object) -> np.ndarray[Any, Any]:
        return np.asarray(value)


class _FutureArray:
    shape = (1,)
    dtype = np.dtype("float64")

    def __array_namespace__(self) -> _FutureNamespace:
        return _FutureNamespace()


class _UnversionedNamespace:
    __name__ = "unversioned_array"


class _UnversionedArray:
    shape = (1,)
    dtype = np.dtype("float64")

    def __array_namespace__(self) -> _UnversionedNamespace:
        return _UnversionedNamespace()


class _PinnedNamespace:
    __name__ = "multi_version_array"
    __array_api_version__ = "2024.12"

    @staticmethod
    def __array_namespace_info__() -> object:
        return object()

    @staticmethod
    def asarray(value: object) -> np.ndarray[Any, Any]:
        return np.asarray(value)


class _DefaultFutureNamespace(_PinnedNamespace):
    __array_api_version__ = "2099.12"


class _MultiVersionArray:
    shape = (1,)
    dtype = np.dtype("float64")

    def __init__(self) -> None:
        self.requests: list[str | None] = []

    def __array_namespace__(self, *, api_version: str | None = None) -> object:
        self.requests.append(api_version)
        return _PinnedNamespace() if api_version == "2024.12" else _DefaultFutureNamespace()


def _contains_python_index(value: object) -> bool:
    if isinstance(value, (slice, type(Ellipsis))):
        return True
    if isinstance(value, Mapping):
        return any(_contains_python_index(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_python_index(item) for item in value)
    return False


def test_expand_dims_and_squeeze_have_explicit_shapes_and_canonical_ids() -> None:
    def transform(x: Any) -> Any:
        xp = x.__array_namespace__()
        return xp.squeeze(xp.expand_dims(x, axis=(0, 2)), axis=(0, 2))

    program = cast(
        "ad.StagedProgram",
        ad.stage(transform, specs=(ad.ArraySpec((2, 3), "float32"),)),
    )
    operation_nodes = [node for node in _graph_nodes(program) if node.op.startswith("array.")]

    assert [(node.op, tuple(node.shape)) for node in operation_nodes] == [
        ("array.expand_dims", (1, 2, 1, 3)),
        ("array.squeeze", (2, 3)),
    ]
    value = strict.reshape(strict.arange(6, dtype=strict.float32), (2, 3))
    result = ad.StagedProgram.from_dict(program.to_dict())(value)
    assert result.shape == (2, 3)


def test_basic_index_is_structural_serializable_and_round_trips() -> None:
    program = cast(
        "ad.StagedProgram",
        ad.stage(
            lambda x: x[1:, None, ..., ::-2],
            specs=(ad.ArraySpec((3, 4, 5), "float32"),),
        ),
    )
    index_node = next(node for node in _graph_nodes(program) if node.op == "advect.getitem")

    assert tuple(index_node.shape) == (2, 1, 4, 3)
    assert not _contains_python_index(index_node.attrs["index"])
    value = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    restored = ad.StagedProgram.from_dict(program.to_dict())
    np.testing.assert_array_equal(restored(value), value[1:, None, ..., ::-2])


def test_staged_calls_validate_device_and_weak_scalar_contracts() -> None:
    device_program = cast(
        "ad.StagedProgram",
        ad.stage(
            lambda x: x,
            specs=(ad.ArraySpec((1,), "float32", device="cuda:0"),),
        ),
    )
    with pytest.raises(ValueError, match=r"device=cuda:0.*device=cpu"):
        device_program(np.ones(1, dtype=np.float32))

    weak_program = cast(
        "ad.StagedProgram",
        ad.stage(
            lambda x: x,
            specs=(ad.ArraySpec((), "float64", weak=True),),
        ),
    )
    with pytest.raises(ValueError, match=r"weak=True.*weak=False"):
        weak_program(np.asarray(1.0, dtype=np.float64))
    assert weak_program(2.0) == 2.0


def test_example_staged_program_accepts_computed_values_in_an_enclosing_stage() -> None:
    inner = cast("ad.StagedProgram", ad.stage(lambda x: 2.0 * x, np.ones(3)))
    assert inner.signature[0][0].device == "cpu"
    value = np.array([0.5, 1.0, 2.0])

    # Computed abstract values carry no device; the outer input check owns it.
    outer = cast("ad.StagedProgram", ad.stage(lambda x: inner(np.cos(x)), value))
    np.testing.assert_allclose(outer(value), 2.0 * np.cos(value))

    known = cast(
        "ad.StagedProgram",
        ad.stage(lambda x: x, specs=(ad.ArraySpec((3,), "float64", device="cuda:0"),)),
    )
    with pytest.raises(ValueError, match=r"device=cuda:0.*device=cpu"):
        ad.stage(lambda x: known(x), value)  # an explicit trace boundary


def test_nested_staged_scalar_preserves_weak_metadata() -> None:
    scalar_spec = ad.ArraySpec((), "float64", weak=True)
    inner = ad.stage(lambda value: value + 1, specs=(scalar_spec,))
    outer = ad.stage(lambda value: 2 * inner(value), specs=(scalar_spec,))

    assert outer.signature[0] == (scalar_spec,)
    assert outer(3.0) == 8.0
    assert ad.StagedProgram.from_dict(outer.to_dict())(3.0) == 8.0


def test_staged_stencil_functionalizes_basic_updates_and_round_trips() -> None:
    def step(u: Any, dt: float) -> Any:
        u = u.copy()
        lap = u[2:] - 2 * u[1:-1] + u[:-2]
        u[1:-1] += dt * lap
        u[0] = -1
        return u

    program = cast(
        "ad.StagedProgram",
        ad.stage(
            step,
            specs=(ad.ArraySpec((6,), "float32"), ad.StaticSpec(0.1)),
        ),
    )
    update_nodes = [node for node in _graph_nodes(program) if node.op == "advect.index_update"]
    assert len(update_nodes) == 2

    value = np.arange(6, dtype=np.float32) ** 2
    expected = value.copy()
    lap = value[2:] - 2 * value[1:-1] + value[:-2]
    expected[1:-1] += np.float32(0.1) * lap
    expected[0] = -1
    restored = ad.StagedProgram.from_dict(program.to_dict())
    np.testing.assert_allclose(restored(value, 0.1), expected)
    np.testing.assert_allclose(
        ad.grad(lambda u: np.sum(restored(u, 0.1)))(value),
        np.array([0.1, 0.9, 1.0, 1.0, 0.9, 1.1], dtype=np.float32),
    )


def test_staged_ufunc_out_preserves_identity_where_and_round_trips() -> None:
    identities: list[bool] = []

    def update(x: Any) -> Any:
        destination = x.copy()
        result = np.add(x, 1, out=destination, where=x > 0)
        identities.append(result is destination)
        return destination

    program = cast(
        "ad.StagedProgram",
        ad.stage(update, specs=(ad.ArraySpec((3,), "float32"),)),
    )
    assert identities == [True]
    assert any(node.op == "array.where" for node in _graph_nodes(program))

    value = np.array([-1.0, 0.0, 2.0], dtype=np.float32)
    restored = ad.StagedProgram.from_dict(program.to_dict())
    np.testing.assert_array_equal(restored(value), np.array([-1.0, 0.0, 3.0]))


def test_staged_views_detect_stale_and_support_named_basic_view_mutation() -> None:
    def stale(x: Any) -> Any:
        x = x.copy()
        view = x[::2]
        x += 1
        return view + 1

    with pytest.raises(ad.StaleViewError, match="used after its base changed"):
        ad.stage(stale, specs=(ad.ArraySpec((4,), "float32"),))

    def named(x: Any) -> Any:
        x = x.copy()
        view = x[::2]
        view += 1
        view *= 2
        return x

    program = cast(
        "ad.StagedProgram",
        ad.stage(named, specs=(ad.ArraySpec((4,), "float32"),)),
    )
    np.testing.assert_array_equal(
        program(np.arange(4, dtype=np.float32)),
        np.array([2, 1, 6, 3], dtype=np.float32),
    )


@pytest.mark.parametrize(
    "key",
    [
        0,
        (slice(None), 1),
        Ellipsis,
        (None, Ellipsis),
        slice(None, None, 2),
    ],
    ids=("integer", "tuple", "ellipsis", "newaxis", "step-slice"),
)
def test_staged_named_view_update_supports_each_basic_index_form(key: object) -> None:
    def update(x: Any) -> Any:
        current = x.copy()
        view = current[key]
        view += 2
        return current

    program = cast(
        "ad.StagedProgram",
        ad.stage(update, specs=(ad.ArraySpec((3, 4), "float32"),)),
    )
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    expected = source.copy()
    expected[key] += 2

    np.testing.assert_array_equal(program(source), expected)


def test_staged_runtime_enforces_pinned_array_api_version() -> None:
    program = cast(
        "ad.StagedProgram",
        ad.stage(lambda x: x, specs=(ad.ArraySpec((1,), "float64"),)),
    )

    with pytest.raises(TypeError, match=r"required Array API 2024\.12"):
        program(_FutureArray())
    with pytest.raises(TypeError, match=r"required Array API 2024\.12"):
        program(_UnversionedArray())


def test_staged_runtime_validates_every_provider_and_negotiates_the_pin() -> None:
    identity = cast(
        "ad.StagedProgram",
        ad.stage(lambda x: x, specs=(ad.ArraySpec((1,), "float64"),)),
    )
    negotiated = _MultiVersionArray()
    assert identity(negotiated) is negotiated
    assert negotiated.requests == ["2024.12"]

    first_of_two = cast(
        "ad.StagedProgram",
        ad.stage(
            lambda x, _y: x,
            specs=(
                ad.ArraySpec((1,), "float64"),
                ad.ArraySpec((1,), "float64"),
            ),
        ),
    )
    with pytest.raises(TypeError, match=r"required Array API 2024\.12"):
        first_of_two(np.ones(1), _FutureArray())


def test_staged_numpy_uses_its_separate_frontend_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(np, "__array_api_version__", "2099.12")
    program = cast(
        "ad.StagedProgram",
        ad.stage(lambda x: x + 1, specs=(ad.ArraySpec((2,), "float64"),)),
    )

    np.testing.assert_array_equal(program(np.arange(2.0)), np.array([1.0, 2.0]))
