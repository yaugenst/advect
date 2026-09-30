"""End-to-end contracts for the backend-neutral Python Array API frontend."""

from __future__ import annotations

import operator
import re
from contextlib import contextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import array_api_strict as xp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from numpy.testing import assert_allclose

import advect as ad
from advect.autodiff import _ephemeral
from advect.autodiff._ephemeral import trace_call
from advect.core._array_api import providers, signatures
from advect.core._array_api.frontend import (
    _ARITHMETIC_OPERATORS,
    _COMPARISON_OPERATORS,
    _FUNCTION_SPECS,
    ArrayAPINamespace,
    _accepts_array_api,
    bind_array_api_call,
)
from advect.core._array_api.profiles import LATEST_ARRAY_API_VERSION, SUPPORTED_ARRAY_API_VERSIONS
from advect.core._errors import MutationError, TracingError

if TYPE_CHECKING:
    from collections.abc import Iterator


def _strict(values: Any, *, dtype: Any = None) -> Any:
    return xp.asarray(values) if dtype is None else xp.asarray(values, dtype=dtype)


@contextmanager
def _traced(function: Any, *values: Any) -> Iterator[Any]:
    """Trace ``function`` on ``values`` and release the tape's payloads afterwards."""
    traced = trace_call(
        function,
        args=values,
        kwargs={},
        argnums=tuple(range(len(values))),
        argnames=None,
    )
    try:
        yield traced
    finally:
        traced.tape.release_payloads()


def _trace(function: Any, *values: Any) -> Any:
    with _traced(function, *values) as traced:
        return traced.output


def test_call_binding_normalizes_standard_and_optional_live_parameters() -> None:
    source = object()
    indices = object()
    maximum = object()

    taken = bind_array_api_call("take", (source, indices), {"axis": 0})
    clipped = bind_array_api_call("clip", (source,), {"min": None, "max": maximum})
    pinv = bind_array_api_call("linalg.pinv", (source,), {"rtol": None})

    assert taken.operands == (source, indices)
    assert taken.attrs == {"axis": 0}
    assert clipped.operands == (source, maximum)
    assert clipped.attrs == {
        "_advect_clip_min_is_input": False,
        "_advect_clip_max_is_input": True,
    }
    assert pinv.operands == (source,)
    assert pinv.attrs == {"_advect_pinv_tolerance": None}


def test_grad_uses_runtime_array_namespace_and_preserves_backend() -> None:
    x = _strict([0.2, -0.3, 0.5], dtype=xp.float32)

    def objective(value: Any) -> Any:
        namespace = value.__array_namespace__()
        return namespace.sum(namespace.sin(value) * value)

    gradient = ad.grad(objective)(x)
    expected = xp.sin(x) + x * xp.cos(x)

    assert type(gradient) is type(x)
    assert gradient.dtype == xp.float32
    assert_allclose(np.asarray(gradient), np.asarray(expected), rtol=1e-6, atol=1e-6)


def test_nested_dynamic_transforms_keep_array_api_backend() -> None:
    x = _strict([1.0, 2.0], dtype=xp.float32)
    vector = _strict([0.5, -0.25], dtype=xp.float32)

    def cubic(value: Any) -> Any:
        namespace = value.__array_namespace__()
        return namespace.sum(value * value * value)

    value, product = ad.hvp(cubic)(x, vectors=vector)

    assert type(value) is type(x)
    assert type(product) is type(x)
    assert product.dtype == xp.float32
    assert_allclose(np.asarray(value), np.asarray(xp.asarray(9.0, dtype=xp.float32)))
    assert_allclose(
        np.asarray(product),
        np.asarray(_strict([3.0, -3.0], dtype=xp.float32)),
    )


def test_asarray_constructs_nested_live_tracer_sequences() -> None:
    value = _strict([1.0, 2.0], dtype=xp.float32)

    def matrix_sum(argument: Any) -> Any:
        namespace = argument.__array_namespace__()
        matrix = namespace.asarray(
            [[argument[0], argument[1]], [argument[1], 2 * argument[0]]],
            dtype=argument.dtype,
        )
        return namespace.sum(matrix)

    assert_allclose(
        np.asarray(ad.grad(matrix_sum)(value)),
        np.asarray([3.0, 2.0], dtype=np.float32),
    )

    program = ad.stage(
        matrix_sum,
        specs=(ad.ArraySpec(value.shape, value.dtype),),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())
    for staged in (program, restored):
        assert_allclose(np.asarray(staged(value)), 7.0)


def test_nested_grad_retains_captured_outer_array_api_tracer() -> None:
    x = _strict([1.0, -2.0, 3.0], dtype=xp.float32)
    ones = xp.ones_like(x)

    def objective(outer: Any) -> Any:
        def inner_loss(inner: Any) -> Any:
            namespace = inner.__array_namespace__()
            return namespace.sum(outer * inner)

        gradient = ad.grad(inner_loss)(ones)
        return gradient.__array_namespace__().sum(gradient)

    assert_allclose(np.asarray(ad.grad(objective)(x)), np.asarray(ones))


@pytest.mark.parametrize(
    "expression",
    [
        lambda _namespace, outer, inner: outer * inner * inner,
        lambda namespace, outer, inner: namespace.multiply(
            namespace.multiply(inner, inner),
            outer,
        ),
    ],
    ids=["scalar-operator", "namespace-function"],
)
def test_nested_array_api_grad_retains_captured_outer_rank_zero_array(
    expression: Any,
) -> None:
    x = _strict([1.0, 2.0], dtype=xp.float32)

    def objective(outer: Any) -> Any:
        def inner_loss(inner: Any) -> Any:
            namespace = inner.__array_namespace__()
            return namespace.sum(expression(namespace, outer, inner))

        gradient = ad.grad(inner_loss)(x)
        return gradient.__array_namespace__().sum(gradient)

    derivative = ad.grad(objective)(_strict(2.0, dtype=xp.float32))

    assert derivative.dtype == xp.float32
    assert_allclose(np.asarray(derivative), np.asarray(_strict(6.0, dtype=xp.float32)))


def test_complex_real_loss_uses_descent_ready_real_adjoint() -> None:
    x = _strict([1.0 + 2.0j, -3.0 + 0.5j], dtype=xp.complex64)

    def squared_norm(value: Any) -> Any:
        namespace = value.__array_namespace__()
        return namespace.sum(namespace.real(namespace.conj(value) * value))

    gradient = ad.grad(squared_norm)(x)

    assert type(gradient) is type(x)
    assert gradient.dtype == xp.complex64
    assert_allclose(np.asarray(gradient), np.asarray(2.0 * x), rtol=1e-6, atol=1e-6)


def test_strict_where_cast_and_shape_transposes() -> None:
    x = _strict([1.0, 2.0, 3.0, 4.0], dtype=xp.float32)
    vector = _strict([0.5, -0.25, 1.0, -2.0], dtype=xp.float32)
    condition = _strict([[True, False], [False, True]], dtype=xp.bool)

    def objective(value: Any) -> Any:
        namespace = value.__array_namespace__()
        matrix = namespace.reshape(value, (2, 2))
        matrix = namespace.permute_dims(matrix, (1, 0))
        matrix = namespace.expand_dims(matrix, axis=0)
        matrix = namespace.squeeze(matrix, axis=0)
        selected = namespace.where(condition, matrix, -matrix)
        widened = namespace.astype(selected, xp.float64)
        return namespace.sum(widened)

    gradient = ad.grad(objective)(x)

    assert type(gradient) is type(x)
    assert gradient.dtype == xp.float32
    assert_allclose(
        np.asarray(gradient),
        np.asarray(_strict([1.0, -1.0, -1.0, 1.0], dtype=xp.float32)),
    )

    def quadratic(value: Any) -> Any:
        namespace = value.__array_namespace__()
        matrix = namespace.permute_dims(namespace.reshape(value, (2, 2)), (1, 0))
        selected = namespace.where(condition, matrix, -matrix)
        widened = namespace.astype(selected, xp.float64)
        return namespace.sum(widened * widened)

    _, product = ad.hvp(quadratic)(x, vectors=vector)
    assert type(product) is type(x)
    assert product.dtype == xp.float32
    assert_allclose(np.asarray(product), np.asarray(2.0 * vector))


def test_expand_dims_accepts_the_official_positional_axis_contract() -> None:
    value = _strict([1.0, 2.0], dtype=xp.float32)

    def objective(argument: Any) -> Any:
        namespace = argument.__array_namespace__()
        return namespace.sum(namespace.expand_dims(argument, 0))

    expected = xp.ones_like(value)
    assert_allclose(np.asarray(ad.grad(objective)(value)), np.asarray(expected))

    program = ad.stage(
        objective,
        specs=(ad.ArraySpec(value.shape, value.dtype),),
    )
    assert_allclose(np.asarray(ad.grad(program)(value)), np.asarray(expected))


def test_python_operators_record_canonical_nodes_with_weak_scalars() -> None:
    x = _strict([1.0, 2.0], dtype=xp.float32)

    with _traced(lambda value: (2.0 * value + 1.0) / 3.0, x) as traced:
        assert type(traced.output) is type(x)
        assert traced.output.dtype == xp.float32
        assert traced.tape.op_names == [
            "advect.input",
            "array.multiply",
            "array.add",
            "array.divide",
        ]
        assert traced.tape.stats()["literal_count"] == 3


def test_static_array_api_arguments_are_node_attributes() -> None:
    x = _strict([[1.0, 2.0], [3.0, 4.0]], dtype=xp.float64)

    def reduce_rows(value: Any) -> Any:
        namespace = value.__array_namespace__()
        return namespace.sum(value, axis=1, keepdims=True)

    with _traced(reduce_rows, x) as traced:
        assert traced.tape.op_names == ["advect.input", "array.sum"]
        assert traced.output.shape == (2, 1)
        assert_allclose(np.asarray(traced.output), [[3.0], [7.0]])


@pytest.mark.parametrize(
    ("operation", "input_value", "op_name", "expected_fields"),
    [
        (
            lambda namespace, value: namespace.linalg.eigh(value),
            _strict([[3.0, 0.5], [0.5, 1.0]], dtype=xp.float32),
            "array_ext.linalg.eigh",
            ("eigenvalues", "eigenvectors"),
        ),
        (
            lambda namespace, value: namespace.linalg.qr(value, mode="complete"),
            _strict([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]], dtype=xp.float32),
            "array_ext.linalg.qr",
            ("Q", "R"),
        ),
        (
            lambda namespace, value: namespace.linalg.slogdet(value),
            _strict([[3.0, 0.5], [0.5, 1.0]], dtype=xp.float32),
            "array_ext.linalg.slogdet",
            ("sign", "logabsdet"),
        ),
        (
            lambda namespace, value: namespace.linalg.svd(
                value,
                full_matrices=False,
            ),
            _strict([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]], dtype=xp.float32),
            "array_ext.linalg.svd",
            ("U", "S", "Vh"),
        ),
    ],
    ids=["eigh", "qr-complete", "slogdet", "svd-reduced"],
)
def test_fixed_arity_linalg_results_trace_with_standard_fields(
    operation: Any,
    input_value: Any,
    op_name: str,
    expected_fields: tuple[str, ...],
) -> None:
    def decompose(value: Any) -> Any:
        return operation(value.__array_namespace__(), value)

    expected = tuple(operation(xp, input_value))
    with _traced(decompose, input_value) as traced:
        assert traced.output._fields == expected_fields
        assert len(traced.output) == len(expected_fields)
        for field in expected_fields:
            assert getattr(traced.output, field) is not None
        for output, expected_output in zip(traced.output, expected, strict=True):
            assert output.shape == expected_output.shape
            assert output.dtype == expected_output.dtype
            assert_allclose(
                np.asarray(output),
                np.asarray(expected_output),
                rtol=1e-5,
                atol=1e-5,
            )
        assert traced.tape.op_names == ["advect.input", op_name, "advect.getoutput"]
        assert traced.tape.node_count == len(expected_fields) + 2


def test_complex_python_scalar_keeps_float32_width() -> None:
    x = _strict([1.0, -2.0], dtype=xp.float32)

    with _traced(lambda value: 1j * value, x) as traced:
        assert traced.output.dtype == xp.complex64
        assert traced.tape.op_names == ["advect.input", "array.multiply"]
        assert traced.tape.stats()["literal_count"] == 1


def test_numpy_coercion_of_array_api_tracer_raises() -> None:
    x = _strict([1.0, 2.0], dtype=xp.float64)

    def silently_detaching(value: Any) -> Any:
        return np.asarray(value)

    with pytest.raises(TracingError, match=r"np\.array\(values, like=x\)"):
        ad.grad(silently_detaching)(x)


def _masked_square(value: Any) -> Any:
    return value.__array_namespace__().sum(value[value > 0] ** 2)


def _sorted_prefix_square(value: Any) -> Any:
    namespace = value.__array_namespace__()
    return namespace.sum(value[namespace.argsort(value)[:2]] ** 2)


def _masked_update_square(value: Any) -> Any:
    namespace = value.__array_namespace__()
    updated = namespace.zeros_like(value)
    updated[value > 0] = value[value > 0]
    return namespace.sum(updated * updated)


@pytest.mark.parametrize(
    ("loss", "expected"),
    [
        (_masked_square, [0.6, 0.0, 4.4, 0.0]),
        (_sorted_prefix_square, [0.0, -3.4, 0.0, -7.8]),
        (_masked_update_square, [0.6, 0.0, 4.4, 0.0]),
    ],
)
def test_traced_index_arrays_select_by_their_values(loss: Any, expected: list[float]) -> None:
    """A mask or integer index computed from traced values indexes concretely."""
    value = _strict([0.3, -1.7, 2.2, -3.9], dtype=xp.float64)

    assert_allclose(np.asarray(ad.grad(loss)(value)), expected)


def test_array_valued_repeats_are_rejected_in_every_lifetime() -> None:
    """The inventory declares them untraceable; no lifetime records them."""
    value = _strict([0.3, -1.7, 2.2], dtype=xp.float64)
    counts = _strict([1, 2, 0])

    def repeated(x: Any) -> Any:
        return x.__array_namespace__().repeat(x, counts)

    with pytest.raises(TracingError, match="only scalar repeats"):
        ad.jvp(repeated)(value, tangents=xp.ones_like(value))
    with pytest.raises(TracingError, match="only scalar repeats"):
        ad.grad(lambda x: xp.sum(repeated(x)))(value)
    with pytest.raises(TracingError, match="only scalar repeats"):
        ad.stage(repeated, value)


def test_dynamic_unique_values_is_traceable() -> None:
    x = _strict([2.0, 1.0], dtype=xp.float64)

    def unique_sum(value: Any) -> Any:
        namespace = value.__array_namespace__()
        return namespace.sum(namespace.unique_values(value))

    np.testing.assert_allclose(np.asarray(ad.grad(unique_sum)(x)), np.ones(2))


def test_dynamic_discrete_result_namespace_remains_traceable() -> None:
    x = _strict([0.0, 2.0, 0.0, 4.0], dtype=xp.float64)

    def selected_values(value: Any) -> Any:
        namespace = value.__array_namespace__()
        indices = namespace.nonzero(value)[0]
        return indices.__array_namespace__().take(value, indices, axis=0)

    assert_allclose(np.asarray(_trace(selected_values, x)), [2.0, 4.0])


def test_escaped_array_api_tracer_rejects_metadata_reads() -> None:
    x = _strict([1.0, 2.0], dtype=xp.float32)
    escaped: list[Any] = []

    def retain_tracer(value: Any) -> Any:
        escaped.append(value)
        return value.__array_namespace__().sum(value)

    ad.grad(retain_tracer)(x)
    tracer = escaped[0]

    for attribute in ("shape", "dtype", "ndim", "size", "device", "raw_namespace"):
        with pytest.raises(TracingError, match="escaped the trace"):
            getattr(tracer, attribute)


@pytest.mark.parametrize(
    ("path", "args", "kwargs", "error", "match"),
    [
        pytest.param(
            "expand_dims",
            (object(), 0),
            {"axis": 1},
            TypeError,
            "received 'axis' twice",
            id="duplicate-positional-attribute",
        ),
        pytest.param(
            "clip",
            (object(), object()),
            {"min": object()},
            TypeError,
            "received 'min' twice",
            id="duplicate-positional-operand",
        ),
        pytest.param(
            "sin",
            (object(), 7),
            {},
            TypeError,
            "takes 1 positional arguments but 2 were given",
            id="excess-positional-argument",
        ),
        pytest.param(
            "sin",
            (),
            {},
            TypeError,
            "missing required argument 'x'",
            id="missing-required-operand",
        ),
        pytest.param(
            "concat",
            (object(),),
            {},
            TypeError,
            "expects 'arrays' to be a list or tuple",
            id="non-sequence-variadic-operand",
        ),
        pytest.param(
            "matrix_transpose",
            (SimpleNamespace(shape=(2,), dtype=np.dtype("float64")),),
            {},
            ValueError,
            "at least two dimensions",
            id="matrix-transpose-rank",
        ),
        pytest.param(
            "future_extension",
            (),
            {},
            NotImplementedError,
            "not traceable yet",
            id="unknown-function",
        ),
    ],
)
def test_call_binding_rejects_invalid_standard_forms(
    path: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        bind_array_api_call(path, args, kwargs)


def test_staging_rejects_an_operand_bound_twice() -> None:
    # Regression: the positional ``min`` used to become a third clip operand
    # and stage a silently wrong program instead of the provider's TypeError.
    value = _strict([-2.0, 0.5, 3.0], dtype=xp.float64)
    bound = _strict(0.0, dtype=xp.float64)

    def clip_twice(source: Any, lower: Any, upper: Any) -> Any:
        return source.__array_namespace__().clip(source, lower, min=upper)

    specs = (
        ad.ArraySpec(value.shape, value.dtype),
        ad.ArraySpec(bound.shape, bound.dtype),
        ad.ArraySpec(bound.shape, bound.dtype),
    )
    with pytest.raises(TypeError, match="received 'min' twice"):
        ad.stage(clip_twice, specs=specs)


def _binding_sentinel(path: str, name: str, spec: Any) -> Any:
    if name in spec.sequence_operands:
        return (object(), object())
    if path.endswith("matrix_transpose") and name == "x":
        return SimpleNamespace(shape=(2, 2))
    return object()


@st.composite
def _array_api_call_spellings(draw: st.DrawFn, path: str) -> tuple[dict[str, Any], list[int]]:
    """Draw one bound call of *path* and several positional/keyword splits of it."""
    spec = _FUNCTION_SPECS[path]
    arguments = signatures._signature_arguments(path, LATEST_ARRAY_API_VERSION)
    positional = (*arguments.posonlyargs, *arguments.args)
    required_positional = len(positional) - len(arguments.defaults)
    keyword_only_required = [default is None for default in arguments.kw_defaults]
    bound_positional = draw(
        st.integers(
            min_value=max(required_positional, 0),
            max_value=len(positional),
        )
    )
    names = [argument.arg for argument in positional[:bound_positional]]
    names.extend(
        argument.arg
        for argument, required in zip(arguments.kwonlyargs, keyword_only_required, strict=True)
        if required or draw(st.booleans())
    )
    values = {name: _binding_sentinel(path, name, spec) for name in names}
    posonly = len(arguments.posonlyargs)
    spellings = [
        draw(st.integers(min_value=min(posonly, bound_positional), max_value=bound_positional))
        for _spelling in range(3)
    ]
    return values, [bound_positional, *spellings]


def _spell(
    path: str,
    values: dict[str, Any],
    positional_count: int,
) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
    names = list(values)
    args = tuple(values[name] for name in names[:positional_count])
    kwargs = {name: values[name] for name in names[positional_count:]}
    return bind_array_api_call(path, args, kwargs), args, kwargs


@pytest.mark.parametrize("path", sorted(_FUNCTION_SPECS))
@settings(derandomize=True, deadline=None, max_examples=10)
@given(data=st.data())
def test_array_api_binding_is_independent_of_the_argument_spelling(
    path: str,
    data: st.DataObject,
) -> None:
    values, spellings = data.draw(_array_api_call_spellings(path))
    spec = _FUNCTION_SPECS[path]
    reference, args, kwargs = _spell(path, values, spellings[0])

    expected: list[Any] = []
    for name in spec.operands:
        if name in values:
            value = values[name]
            expected.extend(value if name in spec.sequence_operands else (value,))
    assert reference.operands == tuple(expected)
    assert reference.op == spec.op
    for positional_count in spellings[1:]:
        binding, _args, _kwargs = _spell(path, values, positional_count)
        assert binding == reference

    if args:
        with pytest.raises(TypeError, match="twice"):
            bind_array_api_call(path, args, {**kwargs, next(iter(values)): object()})
    if len(args) == len(spec.positional):
        with pytest.raises(TypeError, match="positional arguments"):
            bind_array_api_call(path, (*args, object()), kwargs)
    required = [name for name in spec.operands if name not in spec.optional_operands]
    if required:
        without = {name: value for name, value in values.items() if name != required[0]}
        with pytest.raises(TypeError, match="missing required argument"):
            bind_array_api_call(path, (), without)


def test_accumulation_reports_a_missing_provider_dtype() -> None:
    namespace = SimpleNamespace(__name__="minimal", sum=lambda value, **_kwargs: value)
    proxy = ArrayAPINamespace(namespace, array_api_version="2022.12")
    source = SimpleNamespace(shape=(2,), dtype=np.dtype("int32"))

    with pytest.raises(TypeError, match="does not provide dtype 'int64'"):
        proxy.sum(source)


_MATRIX = [[2.0, 0.0], [0.0, 1.0]]


@pytest.mark.parametrize(
    ("path", "result", "value", "operands", "error", "match"),
    [
        ("linalg.eigh", lambda x: [x, x], _MATRIX, (), TypeError, "must return a tuple of 2"),
        ("linalg.eigh", lambda x: (x,), _MATRIX, (), ValueError, "returned 1 outputs, expected 2"),
        (
            "linalg.eigh",
            lambda x: (x, object()),
            _MATRIX,
            (),
            NotImplementedError,
            "object at output 1",
        ),
        ("sin", lambda _x: 1.0, [1.0, 2.0], (), NotImplementedError, "returned float"),
        (
            "add",
            lambda x, _y: x,
            [1.0, 2.0],
            (object(),),
            TypeError,
            "Unsupported dynamic Array API operand",
        ),
    ],
    ids=["wrong-container", "wrong-arity", "non-array-output", "scalar-result", "unknown-operand"],
)
def test_provider_results_are_validated_before_recording(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    result: Any,
    value: Any,
    operands: tuple[object, ...],
    error: type[Exception],
    match: str,
) -> None:
    monkeypatch.setattr(f"array_api_strict.{path}", result)

    def call(argument: Any) -> Any:
        return operator.attrgetter(path)(argument.__array_namespace__())(argument, *operands)

    with pytest.raises(error, match=match):
        _trace(call, _strict(value, dtype=xp.float64))


def _invalid(
    function: Any,
    values: Any,
    error: type[Exception],
    match: str,
    *,
    id: str,  # noqa: A002 - mirrors pytest.param
    staged: bool = True,
) -> Any:
    values = values if isinstance(values, tuple) else (_strict(values, dtype=xp.float64),)
    return pytest.param(function, values, error, match, staged, id=id)


_INVALID_CALLS = [
    _invalid(
        lambda x: x.__array_namespace__().linalg.matrix_power(x, bool(1)),
        [[2.0, 0.0], [0.0, 1.0]],
        TypeError,
        "static integer",
        id="matrix-power-bool",
    ),
    _invalid(
        lambda x: x.__array_namespace__().linalg.matrix_power(x, 2),
        [[1.0, 2.0, 3.0]],
        ValueError,
        "square matrices",
        id="matrix-power-nonsquare",
    ),
    _invalid(
        lambda x: x.__array_namespace__().linalg.matrix_power(x, 2, 3),
        [[2.0, 0.0], [0.0, 1.0]],
        TypeError,
        "matrix and a static integer exponent",
        id="matrix-power-arity",
    ),
    _invalid(
        lambda x: x.__array_namespace__().linalg.matrix_power(x, 2),
        (_strict([[1, 2], [3, 4]], dtype=xp.int64),),
        TypeError,
        "floating-point array",
        id="matrix-power-integer-dtype",
    ),
    _invalid(
        lambda x: x.__array_namespace__().linalg.matrix_rank(x),
        [1.0, 2.0],
        ValueError,
        "at least two dimensions",
        id="matrix-rank-vector",
    ),
    _invalid(
        lambda x: x.__array_namespace__().linalg.matrix_rank(x, axis=0),
        [[2.0, 0.0], [0.0, 1.0]],
        TypeError,
        "one array and optional keyword-only rtol",
        id="matrix-rank-keyword",
    ),
    _invalid(
        lambda x: x.__array_namespace__().broadcast_arrays(x=x),
        [1.0, 2.0],
        TypeError,
        "one or more arrays",
        id="broadcast-arrays-keyword",
    ),
    _invalid(
        lambda x: x.__array_namespace__().meshgrid(x, indexing="invalid"),
        [1.0, 2.0],
        ValueError,
        "indexing must be 'ij' or 'xy'",
        id="meshgrid-indexing",
    ),
    _invalid(
        lambda x: x.__array_namespace__().meshgrid(x),
        [[1.0, 2.0]],
        ValueError,
        "one-dimensional",
        id="meshgrid-rank",
    ),
    _invalid(
        lambda x: x.__array_namespace__().meshgrid(x, unexpected=True),
        [1.0, 2.0],
        TypeError,
        "one or more arrays and optional indexing",
        id="meshgrid-keyword",
    ),
    _invalid(
        lambda x, y: x.__array_namespace__().meshgrid(x, y),
        (_strict([1.0, 2.0], dtype=xp.float32), _strict([3.0, 4.0], dtype=xp.float64)),
        ValueError,
        "same dtype",
        id="meshgrid-mixed-dtype",
    ),
    _invalid(
        lambda x: x.__array_namespace__().unstack(x, x),
        [[1.0, 2.0]],
        TypeError,
        "one array and optional keyword-only axis",
        id="unstack-arity",
    ),
    _invalid(
        lambda x: x.__array_namespace__().cumulative_sum(x, axis=2, include_initial=True),
        [1.0, 2.0],
        ValueError,
        "Axis 2 is out of bounds",
        id="cumulative-axis",
    ),
    _invalid(
        lambda x: x.__array_namespace__().cumulative_sum(x, include_initial=True),
        [[1.0, 2.0]],
        ValueError,
        "require axis=",
        id="cumulative-matrix-without-axis",
    ),
    _invalid(
        lambda x: x.__array_namespace__().diff(x, x, prepend=x),
        [1.0, 2.0],
        TypeError,
        "expects one positional array argument",
        id="diff-boundary-arity",
    ),
    _invalid(
        lambda x: x.__array_namespace__().searchsorted(x, sorter=x),
        [3.0, 1.0, 6.0],
        TypeError,
        "expects two positional array arguments",
        id="searchsorted-sorter-only",
    ),
    _invalid(
        lambda x: x.__array_namespace__().asarray([x[0]], copy=False),
        [3.0, 1.0, 6.0],
        ValueError,
        "cannot construct an array from a sequence",
        id="asarray-copy-false-sequence",
    ),
    _invalid(
        lambda x: x.__array_namespace__().asarray(x, dtype=xp.float64, copy=False),
        (_strict([1.0, 2.0], dtype=xp.float32),),
        ValueError,
        "copy=False",
        id="asarray-copy-false-dtype-change",
    ),
    # Dynamic-only forms: staging reports its own unsupported boundary.
    _invalid(
        lambda x: x.__array_namespace__().asarray([x], x),
        [1.0, 2.0],
        TypeError,
        "expects one positional object argument",
        id="asarray-live-sequence-arity",
        staged=False,
    ),
    _invalid(
        lambda x: x.__array_namespace__().unique_values(x, axis=0),
        [1.0, 2.0],
        TypeError,
        "expects one positional array argument",
        id="dynamic-composite-keyword",
        staged=False,
    ),
]


@pytest.mark.parametrize(("function", "values", "error", "match", "staged"), _INVALID_CALLS)
def test_invalid_calls_fail_with_ordinary_errors(
    function: Any,
    values: tuple[Any, ...],
    error: type[Exception],
    match: str,
    *,
    staged: bool,
) -> None:
    with pytest.raises(error, match=match):
        _trace(function, *values)
    if staged:
        with pytest.raises(error, match=match):
            ad.stage(function, specs=tuple(ad.ArraySpec(v.shape, v.dtype) for v in values))


@pytest.mark.parametrize(
    ("function", "value", "expected"),
    [
        pytest.param(
            lambda x: x.__array_namespace__().diff(x, n=0, prepend=x[:1]),
            [3.0, 1.0, 6.0],
            [3.0, 1.0, 6.0],
            id="diff-zero-order",
        ),
        pytest.param(
            lambda x: x.__array_namespace__().cumulative_sum(x, include_initial=True),
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 3.0, 6.0],
            id="cumulative-initial-vector-default-axis",
        ),
        pytest.param(
            lambda x: x.__array_namespace__().asarray([x, []], dtype=x.dtype),
            [],
            np.empty((2, 0)),
            id="live-sequence-empty-child",
        ),
    ],
)
def test_extended_forms_trace_their_edge_cases(function: Any, value: Any, expected: Any) -> None:
    actual = _trace(function, _strict(value, dtype=xp.float64))

    assert actual.dtype == xp.float64
    np.testing.assert_array_equal(np.asarray(actual), expected, strict=True)


def test_diff_rank_zero_boundaries_agree_across_lifetimes() -> None:
    # Regression: the dynamic lowering concatenated rank-zero boundaries as-is
    # and failed, while the provider and staging broadcast them along the axis.
    value = _strict([1.0, 4.0, 9.0], dtype=xp.float64)
    boundary = _strict(0.5, dtype=xp.float64)

    def objective(source: Any, edge: Any) -> Any:
        namespace = source.__array_namespace__()
        return namespace.sum(namespace.diff(source, prepend=edge, append=edge) ** 2)

    # diff([b, x0, x1, x2, b]) = [0.5, 3, 5, -8.5] for the values above.
    expected_value = 0.25 + 9.0 + 25.0 + 72.25
    expected_gradients = ([-5.0, -4.0, 27.0], -18.0)
    program = ad.stage(
        objective,
        specs=(ad.ArraySpec(value.shape, value.dtype), ad.ArraySpec((), value.dtype)),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())
    for function in (objective, program, restored):
        np.testing.assert_allclose(float(function(value, boundary)), expected_value)
        gradients = ad.grad(function, argnums=(0, 1))(value, boundary)
        for actual, expected in zip(gradients, expected_gradients, strict=True):
            np.testing.assert_allclose(np.asarray(actual), expected)


def test_debug_representation_uses_the_provider_value() -> None:
    representations: list[str] = []
    _trace(
        lambda x: representations.append(repr(x)) or x,
        _strict([1.0], dtype=xp.float64),
    )
    with ad.debug():
        _trace(
            lambda x: representations.append(repr(x)) or x,
            _strict([1.0], dtype=xp.float64),
        )
    assert "values=" not in representations[0]
    assert "values=Array([1.]" in representations[1]


def test_tracer_array_methods_preserve_standard_array_results() -> None:
    value = _strict([1.0, 2.0, 3.0, 4.0], dtype=xp.float32)

    def methods(argument: Any) -> Any:
        namespace = argument.__array_namespace__()
        matrix = argument.reshape(2, 2)
        tuple_reshape = argument.reshape((2, 2))
        return (
            argument.astype(namespace.float64, device=argument.device),
            matrix.T,
            tuple_reshape.mT,
            argument.sum(dtype=namespace.float64),
            argument.item(1),
            matrix.item(0, 1),
            argument[:1].item(),
        )

    (
        converted,
        transposed,
        matrix_transposed,
        total,
        flat_item,
        coordinate_item,
        singleton_item,
    ) = _trace(methods, value)

    assert converted.dtype == xp.float64
    np.testing.assert_array_equal(np.asarray(transposed), [[1.0, 3.0], [2.0, 4.0]])
    np.testing.assert_array_equal(np.asarray(matrix_transposed), np.asarray(transposed))
    assert float(np.asarray(total)) == 10.0
    assert float(np.asarray(flat_item)) == 2.0
    assert float(np.asarray(coordinate_item)) == 2.0
    assert float(np.asarray(singleton_item)) == 1.0


def _operator_operands(stem: str) -> tuple[Any, Any]:
    if stem in {"and", "or", "xor", "lshift", "rshift"}:
        return _strict([1, 2, 3], dtype=xp.int64), 6
    if stem == "matmul":
        return (
            _strict([[2.0, 1.0], [0.5, 3.0]], dtype=xp.float64),
            _strict([[1.0, -1.0], [2.0, 0.25]], dtype=xp.float64),
        )
    return _strict([0.5, 1.5, 2.5], dtype=xp.float64), 1.5


@pytest.mark.parametrize("stem", sorted({**_ARITHMETIC_OPERATORS, **_COMPARISON_OPERATORS}))
def test_tracer_operators_match_the_provider_operators(stem: str) -> None:
    # Regression guard: a reflected operator that dropped its reflection
    # (`3.0 - x` computing `x - 3.0`) survived the whole suite.
    value, other = _operator_operands(stem)
    python_operator = getattr(operator, stem, None) or getattr(operator, f"{stem}_")
    spellings = [(f"__{stem}__", python_operator(value, other))]
    if stem in _ARITHMETIC_OPERATORS:
        spellings.append((f"__r{stem}__", python_operator(other, value)))

    for method, expected in spellings:
        actual = _trace(lambda argument, method=method: getattr(argument, method)(other), value)
        assert actual.dtype == expected.dtype, method
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected), err_msg=method)


def test_tracer_unary_operators_and_complex_views_match_the_provider() -> None:
    real = _strict([-1.5, 2.0], dtype=xp.float64)
    integer = _strict([1, 2], dtype=xp.int64)
    complex_value = _strict([1.0 + 2.0j, -3.0 + 0.5j], dtype=xp.complex128)

    for function, value in (
        (operator.pos, real),
        (operator.neg, real),
        (operator.abs, real),
        (operator.invert, integer),
    ):
        np.testing.assert_array_equal(
            np.asarray(_trace(function, value)), np.asarray(function(value))
        )
    complex_outputs = _trace(lambda value: (value.conj(), value.real, value.imag), complex_value)
    np.testing.assert_array_equal(np.asarray(complex_outputs[0]), np.conj(complex_value))
    np.testing.assert_array_equal(np.asarray(complex_outputs[1]), np.real(complex_value))
    np.testing.assert_array_equal(np.asarray(complex_outputs[2]), np.imag(complex_value))


def test_tracer_sequence_scalar_and_mutation_boundaries() -> None:
    value = _strict([1.0, 2.0], dtype=xp.float64)
    scalar = _strict(1.0, dtype=xp.float64)
    observations: list[tuple[int, tuple[tuple[int, ...], ...], bool]] = []

    def inspect(argument: Any) -> Any:
        observations.append(
            (len(argument), tuple(item.shape for item in argument), bool(argument[0]))
        )
        return argument[0]

    _trace(inspect, value)
    assert observations == [(2, ((), ()), True)]

    with pytest.raises(TypeError, match=r"len\(\) of a 0-dimensional array"):
        _trace(len, scalar)
    with pytest.raises(ValueError, match="array of size 1"):
        _trace(lambda argument: argument.item(), value)

    def mutate(argument: Any) -> Any:
        argument[0] = 0.0
        return argument

    with pytest.raises(MutationError, match="generic Array API inputs"):
        _trace(mutate, value)


def test_tracer_rejects_a_different_revision_request() -> None:
    value = _strict([1.0, 2.0], dtype=xp.float64)

    with pytest.raises(ValueError, match=r"provider exposes '2024\.12'"):
        _trace(
            lambda argument: argument.__array_namespace__(api_version="2022.12").sum(argument),
            value,
        )


def test_namespace_proxy_filters_profiles_and_forwards_info() -> None:
    namespace_2022 = ArrayAPINamespace(xp, array_api_version="2022.12")
    namespace_2024 = ArrayAPINamespace(xp, array_api_version="2024.12")

    assert "cumulative_sum" not in dir(namespace_2022)
    assert "cumulative_prod" in dir(namespace_2024)
    assert "svd" in dir(namespace_2024.linalg)
    with pytest.raises(
        AttributeError,
        match=r"not available in the selected 2022\.12 revision",
    ):
        _ = namespace_2022.cumulative_sum
    assert type(namespace_2024.__array_namespace_info__()) is type(xp.__array_namespace_info__())


def test_operations_reject_a_namespace_different_from_the_traced_input() -> None:
    value = _strict([1.0, 2.0], dtype=xp.float64)
    alternate = ArrayAPINamespace(
        SimpleNamespace(
            __name__="alternate",
            add=xp.add,
            broadcast_arrays=xp.broadcast_arrays,
        )
    )

    with pytest.raises(TypeError, match="different Array API namespaces in add"):
        _trace(lambda argument: alternate.add(argument, argument), value)
    with pytest.raises(TypeError, match="different Array API namespaces in broadcast_arrays"):
        _trace(alternate.broadcast_arrays, value)


class _ProtocolArray:
    __advect_namespace_is_instance_specific__ = True
    shape = (1,)
    dtype = np.dtype("float64")

    def __init__(self, namespace: Any) -> None:
        self.namespace = namespace

    def __array_namespace__(self, *, api_version: str | None = None) -> Any:
        del api_version
        return self.namespace


class _WrappedArray:
    def __init__(self, value: Any) -> None:
        self.value = value

    def _advect_snapshot(self) -> tuple[int, Any]:
        return 1, self.value


def _provider_namespace(
    *,
    name: str | None = "provider",
    version: str | None = "2024.12",
    asarray: bool = True,
    namespace_info: bool = True,
) -> Any:
    attributes: dict[str, Any] = {}
    if name is not None:
        attributes["__name__"] = name
    if version is not None:
        attributes["__array_api_version__"] = version
    if asarray:
        attributes["asarray"] = lambda value: value
    if namespace_info:
        attributes["__array_namespace_info__"] = object
    return SimpleNamespace(**attributes)


def test_provider_namespace_changes_after_negotiation_are_reported() -> None:
    current = _provider_namespace()

    class ChangingArray:
        __advect_namespace_is_instance_specific__ = True
        shape = (1,)
        dtype = np.dtype("float64")

        def __init__(self, replacement: Any) -> None:
            self.calls = 0
            self.replacement = replacement

        def __array_namespace__(self, *, api_version: str | None = None) -> Any:
            del api_version
            self.calls += 1
            return current if self.calls <= 2 else self.replacement

    with pytest.raises(TypeError, match="no longer exposes an Array API namespace"):
        _trace(lambda value: value, ChangingArray(None))
    with pytest.raises(TypeError, match=r"provider exposes '2022\.12'"):
        _trace(lambda value: value, ChangingArray(_provider_namespace(version="2022.12")))


def test_grad_reports_every_attempted_revision_unchanged() -> None:
    outdated = _ProtocolArray(_provider_namespace(version="2021.12"))
    message = re.escape(
        "Array inputs _ProtocolArray cannot serve a common Advect Array API revision; "
        f"attempted {', '.join(reversed(SUPPORTED_ARRAY_API_VERSIONS))}"
    )

    # The unary fast path and the general trace negotiate separately.
    for argnums in (0, (0,)):
        with pytest.raises(TypeError, match=f"^{message}$"):
            ad.grad(lambda value: value, argnums=argnums)(outdated)


def test_namespace_discovery_supports_instance_only_and_wrapped_protocols() -> None:
    providers._clear_array_namespace_caches()
    namespace = _provider_namespace()

    class CountingArray:
        calls = 0

        def __array_namespace__(self, *, api_version: str | None = None) -> Any:
            assert api_version == "2024.12"
            CountingArray.calls += 1
            return namespace

    first = _WrappedArray(_WrappedArray(CountingArray()))
    second = _WrappedArray(_WrappedArray(CountingArray()))
    assert providers._get_array_namespace(first, api_version="2024.12") is namespace
    assert providers._get_array_namespace(second, api_version="2024.12") is namespace
    assert CountingArray.calls == 1

    instance_namespace = _provider_namespace(name="instance-provider")
    wrapped_instance = _WrappedArray(_ProtocolArray(instance_namespace))
    assert (
        providers._get_array_namespace(wrapped_instance, api_version="2024.12")
        is instance_namespace
    )


class _ForwardingArray:
    """Expose the wrapped array's namespace through instance attribute lookup."""

    def __init__(self, value: Any) -> None:
        self.value = value

    def __getattr__(self, name: str) -> Any:
        return getattr(self.value, name)


def test_trace_provider_memo_admits_only_type_level_module_namespaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memo: dict[Any, Any] = {}
    monkeypatch.setattr(_ephemeral, "_TRACE_PROVIDERS", memo)
    standard = _provider_namespace(name="standard-proxy")
    standard.zeros_like = standard.ones_like = lambda value: value

    class ProxyNamespaceArray:
        shape = (1,)
        dtype = np.dtype("float64")

        def __array_namespace__(self, *, api_version: str | None = None) -> Any:
            del api_version
            return standard

    def resolve(*values: Any) -> Any:
        return _ephemeral._trace_provider(values, array_api_version="2024.12")

    forwarded_numpy = resolve(_ForwardingArray(np.ones(2)))
    forwarded_strict = resolve(_ForwardingArray(_strict([1.0, 2.0])))
    assert forwarded_numpy is not None
    assert forwarded_numpy.namespace is np
    assert forwarded_strict is not None
    assert forwarded_strict.namespace is xp
    assert resolve(_WrappedArray(np.ones(2))) is not None
    assert resolve(_ProtocolArray(standard)) is not None
    proxy_provider = resolve(ProxyNamespaceArray())
    assert proxy_provider is not None
    assert proxy_provider.namespace is standard
    assert memo == {}

    provider = resolve(np.ones(2), 1.0, np.ones(3))
    assert memo == {((np.ndarray, float), "2024.12"): provider}
    assert resolve(np.zeros(4), 2.0) is provider


def test_namespace_cache_reset_forgets_memoized_trace_providers() -> None:
    # Scripts reset core's type-level caches when a provider changes profiles;
    # the autodiff provider memo is keyed the same way and must follow.
    provider = _ephemeral._trace_provider((np.ones(2),), array_api_version="2024.12")
    assert provider is not None
    assert _ephemeral._TRACE_PROVIDERS

    providers._clear_array_namespace_caches()

    assert _ephemeral._TRACE_PROVIDERS == {}


def test_namespace_discovery_validates_requests_and_empty_wrappers() -> None:
    with pytest.raises(TypeError, match="Invalid Array API version request"):
        providers._get_array_namespace(object(), api_version=object())

    assert providers._get_array_namespace(_WrappedArray(None), api_version="2024.12") is None


def test_acceptance_reports_provider_revision_and_noninvasive_failures() -> None:
    old = _ProtocolArray(_provider_namespace(version="2022.12"))
    with pytest.raises(TypeError, match=r"selected Array API 2024\.12"):
        _accepts_array_api(old)

    class DefaultOnlyArray:
        shape = (1,)
        dtype = np.dtype("float64")

        def __init__(self, *, fail_default: bool = False) -> None:
            self.fail_default = fail_default

        def __array_namespace__(self, *, api_version: str | None = None) -> Any:
            if api_version is not None or self.fail_default:
                raise ValueError("unsupported request")
            return _provider_namespace()

    assert _accepts_array_api(DefaultOnlyArray())
    assert not _accepts_array_api(DefaultOnlyArray(fail_default=True))
    assert not _accepts_array_api(_ProtocolArray(_provider_namespace(namespace_info=False)))
    assert not _accepts_array_api(SimpleNamespace(shape=(1,), dtype=np.dtype("float64")))
    assert not _accepts_array_api(object())
