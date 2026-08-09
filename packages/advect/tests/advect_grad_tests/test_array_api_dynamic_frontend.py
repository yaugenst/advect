"""End-to-end contracts for the backend-neutral Python Array API frontend."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad
from advect.autodiff._ephemeral import trace_call
from advect.autodiff.rules.array_family.providers import (
    resolve_array_family_backend_provider,
)
from advect.core._array_api.frontend import bind_array_api_call
from advect.core._errors import TracingError

xp = pytest.importorskip("array_api_strict")


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


class _PinnedNamespace(_FutureNamespace):
    __name__ = "multi_version_array"
    __array_api_version__ = "2024.12"


class _MultiVersionArray:
    shape = (1,)
    dtype = np.dtype("float64")

    def __init__(self) -> None:
        self.requests: list[str | None] = []

    def __array_namespace__(self, *, api_version: str | None = None) -> object:
        self.requests.append(api_version)
        return _PinnedNamespace() if api_version == "2024.12" else _FutureNamespace()


def _strict(values: Any, *, dtype: Any = None) -> Any:
    return xp.asarray(values) if dtype is None else xp.asarray(values, dtype=dtype)


def test_binding_uses_the_standard_operand_schema() -> None:
    source = object()
    indices = object()

    binding = bind_array_api_call("take", (source, indices), {"axis": 0})

    assert binding.operands == (source, indices)
    assert binding.attrs == {"axis": 0}


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
    provider = resolve_array_family_backend_provider(gradient)
    assert provider.backend == "array_api_strict"
    assert provider.namespace is xp


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


def test_asarray_copy_false_rejects_dtype_changing_staged_aliases() -> None:
    value = _strict([1.0, 2.0], dtype=xp.float32)

    def invalid(argument: Any) -> Any:
        namespace = argument.__array_namespace__()
        return namespace.asarray(argument, dtype=namespace.float64, copy=False)

    with pytest.raises(ValueError, match=r"copy=False"):
        trace_call(
            invalid,
            args=(value,),
            kwargs={},
            argnums=(0,),
            argnames=None,
        )
    with pytest.raises(ValueError, match=r"copy=False"):
        ad.stage(
            invalid,
            specs=(ad.ArraySpec(value.shape, value.dtype),),
        )


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
    traced = trace_call(
        lambda value: (2.0 * value + 1.0) / 3.0,
        args=(x,),
        kwargs={},
        argnums=(0,),
        argnames=None,
    )

    assert type(traced.output) is type(x)
    assert traced.output.dtype == xp.float32
    try:
        assert traced.tape.op_names == [
            "advect.input",
            "array.multiply",
            "array.add",
            "array.divide",
        ]
        assert traced.tape.stats()["literal_count"] == 3
    finally:
        traced.tape.release_payloads()


def test_static_array_api_arguments_are_node_attributes() -> None:
    x = _strict([[1.0, 2.0], [3.0, 4.0]], dtype=xp.float64)

    def reduce_rows(value: Any) -> Any:
        namespace = value.__array_namespace__()
        return namespace.sum(value, axis=1, keepdims=True)

    traced = trace_call(
        reduce_rows,
        args=(x,),
        kwargs={},
        argnums=(0,),
        argnames=None,
    )
    try:
        assert traced.tape.op_names == ["advect.input", "array.sum"]
        assert traced.output.shape == (2, 1)
        assert_allclose(
            np.asarray(traced.output),
            np.asarray(_strict([[3.0], [7.0]], dtype=xp.float64)),
        )
    finally:
        traced.tape.release_payloads()


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
    traced = trace_call(
        decompose,
        args=(input_value,),
        kwargs={},
        argnums=(0,),
        argnames=None,
    )
    try:
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
    finally:
        traced.tape.release_payloads()


def test_complex_python_scalar_keeps_float32_width() -> None:
    x = _strict([1.0, -2.0], dtype=xp.float32)
    traced = trace_call(
        lambda value: 1j * value,
        args=(x,),
        kwargs={},
        argnums=(0,),
        argnames=None,
    )

    assert traced.output.dtype == xp.complex64
    try:
        assert traced.tape.op_names == ["advect.input", "array.multiply"]
        assert traced.tape.stats()["literal_count"] == 1
    finally:
        traced.tape.release_payloads()


def test_numpy_coercion_of_array_api_tracer_raises() -> None:
    x = _strict([1.0, 2.0], dtype=xp.float64)

    def silently_detaching(value: Any) -> Any:
        return np.asarray(value)

    with pytest.raises(TracingError, match=r"np\.array\(values, like=x\)"):
        ad.grad(silently_detaching)(x)


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

    traced = trace_call(
        selected_values,
        args=(x,),
        kwargs={},
        argnums=(0,),
        argnames=None,
    )
    try:
        assert_allclose(np.asarray(traced.output), np.asarray([2.0, 4.0]))
    finally:
        traced.tape.release_payloads()


@pytest.mark.parametrize(
    ("function", "values", "error", "match"),
    [
        pytest.param(
            lambda left, right: left.__array_namespace__().meshgrid(left, right),
            (
                _strict([1.0, 2.0], dtype=xp.float32),
                _strict([3.0, 4.0], dtype=xp.float64),
            ),
            ValueError,
            "same dtype",
            id="meshgrid-mixed-dtype",
        ),
        pytest.param(
            lambda value: value.__array_namespace__().linalg.matrix_power(value, 2),
            (_strict([[1, 2], [3, 4]], dtype=xp.int64),),
            TypeError,
            "floating-point array",
            id="matrix-power-integer-dtype",
        ),
    ],
)
def test_composite_dtype_contracts(
    function: Any,
    values: tuple[Any, ...],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        trace_call(
            function,
            args=values,
            kwargs={},
            argnums=tuple(range(len(values))),
            argnames=None,
        )

    with pytest.raises(error, match=match):
        ad.stage(
            function,
            specs=tuple(ad.ArraySpec(value.shape, value.dtype) for value in values),
        )


def test_dynamic_trace_reports_all_attempted_array_api_versions() -> None:
    with pytest.raises(TypeError, match=r"attempted 2024\.12, 2023\.12, 2022\.12"):
        ad.grad(lambda value: value)(_FutureArray())


def test_dynamic_trace_negotiates_the_pinned_array_api_version() -> None:
    value = _MultiVersionArray()
    traced = trace_call(
        lambda x: x,
        args=(value,),
        kwargs={},
        argnums=(0,),
        argnames=None,
    )

    assert traced.output is value
    assert value.requests
    assert set(value.requests) == {"2024.12"}


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
