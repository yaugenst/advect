"""End-to-end contracts at NumPy's dynamic and staged protocol boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import array_api_strict as strict
import numpy as np
import pytest

import advect as ad
from advect.core._array_api.profiles import LATEST_ARRAY_API_VERSION
from advect_numpy_tests._assertions import (
    assert_jvp_matches_central_difference,
    assert_staged_round_trip,
    assert_tree_close,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize(
    ("operation", "expected_tangent"),
    [
        (
            lambda x: np.pad(x, 1, mode="constant", constant_values=2.0),
            lambda x: np.pad(x, 1, mode="constant", constant_values=0),
        ),
        (
            lambda x: np.pad(x, (1, 2), mode="constant", constant_values=(-1.0, 2.0)),
            lambda x: np.pad(x, (1, 2), mode="constant", constant_values=0),
        ),
        (
            lambda x: np.pad(
                x,
                ((1, 0), (2, 1)),
                mode="constant",
                constant_values=((1.0, 2.0), (3.0, 4.0)),
            ),
            lambda x: np.pad(x, ((1, 0), (2, 1)), mode="constant", constant_values=0),
        ),
        (
            lambda x: np.partition(x, np.asarray(1), axis=-1),
            lambda x: np.partition(x, np.asarray(1), axis=-1),
        ),
        (
            lambda x: np.partition(x, np.asarray([0, 2]), axis=-1),
            lambda x: np.partition(x, np.asarray([0, 2]), axis=-1),
        ),
        (
            lambda x: np.gradient(x, axis=np.int64(-1)),
            lambda x: np.gradient(x, axis=-1),
        ),
        (
            lambda x: np.gradient(x, axis=(-2, -1)),
            lambda x: np.gradient(x, axis=(-2, -1)),
        ),
    ],
    ids=[
        "scalar-pad-width",
        "paired-pad-width",
        "per-axis-pad-metadata",
        "scalar-kth",
        "array-kth",
        "numpy-integer-axis",
        "axis-tuple",
    ],
)
def test_normalized_numpy_forms_preserve_values_and_tangents(
    operation: Callable[[Any], Any],
    expected_tangent: Callable[[np.ndarray], Any],
) -> None:
    value = np.arange(1.0, 7.0).reshape(2, 3)
    direction = np.ones_like(value)

    primal, tangent = ad.jvp(operation)(value, tangents=direction)
    expected_primal = operation(value)
    tangent_reference = expected_tangent(direction)

    assert_tree_close(primal, expected_primal)
    assert_tree_close(tangent, tangent_reference)


@pytest.mark.parametrize(
    ("function", "stages"),
    [
        pytest.param(lambda x: np.copy(x, "F", False), True, id="copy"),  # noqa: FBT003
        pytest.param(lambda x: np.take(x, [-1, 3], 1, None, "wrap"), True, id="take"),
        pytest.param(lambda x: np.sort(x, 0, "mergesort"), True, id="sort"),
        pytest.param(lambda x: np.reshape(x, (1, 6), "F", copy=True), True, id="reshape"),
        pytest.param(lambda x: np.diag(x, -1), False, id="diag"),
        pytest.param(lambda x: np.trace(x, 1, 0, 1, np.float64), True, id="trace"),
        pytest.param(lambda x: np.diff(x, 2, 1, 0.0, 10.0), True, id="diff"),
        pytest.param(
            lambda x: np.diff(x, 2, 0, np.ones((1, 3)), np.full((1, 3), 2.0)),
            True,
            id="diff-array-boundaries",
        ),
        pytest.param(
            lambda x: np.take_along_axis(x, np.array([[2, 0], [1, 1]]), -1),
            True,
            id="take-along-axis",
        ),
        # Staging still rejects NumPy integer and 0-d array metadata for these two.
        pytest.param(lambda x: np.moveaxis(x, np.int64(0), 1), False, id="moveaxis-numpy-integer"),
        pytest.param(lambda x: np.tile(x, np.asarray(2)), False, id="tile-0d-reps"),
    ],
)
def test_positional_and_numpy_scalar_metadata_trace_and_stage(
    function: Callable[[Any], Any],
    *,
    stages: bool,
) -> None:
    value = np.array([[4.0, 1.0, 7.0], [2.0, 6.0, 3.0]])
    direction = np.arange(0.1, 0.7, 0.1).reshape(2, 3)

    assert_jvp_matches_central_difference(function, (value,), (direction,), rtol=1e-7, atol=1e-9)
    if stages:
        assert_staged_round_trip(function, value)


def test_scalar_array_shape_metadata_normalizes_through_resize() -> None:
    value = np.arange(3.0)
    direction = np.asarray([0.1, 0.2, 0.3])

    primal, tangent = ad.jvp(lambda x: np.resize(x, np.asarray(8)))(
        value,
        tangents=direction,
    )

    np.testing.assert_allclose(primal, np.resize(value, 8))
    np.testing.assert_allclose(tangent, np.resize(direction, 8))


def test_numpy_aliases_preserve_nondefault_metadata_during_tracing() -> None:
    value = np.arange(1.0, 7.0).reshape(2, 3)
    direction = np.linspace(0.1, 0.6, 6).reshape(2, 3)

    def aliases(x: Any) -> tuple[Any, ...]:
        return (
            np.astype(x, np.float32, copy=True, device="cpu"),
            np.permute_dims(x, (1, 0)),
            np.linalg.tensordot(x, x, axes=((1,), (1,))),
            np.linalg.vector_norm(x, axis=(0, 1), keepdims=True, ord=2),
            np.linalg.trace(x @ x.T, offset=1, dtype=np.float32),
        )

    assert_jvp_matches_central_difference(
        aliases, (value,), (direction,), rtol=2e-3, atol=2e-3, step=1e-3
    )


@pytest.mark.parametrize(
    ("operation", "error", "match"),
    [
        (
            lambda x: np.astype(x, np.float64, device="gpu"),
            ad.TracingError,
            "device= must be None or 'cpu'",
        ),
        (
            lambda x: np.astype(x, np.float64, copy=False),
            ad.TracingError,
            "runtime-dependent alias",
        ),
    ],
    ids=["astype-device", "astype-alias"],
)
def test_numpy_aliases_reject_nonportable_runtime_contracts(
    operation: Callable[[Any], Any],
    error: type[Exception],
    match: str,
) -> None:
    value = np.arange(6.0).reshape(2, 3)

    with pytest.raises(error, match=match):
        ad.jvp(operation)(value, tangents=np.ones_like(value))


def test_astype_without_a_copy_reads_a_selected_python_scalar_as_its_array() -> None:
    # np.astype(copy=False) read the dtype of the scalar's Python value: "'float'
    # object has no attribute 'dtype'", where a copying astype, and both calls
    # on a rank-zero array, trace the scalar as that array.
    def narrowed(s: Any) -> Any:
        return np.astype(s, np.float32, copy=False) * 2.0

    gradient = ad.grad(narrowed)(0.5)

    assert type(gradient) is float
    assert gradient == ad.grad(narrowed)(np.asarray(0.5)) == 2.0
    with pytest.raises(ad.TracingError, match="runtime-dependent alias"):
        ad.grad(lambda s: np.astype(s, np.float64, copy=False))(0.5)


def test_staged_aliases_and_evaluator_controls_round_trip() -> None:
    value = np.arange(1.0, 7.0, dtype=np.float32).reshape(2, 3)

    def aliases(x: Any) -> tuple[Any, ...]:
        stacked = np.stack((x, x + 1), axis=0, dtype=np.float64, casting="same_kind")
        return (
            np.concatenate((stacked, stacked), axis=1, dtype=np.float64, casting="same_kind"),
            np.zeros_like(
                x,
                dtype=np.float32,
                order="F",
                subok=False,
                shape=(3, 2),
                device="cpu",
            ),
            np.permute_dims(x, (1, 0)),
            np.linalg.tensordot(x, x, axes=((1,), (1,))),
            np.linalg.trace(x @ x.T, offset=1, dtype=np.float32),
        )

    assert_staged_round_trip(aliases, value)


def test_full_constructors_differentiate_the_fill_and_not_the_like_anchor() -> None:
    anchor = np.arange(6.0).reshape(2, 3)
    fill = np.asarray(2.0)

    def create(array: Any, value: Any) -> tuple[Any, ...]:
        return (
            np.full((2, 3), value, np.float32, "F", device="cpu", like=array),
            np.full_like(
                array,
                value,
                np.float32,
                "F",
                False,  # noqa: FBT003 - exercise NumPy's positional form
                (3, 2),
                device="cpu",
            ),
        )

    primal, tangent = ad.jvp(create, argnums=(0, 1))(
        anchor,
        fill,
        tangents=(np.ones_like(anchor), np.asarray(0.25)),
    )

    for actual, reference in zip(primal, create(anchor, fill), strict=True):
        np.testing.assert_array_equal(actual, reference)
        assert actual.flags.f_contiguous
    for actual in tangent:
        np.testing.assert_array_equal(actual, np.full(actual.shape, 0.25, dtype=np.float32))
    assert_staged_round_trip(create, anchor, fill)


def test_like_constructors_preserve_explicit_copy_dtype_and_boolean_constants() -> None:
    value = np.arange(6.0).reshape(2, 3)
    direction = np.linspace(0.1, 0.6, 6).reshape(2, 3)

    def construct(array: Any) -> tuple[Any, ...]:
        return (
            np.asarray(array, dtype=np.float32, copy=True, like=array),
            np.asarray([True, False], like=array),
            np.asarray(array, order="A", like=array),
        )

    primal, tangent = ad.jvp(construct)(value, tangents=direction)

    np.testing.assert_allclose(primal[0], value.astype(np.float32))
    np.testing.assert_allclose(tangent[0], direction.astype(np.float32))
    np.testing.assert_array_equal(primal[1], [True, False])
    np.testing.assert_array_equal(tangent[1], [False, False])
    np.testing.assert_allclose(primal[2], value)
    np.testing.assert_allclose(tangent[2], direction)


def test_expired_tracers_cannot_cross_array_function_boundaries() -> None:
    captured: list[Any] = []
    value = np.arange(3.0)

    def capture(x: Any) -> Any:
        captured.append(x)
        return x + 0

    ad.jvp(capture)(value, tangents=np.ones_like(value))

    with pytest.raises(ad.TracingError, match="unrelated or expired trace recorder"):
        ad.jvp(lambda x: np.concatenate((x, captured[0])))(
            value,
            tangents=np.ones_like(value),
        )


@pytest.mark.parametrize(
    ("protocol", "write"),
    [
        ("ufunc", lambda y, buffer: np.add(y, 1.0, out=buffer)),
        ("array-function", lambda y, buffer: np.clip(y, 0.0, 1.0, out=buffer)),
    ],
)
def test_inner_traces_cannot_write_into_an_outer_trace_out_buffer(
    protocol: str, write: Callable[[Any, Any], object]
) -> None:
    def outer(x: Any) -> Any:
        buffer = x * 1.0

        def inner(y: Any) -> Any:
            write(y, buffer)
            return np.sum(y)

        return np.sum(ad.grad(inner)(x)) + np.sum(buffer)

    with pytest.raises(ad.TracingError, match=f"{protocol} out= must belong to the current trace"):
        ad.grad(outer)(np.arange(3.0))


def test_debug_mode_uses_the_full_ufunc_protocol_without_changing_results() -> None:
    value = np.asarray([0.2, -0.4, 0.7])
    direction = np.asarray([0.3, 0.1, -0.2])

    with ad.debug():
        primal, tangent = ad.jvp(np.sin)(value, tangents=direction)

    np.testing.assert_allclose(primal, np.sin(value))
    np.testing.assert_allclose(tangent, np.cos(value) * direction)


def test_debug_mode_preserves_the_unsupported_ufunc_error() -> None:
    with ad.debug(), pytest.raises(ad.TracingError, match="Unsupported ufunc: gcd"):
        ad.jvp(lambda array: np.gcd(array, 2))(
            np.asarray([2, 3]),
            tangents=np.zeros(2, dtype=int),
        )


def test_debug_mode_records_the_user_callsite_of_simple_ufuncs() -> None:
    def operation(array: Any) -> Any:
        return np.bitwise_and(np.astype(array, np.int64), 1)

    with ad.debug(), pytest.raises(ad.NoJVPError) as caught:
        ad.jvp(operation)(np.asarray([2.0, 3.0]), tangents=np.ones(2))

    location = caught.value.source_location
    assert location is not None
    assert __file__ in location
    assert "in operation()" in location


def test_ufunc_none_out_sentinel_and_multi_output_controls_remain_traceable() -> None:
    value = np.asarray([1.25, -2.5])
    direction = np.asarray([0.2, -0.3])

    primal, tangent = ad.jvp(lambda x: np.add(x, 2.0, out=(None,)))(
        value,
        tangents=direction,
    )
    np.testing.assert_allclose(primal, value + 2.0)
    np.testing.assert_allclose(tangent, direction)

    (fractional, integral), (fractional_tangent, integral_tangent) = ad.jvp(
        lambda x: np.modf(x, casting="same_kind")
    )(value, tangents=direction)
    expected_fractional, expected_integral = np.modf(value)
    np.testing.assert_allclose(fractional, expected_fractional)
    np.testing.assert_allclose(integral, expected_integral)
    np.testing.assert_allclose(fractional_tangent, direction)
    np.testing.assert_allclose(integral_tangent, np.zeros_like(direction))


@pytest.mark.parametrize("method", ["reduce", "accumulate"])
def test_unsupported_ufunc_reduction_methods_name_the_rejected_form(method: str) -> None:
    value = np.arange(3.0)

    def operation(x: Any) -> Any:
        return getattr(np.maximum, method)(x)

    with pytest.raises(ad.TracingError, match=rf"numpy\.maximum\.{method}.*not supported"):
        ad.jvp(operation)(value, tangents=np.ones_like(value))
    with pytest.raises(ad.TracingError, match=rf"numpy\.maximum\.{method}.*not supported"):
        ad.stage(operation, specs=(ad.ArraySpec(value.shape, value.dtype),))


def test_unsupported_array_function_fails_clearly_in_both_lifetimes() -> None:
    value = np.arange(3.0)

    def operation(x: Any) -> Any:
        return np.packbits(x > 0)

    with pytest.raises(ad.TracingError, match=r"numpy\.packbits.*not yet supported"):
        ad.jvp(operation)(value, tangents=np.ones_like(value))
    with pytest.raises(ad.TracingError, match=r"numpy\.packbits.*not supported during staging"):
        ad.stage(operation, specs=(ad.ArraySpec(value.shape, value.dtype),))


@pytest.mark.parametrize(
    ("operation", "value"),
    [
        pytest.param(np.gradient, [0.25, -0.5, 2.0, 1.0], id="gradient"),
        pytest.param(
            lambda x: np.linalg.matrix_power(x, 0), [[0.25, -0.5], [2.0, 1.0]], id="matrix-power"
        ),
        pytest.param(
            lambda x: np.max(x, where=np.asarray([True, False, True, True]), initial=-1.0),
            [0.25, -0.5, 2.0, 1.0],
            id="max-where-initial",
        ),
    ],
)
def test_numpy_helpers_differentiate_programs_staged_from_another_provider(
    operation: Callable[[Any], Any], value: list[Any]
) -> None:
    # The helpers read the staged dtypes, not the provider dtype objects that
    # staged code sees, so a program staged from strict examples differentiates.
    def loss(x: Any) -> Any:
        return np.sum(operation(x) ** 2)

    example = strict.asarray(value, dtype=strict.float64)

    gradient = ad.grad(ad.stage(loss, example))(example)

    assert gradient.dtype == strict.float64
    np.testing.assert_allclose(np.asarray(gradient), ad.grad(loss)(np.asarray(value)))


def test_numpy_powers_stage_a_dtype_the_examples_provider_lacks() -> None:
    # array_api_strict has no float16, so the NumPy power reads the staged dtype.
    value = np.asarray([0.25, -0.5, 2.0], dtype=np.float32)
    version = min(np.__array_api_version__, LATEST_ARRAY_API_VERSION)

    def graph(example: object) -> object:
        program = ad.stage(
            lambda x: np.astype(x, np.float16) ** 2, example, array_api_version=version
        )
        return program.to_dict()["program"]["graph"]

    assert graph(strict.asarray(value)) == graph(value)


def test_numpy_calls_reject_a_provider_dtype_that_staged_code_passes() -> None:
    # Staged code sees the example provider's dtype objects, which NumPy cannot read.
    value = strict.asarray([0.25, -0.5, 2.0], dtype=strict.float64)

    with pytest.raises(TypeError, match=r"^Cannot interpret 'array_api_strict\.float64' as a"):
        ad.stage(lambda x: x.astype(x.dtype, casting="same_kind"), value)


def test_numpy_tracers_over_staged_provider_values_report_numpy_dtypes() -> None:
    # grad wraps a staged strict value in an Array API tracer; a NumPy call on
    # it returns a NumPy tracer, which reports NumPy dtypes whatever the provider.
    value = strict.asarray([0.25, -0.5, 2.0], dtype=strict.float64)
    seen: list[object] = []

    def loss(x: Any) -> Any:
        total = np.cumsum(x)
        seen.append(total.dtype)
        return np.sum(total**2)

    program = ad.stage(ad.grad(loss), value)

    assert seen == [np.dtype(np.float64)]
    with pytest.raises(TypeError, match=r"NumPy-authored node 'array\.cumsum' requires NumPy"):
        program(value)


@pytest.mark.parametrize("function", [np.ones_like, np.zeros_like, np.imag])
def test_numpy_constants_under_grad_replay_on_the_staged_provider(
    function: Callable[[Any], Any],
) -> None:
    value = strict.asarray([0.25, -0.5, 2.0], dtype=strict.float64)
    program = ad.stage(ad.grad(lambda x: np.sum(function(x) ** 2)), value)

    gradient = program(value)

    assert gradient.dtype == strict.float64
    np.testing.assert_array_equal(np.asarray(gradient), np.zeros(3))


@pytest.mark.parametrize(
    ("construct", "source"),
    [
        pytest.param(lambda x: np.asarray(x, like=x), strict.float64, id="own-dtype"),
        pytest.param(
            lambda x: np.asarray(x, dtype=np.float64, like=x), strict.float64, id="same-dtype"
        ),
        pytest.param(
            lambda x: np.asarray(x, dtype="float64", like=x), strict.float64, id="same-dtype-name"
        ),
        pytest.param(
            lambda x: np.array(x, dtype=np.float64, copy=False, like=x),
            strict.float64,
            id="same-dtype-without-copy",
        ),
        pytest.param(
            lambda x: np.asarray(x, np.float32, like=x), strict.float64, id="positional-dtype"
        ),
        pytest.param(
            lambda x: np.array(x, dtype=np.float32, like=x), strict.float64, id="array-dtype"
        ),
        pytest.param(
            lambda x: np.asarray(x, dtype=np.float64, like=x), strict.int64, id="int-to-float"
        ),
    ],
)
def test_like_constructors_keep_a_staged_values_provider(
    construct: Callable[[Any], Any], source: object
) -> None:
    value = strict.asarray([1, -2, 3], dtype=source)
    expected = construct(np.asarray(value)) * 2
    program = ad.stage(lambda x: construct(x) * 2, value)

    actual = program(value)

    assert actual.dtype == getattr(strict, expected.dtype.name)
    np.testing.assert_array_equal(np.asarray(actual), expected)
    assert np.asarray(actual).dtype == expected.dtype


@pytest.mark.parametrize("order", ["c", "F", None])
def test_astype_accepts_numpy_memory_orders_in_both_lifetimes(order: str | None) -> None:
    value = np.arange(6.0).reshape(2, 3)
    expected = value.astype(np.float32, order=order)

    def operation(array: Any) -> Any:
        return array.astype(np.float32, order=order)

    dynamic, _tangent = ad.jvp(operation)(value, tangents=np.ones_like(value))
    staged = ad.stage(operation, specs=(ad.ArraySpec((2, 3), "float64"),))(value)

    for actual in (dynamic, staged):
        np.testing.assert_array_equal(actual, expected)
        assert np.asarray(actual).dtype == expected.dtype


_MATRIX = ad.ArraySpec((2, 3), "float64")
_SQUARE = ad.ArraySpec((2, 2), "float64")
_SCALAR = ad.ArraySpec((), "float64")
_VECTOR = ad.ArraySpec((3,), "float64")
_EMPTY = ad.ArraySpec((0,), "float64")
type _Rejection = tuple[type[Exception], str] | None
# Each lifetime's rejection as it stands: (error, match), or None where that
# lifetime accepts the call or owns no such contract. Dynamic tracing often
# wraps NumPy's ValueError or TypeError in TracingError; staging mirrors NumPy.
_REJECTIONS: dict[
    str, tuple[Callable[..., Any], tuple[ad.ArraySpec, ...], _Rejection, _Rejection]
] = {
    "gradient-duplicate-axis": (
        lambda x: np.gradient(x, axis=(0, 0)),
        (_MATRIX,),
        (ad.TracingError, "axis contains duplicates"),
        (ValueError, "invalid axes"),
    ),
    "gradient-axis-bounds": (
        lambda x: np.gradient(x, axis=3),
        (_MATRIX,),
        (ad.TracingError, "out of bounds"),
        (ValueError, "invalid axes 3"),
    ),
    "gradient-axis-type": (
        lambda x: np.gradient(x, axis="rows"),
        (_MATRIX,),
        (ad.TracingError, "Unsupported axis value"),
        None,
    ),
    "gradient-edge-order": (
        lambda x: np.gradient(x, edge_order=3),
        (_MATRIX,),
        (ad.TracingError, "edge_order must be 1 or 2"),
        (ValueError, "edge_order must be 1 or 2"),
    ),
    "gradient-spacing-count": (
        lambda x: np.gradient(x, 1.0, 2.0, 3.0, axis=(0, 1)),
        (_MATRIX,),
        (ad.TracingError, "one spacing per gradient axis"),
        (TypeError, "one spacing per gradient axis"),
    ),
    "gradient-coordinate-length": (
        lambda x: np.gradient(x, np.ones(2), axis=1),
        (_MATRIX,),
        (ValueError, "must be one-dimensional and match axis 1 length 3"),
        (ValueError, "must be one-dimensional and match axis 1 length 3"),
    ),
    "pad-width": (
        lambda x: np.pad(x, (1, 2, 3)),
        (_MATRIX,),
        (ad.TracingError, "Unsupported pad_width shape"),
        None,
    ),
    "pad-unsupported-mode": (
        lambda x: np.pad(x, 1, mode="empty"),
        (_VECTOR,),
        (ad.TracingError, "not differentiable"),
        None,
    ),
    "pad-negative-width": (
        lambda x: np.pad(x, (-1, 1), mode="edge"),
        (_VECTOR,),
        (ValueError, "index can't contain negative values"),
        None,
    ),
    "pad-reflect-type": (
        lambda x: np.pad(x, 1, mode="reflect", reflect_type="neither"),
        (_VECTOR,),
        (ad.TracingError, "reflect_type"),
        None,
    ),
    "pad-empty-edge": (
        lambda x: np.pad(x, 1, mode="edge"),
        (_EMPTY,),
        (ValueError, "can't extend empty axis"),
        None,
    ),
    "pad-static-boundary-shape": (
        lambda x: np.pad(x, ((1, 1), (1, 1)), mode="linear_ramp", end_values=(1.0, 2.0, 3.0)),
        (_MATRIX,),
        (ad.TracingError, "end_values shape"),
        None,
    ),
    "pad-live-boundary-shape": (
        lambda x, edge: np.pad(x, 1, mode="constant", constant_values=edge),
        (_MATRIX, _VECTOR),
        (ad.TracingError, "constant_values shape"),
        None,
    ),
    "pad-stat-length-shape": (
        lambda x: np.pad(x, ((1, 1), (1, 1)), mode="mean", stat_length=(1, 2, 3)),
        (_MATRIX,),
        (ad.TracingError, "stat_length shape"),
        None,
    ),
    "pad-statistical-negative-width": (
        lambda x: np.pad(x, (-1, 1), mode="mean"),
        (_VECTOR,),
        (ValueError, "index can't contain negative values"),
        None,
    ),
    "pad-negative-stat-length": (
        lambda x: np.pad(x, 1, mode="mean", stat_length=-1),
        (_VECTOR,),
        (ValueError, "index can't contain negative values"),
        None,
    ),
    "pad-empty-extremum-region": (
        lambda x: np.pad(x, 1, mode="maximum", stat_length=0),
        (_VECTOR,),
        (ValueError, "stat_length of 0 yields no value"),
        None,
    ),
    "pad-unpadded-empty-extremum-region": (
        lambda x: np.pad(x, ((0, 0), (1, 1)), mode="minimum", stat_length=((0, 0), (1, 1))),
        (_MATRIX,),
        (ValueError, "stat_length of 0 yields no value"),
        None,
    ),
    "pad-empty-mean": (
        lambda x: np.pad(x, 1, mode="mean"),
        (_EMPTY,),
        (ValueError, "can't extend empty axis"),
        None,
    ),
    "compress-live-condition": (
        lambda x, condition: np.compress(condition, x, axis=1),
        (_MATRIX, ad.ArraySpec((3,), "bool")),
        None,
        (ad.TracingError, "data-dependent output shape"),
    ),
    "compress-condition-rank": (
        lambda x: np.compress(np.asarray(1, dtype=bool), x),
        (_MATRIX,),
        (ValueError, "condition must be a 1-d array"),
        (ValueError, "condition must be a 1-d array"),
    ),
    "cumulative-axis": (
        lambda x: np.cumulative_sum(x, include_initial=True),
        (_MATRIX,),
        (ValueError, "more than one dimension ``axis`` argument is required"),
        (ValueError, "more than one dimension ``axis`` argument is required"),
    ),
    "diff-order": (
        lambda x: np.diff(x, n=-1),
        (_MATRIX,),
        (ad.TracingError, "requires n >= 0"),
        (ValueError, "non-negative integer"),
    ),
    "matrix-transpose-rank": (
        np.matrix_transpose,
        (ad.ArraySpec((3,), "float64"),),
        (ValueError, "at least 2-dimensional"),
        (ValueError, "at least two dimensions"),
    ),
    "matrix-power-shape": (
        lambda x: np.linalg.matrix_power(x, 2),
        (_MATRIX,),
        (ad.TracingError, "requires square matrices"),
        (ValueError, "requires square matrices"),
    ),
    "matrix-power-traced-exponent": (
        np.linalg.matrix_power,
        (_SQUARE, _SCALAR),
        (ad.TracingError, "exponent must be a static integer"),
        (TypeError, "exponent must be a static integer"),
    ),
    "matrix-power-traced-integer-exponent": (
        np.linalg.matrix_power,
        (_SQUARE, ad.ArraySpec((), "int64")),
        None,
        (TypeError, "exponent must be a static integer"),
    ),
    "pinv-tolerance": (
        lambda x: np.linalg.pinv(x, rcond=1e-4, rtol=1e-4),
        (_SQUARE,),
        (ad.TracingError, "only one of rcond= and rtol="),
        (TypeError, "only one of rcond= and rtol="),
    ),
    "variance-traced-ddof": (
        lambda x, ddof: np.var(x, ddof=ddof, correction=1),
        (_MATRIX, _SCALAR),
        (ad.TracingError, "requires a static ddof= beside correction="),
        (TypeError, "requires a static ddof= beside correction="),
    ),
    "variance-traced-ddof-where": (
        lambda x, ddof: np.var(x, ddof=ddof, correction=1, where=True),
        (_MATRIX, _SCALAR),
        (ad.TracingError, "requires a static ddof= beside correction="),
        (TypeError, "requires a static ddof= beside correction="),
    ),
    "where-one-argument": (
        lambda x: np.where(x > 0),
        (_MATRIX,),
        (ad.TracingError, "only supported during tracing in its 3-argument form"),
        (TypeError, r"where\(\) requires 3 array operands"),
    ),
    "extrema-where-initial": (
        lambda x: np.max(x, axis=1, where=x > 0),
        (_MATRIX,),
        (ad.TracingError, r"numpy\.max with where=.*requires initial="),
        (TypeError, "with where= requires initial="),
    ),
    "eye-order": (
        lambda x: np.eye(2, order="F", like=x),
        (_MATRIX,),
        None,
        (TypeError, "supports only order='C'"),
    ),
    "eye-device": (
        lambda x: np.eye(2, like=x, device="gpu"),
        (_MATRIX,),
        None,
        (TypeError, "device='cpu'"),
    ),
    "asarray-copy-free-order": (
        lambda x: np.asarray(x, order="F", copy=False, like=x),
        (_MATRIX,),
        (ValueError, "avoid copy"),
        (ad.TracingError, "layout-constraining order"),
    ),
    "copy-order-type": (
        lambda x: x.copy(order=1),
        (_MATRIX,),
        (TypeError, "order must be str"),
        (TypeError, "order must be str"),
    ),
    "copy-order": (
        lambda x: x.copy(order="Z"),
        (_MATRIX,),
        (ValueError, "order must be one of"),
        (ValueError, "order must be one of"),
    ),
    "astype-order": (
        lambda x: x.astype(np.float32, order="Z"),
        (_MATRIX,),
        (ValueError, "order must be one of"),
        (ValueError, "order must be one of"),
    ),
    "astype-casting": (
        lambda x: x.astype(np.float32, casting="banana"),
        (_MATRIX,),
        (ValueError, "casting must be one of"),
        (ValueError, "invalid casting rule"),
    ),
    "astype-safe": (
        lambda x: x.astype(np.int32, casting="safe"),
        (_MATRIX,),
        (TypeError, "according to the rule 'safe'"),
        (TypeError, "according to the 'safe' rule"),
    ),
    "astype-copy": (
        lambda x: x.astype(np.float32, casting="unsafe", copy=1),
        (_MATRIX,),
        None,
        (TypeError, "copy must be a bool"),
    ),
    "astype-subok": (
        lambda x: x.astype(np.float32, subok=1),
        (_MATRIX,),
        None,
        (TypeError, "subok must be a bool"),
    ),
    "like-non-numeric-constant": (
        lambda x: np.asarray(["not", "numeric"], like=x),
        (_VECTOR,),
        (TypeError, "numeric and boolean"),
        (TypeError, "numeric and boolean"),
    ),
    "like-copy-false-dtype": (
        lambda x: np.asarray(x, dtype=np.float32, copy=False, like=x),
        (_VECTOR,),
        (ValueError, "avoid copy"),
        (ValueError, "avoid copy"),
    ),
    "like-negative-ndmin": (
        lambda x: np.array(x, ndmin=-1, like=x),
        (_VECTOR,),
        (ValueError, "ndmin must be non-negative"),
        (ValueError, "ndmin must be non-negative"),
    ),
}


@pytest.mark.parametrize(
    ("operation", "specs", "dynamic", "staged"), _REJECTIONS.values(), ids=_REJECTIONS.keys()
)
def test_each_lifetime_rejects_invalid_public_forms(
    operation: Callable[..., Any],
    specs: tuple[ad.ArraySpec, ...],
    dynamic: _Rejection,
    staged: _Rejection,
) -> None:
    values = tuple(np.ones(spec.shape, dtype=spec.dtype) for spec in specs)
    argnums = tuple(index for index, value in enumerate(values) if value.dtype.kind == "f")
    if dynamic is not None:
        with pytest.raises(dynamic[0], match=dynamic[1]):
            ad.jvp(operation, argnums=argnums)(
                *values, tangents=tuple(np.ones_like(values[index]) for index in argnums)
            )
    if staged is not None:
        with pytest.raises(staged[0], match=staged[1]):
            ad.stage(operation, specs=specs)


def test_dynamic_protocol_rejects_complex_padding_metadata() -> None:
    value = np.arange(6.0).reshape(2, 3)

    with (
        pytest.warns(np.exceptions.ComplexWarning),
        pytest.raises(
            ad.TracingError,
            match="Unsupported constant_values scalar",
        ),
    ):
        ad.jvp(lambda x: np.pad(x, 1, mode="constant", constant_values=1 + 2j))(
            value,
            tangents=np.ones_like(value),
        )
