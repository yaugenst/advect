"""End-to-end contracts at NumPy's dynamic and staged protocol boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import advect as ad

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

    if isinstance(primal, tuple):
        for actual, reference in zip(primal, expected_primal, strict=True):
            np.testing.assert_allclose(actual, reference)
        for actual, reference in zip(tangent, tangent_reference, strict=True):
            np.testing.assert_allclose(actual, reference)
    else:
        np.testing.assert_allclose(primal, expected_primal)
        np.testing.assert_allclose(tangent, tangent_reference)


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

    primal, tangent = ad.jvp(aliases)(value, tangents=direction)
    expected_primal = aliases(value)
    epsilon = 1e-3
    expected_tangent = tuple(
        (upper - lower) / (2 * epsilon)
        for upper, lower in zip(
            aliases(value + epsilon * direction),
            aliases(value - epsilon * direction),
            strict=True,
        )
    )

    for actual, expected in zip(primal, expected_primal, strict=True):
        np.testing.assert_allclose(actual, expected)
    for actual, expected in zip(tangent, expected_tangent, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("operation", [np.cumulative_sum, np.cumulative_prod])
def test_cumulative_aliases_include_the_initial_identity_dynamically(
    operation: Callable[..., Any],
) -> None:
    value = np.arange(1.0, 7.0).reshape(2, 3)
    direction = np.full_like(value, 0.1)

    def cumulative(x: Any) -> Any:
        return operation(x, axis=-1, dtype=np.float64, include_initial=True)

    primal, tangent = ad.jvp(cumulative)(value, tangents=direction)
    epsilon = 1e-6

    np.testing.assert_allclose(primal, cumulative(value))
    np.testing.assert_allclose(
        tangent,
        (cumulative(value + epsilon * direction) - cumulative(value - epsilon * direction))
        / (2 * epsilon),
        rtol=2e-6,
        atol=2e-6,
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
        (
            lambda x: np.cumulative_sum(x, include_initial=True),
            ValueError,
            "axis.*required",
        ),
    ],
    ids=["astype-device", "astype-alias", "cumulative-axis"],
)
def test_numpy_aliases_reject_nonportable_runtime_contracts(
    operation: Callable[[Any], Any],
    error: type[Exception],
    match: str,
) -> None:
    value = np.arange(6.0).reshape(2, 3)

    with pytest.raises(error, match=match):
        ad.jvp(operation)(value, tangents=np.ones_like(value))


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

    program = ad.stage(aliases, specs=(ad.ArraySpec(value.shape, value.dtype),))
    restored = ad.StagedProgram.from_dict(program.to_dict())
    expected = aliases(value)

    for staged in (program, restored):
        for actual, reference in zip(staged(value), expected, strict=True):
            np.testing.assert_allclose(actual, reference)


def test_staged_diff_accepts_live_prepend_and_append_operands() -> None:
    value = np.arange(6.0).reshape(2, 3)
    prepend = np.asarray([[10.0], [20.0]])
    append = np.asarray([[30.0], [40.0]])

    def difference(x: Any, before: Any, after: Any) -> Any:
        return np.diff(x, n=2, axis=1, prepend=before, append=after)

    directions = (
        np.full_like(value, 0.1),
        np.full_like(prepend, 0.2),
        np.full_like(append, -0.3),
    )
    primal, tangent = ad.jvp(difference, argnums=(0, 1, 2))(
        value,
        prepend,
        append,
        tangents=directions,
    )
    np.testing.assert_allclose(primal, difference(value, prepend, append))
    np.testing.assert_allclose(tangent, difference(*directions))

    program = ad.stage(
        difference,
        specs=tuple(ad.ArraySpec(item.shape, item.dtype) for item in (value, prepend, append)),
    )
    expected = difference(value, prepend, append)

    np.testing.assert_allclose(program(value, prepend, append), expected)
    restored = ad.StagedProgram.from_dict(program.to_dict())
    np.testing.assert_allclose(restored(value, prepend, append), expected)


def test_dynamic_full_differentiates_fill_and_not_like_dispatch_anchor() -> None:
    anchor = np.zeros(2)
    fill = np.asarray(3.0)

    def filled(like: Any, value: Any) -> Any:
        return np.full(
            (2, 2),
            value,
            dtype=np.float64,
            order="F",
            device="cpu",
            like=like,
        )

    primal, tangent = ad.jvp(filled, argnums=(0, 1))(
        anchor,
        fill,
        tangents=(np.ones_like(anchor), np.asarray(0.5)),
    )
    np.testing.assert_allclose(primal, np.full((2, 2), fill))
    np.testing.assert_allclose(tangent, np.full((2, 2), 0.5))


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


def test_debug_mode_uses_the_full_ufunc_protocol_without_changing_results() -> None:
    value = np.asarray([0.2, -0.4, 0.7])
    direction = np.asarray([0.3, 0.1, -0.2])

    with ad.debug():
        primal, tangent = ad.jvp(np.sin)(value, tangents=direction)

    np.testing.assert_allclose(primal, np.sin(value))
    np.testing.assert_allclose(tangent, np.cos(value) * direction)


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
    ("order", "error", "match"),
    [(1, TypeError, "order must be str"), ("Z", ValueError, "order must be one of")],
)
def test_array_copy_rejects_invalid_memory_orders_in_both_lifetimes(
    order: object,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        ad.jvp(lambda x: x.copy(order=order))(
            np.arange(4.0),
            tangents=np.ones(4),
        )
    with pytest.raises(error, match=match):
        ad.stage(
            lambda x: x.copy(order=order),
            specs=(ad.ArraySpec((2, 3), "float64"),),
        )


@pytest.mark.parametrize(
    ("operation", "specs", "error", "match"),
    [
        (
            lambda x: np.gradient(x, axis=(0, 0)),
            (ad.ArraySpec((2, 3), "float64"),),
            ValueError,
            "invalid axes",
        ),
        (
            lambda x: np.gradient(x, edge_order=3),
            (ad.ArraySpec((2, 3), "float64"),),
            ValueError,
            "edge_order must be 1 or 2",
        ),
        (
            lambda x: np.gradient(x, 1.0, 2.0, 3.0, axis=(0, 1)),
            (ad.ArraySpec((2, 3), "float64"),),
            TypeError,
            "one spacing per gradient axis",
        ),
        (
            lambda x, condition: np.compress(condition, x, axis=1),
            (ad.ArraySpec((2, 3), "float64"), ad.ArraySpec((3,), "bool")),
            ad.TracingError,
            "data-dependent output shape",
        ),
        (
            lambda x: np.cumulative_sum(x, include_initial=True),
            (ad.ArraySpec((2, 3), "float64"),),
            ValueError,
            "axis=",
        ),
        (
            np.matrix_transpose,
            (ad.ArraySpec((3,), "float64"),),
            ValueError,
            "at least two dimensions",
        ),
        (
            np.linalg.matrix_power,
            (ad.ArraySpec((2, 2), "float64"), ad.ArraySpec((), "int64")),
            TypeError,
            "static integer",
        ),
        (
            lambda x: x.astype(np.float32, order="Z"),
            (ad.ArraySpec((2, 3), "float64"),),
            ValueError,
            "invalid order",
        ),
        (
            lambda x: x.astype(np.float32, casting="unsafe", copy=1),
            (ad.ArraySpec((2, 3), "float64"),),
            TypeError,
            "copy must be a bool",
        ),
        (
            lambda x: np.eye(2, like=x, device="gpu"),
            (ad.ArraySpec((2, 3), "float64"),),
            TypeError,
            "device='cpu'",
        ),
        (
            lambda x: np.linalg.pinv(x, rcond=1e-4, rtol=1e-4),
            (ad.ArraySpec((2, 2), "float64"),),
            TypeError,
            "only one of rcond= and rtol=",
        ),
    ],
    ids=[
        "gradient-duplicate-axis",
        "gradient-edge-order",
        "gradient-spacing-count",
        "compress-live-condition",
        "cumulative-axis",
        "matrix-transpose-rank",
        "matrix-power-static-exponent",
        "astype-order",
        "astype-copy",
        "eye-device",
        "pinv-tolerance",
    ],
)
def test_staging_rejects_invalid_or_runtime_dependent_numpy_contracts(
    operation: Callable[..., Any],
    specs: tuple[ad.ArraySpec, ...],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        ad.stage(operation, specs=specs)


@pytest.mark.parametrize(
    ("operation", "match"),
    [
        (lambda x: np.pad(x, (1, 2, 3)), "Unsupported pad_width shape"),
        (lambda x: np.gradient(x, axis="rows"), "Unsupported axis value"),
        (lambda x: np.gradient(x, axis=(0, 0)), "axis contains duplicates"),
        (lambda x: np.gradient(x, axis=3), "out of bounds"),
    ],
    ids=["pad-width", "axis-type", "duplicate-axis", "axis-bounds"],
)
def test_dynamic_protocol_rejects_invalid_normalized_metadata(
    operation: Callable[[Any], Any],
    match: str,
) -> None:
    value = np.arange(6.0).reshape(2, 3)

    with pytest.raises(ad.TracingError, match=match):
        ad.jvp(operation)(value, tangents=np.ones_like(value))


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
