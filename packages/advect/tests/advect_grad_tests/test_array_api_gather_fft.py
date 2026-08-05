"""Portable gather and Hermitian-FFT derivative contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import array_api_strict as strict
import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad

if TYPE_CHECKING:
    from collections.abc import Callable


def _round_tripped_staged_gradient(
    loss: Callable[..., Any],
    *values: Any,
    argnums: int = 0,
) -> tuple[Any, Any, Any]:
    specs = tuple(ad.ArraySpec(value.shape, value.dtype) for value in values)
    program = ad.stage(loss, specs=specs)
    gradient = ad.grad(program, argnums=argnums)
    restored = ad.StagedProgram.from_dict(gradient.to_dict())
    return (
        ad.grad(loss, argnums=argnums)(*values),
        gradient(*values),
        restored(*values),
    )


@pytest.mark.parametrize("operation", ["take", "take_along_axis"])
def test_gather_gradients_accumulate_duplicate_indices(operation: str) -> None:
    if operation == "take":
        value = strict.asarray(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=strict.float64,
        )
        indices = strict.asarray([2, 0, 2, 1], dtype=strict.int64)
        weights = strict.asarray(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            dtype=strict.float64,
        )
        expected = np.asarray([[2.0, 4.0, 4.0], [6.0, 8.0, 12.0]])

        def loss(argument: Any, index: Any) -> Any:
            namespace = argument.__array_namespace__()
            return namespace.sum(namespace.take(argument, index, axis=1) * weights)

    else:
        value = strict.asarray(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            dtype=strict.float64,
        )
        indices = strict.asarray(
            [[2, 0, 2], [1, 1, 3]],
            dtype=strict.int64,
        )
        weights = strict.asarray(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=strict.float64,
        )
        expected = np.asarray([[2.0, 0.0, 4.0, 0.0], [0.0, 9.0, 0.0, 6.0]])

        def loss(argument: Any, index: Any) -> Any:
            namespace = argument.__array_namespace__()
            gathered = namespace.take_along_axis(argument, index, axis=1)
            return namespace.sum(gathered * weights)

    for actual in _round_tripped_staged_gradient(loss, value, indices):
        assert type(actual) is type(value)
        assert actual.dtype == value.dtype
        assert_allclose(np.asarray(actual), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("operation", ["take", "take_along_axis"])
def test_gather_jvp_gathers_the_source_tangent(operation: str) -> None:
    value = strict.asarray(
        [[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]],
        dtype=strict.float64,
    )
    tangent = strict.asarray(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        dtype=strict.float64,
    )
    if operation == "take":
        indices = strict.asarray([2, 0], dtype=strict.int64)

        def gather(argument: Any) -> Any:
            return argument.__array_namespace__().take(argument, indices, axis=1)

        expected = strict.take(tangent, indices, axis=1)
    else:
        indices = strict.asarray([[2, 0], [1, 1]], dtype=strict.int64)

        def gather(argument: Any) -> Any:
            return argument.__array_namespace__().take_along_axis(
                argument,
                indices,
                axis=1,
            )

        expected = strict.take_along_axis(tangent, indices, axis=1)

    _primal, actual = ad.jvp(gather)(value, tangents=tangent)

    assert_allclose(np.asarray(actual), np.asarray(expected))


def test_take_along_axis_pullback_reduces_broadcast_axes_and_duplicates() -> None:
    value = strict.asarray([[1.0, 2.0, 3.0]], dtype=strict.float64)
    indices = strict.asarray([[0, 0], [2, 0]], dtype=strict.int64)

    def loss(argument: Any) -> Any:
        namespace = argument.__array_namespace__()
        return namespace.sum(namespace.take_along_axis(argument, indices, axis=1))

    for actual in _round_tripped_staged_gradient(loss, value):
        assert_allclose(np.asarray(actual), np.asarray([[3.0, 0.0, 1.0]]))


@pytest.mark.parametrize("operation", ["hfft", "ihfft"])
def test_hermitian_fft_gradients_obey_real_inner_product(operation: str) -> None:
    if operation == "hfft":
        value = strict.asarray(
            [1.0 + 0.0j, 0.5 - 0.25j, -0.75 + 0.4j, 0.2 + 0.0j],
            dtype=strict.complex128,
        )
        tangent = strict.asarray(
            [0.1 + 0.2j, -0.3 + 0.05j, 0.25 - 0.1j, -0.15 + 0.0j],
            dtype=strict.complex128,
        )
        weights = strict.asarray(
            [0.5, -1.0, 1.5, 0.25, -0.75, 2.0],
            dtype=strict.float64,
        )

        def loss(argument: Any) -> Any:
            namespace = argument.__array_namespace__()
            return namespace.sum(namespace.fft.hfft(argument, n=6) * weights)

    else:
        value = strict.asarray(
            [0.25, -0.5, 1.0, 0.75, -0.2, 0.4],
            dtype=strict.float64,
        )
        tangent = strict.asarray(
            [-0.2, 0.1, 0.05, -0.15, 0.3, 0.25],
            dtype=strict.float64,
        )
        weights = strict.asarray(
            [0.5 + 0.25j, -1.0 + 0.5j, 0.75 - 0.2j, 1.25 + 0.0j],
            dtype=strict.complex128,
        )

        def loss(argument: Any) -> Any:
            namespace = argument.__array_namespace__()
            transformed = namespace.fft.ihfft(argument, n=6)
            return namespace.sum(namespace.real(transformed * namespace.conj(weights)))

    dynamic, staged, restored = _round_tripped_staged_gradient(loss, value)
    epsilon = 1e-6
    finite_difference = (
        np.asarray(loss(value + epsilon * tangent)) - np.asarray(loss(value - epsilon * tangent))
    ) / (2 * epsilon)
    directional = np.real(np.vdot(np.asarray(dynamic), np.asarray(tangent)))

    assert_allclose(directional, finite_difference, rtol=2e-8, atol=2e-8)
    assert_allclose(np.asarray(staged), np.asarray(dynamic), rtol=1e-11, atol=1e-11)
    assert_allclose(np.asarray(restored), np.asarray(dynamic), rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("operation", ["hfft", "ihfft"])
def test_hermitian_fft_jvps_cover_odd_lengths_and_ortho_norm(operation: str) -> None:
    if operation == "hfft":
        value = strict.asarray(
            [1.0 + 0.0j, 0.5 - 0.25j, -0.75 + 0.0j],
            dtype=strict.complex128,
        )
        tangent = strict.asarray(
            [0.1 + 0.2j, -0.3 + 0.05j, 0.25 + 0.0j],
            dtype=strict.complex128,
        )

        def transform(argument: Any) -> Any:
            return argument.__array_namespace__().fft.hfft(
                argument,
                n=5,
                norm="ortho",
            )

        expected = strict.fft.hfft(tangent, n=5, norm="ortho")
    else:
        value = strict.asarray([0.25, -0.5, 1.0, 0.75, -0.2], dtype=strict.float64)
        tangent = strict.asarray([-0.2, 0.1, 0.05, -0.15, 0.3], dtype=strict.float64)

        def transform(argument: Any) -> Any:
            return argument.__array_namespace__().fft.ihfft(
                argument,
                n=5,
                norm="ortho",
            )

        expected = strict.fft.ihfft(tangent, n=5, norm="ortho")

    _primal, actual = ad.jvp(transform)(value, tangents=tangent)

    assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-12, atol=1e-12)
