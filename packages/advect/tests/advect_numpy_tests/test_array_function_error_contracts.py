"""Public error contracts for NumPy array-function tracing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import advect as ad

if TYPE_CHECKING:
    from collections.abc import Callable


def _run_traced(function: Callable[[Any], Any]) -> None:
    value = np.arange(6.0).reshape(2, 3)
    ad.jvp(function)(value, tangents=np.ones_like(value))


def test_array_function_rejects_unknown_handler_keyword() -> None:
    with pytest.raises(ad.TracingError, match=r"kwargs are not supported.*order"):
        _run_traced(lambda value: np.clip(value, 0.0, 1.0, order="K"))


# NumPy's dispatcher rejects each malformed call before the tracer sees it.
_PREEMPTED_CALLS: dict[str, Callable[[Any], Any]] = {
    "sum": lambda value: np.sum(value, 0, axis=1),
    "polyfit": lambda value: np.polyfit(value[0], value[1], 1, None, rcond=None),
    "median": lambda value: np.median(value, 0, axis=0),
    "repeat": lambda value: np.repeat(value, 2, repeats=3),
    "unique": lambda value: np.unique(value, True, return_index=True),  # noqa: FBT003
    "fft": lambda value: np.fft.fft(value, 4, n=4),
    "convolve": lambda value: np.convolve(value[0], value[1], "full", mode="full"),
    "zeros_like": lambda value: np.zeros_like(value, float, dtype=float),
    "polyval": lambda value: np.polyval(value[0], 1.0, extra=1),
    "argsort": lambda value: np.argsort(value, 0, axis=0),
    "roll": lambda value: np.roll(value, 1, shift=1),
}


@pytest.mark.parametrize("call", _PREEMPTED_CALLS.values(), ids=list(_PREEMPTED_CALLS))
def test_numpy_binds_the_public_signature_before_the_tracer(call: Callable[[Any], Any]) -> None:
    with pytest.raises(TypeError):
        _run_traced(call)


def test_array_function_rejects_concrete_out_destination() -> None:
    with pytest.raises(ad.TracingError, match=r"out=.*one TracedArray"):
        _run_traced(lambda value: np.sum(value, axis=0, out=np.empty(3)))


@pytest.mark.parametrize(
    "fill",
    [
        pytest.param(lambda value: np.full_like(np.ones(3), value), id="full_like"),
        pytest.param(lambda value: np.full(3, value, dtype=np.float64), id="full"),
        pytest.param(lambda value: np.copyto(np.ones(3), value), id="copyto"),
    ],
)
def test_a_traced_fill_of_an_untraced_array_names_a_traceable_spelling(
    fill: Callable[[Any], Any],
) -> None:
    # NumPy fills an untraced array through copyto, which cannot hold the traced
    # fill value. full_like dispatches on its template alone, and full on none.
    template = np.ones(3)

    with pytest.raises(ad.TracingError, match=r"untraced array.*broadcast_to\(value, shape\)"):
        ad.grad(lambda value: np.sum(fill(value)))(0.5)
    gradient = ad.grad(
        lambda fill: np.sum(np.broadcast_to(fill, template.shape).astype(template.dtype))
    )(0.5)
    assert gradient == 3.0


def test_array_function_rejects_traced_static_control() -> None:
    xp = np.array([0.0, 5.0])
    fp = np.array([0.0, 1.0])

    with pytest.raises(ad.TracingError, match=r"period=.*must be static"):
        _run_traced(lambda value: np.interp(value, xp, fp, period=value[0, 0]))
