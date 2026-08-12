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


def test_array_function_rejects_duplicate_positional_and_keyword_argument() -> None:
    with pytest.raises(TypeError, match=r"multiple values.*axis"):
        _run_traced(lambda value: np.sum(value, 0, axis=1))


def test_array_function_rejects_concrete_out_destination() -> None:
    with pytest.raises(ad.TracingError, match=r"out=.*one TracedArray"):
        _run_traced(lambda value: np.sum(value, axis=0, out=np.empty(3)))


def test_array_function_reports_invalid_reduction_controls() -> None:
    mask = np.array([[True, False, True], [False, True, True]])

    with pytest.raises(ad.TracingError, match=r"numpy\.max with where=.*requires initial="):
        _run_traced(lambda value: np.max(value, axis=0, where=mask))


def test_array_function_rejects_traced_static_control() -> None:
    xp = np.array([0.0, 5.0])
    fp = np.array([0.0, 1.0])

    with pytest.raises(ad.TracingError, match=r"period=.*must be static"):
        _run_traced(lambda value: np.interp(value, xp, fp, period=value[0, 0]))
