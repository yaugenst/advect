"""Smoke-test Advect's documented gradient in a real Pyodide runtime."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

import advect as ad

if TYPE_CHECKING:
    import numpy.typing as npt


def _require(*, condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    """Verify the browser runtime, NumPy profile, and gradient result."""
    _require(
        condition=sys.platform == "emscripten",
        message=f"Expected an Emscripten runtime, found {sys.platform!r}",
    )
    _require(
        condition=np.__version__.split(".")[:2] == ["2", "4"],
        message=f"Expected NumPy 2.4 in Pyodide, found {np.__version__}",
    )

    def function(value: npt.ArrayLike) -> npt.NDArray[np.floating]:
        return np.sin(value) ** 2

    value = np.asarray(0.55)
    derivative = np.asarray(ad.grad(function)(value))
    expected = 2 * np.sin(value) * np.cos(value)
    np.testing.assert_allclose(derivative, expected)


if __name__ == "__main__":
    main()
