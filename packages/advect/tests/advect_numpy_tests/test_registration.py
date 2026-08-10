"""Tests for import-time NumPy frontend registration."""

from __future__ import annotations

from advect.core._backends import get_hook
from advect.numpy._abstract_protocol import _NumpyAbstractArray


def test_numpy_frontend_registers_abstract_array_factory_on_import() -> None:
    assert get_hook("advect.abstract_array_factory") is _NumpyAbstractArray
