"""Tests for import-time NumPy frontend registration."""

from __future__ import annotations

import advect.numpy as advect_numpy
from advect.core._backends import get_hook
from advect.numpy._abstract_protocol import _NumpyAbstractArray


def test_numpy_frontend_registration_is_import_time_only() -> None:
    assert "register_backend" not in vars(advect_numpy)
    assert get_hook("advect.abstract_array_factory") is _NumpyAbstractArray
    assert get_hook("advect.abstract_array_method") is None
    assert get_hook("advect.stage_context") is None
    assert {
        "__array_function__",
        "__array_ufunc__",
        "_advect_stage_context",
        "astype",
        "copy",
        "mean",
        "sum",
    } <= vars(_NumpyAbstractArray).keys()
