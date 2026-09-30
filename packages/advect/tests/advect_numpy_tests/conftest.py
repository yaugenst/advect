"""Pytest setup for Advect's NumPy frontend tests."""

from __future__ import annotations

import importlib

import pytest

# Report helper assertion operands the way test-module assertions report them.
pytest.register_assert_rewrite("advect_numpy_tests._assertions")
# Ensure NumPy hooks are registered before test-module constants are built.
importlib.import_module("advect.numpy")
