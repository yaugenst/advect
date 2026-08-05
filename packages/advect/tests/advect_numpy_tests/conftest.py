"""Pytest setup for Advect's NumPy frontend tests."""

from __future__ import annotations

import importlib

# Ensure NumPy hooks are registered before test-module constants are built.
importlib.import_module("advect.numpy")
