"""Pytest configuration for Advect tests."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest
from hypothesis import settings

if TYPE_CHECKING:
    from collections.abc import Iterator

_BUILTIN_DEFAULT = settings.get_profile("default")

settings.register_profile(
    "advect",
    parent=_BUILTIN_DEFAULT,
    max_examples=100,
    deadline=500,
)
# The conformance matrix searches a large space; the Advect profile keeps the
# whole suite in seconds, while CI's `--hypothesis-profile=thorough` run is the
# deep search. Local failures replay through the `.hypothesis` example database;
# CI discoveries must become explicit examples or focused regression tests.
settings.register_profile(
    "thorough",
    parent=_BUILTIN_DEFAULT,
    max_examples=1000,
    deadline=None,
)
settings.load_profile("advect")


@pytest.fixture(autouse=True)
def _restore_array_api_strict_flags() -> Iterator[None]:
    """Keep the reference provider's process-global revision test-local.

    ``array-api-strict`` switches its global flags whenever a namespace for
    another revision is requested, so one test could otherwise change which
    functions the next test may call.
    """
    strict = sys.modules.get("array_api_strict")
    flags = None if strict is None else strict.get_array_api_strict_flags()
    yield
    if strict is not None and strict.get_array_api_strict_flags() != flags:
        strict.set_array_api_strict_flags(**flags)
