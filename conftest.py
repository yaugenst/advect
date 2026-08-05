"""Pytest configuration for Advect tests."""

from __future__ import annotations

from hypothesis import settings

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
