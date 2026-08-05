"""Narrow contracts for the native store behind staged programs."""

from __future__ import annotations

from advect.core._native import native_build_info


def test_native_build_reports_provenance() -> None:
    info = native_build_info()
    assert info["version"]
    assert info["build_profile"] in {"debug", "release"}
