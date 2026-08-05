"""Shared provenance for generated numerical evidence."""

from __future__ import annotations

import os
import platform

_UNRECORDED_SOURCE = "unrecorded"


def source_revision() -> str:
    """Return the caller-supplied source identity or an explicit sentinel."""
    for variable in ("ADVECT_SOURCE_REVISION", "GITHUB_SHA"):
        if value := os.environ.get(variable, "").strip():
            return value
    return _UNRECORDED_SOURCE


def source_revision_is_recorded(value: object) -> bool:
    """Return whether *value* identifies the measured source state."""
    return isinstance(value, str) and bool(value) and value != _UNRECORDED_SOURCE


def machine_metadata() -> dict[str, str]:
    """Describe the machine closely enough to contextualize numerical evidence."""
    return {
        "node": platform.node(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
    }


def evidence_environment() -> dict[str, object]:
    """Return the provenance fields shared by every generated report."""
    return {
        "source_revision": source_revision(),
        "python": platform.python_version(),
        "machine": machine_metadata(),
    }


def evidence_report_header(*, schema_version: int, report_kind: str) -> dict[str, object]:
    """Return the identity and source environment shared by evidence reports."""
    return {
        "schema_version": schema_version,
        "report_kind": report_kind,
        "environment": evidence_environment(),
    }
