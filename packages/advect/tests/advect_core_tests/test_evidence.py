"""Contracts shared by generated numerical evidence."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from scripts._support import evidence

if TYPE_CHECKING:
    import pytest


def test_source_revision_prefers_explicit_identity_then_ci(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ADVECT_SOURCE_REVISION", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    assert evidence.source_revision() == "unrecorded"

    monkeypatch.setenv("GITHUB_SHA", "ci-revision")
    assert evidence.source_revision() == "ci-revision"

    monkeypatch.setenv("ADVECT_SOURCE_REVISION", "explicit-source-state")
    assert evidence.source_revision() == "explicit-source-state"


def test_evidence_environment_has_common_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ADVECT_SOURCE_REVISION", "source-state")

    environment = evidence.evidence_environment()

    assert environment["source_revision"] == "source-state"
    assert environment["python"]
    assert set(environment["machine"]) == {
        "architecture",
        "node",
        "platform",
        "processor",
    }


def test_evidence_report_header_has_a_shared_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ADVECT_SOURCE_REVISION", "source-state")

    header = evidence.evidence_report_header(
        schema_version=2,
        report_kind="advect.example",
    )

    assert header["schema_version"] == 2
    assert header["report_kind"] == "advect.example"
    assert header["environment"]["source_revision"] == "source-state"


def test_official_suite_report_uses_the_shared_evidence_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_array_api_conformance  # noqa: PLC0415

    monkeypatch.setenv("ADVECT_SOURCE_REVISION", "source-state")

    report = run_array_api_conformance._build_report(
        arguments=argparse.Namespace(
            array_api_version="2023.12",
            max_examples=10,
            shard_count=1,
            shard_index=0,
        ),
        baseline={"passed_test_count": 1},
        collected_test_count=1,
        results=[{"passed": True}],
        signature_snapshot={"verified": True},
        suite_revision="suite-state",
        test_count=1,
    )

    assert report["schema_version"] == 1
    assert report["report_kind"] == "advect.array-api-official-suite"
    assert report["environment"]["source_revision"] == "source-state"
    assert report["suite_revision"] == "suite-state"
    assert report["passed"] is True
