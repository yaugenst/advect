"""Smoke contract for the SciPy runtime evidence matrix."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts import bench_scipy_runtime as benchmark

from advect.scipy import ndimage

if TYPE_CHECKING:
    import pytest


def test_every_ndimage_and_special_case_runs_with_staged_dynamic_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ADVECT_SOURCE_REVISION", "source-state")
    config = benchmark._Config(
        warmup=1, rounds=1, target_ms=0.1, max_block_size=1, derivative_limit=8.0
    )

    report = benchmark._report(8, config)

    assert report["schema_version"] == 1
    assert report["report_kind"] == "advect.scipy-runtime"
    assert report["environment"]["source_revision"] == "source-state"
    names = {result["name"] for result in report["results"]}
    assert set(ndimage.__all__) <= names
    assert len(names) == len(report["results"])
    assert all(result["gradient_graph_nodes"] > 0 for result in report["results"])
