"""Focused contracts for the non-gating Advect/HIPS comparison."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts._support import bench_autodiff_runtime as benchmark

if TYPE_CHECKING:
    import pytest


def _result(name: str, group: str, ratio: float) -> dict[str, object]:
    return {
        "name": name,
        "group": group,
        "advect_over_hips": ratio,
        "advect": {"median_us": 10.0 * ratio},
        "hips_autograd": {"median_us": 10.0},
    }


def _result_with_phases(name: str, ratio: float) -> dict[str, object]:
    result = _result(name, "core", ratio)
    result["phases"] = {
        "linearize_and_release": _result("linearize_and_release", "core", ratio),
        "reused_reverse": _result("reused_reverse", "core", ratio * 2),
    }
    return result


def _payload(ratio: float) -> dict[str, object]:
    return {
        "workloads": [
            _result("elementwise", "core", ratio),
            _result("stencil", "core", ratio),
        ],
        "geometric_mean_advect_over_hips": ratio,
        "group_geometric_means_advect_over_hips": {"core": ratio},
    }


def test_reused_reverse_phase_keeps_one_linearization_alive() -> None:
    workload = benchmark._core_workloads(4)[0]

    result = benchmark._measure_phases(
        workload,
        warmup=1,
        rounds=1,
        block_size=2,
    )

    assert set(result) == {"linearize_and_release", "reused_reverse"}
    for phase in result.values():
        assert phase["advect"]["median_us"] > 0
        assert phase["hips_autograd"]["median_us"] > 0


def test_timing_summary_reports_only_a_consumed_statistic() -> None:
    assert benchmark._summary([1.0, 9.0, 2.0]) == {"median_us": 2.0}


def test_warmed_replicates_use_medians_without_claiming_process_isolation() -> None:
    payload = benchmark._aggregate_warmed_replicates([_payload(1.0), _payload(1.1), _payload(4.0)])

    assert payload["geometric_mean_advect_over_hips"] == 1.1
    assert payload["workloads"][0]["advect_over_hips"] == 1.1
    assert payload["warmed_replicates"] == {
        "count": 3,
        "process_isolation": False,
        "runs": [
            {
                "geometric_mean_advect_over_hips": 1.0,
                "workload_ratios_advect_over_hips": {
                    "elementwise": 1.0,
                    "stencil": 1.0,
                },
            },
            {
                "geometric_mean_advect_over_hips": 1.1,
                "workload_ratios_advect_over_hips": {
                    "elementwise": 1.1,
                    "stencil": 1.1,
                },
            },
            {
                "geometric_mean_advect_over_hips": 4.0,
                "workload_ratios_advect_over_hips": {
                    "elementwise": 4.0,
                    "stencil": 4.0,
                },
            },
        ],
    }


def test_warmed_replicates_aggregate_phase_medians() -> None:
    payloads = [
        {
            "workloads": [_result_with_phases("elementwise", ratio)],
            "geometric_mean_advect_over_hips": ratio,
            "group_geometric_means_advect_over_hips": {"core": ratio},
        }
        for ratio in (1.0, 1.5, 4.0)
    ]

    payload = benchmark._aggregate_warmed_replicates(payloads)

    phases = payload["workloads"][0]["phases"]
    assert phases["linearize_and_release"]["advect_over_hips"] == 1.5
    assert phases["reused_reverse"]["advect_over_hips"] == 3.0


def test_text_report_calls_the_hips_comparison_non_gating(
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = benchmark._aggregate_warmed_replicates([_payload(1.0), _payload(1.1)])
    payload.update(
        {
            "size": 4,
            "rounds": 1,
            "block_size": 1,
            "suite": "core",
            "environment": {"advect_native": {"version": "test", "build_profile": "release"}},
        }
    )

    benchmark._print_text(payload)

    rendered = capsys.readouterr().out
    assert "comparison=non-gating" in rendered
    assert "warmed same-process replicates" in rendered
    assert "acceptance" not in rendered
