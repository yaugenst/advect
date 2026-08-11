"""Focused tests for the isolated runtime-memory benchmark."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
from scripts import bench_runtime_memory as benchmark


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1024", 1024),
        ("1 KiB", 1024),
        ("1.5MiB", 1_572_864),
        ("2_gib", 2 * 1024**3),
    ],
)
def test_parse_byte_size(raw: str, expected: int) -> None:
    assert benchmark._parse_byte_size(raw) == expected


def test_byte_budget_sizing_respects_factor_and_cap() -> None:
    assert (
        benchmark._elements_for_budget(
            64 * 1024**2,
            live_array_factor=4,
            max_bytes=128 * 1024**2,
        )
        == 2_097_152
    )
    with pytest.raises(ValueError, match="exceeds explicit cap"):
        benchmark._elements_for_budget(
            129 * 1024**2,
            live_array_factor=4,
            max_bytes=128 * 1024**2,
        )


def test_unique_provider_bytes_deduplicate_numpy_views() -> None:
    owner = np.arange(128, dtype=np.float64)
    first = owner[::2]
    second = owner[1::2]

    assert sum(benchmark._buffer_map((owner, first, second)).values()) == owner.nbytes


def test_named_acceptance_profiles_own_exact_case_matrices() -> None:
    cpu = benchmark._ACCEPTANCE_PROFILES["cpu-runtime"]
    cupy = benchmark._ACCEPTANCE_PROFILES["cupy-donation"]

    assert [case.name for case in cpu.cases] == [
        "allocation_probe:python:allocate:host",
        "elementwise:advect:dynamic:numpy",
        "stencil:advect:dynamic:numpy",
        "checkpoint:advect:plain:numpy",
        "checkpoint:advect:checkpoint:numpy",
        "residual:advect:retained:numpy",
        "linear_map:advect:reusable:numpy",
        "captured_constant:advect:staged:numpy",
    ]
    assert [case.name for case in cupy.cases] == [
        "allocation_probe:python:allocate:host",
        "functional_updates:advect:donation:cupy",
        "functional_updates:advect:forced_fresh:cupy",
    ]
    assert cpu.timed_cases == (
        ("checkpoint", "plain"),
        ("checkpoint", "checkpoint"),
    )
    assert cupy.timed_cases == (
        ("functional_updates", "donation"),
        ("functional_updates", "forced_fresh"),
    )


def test_every_profile_advect_case_receives_a_correctness_preflight() -> None:
    cases = benchmark._ACCEPTANCE_PROFILES["cpu-runtime"].cases

    assert [case.name for case in benchmark._correctness_preflight_cases(cases)] == [
        "elementwise:advect:dynamic:numpy",
        "stencil:advect:dynamic:numpy",
        "checkpoint:advect:plain:numpy",
        "checkpoint:advect:checkpoint:numpy",
        "residual:advect:retained:numpy",
        "linear_map:advect:reusable:numpy",
        "captured_constant:advect:staged:numpy",
    ]


def test_cpu_donation_gate_is_explicitly_not_applicable() -> None:
    case: dict[str, object] = {
        "memory": {
            "summary": dict[str, object](),
            "runs": list[object](),
        },
        "timing": {
            "summary_seconds_per_call": dict[str, float](),
            "runs": list[object](),
        },
    }

    check = benchmark._donation_acceptance_check(case, case, provider="numpy")

    assert check["status"] == "not_applicable"
    assert "CuPy" in check["reason"]


def test_checkpoint_gate_uses_paired_memory_and_timing() -> None:
    def case(rss: float, runtime: float) -> dict[str, object]:
        return {
            "memory": {
                "summary": {
                    "peak_rss_delta_bytes": {"median": rss},
                },
                "runs": [],
            },
            "timing": {
                "summary_seconds_per_call": {"median": runtime},
                "runs": [],
            },
        }

    passed = benchmark._checkpoint_acceptance_check(
        case(100.0, 1.0),
        case(70.0, 1.3),
    )
    failed = benchmark._checkpoint_acceptance_check(
        case(100.0, 1.0),
        case(90.0, 1.1),
    )

    assert passed["status"] == "passed"
    assert failed["status"] == "failed"


def test_cupy_donation_gate_uses_reserved_pool_high_water() -> None:
    def case(
        *,
        reserved: float,
        used: float,
        runtime: float,
        input_bytes: int,
    ) -> dict[str, object]:
        return {
            "memory": {
                "summary": {
                    "peak_provider_pool_reserved_delta_bytes": {"median": reserved},
                    "peak_provider_pool_used_delta_bytes": {"median": used},
                    "peak_device_used_delta_bytes": {"median": reserved},
                },
                "runs": [{"input_bytes": input_bytes}],
            },
            "timing": {
                "summary_seconds_per_call": {"median": runtime},
                "runs": [],
            },
        }

    donated = case(reserved=80.0, used=100.0, runtime=1.04, input_bytes=20)
    forced_fresh = case(reserved=100.0, used=100.0, runtime=1.0, input_bytes=20)

    check = benchmark._donation_acceptance_check(
        donated,
        forced_fresh,
        provider="cupy",
    )

    assert check["status"] == "passed"
    assert check["donation_over_forced_fresh_provider_memory"] == pytest.approx(0.8)


def test_cupy_acceptance_uses_stable_device_pool_variation() -> None:
    memory_runs = [
        {
            "status": "ok",
            "peak_rss_delta_bytes": 10,
            "input_bytes": 8,
            "markers": [
                {
                    "phase": "baseline",
                    "provider_pool_reserved_bytes": 100,
                },
                {
                    "phase": "closed",
                    "provider_pool_reserved_bytes": 110,
                    "provider_owned_live_bytes": 0,
                },
            ],
            "environment": {
                "advect_native": {"build_profile": "release"},
                "advect_debug": False,
            },
        }
        for _ in range(5)
    ]
    timing_runs = [
        {
            "status": "ok",
            "seconds_per_call": 1.0,
            "environment": {
                "advect_native": {"build_profile": "release"},
                "advect_debug": False,
            },
        }
        for _ in range(5)
    ]
    case = {
        "name": "functional_updates:advect:donation:cupy",
        "case": {
            "framework": "advect",
            "mode": "donation",
            "provider": "cupy",
            "workload": "functional_updates",
        },
        "status": "ok",
        "memory": {
            "summary": {
                "peak_rss_delta_bytes": {
                    "median": 10.0,
                    "variation": 0.2,
                    "median_absolute_variation": 0.2,
                },
                "peak_provider_pool_reserved_delta_bytes": {
                    "median": 10.0,
                    "variation": 0.0,
                    "median_absolute_variation": 0.0,
                },
            },
            "runs": memory_runs,
        },
        "timing": {
            "summary_seconds_per_call": {
                "median": 1.0,
                "median_absolute_variation": 0.0,
            },
            "runs": timing_runs,
        },
    }
    payload = {
        "config": {
            "profile": "cupy-donation",
            "runs": 5,
            "timing_runs": 5,
            "timing_iterations": 10,
            "no_timing": False,
            "byte_budget": 64 * 1024**2,
        },
        "environment": {"source_revision": "test-source"},
        "correctness_preflights": [],
        "cases": [case],
    }

    violations = benchmark._acceptance_violations(
        payload,
        requested=True,
        gated_checks=(),
    )

    assert not any("variation" in violation for violation in violations)


def test_acceptance_requires_every_profile_worker_to_succeed() -> None:
    runs = [{"status": "ok", "peak_rss_delta_bytes": 64 * 1024**2} for _ in range(4)]
    runs.append({"status": "skipped"})
    case = {
        "name": "allocation_probe:python:allocate:host",
        "case": {
            "framework": "python",
            "mode": "allocate",
            "provider": "host",
            "workload": "allocation_probe",
        },
        "memory": {"runs": runs},
        "timing": {"runs": []},
    }

    violations = benchmark._run_contract_violations(
        case,
        profile=benchmark._ACCEPTANCE_PROFILES["cpu-runtime"],
    )

    assert violations == [
        (
            "allocation_probe:python:allocate:host requires exactly 5 successful memory runs; "
            "recorded=5, successful=4"
        )
    ]


def test_acceptance_requires_each_lifetime_metric_in_each_run() -> None:
    runs = [
        {
            "status": "ok",
            "peak_rss_delta_bytes": 1,
            "markers": [
                {"phase": "closed", "provider_owned_live_bytes": 0},
                {"phase": "reverse_retained", "provider_owned_live_bytes": 1},
            ],
        }
        for _ in range(5)
    ]
    runs[-1]["markers"] = [{"phase": "closed", "provider_owned_live_bytes": 0}]
    case = {
        "name": "linear_map:advect:reusable:numpy",
        "case": {
            "framework": "advect",
            "mode": "reusable",
            "provider": "numpy",
            "workload": "linear_map",
        },
        "memory": {"runs": runs},
        "timing": {"runs": []},
    }

    violations = benchmark._run_contract_violations(
        case,
        profile=benchmark._ACCEPTANCE_PROFILES["cpu-runtime"],
    )

    assert violations == [
        (
            "linear_map:advect:reusable:numpy memory run 5 is missing "
            "reverse_retained.provider_owned_live_bytes"
        )
    ]


def test_memory_stability_uses_robust_median_absolute_variation() -> None:
    summary = benchmark._numeric_summary([99.0, 100.0, 100.0, 101.0, 150.0])

    assert summary["variation"] == pytest.approx(0.51)
    assert summary["median_absolute_deviation"] == pytest.approx(1.0)
    assert summary["median_absolute_variation"] == pytest.approx(0.01)


def test_acceptance_rejects_unrecorded_source_revision() -> None:
    payload = {
        "config": {
            "profile": "cpu-runtime",
            "runs": 5,
            "timing_runs": 5,
            "timing_iterations": 10,
            "no_timing": False,
            "byte_budget": 64 * 1024**2,
        },
        "environment": {"source_revision": "unrecorded"},
        "correctness_preflights": [],
        "cases": [],
    }

    violations = benchmark._acceptance_violations(
        payload,
        requested=True,
        gated_checks=(),
    )

    assert (
        "source revision is unrecorded; set ADVECT_SOURCE_REVISION to the exact source state"
        in violations
    )


def test_provider_cache_is_reported_but_excluded_from_owned_bytes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    caller = np.ones(32, dtype=np.float64)
    provider_cache = np.ones(64, dtype=np.float64)
    reporter = benchmark._Reporter(np, hold_seconds=0.0)
    reporter.start(roots=(caller,))

    reporter.mark(
        "closed",
        roots=(caller,),
        excluded_roots=(caller,),
        provider_cache_roots=(provider_cache,),
    )
    reporter.stop()

    marker = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert marker["provider_owned_live_bytes"] == 0
    assert marker["provider_cache_live_bytes"] == provider_cache.nbytes


@pytest.mark.skipif(sys.platform != "linux", reason="parent RSS sampling uses Linux procfs")
def test_allocation_probe_runs_in_an_isolated_child() -> None:
    case = benchmark._Case("allocation_probe", "python", "allocate", "host")
    spec = benchmark._WorkerSpec(
        case=case,
        byte_budget=8 * 1024**2,
        max_bytes=16 * 1024**2,
        sample_hold_seconds=0.01,
        measurement="memory",
        timing_iterations=1,
    )

    result = benchmark._run_isolated(
        spec,
        sample_interval_seconds=0.001,
        include_samples=True,
    )

    assert result["status"] == "ok"
    assert result["peak_rss_delta_bytes"] >= 6 * 1024**2
    assert len(result["rss_samples"]) > 1
    phases = [marker["phase"] for marker in result["markers"]]
    assert phases == ["baseline", "allocated", "closed"]


@pytest.mark.skipif(sys.platform != "linux", reason="parent RSS sampling uses Linux procfs")
def test_partitioned_checkpoint_reports_four_recomputations() -> None:
    case = benchmark._Case("checkpoint", "advect", "checkpoint", "numpy")
    spec = benchmark._WorkerSpec(
        case=case,
        byte_budget=2 * 1024**2,
        max_bytes=8 * 1024**2,
        sample_hold_seconds=0.001,
        measurement="memory",
        timing_iterations=1,
    )

    result = benchmark._run_isolated(
        spec,
        sample_interval_seconds=0.001,
        include_samples=False,
    )

    assert result["status"] == "ok"
    reverse = next(marker for marker in result["markers"] if marker["phase"] == "reverse")
    assert reverse["forward_calls"] == 4
    assert reverse["recomputation_count"] == 4


def test_json_smoke_reports_separate_memory_metrics() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.bench_runtime_memory",
            "--profile",
            "cpu-runtime",
            "--smoke",
            "--no-timing",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    summary = payload["cases"][0]["memory"]["summary"]

    assert payload["schema_version"] == 2
    assert payload["report_kind"] == "advect.runtime-memory"
    assert payload["environment"]["source_revision"]
    assert payload["environment"]["python"]
    assert set(payload["environment"]["machine"]) == {
        "architecture",
        "node",
        "platform",
        "processor",
    }
    assert payload["cases"][0]["status"] == "ok"
    assert "peak_rss_delta_bytes" in summary
    assert "peak_tracemalloc_delta_bytes" in summary
    assert "peak_provider_delta_bytes" in summary


def test_controller_requires_a_named_profile() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.bench_runtime_memory", "--smoke"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "--profile is required" in completed.stderr
