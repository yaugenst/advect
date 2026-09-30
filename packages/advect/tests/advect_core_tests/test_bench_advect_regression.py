"""Focused contracts for isolated Advect regression evidence."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest
from hypothesis import given, strategies as st
from scripts import bench_advect_regression as benchmark

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

_THRESHOLDS = benchmark._ThresholdConfig(0.05, 0.20, 6.0)
_MEDIANS = st.floats(1e-3, 1e6)


def _sample(value: float) -> dict[str, object]:
    return {
        "status": "ok",
        "correctness": {"passed": True, "gradient": [1.0, 2.0]},
        "environment": {
            "python": "3.12.0",
            "platform": "test-host",
            "numpy": "2.4.0",
            "advect_native": {"build_profile": "release"},
            "advect_debug": False,
        },
        "workloads": [
            {
                "name": "stencil",
                "lifetime": "dynamic",
                "phases": {"one_shot_gradient": {"median_us": value}},
            }
        ],
    }


def _report(
    reference: list[dict[str, object]],
    candidate: list[dict[str, object]],
) -> dict[str, object]:
    count = len(reference)
    return benchmark._build_report(
        reference_samples=reference,
        candidate_samples=candidate,
        orders=(("reference", "candidate"),) * count,
        reference_artifact={"source_revision": "ref", "sha256": "a"},
        candidate_artifact={"source_revision": "candidate", "sha256": "b"},
        measurement=benchmark._MeasurementConfig(32, 10, 7, 50, count),
        thresholds=benchmark._ThresholdConfig(0.05, 0.20, 6.0),
        acceptance=True,
    )


def test_artifact_provenance_hashes_the_exact_wheel(tmp_path: Path) -> None:
    wheel = tmp_path / "advect-0.1-py3-none-any.whl"
    wheel.write_bytes(b"wheel bytes")

    provenance = benchmark._artifact_provenance(wheel, source_revision="abc123")

    assert provenance["path"] == str(wheel.resolve())
    assert provenance["bytes"] == len(b"wheel bytes")
    assert provenance["source_revision"] == "abc123"
    assert len(provenance["sha256"]) == 64


def test_non_finite_noise_multiplier_is_an_argument_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["bench_advect_regression", "--noise-multiplier", "nan"])

    with pytest.raises(SystemExit) as raised:
        benchmark._arguments()

    assert raised.value.code == 2
    assert "expected a finite positive number, got 'nan'" in capsys.readouterr().err


def test_worker_command_uses_an_isolated_non_project_environment(tmp_path: Path) -> None:
    wheel = tmp_path / "advect.whl"
    spec = benchmark._WorkerSpec("reference", "abc", "digest", 8, 1, 1, 1)

    command = benchmark._worker_command(uv="uv", wheel=wheel, spec=spec)

    assert command[:6] == ["uv", "run", "--isolated", "--no-project", "--with", str(wheel)]
    assert command[-2] == "--_worker-spec"


@pytest.mark.parametrize(
    ("candidate_values", "passed", "violation"),
    [
        ((103.0,) * 5, True, None),
        ((110.0,) * 5, False, "regressed to 1.100x"),
        ((80.0, 90.0, 100.0, 110.0, 120.0), False, "stability ceiling"),
    ],
)
def test_phase_gate_uses_paired_noise_with_a_stability_ceiling(
    candidate_values: tuple[float, ...],
    passed: object,
    violation: str | None,
) -> None:
    reference = [_sample(100.0) for _ in candidate_values]
    candidate = [_sample(value) for value in candidate_values]

    comparisons, violations = benchmark._compare_phases(
        reference,
        candidate,
        thresholds=benchmark._ThresholdConfig(0.05, 0.20, 6.0),
    )

    assert comparisons[0]["passed"] is passed
    assert any(violation in item for item in violations) if violation else violations == []


@st.composite
def _paired_medians(draw: st.DrawFn) -> tuple[list[float], list[float]]:
    """Draw paired replicate medians, mostly near one common candidate/reference ratio."""
    count = draw(st.integers(1, 9))
    reference = draw(st.lists(_MEDIANS, min_size=count, max_size=count))
    if draw(st.booleans()):
        return reference, draw(st.lists(_MEDIANS, min_size=count, max_size=count))
    factor = draw(st.floats(0.5, 2.0))
    jitter = st.floats(-0.05, 0.05)
    return reference, [value * factor * (1 + draw(jitter)) for value in reference]


def _compare(
    reference: Sequence[float],
    candidate: Sequence[float],
) -> tuple[list[dict[str, object]], list[str]]:
    return benchmark._compare_phases(
        [_sample(value) for value in reference],
        [_sample(value) for value in candidate],
        thresholds=_THRESHOLDS,
    )


@given(pairs=_paired_medians(), data=st.data())
def test_phase_gate_ignores_replicate_order(
    pairs: tuple[list[float], list[float]],
    data: st.DataObject,
) -> None:
    reference, candidate = pairs
    order = data.draw(st.permutations(range(len(reference))))

    permuted = _compare([reference[i] for i in order], [candidate[i] for i in order])

    assert permuted == _compare(reference, candidate)


@given(pairs=_paired_medians(), exponent=st.integers(-20, 20))
def test_phase_gate_verdict_is_independent_of_the_time_unit(
    pairs: tuple[list[float], list[float]],
    exponent: int,
) -> None:
    reference, candidate = pairs
    scale = 2.0**exponent
    keys = ("candidate_over_reference", "effective_threshold", "stable", "passed")

    comparisons, violations = _compare(reference, candidate)
    scaled, scaled_violations = _compare(
        [value * scale for value in reference], [value * scale for value in candidate]
    )

    assert [{key: item[key] for key in keys} for item in scaled] == [
        {key: item[key] for key in keys} for item in comparisons
    ]
    assert scaled_violations == violations


@given(reference=st.lists(_MEDIANS, min_size=1, max_size=9))
def test_identical_artifacts_pass_at_the_minimum_threshold(reference: list[float]) -> None:
    (comparison,), violations = _compare(reference, reference)

    assert violations == []
    assert comparison["candidate_over_reference"] == 1.0
    assert comparison["effective_threshold"] == _THRESHOLDS.minimum
    assert comparison["passed"] is True


@given(pairs=_paired_medians())
def test_two_artifacts_never_regress_against_each_other(
    pairs: tuple[list[float], list[float]],
) -> None:
    first, second = pairs

    def regressed(reference: list[float], candidate: list[float]) -> bool:
        (comparison,), _violations = _compare(reference, candidate)
        return comparison["stable"] is True and comparison["passed"] is False

    assert not (regressed(first, second) and regressed(second, first))


def test_acceptance_requires_five_comparable_replicates() -> None:
    report = _report([_sample(100.0)] * 2, [_sample(100.0)] * 2)

    assert report["acceptance"] == {
        "requested": True,
        "valid": False,
        "violations": ["acceptance requires at least 5 warmed replicates"],
    }
    assert report["comparisons"] == []


@pytest.mark.parametrize("field", ["environment", "correctness"])
def test_incomparable_artifacts_have_no_performance_verdict(field: str) -> None:
    reference = [_sample(100.0) for _ in range(5)]
    candidate = [_sample(100.0) for _ in range(5)]
    if field == "environment":
        candidate[0]["environment"]["numpy"] = "2.3.0"
    else:
        candidate[0]["correctness"]["gradient"] = [2.0]

    report = _report(reference, candidate)

    assert report["acceptance"]["valid"] is False
    assert report["comparisons"] == []


def test_worker_runs_one_dynamic_and_one_staged_workload() -> None:
    payload = benchmark._worker_payload(
        benchmark._WorkerSpec("candidate", "test", "digest", 8, 1, 1, 1)
    )

    assert payload["status"] == "ok"
    assert payload["schema_version"] == 2
    assert payload["measurement"] == {
        "size": 8,
        "warmup": 1,
        "rounds": 1,
        "block_size": 1,
    }
    assert payload["correctness"]["passed"] is True
    workloads = {item["lifetime"]: item["phases"] for item in payload["workloads"]}
    assert set(workloads["dynamic"]) == {
        "one_shot_gradient",
        "linearize_and_release",
        "reused_reverse",
    }
    assert set(workloads["staged"]) == {
        "compile_gradient",
        "warm_gradient_execution",
        "roundtrip_and_execute",
    }


def test_regression_report_records_every_measurement_input() -> None:
    report = _report([_sample(100.0)] * 5, [_sample(100.0)] * 5)

    assert report["schema_version"] == 2
    assert report["measurement"] == {
        "size": 32,
        "warmup": 10,
        "rounds": 7,
        "block_size": 50,
        "warmed_replicates": 5,
        "orders": [["reference", "candidate"]] * 5,
        "thresholds": {"minimum": 0.05, "maximum": 0.2, "noise_multiplier": 6.0},
    }
