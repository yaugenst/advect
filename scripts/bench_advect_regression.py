"""Compare one reference and candidate Advect wheel on the same host."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import platform
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

_SCHEMA_VERSION = 2
_MIN_ACCEPTANCE_REPLICATES = 5
_DEFAULT_MINIMUM_THRESHOLD = 0.05
_DEFAULT_MAXIMUM_THRESHOLD = 0.20
_DEFAULT_NOISE_MULTIPLIER = 6.0


@dataclass(frozen=True, slots=True)
class _WorkerSpec:
    label: str
    source_revision: str
    wheel_sha256: str
    size: int
    warmup: int
    rounds: int
    block_size: int


@dataclass(frozen=True, slots=True)
class _ThresholdConfig:
    minimum: float
    maximum: float
    noise_multiplier: float


@dataclass(frozen=True, slots=True)
class _MeasurementConfig:
    size: int
    warmup: int
    rounds: int
    block_size: int
    warmed_replicates: int


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        message = f"expected a positive integer, got {value!r}"
        raise argparse.ArgumentTypeError(message)
    return parsed


def _fraction(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed < 1:
        message = f"expected a fraction between zero and one, got {value!r}"
        raise argparse.ArgumentTypeError(message)
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        message = f"expected a positive number, got {value!r}"
        raise argparse.ArgumentTypeError(message)
    return parsed


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-wheel", type=Path)
    parser.add_argument("--candidate-wheel", type=Path)
    parser.add_argument("--reference-revision")
    parser.add_argument("--candidate-revision")
    parser.add_argument("--uv", default=shutil.which("uv") or "uv")
    parser.add_argument("--size", type=_positive_int, default=32)
    parser.add_argument("--warmup", type=_positive_int, default=10)
    parser.add_argument("--rounds", type=_positive_int, default=7)
    parser.add_argument("--block-size", type=_positive_int, default=50)
    parser.add_argument("--warmed-replicates", type=_positive_int, default=5)
    parser.add_argument("--minimum-threshold", type=_fraction, default=_DEFAULT_MINIMUM_THRESHOLD)
    parser.add_argument("--maximum-threshold", type=_fraction, default=_DEFAULT_MAXIMUM_THRESHOLD)
    parser.add_argument(
        "--noise-multiplier", type=_positive_float, default=_DEFAULT_NOISE_MULTIPLIER
    )
    parser.add_argument("--acceptance", action="store_true")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--_worker-spec", dest="worker_spec", help=argparse.SUPPRESS)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_provenance(path: Path, *, source_revision: str) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.suffix != ".whl":
        message = f"benchmark artifact is not a wheel: {resolved}"
        raise ValueError(message)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
        "source_revision": source_revision,
    }


def _measure(
    call: Callable[[], object],
    *,
    warmup: int,
    rounds: int,
    block_size: int,
) -> dict[str, object]:
    for _ in range(warmup):
        call()
    samples: list[float] = []
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(rounds):
            started = time.perf_counter_ns()
            for _ in range(block_size):
                call()
            samples.append((time.perf_counter_ns() - started) / (1_000.0 * block_size))
    finally:
        if gc_enabled:
            gc.enable()
    return {"median_us": statistics.median(samples), "samples_us": samples}


def _stencil(np: Any, size: int) -> tuple[Callable[[object], object], object]:  # noqa: ANN401
    value = np.sin(np.linspace(-1.2, 1.1, size, dtype=np.float64))

    def loss(current: object) -> object:
        laplacian = current[2:] - 2.0 * current[1:-1] + current[:-2]  # type: ignore[index,operator]
        return np.sum(laplacian**2)  # type: ignore[operator]

    return loss, value


def _worker_payload(spec: _WorkerSpec) -> dict[str, object]:
    np = importlib.import_module("numpy")
    ad = importlib.import_module("advect")
    context = importlib.import_module("advect.core._context")
    native = importlib.import_module("advect.core._native")
    loss, value = _stencil(np, spec.size)

    gradient = ad.grad(loss)
    expected = gradient(value)
    direction = np.linspace(0.2, 0.8, value.size, dtype=value.dtype)
    epsilon = 1e-5
    finite_difference = (loss(value + epsilon * direction) - loss(value - epsilon * direction)) / (
        2 * epsilon
    )
    np.testing.assert_allclose(
        np.sum(expected * direction), finite_difference, rtol=2e-5, atol=2e-7
    )

    def linearize_and_close() -> None:
        _output, linear = ad.linearize(loss, value)
        linear.close()

    _output, linear = ad.linearize(loss, value)
    pullback = linear.transpose()
    pullback(1.0)
    try:
        dynamic_phases = {
            "one_shot_gradient": _measure(
                lambda: gradient(value),
                warmup=spec.warmup,
                rounds=spec.rounds,
                block_size=spec.block_size,
            ),
            "linearize_and_release": _measure(
                linearize_and_close,
                warmup=spec.warmup,
                rounds=spec.rounds,
                block_size=spec.block_size,
            ),
            "reused_reverse": _measure(
                lambda: pullback(1.0),
                warmup=spec.warmup,
                rounds=spec.rounds,
                block_size=spec.block_size,
            ),
        }
    finally:
        linear.close()

    def compile_gradient() -> object:
        return ad.grad(ad.stage(loss, value))

    staged_gradient = compile_gradient()
    artifact = staged_gradient.to_dict()
    restored = ad.StagedProgram.from_dict(artifact)
    input_before = np.array(value, copy=True)
    np.testing.assert_allclose(staged_gradient(value), expected, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(restored(value), expected, rtol=1e-11, atol=1e-12)
    np.testing.assert_array_equal(value, input_before)
    staged_phases = {
        "compile_gradient": _measure(
            compile_gradient,
            warmup=1,
            rounds=spec.rounds,
            block_size=1,
        ),
        "warm_gradient_execution": _measure(
            lambda: staged_gradient(value),
            warmup=spec.warmup,
            rounds=spec.rounds,
            block_size=spec.block_size,
        ),
        "roundtrip_and_execute": _measure(
            lambda: ad.StagedProgram.from_dict(artifact)(value),
            warmup=1,
            rounds=spec.rounds,
            block_size=1,
        ),
    }
    return {
        "schema_version": _SCHEMA_VERSION,
        "report_kind": "advect.performance-sample",
        "status": "ok",
        "artifact": {
            "label": spec.label,
            "source_revision": spec.source_revision,
            "wheel_sha256": spec.wheel_sha256,
        },
        "measurement": {
            "size": spec.size,
            "warmup": spec.warmup,
            "rounds": spec.rounds,
            "block_size": spec.block_size,
        },
        "correctness": {"passed": True, "gradient": np.asarray(expected).tolist()},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "advect_native": native.native_build_info(),
            "advect_debug": context.is_debug(),
        },
        "workloads": [
            {"name": "stencil", "lifetime": "dynamic", "phases": dynamic_phases},
            {"name": "stencil", "lifetime": "staged", "phases": staged_phases},
        ],
    }


def _worker_main(raw_spec: str) -> int:
    spec = _WorkerSpec(**json.loads(raw_spec))
    try:
        payload = _worker_payload(spec)
    except Exception as error:  # noqa: BLE001 - preserve worker diagnostics
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "report_kind": "advect.performance-sample",
            "status": "error",
            "reason": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0 if payload["status"] == "ok" else 1


def _worker_command(*, uv: str, wheel: Path, spec: _WorkerSpec) -> list[str]:
    return [
        uv,
        "run",
        "--isolated",
        "--no-project",
        "--with",
        str(wheel.resolve()),
        "python",
        "-m",
        "scripts.bench_advect_regression",
        "--_worker-spec",
        json.dumps(asdict(spec), sort_keys=True),
    ]


def _run_artifact(*, uv: str, wheel: Path, spec: _WorkerSpec) -> dict[str, object]:
    completed = subprocess.run(  # noqa: S603 - exact command is constructed above
        _worker_command(uv=uv, wheel=wheel, spec=spec),
        check=False,
        capture_output=True,
        text=True,
    )
    lines = completed.stdout.strip().splitlines()
    if not lines:
        return {"status": "error", "reason": completed.stderr.strip() or "worker produced no JSON"}
    return cast("dict[str, object]", json.loads(lines[-1]))


def _phase_values(
    samples: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str, str], list[float]]:
    values: dict[tuple[str, str, str], list[float]] = {}
    for sample in samples:
        for workload in cast("Sequence[Mapping[str, object]]", sample["workloads"]):
            phases = cast("Mapping[str, Mapping[str, object]]", workload["phases"])
            for phase, measurement in phases.items():
                key = (str(workload["name"]), str(workload["lifetime"]), phase)
                values.setdefault(key, []).append(float(measurement["median_us"]))
    return values


def _relative_mad(values: Sequence[float]) -> float:
    median = statistics.median(values)
    if median == 0:
        return 0.0
    return statistics.median(abs(value - median) for value in values) / abs(median)


def _compare_phases(
    reference: Sequence[Mapping[str, object]],
    candidate: Sequence[Mapping[str, object]],
    *,
    thresholds: _ThresholdConfig,
) -> tuple[list[dict[str, object]], list[str]]:
    reference_values = _phase_values(reference)
    candidate_values = _phase_values(candidate)
    comparisons: list[dict[str, object]] = []
    violations: list[str] = []
    if reference_values.keys() != candidate_values.keys():
        return [], ["reference and candidate reported different workload phases"]
    for key in sorted(reference_values):
        reference_samples = reference_values[key]
        candidate_samples = candidate_values[key]
        if len(reference_samples) != len(candidate_samples):
            return [], ["reference and candidate reported different replicate counts"]
        ratios = [
            candidate_value / reference_value
            for reference_value, candidate_value in zip(
                reference_samples, candidate_samples, strict=True
            )
        ]
        ratio = statistics.median(ratios)
        threshold = max(thresholds.minimum, thresholds.noise_multiplier * _relative_mad(ratios))
        stable = threshold <= thresholds.maximum
        passed = stable and ratio <= 1.0 + threshold
        label = "/".join(key)
        if not stable:
            violations.append(
                f"{label} noise-derived threshold {threshold:.1%} exceeds "
                f"the {thresholds.maximum:.1%} stability ceiling"
            )
        elif not passed:
            violations.append(f"{label} regressed to {ratio:.3f}x; limit is {1.0 + threshold:.3f}x")
        comparisons.append(
            {
                "workload": key[0],
                "lifetime": key[1],
                "phase": key[2],
                "reference_median_us": statistics.median(reference_samples),
                "candidate_median_us": statistics.median(candidate_samples),
                "candidate_over_reference": ratio,
                "effective_threshold": threshold,
                "stable": stable,
                "passed": passed,
            }
        )
    return comparisons, violations


def _numbers_close(left: object, right: object) -> bool:
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _numbers_close(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-12)
    return left == right


def _evidence_violations(
    reference: Sequence[Mapping[str, object]],
    candidate: Sequence[Mapping[str, object]],
    *,
    acceptance: bool,
) -> list[str]:
    violations: list[str] = []
    if acceptance and len(reference) < _MIN_ACCEPTANCE_REPLICATES:
        violations.append(
            f"acceptance requires at least {_MIN_ACCEPTANCE_REPLICATES} warmed replicates"
        )
    for label, samples in (("reference", reference), ("candidate", candidate)):
        for index, sample in enumerate(samples, start=1):
            if sample.get("status") != "ok":
                violations.append(f"{label} replicate {index} failed: {sample.get('reason')}")
                continue
            environment = cast("Mapping[str, object]", sample["environment"])
            native = cast("Mapping[str, object]", environment["advect_native"])
            if acceptance and native.get("build_profile") != "release":
                violations.append(f"{label} replicate {index} did not use a release wheel")
            if acceptance and environment.get("advect_debug") is not False:
                violations.append(f"{label} replicate {index} enabled Advect diagnostics")
    for index, (reference_sample, candidate_sample) in enumerate(
        zip(reference, candidate, strict=False), start=1
    ):
        if reference_sample.get("status") != "ok" or candidate_sample.get("status") != "ok":
            continue
        if not _numbers_close(
            cast("Mapping[str, object]", reference_sample["correctness"])["gradient"],
            cast("Mapping[str, object]", candidate_sample["correctness"])["gradient"],
        ):
            violations.append(f"replicate {index} reference/candidate gradients differ")
        reference_environment = cast("Mapping[str, object]", reference_sample["environment"])
        candidate_environment = cast("Mapping[str, object]", candidate_sample["environment"])
        violations.extend(
            f"replicate {index} environment differs for {key}"
            for key in ("python", "platform", "numpy")
            if reference_environment.get(key) != candidate_environment.get(key)
        )
    return violations


def _build_report(
    *,
    reference_samples: Sequence[Mapping[str, object]],
    candidate_samples: Sequence[Mapping[str, object]],
    orders: Sequence[Sequence[str]],
    reference_artifact: Mapping[str, object],
    candidate_artifact: Mapping[str, object],
    measurement: _MeasurementConfig,
    thresholds: _ThresholdConfig,
    acceptance: bool,
) -> dict[str, object]:
    violations = _evidence_violations(reference_samples, candidate_samples, acceptance=acceptance)
    comparisons: list[dict[str, object]] = []
    if not violations:
        comparisons, comparison_violations = _compare_phases(
            reference_samples, candidate_samples, thresholds=thresholds
        )
        violations.extend(comparison_violations)
    return {
        "schema_version": _SCHEMA_VERSION,
        "report_kind": "advect.performance-regression",
        "artifacts": {
            "reference": dict(reference_artifact),
            "candidate": dict(candidate_artifact),
        },
        "measurement": {
            **asdict(measurement),
            "orders": [list(order) for order in orders],
            "thresholds": asdict(thresholds),
        },
        "comparisons": comparisons,
        "samples": {"reference": list(reference_samples), "candidate": list(candidate_samples)},
        "acceptance": {"requested": acceptance, "valid": not violations, "violations": violations},
    }


def _print_text(report: Mapping[str, object]) -> None:
    for comparison in cast("Sequence[Mapping[str, object]]", report["comparisons"]):
        print(
            f"{comparison['workload']}/{comparison['lifetime']}/{comparison['phase']}: "
            f"{float(comparison['candidate_over_reference']):.3f}x "
            f"(limit {1.0 + float(comparison['effective_threshold']):.3f}x)"
        )
    acceptance = cast("Mapping[str, object]", report["acceptance"])
    print(f"acceptance-valid={str(acceptance['valid']).lower()}")
    for violation in cast("Sequence[str]", acceptance["violations"]):
        print(f"- {violation}")


def _controller_main(args: argparse.Namespace) -> int:
    required = {
        "--reference-wheel": args.reference_wheel,
        "--candidate-wheel": args.candidate_wheel,
        "--reference-revision": args.reference_revision,
        "--candidate-revision": args.candidate_revision,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        message = f"missing required controller arguments: {', '.join(missing)}"
        raise ValueError(message)
    thresholds = _ThresholdConfig(
        args.minimum_threshold, args.maximum_threshold, args.noise_multiplier
    )
    if thresholds.minimum > thresholds.maximum:
        message = "--minimum-threshold cannot exceed --maximum-threshold"
        raise ValueError(message)
    reference_artifact = _artifact_provenance(
        cast("Path", args.reference_wheel), source_revision=cast("str", args.reference_revision)
    )
    candidate_artifact = _artifact_provenance(
        cast("Path", args.candidate_wheel), source_revision=cast("str", args.candidate_revision)
    )
    artifacts = {
        "reference": (cast("Path", args.reference_wheel), reference_artifact),
        "candidate": (cast("Path", args.candidate_wheel), candidate_artifact),
    }
    samples: dict[str, list[dict[str, object]]] = {"reference": [], "candidate": []}
    orders: list[list[str]] = []
    for replicate in range(args.warmed_replicates):
        order = ["reference", "candidate"] if replicate % 2 == 0 else ["candidate", "reference"]
        orders.append(order)
        for label in order:
            wheel, artifact = artifacts[label]
            spec = _WorkerSpec(
                label,
                cast("str", artifact["source_revision"]),
                cast("str", artifact["sha256"]),
                args.size,
                args.warmup,
                args.rounds,
                args.block_size,
            )
            samples[label].append(_run_artifact(uv=args.uv, wheel=wheel, spec=spec))
    report = _build_report(
        reference_samples=samples["reference"],
        candidate_samples=samples["candidate"],
        orders=orders,
        reference_artifact=reference_artifact,
        candidate_artifact=candidate_artifact,
        measurement=_MeasurementConfig(
            size=args.size,
            warmup=args.warmup,
            rounds=args.rounds,
            block_size=args.block_size,
            warmed_replicates=args.warmed_replicates,
        ),
        thresholds=thresholds,
        acceptance=args.acceptance,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    elif args.format == "json":
        print(rendered)
    else:
        _print_text(report)
    acceptance = cast("Mapping[str, object]", report["acceptance"])
    return 0 if not acceptance["requested"] or acceptance["valid"] else 2


def main() -> int:
    """Run one isolated reference-versus-candidate comparison."""
    args = _arguments()
    if args.worker_spec is not None:
        return _worker_main(args.worker_spec)
    try:
        return _controller_main(args)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
