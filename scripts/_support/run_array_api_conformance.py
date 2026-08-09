"""Run the pinned official Array API suite through Advect's supported surface."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from advect.core._array_api.frontend import (
    _ARRAY_API_COMPOSITES,
    _ARRAY_API_META_FUNCTIONS,
    _FUNCTION_SPECS,
)
from advect.core._array_api.profiles import (
    LATEST_ARRAY_API_VERSION,
    SUPPORTED_ARRAY_API_VERSIONS,
)
from advect.core._array_api.signatures import official_signatures
from advect.core._array_api.support import build_support_profile
from scripts._support.evidence import evidence_report_header

_PINNED_SUITE_REVISION = "5d0b701b0c4ab6ec98794068cf7af393a8a51c61"
_RESULT_PLUGIN = "_support.array_api_pytest_results"
_RESULT_ENV = "ADVECT_ARRAY_API_PYTEST_RESULTS"
_METADATA_QUALIFIED_ELSEWHERE = tuple(sorted(_ARRAY_API_META_FUNCTIONS))
_COLLECTION_EXCLUSIONS = {
    # The module is marked 2023.12+, but constructs a Hypothesis strategy by
    # calling the 2023.12 inspection API at import time, before pytest can skip
    # it for a 2022.12 run.
    "2022.12": ("array_api_tests/test_inspection_functions.py",),
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        msg = f"expected a positive integer, got {value!r}"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        msg = f"expected a non-negative integer, got {value!r}"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-path", required=True, type=Path)
    parser.add_argument(
        "--array-api-version",
        choices=SUPPORTED_ARRAY_API_VERSIONS,
        default=LATEST_ARRAY_API_VERSION,
    )
    parser.add_argument(
        "--mode",
        choices=("dynamic", "stage", "serialized", "all"),
        default="all",
    )
    parser.add_argument("--max-examples", type=_positive_int, default=10)
    parser.add_argument("--shard-count", type=_positive_int, default=1)
    parser.add_argument("--shard-index", type=_nonnegative_int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _revision(suite_path: Path) -> str:
    git = shutil.which("git")
    if git is None:
        msg = "git is required to identify the Array API test-suite revision"
        raise RuntimeError(msg)
    completed = subprocess.run(  # noqa: S603
        [git, "-C", str(suite_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _normalized_signature(function: object) -> str:
    signature = inspect.signature(function)
    parameters = tuple(
        parameter.replace(annotation=inspect.Parameter.empty)
        for parameter in signature.parameters.values()
    )
    return str(
        signature.replace(
            parameters=parameters,
            return_annotation=inspect.Signature.empty,
        )
    )


def _verify_official_signature_snapshot(
    suite_path: Path,
    *,
    array_api_version: str,
) -> dict[str, object]:
    source_root = suite_path / "array-api" / "src"
    if not source_root.is_dir():
        message = "The pinned Array API suite checkout is missing its official stub submodule"
        raise RuntimeError(message)
    sys.path.insert(0, str(source_root))
    try:
        stubs = importlib.import_module(f"array_api_stubs._{array_api_version.replace('.', '_')}")
    finally:
        sys.path.remove(str(source_root))

    observed: dict[str, str] = {}
    signatures = official_signatures(array_api_version)
    for path in signatures:
        function = stubs
        for component in path.split("."):
            function = getattr(function, component)
        if not callable(function):
            message = f"Official Array API stub {path!r} is not callable"
            raise TypeError(message)
        observed[path] = _normalized_signature(function)
    if observed != signatures:
        drift = {
            path: {
                "expected": signatures.get(path),
                "observed": observed.get(path),
            }
            for path in sorted(set(observed) | set(signatures))
            if observed.get(path) != signatures.get(path)
        }
        message = (
            f"The pinned official Array API {array_api_version} stub signatures drifted from "
            f"Advect's snapshot: {json.dumps(drift, sort_keys=True)}"
        )
        raise RuntimeError(message)
    return {
        "function_count": len(observed),
        "source": f"array-api/src/array_api_stubs/_{array_api_version.replace('.', '_')}",
        "source_revision": _revision(suite_path / "array-api"),
        "verified": True,
    }


def _read_calls(path: Path) -> Counter[str]:
    calls: Counter[str] = Counter()
    if not path.exists():
        return calls
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        calls[str(payload["operation"])] += 1
    return calls


def _operations_for_mode(
    mode: str,
    array_api_version: str = LATEST_ARRAY_API_VERSION,
) -> tuple[str, ...]:
    manifest_mode = "staged" if mode == "stage" else mode
    profile = build_support_profile(array_api_version)
    callables = profile["callables"]
    if not isinstance(callables, list):
        message = "Array API support profile callables must be a list"
        raise TypeError(message)
    operations = [
        str(row["path"])
        for row in callables
        if isinstance(row, dict)
        and row.get("complete") is True
        and manifest_mode in row.get("modes", ())
        and row.get("path") in set(_FUNCTION_SPECS) | _ARRAY_API_COMPOSITES
    ]
    return tuple(sorted(operations))


def _metadata_qualification() -> dict[str, object]:
    return {
        "evidence": (
            "packages/advect/tests/advect_grad_tests/test_array_api_operation_qualification.py"
        ),
        "lifetimes": ["dynamic", "staged", "serialized"],
        "operations": list(_METADATA_QUALIFIED_ELSEWHERE),
        "reason": (
            "compile-time metadata calls do not return arrays and therefore "
            "cannot be reconstructed by the official-suite bridge"
        ),
    }


def _qualification_environment(
    *,
    mode: str,
    operations: tuple[str, ...],
    log_path: Path,
    array_api_version: str,
) -> dict[str, str]:
    scripts_path = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    current_pythonpath = environment.get("PYTHONPATH")
    environment.update(
        {
            "ARRAY_API_TESTS_MODULE": "array_api_trace_namespace",
            "ARRAY_API_TESTS_VERSION": array_api_version,
            "ARRAY_API_STRICT_API_VERSION": array_api_version,
            "ADVECT_ARRAY_API_QUALIFICATION_MODE": mode,
            "ADVECT_ARRAY_API_QUALIFICATION_OPS": ",".join(operations),
            "ADVECT_ARRAY_API_TRACE_LOG": str(log_path),
            "PYTHONPATH": (
                str(scripts_path)
                if not current_pythonpath
                else f"{scripts_path}{os.pathsep}{current_pythonpath}"
            ),
        }
    )
    return environment


def _baseline_environment(
    *,
    result_path: Path,
    array_api_version: str,
) -> dict[str, str]:
    scripts_path = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    current_pythonpath = environment.get("PYTHONPATH")
    for name in tuple(environment):
        if name.startswith("ADVECT_ARRAY_API_"):
            environment.pop(name)
    environment.update(
        {
            "ARRAY_API_TESTS_MODULE": "array_api_strict",
            "ARRAY_API_TESTS_VERSION": array_api_version,
            "ARRAY_API_STRICT_API_VERSION": array_api_version,
            _RESULT_ENV: str(result_path),
            "PYTHONPATH": (
                str(scripts_path)
                if not current_pythonpath
                else f"{scripts_path}{os.pathsep}{current_pythonpath}"
            ),
        }
    )
    return environment


def _pytest_command(
    *,
    max_examples: int,
    test_nodes: tuple[str, ...],
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--hypothesis-derandomize",
        "--hypothesis-disable-deadline",
        "--hypothesis-max-examples",
        str(max_examples),
        "-p",
        _RESULT_PLUGIN,
    ]
    command.extend(test_nodes)
    return command


def _run_baseline(
    suite_path: Path,
    *,
    max_examples: int,
    result_path: Path,
    test_nodes: tuple[str, ...],
    array_api_version: str,
) -> tuple[dict[str, object], tuple[str, ...]]:
    environment = _baseline_environment(
        result_path=result_path,
        array_api_version=array_api_version,
    )
    started = time.perf_counter()
    completed = subprocess.run(  # noqa: S603
        _pytest_command(
            max_examples=max_examples,
            test_nodes=test_nodes,
        ),
        cwd=suite_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if not result_path.exists():
        detail = completed.stderr.strip() or completed.stdout.strip()
        message = f"Array API baseline produced no result record: {detail}"
        raise RuntimeError(message)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    raw_results = payload.get("results")
    if not isinstance(raw_results, dict):
        message = "Array API baseline result record has no results mapping"
        raise TypeError(message)
    statuses = {str(node): str(status) for node, status in raw_results.items()}
    expected = set(test_nodes)
    unexpected = sorted(set(statuses).difference(expected))
    missing = sorted(expected.difference(statuses))
    if unexpected or missing:
        message = (
            "Array API baseline did not report the selected nodes exactly; "
            f"unexpected={unexpected}, missing={missing}"
        )
        raise RuntimeError(message)
    passing = tuple(node for node in test_nodes if statuses[node] == "passed")
    failures = tuple(node for node in test_nodes if statuses[node] == "failed")
    skipped = tuple(node for node in test_nodes if statuses[node] == "skipped")
    unknown = sorted(set(statuses.values()).difference({"failed", "passed", "skipped"}))
    if unknown:
        message = f"Array API baseline reported unknown outcomes: {unknown}"
        raise RuntimeError(message)
    if completed.returncode != 0 and not failures:
        message = "Array API baseline exited unsuccessfully without an attributable test failure"
        raise RuntimeError(message)
    return (
        {
            "elapsed_seconds": elapsed,
            "failed_test_nodes": list(failures),
            "failure_count": len(failures),
            "passed_test_count": len(passing),
            "provider": "array-api-strict",
            "pytest_returncode": completed.returncode,
            "sampling": "sha256-nodeid-v1",
            "skipped_test_count": len(skipped),
            "skipped_test_nodes": list(skipped),
            "test_count": len(test_nodes),
        },
        passing,
    )


def _collect_test_nodes(
    suite_path: Path,
    *,
    environment: dict[str, str],
    array_api_version: str,
) -> tuple[str, ...]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "array_api_tests",
    ]
    command.extend(f"--ignore={path}" for path in _COLLECTION_EXCLUSIONS.get(array_api_version, ()))
    completed = subprocess.run(  # noqa: S603
        command,
        cwd=suite_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        msg = f"Could not collect the official Array API suite: {detail}"
        raise RuntimeError(msg)
    nodes = tuple(
        sorted(
            line.strip()
            for line in completed.stdout.splitlines()
            if line.startswith("array_api_tests/") and "::" in line
        )
    )
    if not nodes:
        message = "The official Array API suite collected no tests"
        raise RuntimeError(message)
    return nodes


def _shard_test_nodes(
    nodes: tuple[str, ...],
    *,
    shard_count: int,
    shard_index: int,
) -> tuple[str, ...]:
    return tuple(
        node for position, node in enumerate(nodes) if position % shard_count == shard_index
    )


def _run_mode(
    suite_path: Path,
    *,
    mode: str,
    max_examples: int,
    log_path: Path,
    test_nodes: tuple[str, ...],
    array_api_version: str,
) -> dict[str, object]:
    operations = _operations_for_mode(mode, array_api_version)
    environment = _qualification_environment(
        mode=mode,
        operations=operations,
        log_path=log_path,
        array_api_version=array_api_version,
    )
    command = _pytest_command(max_examples=max_examples, test_nodes=test_nodes)
    started = time.perf_counter()
    completed = subprocess.run(  # noqa: S603
        command,
        cwd=suite_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)

    calls = _read_calls(log_path)
    unobserved = sorted(set(operations).difference(calls))
    passed = completed.returncode == 0
    return {
        "call_counts": dict(sorted(calls.items())),
        "elapsed_seconds": elapsed,
        "mode": mode,
        "operations": list(operations),
        "passed": passed,
        "pytest_returncode": completed.returncode,
        "test_count": len(test_nodes),
        # Some creation and metadata calls have no array operand and therefore
        # cannot enter a concrete trace from the official suite. Deterministic
        # zero-operand staging cases cover those separately.
        "unobserved_operations": unobserved,
    }


def _build_report(
    *,
    arguments: argparse.Namespace,
    baseline: dict[str, object],
    collected_test_count: int,
    results: list[dict[str, object]],
    signature_snapshot: dict[str, object],
    suite_revision: str,
    test_count: int,
) -> dict[str, object]:
    return {
        **evidence_report_header(
            schema_version=1,
            report_kind="advect.array-api-official-suite",
        ),
        "api_version": arguments.array_api_version,
        "baseline": baseline,
        "collected_test_count": collected_test_count,
        "collection_exclusions": list(_COLLECTION_EXCLUSIONS.get(arguments.array_api_version, ())),
        "max_examples": arguments.max_examples,
        "metadata_qualified_elsewhere": _metadata_qualification(),
        "passed": all(bool(result["passed"]) for result in results),
        "results": results,
        "shard_count": arguments.shard_count,
        "shard_index": arguments.shard_index,
        "signature_snapshot": signature_snapshot,
        "suite_revision": suite_revision,
        "test_count": test_count,
    }


def main() -> int:
    """Run the selected modes and write their reproducible evidence."""
    arguments = _arguments()
    if arguments.shard_index >= arguments.shard_count:
        msg = (
            f"shard index {arguments.shard_index} must be less than "
            f"shard count {arguments.shard_count}"
        )
        raise ValueError(msg)
    suite_path = arguments.suite_path.resolve()
    if not (suite_path / "array_api_tests").is_dir():
        msg = f"{suite_path} is not an array-api-tests checkout"
        raise RuntimeError(msg)
    revision = _revision(suite_path)
    if revision != _PINNED_SUITE_REVISION:
        msg = (
            f"Expected array-api-tests revision {_PINNED_SUITE_REVISION}, "
            f"found {revision}. Update the pin and qualification record intentionally."
        )
        raise RuntimeError(msg)
    signature_snapshot = _verify_official_signature_snapshot(
        suite_path,
        array_api_version=arguments.array_api_version,
    )

    modes = ("dynamic", "stage", "serialized") if arguments.mode == "all" else (arguments.mode,)
    with tempfile.TemporaryDirectory(prefix="advect-array-api-") as temporary:
        temporary_path = Path(temporary)
        collection_environment = _baseline_environment(
            result_path=temporary_path / "collection-results.json",
            array_api_version=arguments.array_api_version,
        )
        all_test_nodes = _collect_test_nodes(
            suite_path,
            environment=collection_environment,
            array_api_version=arguments.array_api_version,
        )
        test_nodes = _shard_test_nodes(
            all_test_nodes,
            shard_count=arguments.shard_count,
            shard_index=arguments.shard_index,
        )
        if not test_nodes:
            message = "The selected Array API conformance shard is empty"
            raise RuntimeError(message)
        baseline, qualified_test_nodes = _run_baseline(
            suite_path,
            max_examples=arguments.max_examples,
            result_path=temporary_path / "baseline-results.json",
            test_nodes=test_nodes,
            array_api_version=arguments.array_api_version,
        )
        if not qualified_test_nodes:
            message = "The selected Array API shard has no baseline-passing tests"
            raise RuntimeError(message)
        results = [
            _run_mode(
                suite_path,
                mode=mode,
                max_examples=arguments.max_examples,
                log_path=temporary_path / f"{mode}.jsonl",
                test_nodes=qualified_test_nodes,
                array_api_version=arguments.array_api_version,
            )
            for mode in modes
        ]

    report = _build_report(
        arguments=arguments,
        baseline=baseline,
        collected_test_count=len(all_test_nodes),
        results=results,
        signature_snapshot=signature_snapshot,
        suite_revision=revision,
        test_count=len(test_nodes),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(f"{rendered}\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
