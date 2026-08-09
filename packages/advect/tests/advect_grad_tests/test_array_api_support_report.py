"""Tests for the generated Array API support inventory."""

from __future__ import annotations

import ast
from collections import Counter
from typing import TYPE_CHECKING

import pytest
from scripts._support import report_array_api_support, run_array_api_conformance

from advect.core._array_api.frontend import _ARRAY_API_META_FUNCTIONS
from advect.core._array_api.profiles import (
    LATEST_ARRAY_API_VERSION,
    materialize_array_api_profile,
)
from advect.core._array_api.signatures import OFFICIAL_SIGNATURES, official_parameter_names
from advect.core._array_api.support import build_support_profile

if TYPE_CHECKING:
    from types import ModuleType


@pytest.fixture(scope="module")
def reporter() -> ModuleType:
    return report_array_api_support


def test_discovers_the_complete_installed_2024_12_function_surface(
    reporter: ModuleType,
) -> None:
    functions = reporter._official_functions()
    paths = [function.path for function in functions]

    assert len(paths) == 170
    assert len(paths) == len(set(paths))
    assert paths == sorted(paths)
    assert "pow" in paths
    assert "fft.fft" in paths
    assert "linalg.solve" in paths
    assert "__array_namespace_info__" not in paths
    assert "set_array_api_strict_flags" not in paths


def _signature_parameter_names(signature: str) -> tuple[str, ...]:
    parsed = ast.parse(f"def operation{signature}:\n    pass\n")
    function = parsed.body[0]
    assert isinstance(function, ast.FunctionDef)
    arguments = function.args
    return tuple(
        argument.arg
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *((arguments.vararg,) if arguments.vararg is not None else ()),
            *arguments.kwonlyargs,
            *((arguments.kwarg,) if arguments.kwarg is not None else ()),
        )
    )


def test_runtime_manifest_snapshots_the_official_stub_contract(
    reporter: ModuleType,
) -> None:
    profile = build_support_profile()
    snapshotted = {str(row["path"]): str(row["signature"]) for row in profile["callables"]}
    snapshotted_parameters = {
        str(row["path"]): tuple(str(parameter["name"]) for parameter in row["parameters"])
        for row in profile["callables"]
    }

    assert snapshotted == OFFICIAL_SIGNATURES
    official_parameters = {
        path: official_parameter_names(path, LATEST_ARRAY_API_VERSION)
        for path in OFFICIAL_SIGNATURES
    }
    assert {
        path: _signature_parameter_names(signature)
        for path, signature in OFFICIAL_SIGNATURES.items()
    } == official_parameters
    assert snapshotted_parameters == official_parameters

    rows = {row["path"]: row for row in reporter.build_report()["functions"]}
    assert rows["expand_dims"]["signature"] == "(x, /, axis)"
    assert rows["expand_dims"]["provider_signature"] == "(x, /, *, axis)"
    assert rows["expand_dims"]["provider_signature_deviation"] is True
    assert "expand_dims" in reporter.build_report()["provider_signature_deviations"]


def test_official_runner_selects_the_declared_operations_for_each_mode() -> None:
    dynamic = set(run_array_api_conformance._operations_for_mode("dynamic"))
    staged = set(run_array_api_conformance._operations_for_mode("stage"))
    serialized = set(run_array_api_conformance._operations_for_mode("serialized"))

    assert staged
    assert staged == serialized
    assert staged < dynamic
    assert not set(_ARRAY_API_META_FUNCTIONS) & dynamic
    metadata = run_array_api_conformance._metadata_qualification()
    assert set(metadata["operations"]) == _ARRAY_API_META_FUNCTIONS
    assert metadata["lifetimes"] == ["dynamic", "staged", "serialized"]


def test_live_nondifferentiable_parameters_are_not_reported_as_static() -> None:
    rows = {row["path"]: row for row in build_support_profile()["callables"]}
    searchsorted = {
        parameter["name"]: parameter["role"] for parameter in rows["searchsorted"]["parameters"]
    }
    result_type = rows["result_type"]
    result_type_roles = {
        parameter["name"]: parameter["role"] for parameter in result_type["parameters"]
    }

    assert searchsorted["sorter"] == "nondifferentiable"
    assert result_type["signature"] == "(*arrays_and_dtypes)"
    assert result_type_roles["arrays_and_dtypes"] == "nondifferentiable"


def test_report_classification_is_exhaustive_and_mutually_exclusive(
    reporter: ModuleType,
) -> None:
    report = reporter.build_report()
    rows = report["functions"]
    summary = report["summary"]
    classifications = Counter(row["classification"] for row in rows)
    paths = [row["path"] for row in rows]

    assert report["schema_version"] == 3
    assert report["report_kind"] == "advect.array-api-support"
    assert report["environment"]["source_revision"]
    assert report["environment"]["python"]
    assert report["environment"]["machine"]["platform"]
    assert len(rows) == summary["official_functions"]
    assert len(paths) == len(set(paths))
    assert sum(classifications.values()) == summary["official_functions"]
    assert set(classifications).issubset(reporter._SUPPORT_CLASSIFICATIONS)
    assert all(row["classification"] in reporter._SUPPORT_CLASSIFICATIONS for row in rows)
    assert dict(sorted(classifications.items())) == summary["classifications"]
    assert (
        summary["supported_transform_functions"] + summary["unsupported_transform_functions"]
        == summary["transform_applicable_functions"]
    )
    assert (
        summary["transform_applicable_functions"] + classifications["provider_passthrough"]
        == summary["official_functions"]
    )
    assert all(row["canonical_op"] is not None for row in rows if row["binding"] == "operation")
    assert all(row["canonical_op"] is None for row in rows if row["binding"] == "composite")
    assert all(
        row["result_kind"] == "multiple_arrays"
        for row in rows
        if row["classification"] == "multi_output_unsupported"
    )
    assert all(
        not row["has_array_operand"] and row["binding"] == "none"
        for row in rows
        if row["classification"] == "provider_passthrough"
    )

    official_paths = {row["path"] for row in rows}
    expected_extras = [
        {
            "canonical_op": reporter._FUNCTION_SPECS[path].op,
            "path": path,
        }
        for path in sorted(set(reporter._FUNCTION_SPECS).difference(official_paths))
    ]
    assert report["extra_catalog_paths"] == expected_extras


def test_compile_time_metadata_uses_the_explicit_runtime_surface(
    reporter: ModuleType,
) -> None:
    rows = {row["path"]: row for row in reporter.build_report()["functions"]}
    classified = {
        path for path, row in rows.items() if row["classification"] == "compile_time_metadata"
    }

    assert classified == reporter._ARRAY_API_META_FUNCTIONS
    assert all(rows[path]["binding"] == "compile_time_metadata" for path in classified)
    assert all(rows[path]["canonical_op"] is None for path in classified)
    assert all(rows[path]["derivative_status"] == "not_applicable" for path in classified)
    assert rows["can_cast"]["classification"] == "compile_time_metadata"
    assert rows["finfo"]["classification"] == "compile_time_metadata"
    assert rows["iinfo"]["classification"] == "compile_time_metadata"


def test_registered_operations_have_complete_derivative_classification(
    reporter: ModuleType,
) -> None:
    report = reporter.build_report()
    rows = report["functions"]
    unclassified = {row["path"] for row in rows if row["derivative_status"] == "unclassified"}
    constant_only_ruleless = {
        row["path"]
        for row in rows
        if row["registry_op"]
        and not reporter._FUNCTION_SPECS[row["path"]].operands
        and not row["jvp_registered"]
        and not row["vjp_registered"]
        and not row["non_differentiable"]
        and row["canonical_op"] not in reporter.STRUCTURAL_OPS
    }
    structural = [row["path"] for row in rows if row["canonical_op"] in reporter.STRUCTURAL_OPS]

    assert unclassified == constant_only_ruleless
    assert structural
    assert all(
        row["derivative_status"] == "not_applicable"
        for row in rows
        if row["canonical_op"] in reporter.STRUCTURAL_OPS
    )
    assert report["summary"]["derivative_statuses"].get("unclassified", 0) == len(
        constant_only_ruleless
    )


def test_fixed_arity_multi_output_ops_are_reported_as_staged(reporter: ModuleType) -> None:
    rows = {row["path"]: row for row in reporter.build_report()["functions"]}

    for path in ("linalg.eigh", "linalg.qr", "linalg.slogdet", "linalg.svd"):
        row = rows[path]
        assert row["result_kind"] == "multiple_arrays"
        assert row["declared_num_outputs"] > 1
        assert row["classification"] == "staged"

    assert all(
        row["declared_num_outputs"] is None
        for row in rows.values()
        if row["classification"] == "multi_output_unsupported"
    )


def test_execution_catalog_accounts_for_every_staged_function(
    reporter: ModuleType,
) -> None:
    rows = {row["path"]: row for row in reporter.build_report()["functions"]}
    staged = {path for path, row in rows.items() if row["classification"] == "staged"}
    executable = {
        path for path, row in rows.items() if row["execution_qualification"] == "executable"
    }
    declared_gaps = {
        path
        for path, row in rows.items()
        if row["execution_qualification"] == "declared_not_executable"
    }
    portable = {path for path, row in rows.items() if row["portable_execution_case"]}

    assert executable | declared_gaps == staged
    assert executable & declared_gaps == set()
    assert declared_gaps == set()
    assert len(executable) == len(staged)
    assert len(portable) == sum(case.portable for case in reporter._execution_cases().values())


def test_human_report_renders_the_live_registry(reporter: ModuleType) -> None:
    rendered = reporter._human_report(reporter.build_report())
    assert "Array API 2024.12 support" in rendered
    assert "Support classifications:" in rendered
    assert "Derivative status:" in rendered


@pytest.mark.parametrize(
    ("version", "expected_count"),
    [("2022.12", 152), ("2023.12", 164), ("2024.12", 170)],
)
def test_report_materializes_each_declared_revision(
    reporter: ModuleType,
    version: str,
    expected_count: int,
) -> None:
    report = reporter.build_report(version)
    rows = report["functions"]
    profile = materialize_array_api_profile(version)

    assert report["api_version"] == version
    assert report["summary"]["official_functions"] == expected_count
    assert {row["path"] for row in rows} == set(profile.signatures)
