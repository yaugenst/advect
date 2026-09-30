"""Tests for the generated Array API support inventory."""

from __future__ import annotations

import ast
from typing import Any

import array_api_strict
import pytest
from scripts import report_array_api_support as reporter, run_array_api_conformance

import advect as ad
from advect.core._array_api.evidence import operation_cases
from advect.core._array_api.frontend import _ARRAY_API_META_FUNCTIONS
from advect.core._array_api.profiles import (
    LATEST_ARRAY_API_VERSION,
    SUPPORTED_ARRAY_API_VERSIONS,
)
from advect.core._array_api.signatures import OFFICIAL_SIGNATURES, official_parameter_names
from advect.core._array_api.support import build_support_profile

_OFFICIAL_FUNCTION_COUNTS = {"2022.12": 152, "2023.12": 164, "2024.12": 170}
_CATALOG_COLUMNS = ("lowering", "abstract", "jvp", "vjp")


@pytest.fixture(scope="module")
def reports() -> dict[str, dict[str, Any]]:
    return {version: reporter.build_report(version) for version in SUPPORTED_ARRAY_API_VERSIONS}


@pytest.fixture(scope="module")
def catalog() -> dict[str, dict[str, Any]]:
    extension = ad.support_catalog()["extensions"]["array_api"]
    return {row["callable"]: row for row in extension["functions"]}


def test_discovers_the_complete_installed_2024_12_function_surface() -> None:
    paths = [path for path, _function in reporter._official_functions()]

    assert len(paths) == 170
    assert len(paths) == len(set(paths))
    assert paths == sorted(paths)
    assert "pow" in paths
    assert "fft.fft" in paths
    assert "linalg.solve" in paths
    assert "__array_namespace_info__" not in paths
    assert "set_array_api_strict_flags" not in paths


def test_reporting_an_older_revision_keeps_the_reference_provider_flags() -> None:
    flags = array_api_strict.get_array_api_strict_flags()

    reporter._official_functions("2022.12")

    assert array_api_strict.get_array_api_strict_flags() == flags


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
    reports: dict[str, dict[str, Any]],
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

    report = reports[LATEST_ARRAY_API_VERSION]
    rows = {row["path"]: row for row in report["functions"]}
    assert rows["expand_dims"]["signature"] == "(x, /, axis)"
    assert rows["expand_dims"]["provider_signature"] == "(x, /, *, axis)"
    assert rows["expand_dims"]["provider_signature_deviation"] is True
    assert "expand_dims" in report["provider_signature_deviations"]


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


@pytest.mark.parametrize("version", SUPPORTED_ARRAY_API_VERSIONS)
def test_report_joins_each_revision_profile_to_the_public_catalog(
    reports: dict[str, dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
    version: str,
) -> None:
    report = reports[version]
    rows = report["functions"]
    paths = [row["path"] for row in rows]
    profile = {row["path"]: row for row in build_support_profile(version)["callables"]}
    summary = report["summary"]

    assert report["schema_version"] == 4
    assert report["report_kind"] == "advect.array-api-support"
    assert report["environment"]["source_revision"]
    assert report["environment"]["python"]
    assert report["environment"]["machine"]["platform"]
    assert report["api_version"] == version
    assert paths == sorted(profile)
    assert summary["official_functions"] == len(rows) == _OFFICIAL_FUNCTION_COUNTS[version]
    assert sum(summary["classifications"].values()) == len(rows)
    assert set(summary["classifications"]) <= set(reporter._SUPPORT_CLASSIFICATIONS)
    for row in rows:
        claim = profile[row["path"]]
        entry = catalog.get(row["path"], dict.fromkeys(_CATALOG_COLUMNS))
        assert [row[key] for key in ("signature", "modes", "complete")] == [
            claim[key] for key in ("signature", "modes", "complete")
        ]
        assert [row[key] for key in _CATALOG_COLUMNS] == [entry[key] for key in _CATALOG_COLUMNS]
    assert report["extra_catalog_paths"] == [
        {"lowering": catalog[path]["lowering"], "path": path}
        for path in sorted(set(catalog).difference(paths))
    ]


def test_report_rejects_a_revision_claim_absent_from_the_latest_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = ad.support_catalog()
    extension = catalog["extensions"]["array_api"]
    extension["functions"] = [row for row in extension["functions"] if row["callable"] != "abs"]
    monkeypatch.setattr(reporter, "support_catalog", lambda: catalog)

    with pytest.raises(RuntimeError, match=r"does not list: \['abs'\]"):
        reporter.build_report("2022.12")


def test_report_rejects_an_unclaimed_callable_the_latest_revision_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An incomplete callable has no modes, but the latest-only catalog join
    # would still misclassify it if a later revision removed it.
    def with_retired_callable(version: str) -> dict[str, Any]:
        profile = build_support_profile(version)
        retired = {**profile["callables"][0], "complete": False, "modes": [], "path": "retired"}
        return {**profile, "callables": [*profile["callables"], retired]}

    monkeypatch.setattr(reporter, "build_support_profile", with_retired_callable)

    with pytest.raises(RuntimeError, match=r"does not list: \['retired'\]"):
        reporter.build_report("2022.12")


@pytest.mark.parametrize("version", SUPPORTED_ARRAY_API_VERSIONS)
def test_classification_follows_the_catalog_abstract_rule(
    reports: dict[str, dict[str, Any]],
    version: str,
) -> None:
    rows = {row["path"]: row for row in reports[version]["functions"]}

    def classified(classification: str) -> set[str]:
        return {path for path, row in rows.items() if row["classification"] == classification}

    assert classified("compile_time_metadata") == _ARRAY_API_META_FUNCTIONS
    assert all(rows[path]["lowering"] == "metadata" for path in _ARRAY_API_META_FUNCTIONS)
    assert classified("dynamic_only") == {
        path for path, row in rows.items() if row["abstract"] == "no"
    }
    assert classified("staged") == {
        path for path, row in rows.items() if row["abstract"] in {"yes", "composite"}
    }
    assert rows["from_dlpack"]["classification"] == "provider_passthrough"
    assert all(
        not rows[path]["has_array_operand"] and rows[path]["lowering"] is None
        for path in classified("provider_passthrough")
    )
    for path in ("linalg.eigh", "linalg.qr", "linalg.slogdet", "linalg.svd"):
        assert rows[path]["result_kind"] == "multiple_arrays"
        assert rows[path]["classification"] == "staged"


@pytest.mark.parametrize("version", SUPPORTED_ARRAY_API_VERSIONS)
def test_execution_catalog_accounts_for_every_staged_function(
    reports: dict[str, dict[str, Any]],
    version: str,
) -> None:
    rows = reports[version]["functions"]
    staged = {row["path"] for row in rows if row["classification"] == "staged"}
    executable = {row["path"] for row in rows if row["execution_qualification"] == "executable"}
    portable = {row["path"] for row in rows if row["portable_execution_case"]}

    assert executable == staged
    assert portable == {case.path for case in operation_cases(version) if case.portable}


@pytest.mark.parametrize("version", SUPPORTED_ARRAY_API_VERSIONS)
def test_only_structural_ops_lack_derivative_rules(
    reports: dict[str, dict[str, Any]],
    version: str,
) -> None:
    rows = reports[version]["functions"]
    ruleless = {row["path"] for row in rows if "no" in {row["jvp"], row["vjp"]}}

    assert ruleless
    assert ruleless <= {row["path"] for row in rows if row["structural"]}


def test_human_report_renders_the_catalog_join(reports: dict[str, dict[str, Any]]) -> None:
    rendered = reporter._human_report(reports[LATEST_ARRAY_API_VERSION])

    assert "Array API 2024.12 support" in rendered
    assert "Support classifications:" in rendered
    assert "JVP:" in rendered
    assert "VJP:" in rendered
    assert "Structural ops without derivative rules (no differentiable input):" in rendered
