"""Report Advect support for one declared Array API revision."""

from __future__ import annotations

import argparse
import inspect
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, cast, get_origin

import array_api_strict

from advect import support_catalog
from advect.core._array_api.evidence import operation_cases
from advect.core._array_api.profiles import (
    LATEST_ARRAY_API_VERSION,
    SUPPORTED_ARRAY_API_VERSIONS,
    materialize_array_api_profile,
)
from advect.core._array_api.support import build_support_profile
from advect.core._primitive_classification import STRUCTURAL_OPS
from scripts._support.array_api import normalized_signature
from scripts._support.evidence import evidence_report_header

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import ModuleType

    from advect.core._array_api.evidence import OperationCase

_STRICT_HELPER_MODULES = frozenset(
    {
        "array_api_strict._flags",
        "array_api_strict._info",
    }
)
_SUPPORT_CLASSIFICATIONS = (
    "staged",
    "dynamic_only",
    "compile_time_metadata",
    "provider_passthrough",
    "missing_binder",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("human", "json"), default="human")
    parser.add_argument(
        "--array-api-version",
        choices=SUPPORTED_ARRAY_API_VERSIONS,
        default=LATEST_ARRAY_API_VERSION,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _category(path: str, source_module: str) -> str:
    namespace = path.rpartition(".")[0]
    leaf = source_module.rsplit(".", 1)[-1]
    return namespace or leaf.removeprefix("_").removesuffix("_functions")


def _public_functions(module: ModuleType, prefix: str = "") -> list[tuple[str, object]]:
    names = cast("tuple[str, ...] | list[str]", getattr(module, "__all__", ()))
    return [
        (f"{prefix}{name}", value)
        for name in names
        if inspect.isfunction(value := getattr(module, name, None))
    ]


def _official_functions(
    array_api_version: str = LATEST_ARRAY_API_VERSION,
) -> list[tuple[str, object]]:
    """Return the reference provider's ``(path, function)`` pairs for one revision."""
    official = frozenset(materialize_array_api_profile(array_api_version).signatures)
    with array_api_strict.ArrayAPIStrictFlags(api_version=array_api_version):
        api_version = getattr(array_api_strict, "__array_api_version__", None)
    if api_version != array_api_version:
        msg = (
            f"Expected array-api-strict to expose Array API {array_api_version}, "
            f"found {api_version!r}"
        )
        raise RuntimeError(msg)

    functions = [
        (path, function)
        for path, function in (
            *_public_functions(array_api_strict),
            *_public_functions(array_api_strict.fft, "fft."),
            *_public_functions(array_api_strict.linalg, "linalg."),
        )
        if path in official and getattr(function, "__module__", "") not in _STRICT_HELPER_MODULES
    ]
    functions.sort(key=lambda item: item[0])
    paths = [path for path, _function in functions]
    if len(paths) != len(set(paths)):
        duplicates = sorted(path for path, count in Counter(paths).items() if count > 1)
        msg = f"array-api-strict exposed duplicate function paths: {duplicates}"
        raise RuntimeError(msg)
    missing = sorted(official - set(paths))
    if missing:
        message = (
            f"array-api-strict does not expose the frozen Array API {array_api_version} "
            f"callables: {missing!r}"
        )
        raise RuntimeError(message)
    return functions


def _annotation_text(annotation: object) -> str:
    if annotation is inspect.Signature.empty:
        return ""
    return str(annotation)


def _has_array_operand(function: object) -> bool:
    signature = inspect.signature(function)
    return any(
        "Array" in _annotation_text(parameter.annotation)
        for parameter in signature.parameters.values()
    )


def _result_kind(function: object) -> str:
    annotation = inspect.signature(function).return_annotation
    origin = get_origin(annotation)
    if origin in {list, tuple}:
        return "multiple_arrays"
    if isinstance(annotation, str) and annotation.startswith(("list[", "tuple[")):
        return "multiple_arrays"
    if isinstance(annotation, type) and issubclass(annotation, tuple):
        return "multiple_arrays"
    if "Array" in _annotation_text(annotation):
        return "array"
    return "metadata"


def _classification(catalog: dict[str, object] | None, *, has_array_operand: bool) -> str:
    """Classify one callable by what its public catalog row can stage structurally.

    ``staged`` means one registered invocation has an abstract rule; the
    revision profile's ``modes`` say whether the whole callable contract is
    claimed.
    """
    if catalog is None:
        return "missing_binder" if has_array_operand else "provider_passthrough"
    if catalog["kind"] == "metadata":
        return "compile_time_metadata"
    return "staged" if catalog["abstract"] in {"yes", "composite"} else "dynamic_only"


def _function_row(
    path: str,
    function: object,
    *,
    catalog: dict[str, object] | None,
    profile: dict[str, object],
    execution_case: OperationCase | None,
) -> dict[str, object]:
    source_module = str(getattr(function, "__module__", ""))
    has_array_operand = _has_array_operand(function)
    classification = _classification(catalog, has_array_operand=has_array_operand)
    provider_signature = normalized_signature(function)
    return {
        "abstract": None if catalog is None else catalog["abstract"],
        "category": _category(path, source_module),
        "classification": classification,
        "complete": profile["complete"],
        "execution_qualification": (
            "not_catalogued"
            if execution_case is None
            else "executable"
            if classification == "staged"
            else "dynamic_executable"
        ),
        "has_array_operand": has_array_operand,
        "jvp": None if catalog is None else catalog["jvp"],
        "lowering": None if catalog is None else catalog["lowering"],
        "modes": profile["modes"],
        "path": path,
        "portable_execution_case": execution_case is not None and execution_case.portable,
        "provider_signature": provider_signature,
        "provider_signature_deviation": provider_signature != profile["signature"],
        "result_kind": _result_kind(function),
        "signature": profile["signature"],
        "source_module": source_module,
        "structural": catalog is not None and catalog["lowering"] in STRUCTURAL_OPS,
        "vjp": None if catalog is None else catalog["vjp"],
    }


def _counts(values: Iterable[object]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def build_report(
    array_api_version: str = LATEST_ARRAY_API_VERSION,
) -> dict[str, object]:
    """Join the reference namespace to Advect's public catalog and revision profile."""
    extensions = cast("dict[str, dict[str, object]]", support_catalog()["extensions"])
    catalog = {
        str(row["callable"]): row
        for row in cast("list[dict[str, object]]", extensions["array_api"]["functions"])
    }
    profile = {
        str(row["path"]): row
        for row in cast(
            "list[dict[str, object]]",
            build_support_profile(array_api_version)["callables"],
        )
    }
    # The catalog lists only callables the latest revision admits, so a
    # callable a later revision dropped could not join even while unclaimed.
    latest = materialize_array_api_profile(LATEST_ARRAY_API_VERSION)
    uncatalogued = sorted(
        path
        for path, claim in profile.items()
        if (claim["modes"] and path not in catalog) or not latest.admits(path)
    )
    if uncatalogued:
        msg = (
            f"Array API {array_api_version} profiles callables that the "
            f"{LATEST_ARRAY_API_VERSION} support catalog does not list: {uncatalogued!r}"
        )
        raise RuntimeError(msg)
    execution_cases = {case.path: case for case in operation_cases(array_api_version)}
    rows = [
        _function_row(
            path,
            function,
            catalog=catalog.get(path),
            profile=profile[path],
            execution_case=execution_cases.get(path),
        )
        for path, function in _official_functions(array_api_version)
    ]
    extra_catalog_paths = [
        {"lowering": row["lowering"], "path": path}
        for path, row in sorted(catalog.items())
        if path not in profile
    ]
    provider_signature_deviations = [
        str(row["path"]) for row in rows if row["provider_signature_deviation"]
    ]
    catalogued = [row for row in rows if row["lowering"] is not None]

    return {
        **evidence_report_header(
            schema_version=4,
            report_kind="advect.array-api-support",
        ),
        "api_version": array_api_version,
        "array_api_strict_version": str(getattr(array_api_strict, "__version__", "unknown")),
        "extra_catalog_paths": extra_catalog_paths,
        "functions": rows,
        "provider_signature_deviations": provider_signature_deviations,
        "summary": {
            "classifications": _counts(row["classification"] for row in rows),
            "complete_functions": sum(row["complete"] is True for row in rows),
            "execution_qualifications": _counts(row["execution_qualification"] for row in rows),
            "jvp": _counts(row["jvp"] for row in catalogued),
            "official_functions": len(rows),
            "portable_execution_cases": sum(row["portable_execution_case"] is True for row in rows),
            "provider_signature_deviations": len(provider_signature_deviations),
            "result_kinds": _counts(row["result_kind"] for row in rows),
            "vjp": _counts(row["vjp"] for row in catalogued),
        },
    }


def _human_report(report: dict[str, object]) -> str:
    summary = cast("dict[str, object]", report["summary"])
    classifications = cast("dict[str, int]", summary["classifications"])
    execution = cast("dict[str, int]", summary["execution_qualifications"])
    rows = cast("list[dict[str, object]]", report["functions"])
    ruleless = "Structural ops without derivative rules (no differentiable input)"
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for row in rows:
        grouped[str(row["classification"])].append(str(row["path"]))
        if row["structural"] and "no" in {row["jvp"], row["vjp"]}:
            grouped[ruleless].append(str(row["path"]))

    lines = [
        (
            f"Array API {report['api_version']} support "
            f"(array-api-strict {report['array_api_strict_version']})"
        ),
        f"Official functions: {summary['official_functions']}",
        f"Complete callable contracts: {summary['complete_functions']}",
        f"Executable staged cases: {execution.get('executable', 0)}",
        f"Portable execution cases: {summary['portable_execution_cases']}",
        f"Provider signature deviations: {summary['provider_signature_deviations']}",
        "",
        "Support classifications:",
    ]
    lines.extend(
        f"  {classification}: {classifications.get(classification, 0)}"
        for classification in _SUPPORT_CLASSIFICATIONS
    )
    for title, key in (
        ("JVP", "jvp"),
        ("VJP", "vjp"),
        ("Execution qualification", "execution_qualifications"),
    ):
        counts = cast("dict[str, int]", summary[key])
        lines.extend(["", f"{title}:"])
        lines.extend(f"  {status}: {count}" for status, count in counts.items())

    for title in (*_SUPPORT_CLASSIFICATIONS, ruleless):
        paths = grouped[title]
        if paths:
            lines.extend(["", f"{title}:", f"  {', '.join(paths)}"])

    extra_catalog_paths = cast("list[dict[str, str]]", report["extra_catalog_paths"])
    if extra_catalog_paths:
        rendered = ", ".join(
            f"{item['path']} -> {item['lowering']}" for item in extra_catalog_paths
        )
        lines.extend(["", "Nonstandard catalog paths:", f"  {rendered}"])
    provider_deviations = cast("list[str]", report["provider_signature_deviations"])
    if provider_deviations:
        lines.extend(
            [
                "",
                "array-api-strict signature deviations from the official stubs:",
                f"  {', '.join(provider_deviations)}",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    """Render the live report and optionally write it to disk."""
    arguments = _arguments()
    report = build_report(arguments.array_api_version)
    rendered = (
        json.dumps(report, indent=2, sort_keys=True)
        if arguments.format == "json"
        else _human_report(report)
    )
    print(rendered)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(f"{rendered}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
