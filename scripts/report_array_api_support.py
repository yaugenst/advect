"""Report Advect support for one declared Array API revision."""

from __future__ import annotations

import argparse
import inspect
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast, get_origin

import array_api_strict

from advect.core._array_api.evidence import operation_cases
from advect.core._array_api.frontend import (
    _ARRAY_API_COMPOSITES,
    _ARRAY_API_META_FUNCTIONS,
    _FUNCTION_SPECS,
    _NONDIFFERENTIABLE_ARRAY_API_COMPOSITES,
    _STAGED_ARRAY_API_COMPOSITES,
)
from advect.core._array_api.profiles import (
    LATEST_ARRAY_API_VERSION,
    SUPPORTED_ARRAY_API_VERSIONS,
    materialize_array_api_profile,
)
from advect.core._array_api.signatures import official_signatures
from advect.core._primitive_classification import STRUCTURAL_OPS
from advect.core._registry import get_registry
from scripts._support.evidence import evidence_report_header

if TYPE_CHECKING:
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
    "multi_output_unsupported",
    "non_array_result_unsupported",
)


@dataclass(frozen=True, slots=True)
class _OfficialFunction:
    path: str
    category: str
    function: object
    source_module: str


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


def _category(source_module: str) -> str:
    leaf = source_module.rsplit(".", 1)[-1]
    return leaf.removeprefix("_").removesuffix("_functions")


def _public_functions(module: ModuleType) -> list[tuple[str, object]]:
    names = cast("tuple[str, ...] | list[str]", getattr(module, "__all__", ()))
    return [
        (name, value) for name in names if inspect.isfunction(value := getattr(module, name, None))
    ]


def _official_functions(
    array_api_version: str = LATEST_ARRAY_API_VERSION,
) -> tuple[_OfficialFunction, ...]:
    profile = materialize_array_api_profile(array_api_version)
    array_api_strict.set_array_api_strict_flags(api_version=array_api_version)
    api_version = getattr(array_api_strict, "__array_api_version__", None)
    if api_version != array_api_version:
        msg = (
            f"Expected array-api-strict to expose Array API {array_api_version}, "
            f"found {api_version!r}"
        )
        raise RuntimeError(msg)

    functions: list[_OfficialFunction] = []
    for name, function in _public_functions(array_api_strict):
        if name not in profile.signatures:
            continue
        source_module = str(getattr(function, "__module__", ""))
        if source_module in _STRICT_HELPER_MODULES:
            continue
        functions.append(
            _OfficialFunction(
                path=name,
                category=_category(source_module),
                function=function,
                source_module=source_module,
            )
        )

    for namespace in ("fft", "linalg"):
        module = cast("ModuleType", getattr(array_api_strict, namespace))
        functions.extend(
            _OfficialFunction(
                path=f"{namespace}.{name}",
                category=namespace,
                function=function,
                source_module=str(getattr(function, "__module__", "")),
            )
            for name, function in _public_functions(module)
            if f"{namespace}.{name}" in profile.signatures
        )

    functions.sort(key=lambda item: item.path)
    paths = [item.path for item in functions]
    if len(paths) != len(set(paths)):
        duplicates = sorted(path for path, count in Counter(paths).items() if count > 1)
        msg = f"array-api-strict exposed duplicate function paths: {duplicates}"
        raise RuntimeError(msg)
    missing = sorted(set(profile.signatures) - set(paths))
    if missing:
        message = (
            f"array-api-strict does not expose the frozen Array API {array_api_version} "
            f"callables: {missing!r}"
        )
        raise RuntimeError(message)
    return tuple(functions)


def _annotation_text(annotation: object) -> str:
    if annotation is inspect.Signature.empty:
        return ""
    return str(annotation)


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


def _derivative_status(
    *,
    jvp_registered: bool,
    vjp_registered: bool,
    non_differentiable: bool,
    catalogued: bool,
    compile_time_metadata: bool,
    composite: bool,
    structural: bool,
) -> str:
    if compile_time_metadata or structural:
        return "not_applicable"
    if not catalogued:
        return "not_catalogued"
    if non_differentiable:
        return "non_differentiable"
    if composite:
        return "composite"
    if jvp_registered and vjp_registered:
        return "jvp_and_vjp"
    if jvp_registered:
        return "jvp"
    if vjp_registered:
        return "vjp"
    return "unclassified"


def _support_classification(
    *,
    catalogued: bool,
    staged: bool,
    has_array_operand: bool,
    result_kind: str,
    fixed_output_arity: bool,
    compile_time_metadata: bool,
    composite: bool,
) -> str:
    if compile_time_metadata:
        return "compile_time_metadata"
    if composite:
        return "staged" if staged else "dynamic_only"
    if not catalogued and not has_array_operand:
        return "provider_passthrough"
    if result_kind == "multiple_arrays" and not fixed_output_arity:
        return "multi_output_unsupported"
    if result_kind == "metadata":
        return "non_array_result_unsupported"
    if staged:
        return "staged"
    if catalogued:
        return "dynamic_only"
    return "missing_binder"


def _function_row(
    function: _OfficialFunction,
    *,
    array_api_version: str,
    execution_cases: dict[str, OperationCase],
) -> dict[str, object]:
    function_spec = _FUNCTION_SPECS.get(function.path)
    canonical_op = function_spec.op if function_spec is not None else None
    composite = function.path in _ARRAY_API_COMPOSITES
    catalogued = canonical_op is not None or composite
    registry = get_registry()
    op_definition = registry.get_optional(canonical_op) if canonical_op is not None else None
    staged = (
        function.path in _STAGED_ARRAY_API_COMPOSITES
        if composite
        else op_definition is not None and op_definition.abstract_evaluator is not None
    )
    jvp_registered = op_definition is not None and op_definition.jvp is not None
    vjp_registered = op_definition is not None and op_definition.vjp is not None
    non_differentiable = (
        op_definition is not None and op_definition.non_differentiable_reason is not None
    ) or function.path in _NONDIFFERENTIABLE_ARRAY_API_COMPOSITES
    has_array_operand = _has_array_operand(function.function)
    result_kind = _result_kind(function.function)
    compile_time_metadata = function.path in _ARRAY_API_META_FUNCTIONS
    structural = canonical_op in STRUCTURAL_OPS
    declared_num_outputs = op_definition.num_outputs if op_definition is not None else None
    fixed_output_arity = declared_num_outputs is not None and declared_num_outputs > 1
    execution_case = execution_cases.get(function.path)
    official_signature = official_signatures(array_api_version)[function.path]
    provider_signature = _normalized_signature(function.function)
    classification = _support_classification(
        catalogued=catalogued,
        staged=staged,
        has_array_operand=has_array_operand,
        result_kind=result_kind,
        fixed_output_arity=fixed_output_arity,
        compile_time_metadata=compile_time_metadata,
        composite=composite,
    )
    return {
        "abstract_rule": staged,
        "binding": (
            "composite"
            if composite
            else "operation"
            if canonical_op is not None
            else "compile_time_metadata"
            if compile_time_metadata
            else "none"
        ),
        "canonical_op": canonical_op,
        "category": function.category,
        "classification": classification,
        "compile_time_metadata": compile_time_metadata,
        "declared_num_outputs": declared_num_outputs,
        "derivative_status": _derivative_status(
            jvp_registered=jvp_registered,
            vjp_registered=vjp_registered,
            non_differentiable=non_differentiable,
            catalogued=catalogued,
            compile_time_metadata=compile_time_metadata,
            composite=composite,
            structural=structural,
        ),
        "execution_qualification": (
            "executable"
            if execution_case is not None and classification == "staged"
            else "dynamic_executable"
            if execution_case is not None
            else "not_catalogued"
        ),
        "has_array_operand": has_array_operand,
        "jvp_registered": jvp_registered,
        "non_differentiable": non_differentiable,
        "path": function.path,
        "portable_execution_case": (
            execution_case.portable if execution_case is not None else False
        ),
        "registry_op": op_definition is not None,
        "result_kind": result_kind,
        "signature": official_signature,
        "source_module": function.source_module,
        "provider_signature": provider_signature,
        "provider_signature_deviation": provider_signature != official_signature,
        "vjp_registered": vjp_registered,
    }


def _execution_cases(
    array_api_version: str = LATEST_ARRAY_API_VERSION,
) -> dict[str, OperationCase]:
    return {case.path: case for case in operation_cases(array_api_version)}


def _counts(rows: list[dict[str, object]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def build_report(
    array_api_version: str = LATEST_ARRAY_API_VERSION,
) -> dict[str, object]:
    """Build a deterministic support report from the installed and live registries."""
    official_functions = _official_functions(array_api_version)
    execution_cases = _execution_cases(array_api_version)
    rows = [
        _function_row(
            function,
            array_api_version=array_api_version,
            execution_cases=execution_cases,
        )
        for function in official_functions
    ]
    official_paths = {function.path for function in official_functions}
    extra_catalog_paths = [
        {
            "canonical_op": _FUNCTION_SPECS[path].op,
            "path": path,
        }
        for path in sorted(set(_FUNCTION_SPECS).difference(official_paths))
    ]
    classifications = _counts(rows, "classification")
    derivative_statuses = _counts(rows, "derivative_status")
    execution_qualifications = _counts(rows, "execution_qualification")
    provider_signature_deviations = [
        str(row["path"]) for row in rows if row["provider_signature_deviation"]
    ]

    return {
        **evidence_report_header(
            schema_version=3,
            report_kind="advect.array-api-support",
        ),
        "api_version": array_api_version,
        "array_api_strict_version": str(getattr(array_api_strict, "__version__", "unknown")),
        "extra_catalog_paths": extra_catalog_paths,
        "functions": rows,
        "provider_signature_deviations": provider_signature_deviations,
        "summary": {
            "catalog_paths": len(_FUNCTION_SPECS) + len(_ARRAY_API_COMPOSITES),
            "catalogued_official_functions": sum(
                row["binding"] in {"operation", "composite"} for row in rows
            ),
            "classifications": classifications,
            "compile_time_metadata_functions": classifications.get("compile_time_metadata", 0),
            "derivative_statuses": derivative_statuses,
            "execution_qualifications": execution_qualifications,
            "official_functions": len(rows),
            "result_kinds": _counts(rows, "result_kind"),
            "portable_execution_cases": sum(bool(row["portable_execution_case"]) for row in rows),
            "provider_signature_deviations": len(provider_signature_deviations),
            "staged_functions": classifications.get("staged", 0),
            "supported_transform_functions": sum(
                classifications.get(classification, 0)
                for classification in ("staged", "dynamic_only", "compile_time_metadata")
            ),
            "transform_applicable_functions": sum(
                bool(row["has_array_operand"]) or row["binding"] != "none" for row in rows
            ),
            "uncatalogued_official_functions": sum(
                row["binding"] not in {"operation", "composite"} for row in rows
            ),
            "unsupported_transform_functions": sum(
                classifications.get(classification, 0)
                for classification in (
                    "missing_binder",
                    "multi_output_unsupported",
                    "non_array_result_unsupported",
                )
            ),
        },
    }


def _human_report(report: dict[str, object]) -> str:
    summary = cast("dict[str, object]", report["summary"])
    classifications = cast("dict[str, int]", summary["classifications"])
    derivatives = cast("dict[str, int]", summary["derivative_statuses"])
    rows = cast("list[dict[str, object]]", report["functions"])
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for row in rows:
        grouped[str(row["classification"])].append(str(row["path"]))

    lines = [
        (
            f"Array API {report['api_version']} support "
            f"(array-api-strict {report['array_api_strict_version']})"
        ),
        f"Official functions: {summary['official_functions']}",
        f"Transform-applicable functions: {summary['transform_applicable_functions']}",
        f"Catalogued official functions: {summary['catalogued_official_functions']}",
        f"Compile-time metadata functions: {summary['compile_time_metadata_functions']}",
        f"Abstractly staged functions: {summary['staged_functions']}",
        f"Supported transform functions: {summary['supported_transform_functions']}",
        f"Unsupported transform functions: {summary['unsupported_transform_functions']}",
        f"Executable staged cases: {summary['execution_qualifications'].get('executable', 0)}",
        f"Portable execution cases: {summary['portable_execution_cases']}",
        f"Provider signature deviations: {summary['provider_signature_deviations']}",
        "",
        "Support classifications:",
    ]
    lines.extend(
        f"  {classification}: {classifications.get(classification, 0)}"
        for classification in _SUPPORT_CLASSIFICATIONS
    )
    lines.extend(["", "Derivative status:"])
    lines.extend(f"  {status}: {count}" for status, count in derivatives.items())
    execution = cast("dict[str, int]", summary["execution_qualifications"])
    lines.extend(["", "Execution qualification:"])
    lines.extend(f"  {status}: {count}" for status, count in execution.items())

    for classification in _SUPPORT_CLASSIFICATIONS:
        paths = grouped[classification]
        if not paths:
            continue
        lines.extend(["", f"{classification}:", f"  {', '.join(paths)}"])

    extra_catalog_paths = cast("list[dict[str, str]]", report["extra_catalog_paths"])
    if extra_catalog_paths:
        rendered = ", ".join(
            f"{item['path']} -> {item['canonical_op']}" for item in extra_catalog_paths
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
