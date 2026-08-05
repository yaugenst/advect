"""Versioned callable contract for the Python Array API frontend."""

from __future__ import annotations

from typing import TYPE_CHECKING

from advect.core._array_api import (
    _ARRAY_API_COMPOSITE_OPERANDS,
    _ARRAY_API_COMPOSITES,
    _ARRAY_API_META_FUNCTIONS,
    _FUNCTION_SPECS,
    _NONDIFFERENTIABLE_ARRAY_API_COMPOSITES,
    _STAGED_ARRAY_API_COMPOSITES,
)
from advect.core._array_api_evidence import (
    case_parameter_values,
    input_indices,
    metadata_cases,
    operation_evidence_cases,
    operation_modes,
    static_variant,
    static_variant_requirements,
)
from advect.core._array_api_profiles import (
    LATEST_ARRAY_API_VERSION,
    materialize_array_api_profile,
)
from advect.core._array_api_signatures import official_parameter_names
from advect.core._registry import get_registry

if TYPE_CHECKING:
    from advect.core._array_api_evidence import OperationCase

_NONDIFFERENTIABLE_FUNCTIONS = frozenset(
    {
        "all",
        "any",
        "argmax",
        "argmin",
        "argsort",
        "bitwise_and",
        "bitwise_invert",
        "bitwise_left_shift",
        "bitwise_or",
        "bitwise_right_shift",
        "bitwise_xor",
        "count_nonzero",
        "equal",
        "greater",
        "greater_equal",
        "isfinite",
        "isinf",
        "isnan",
        "less",
        "less_equal",
        "logical_and",
        "logical_not",
        "logical_or",
        "logical_xor",
        "not_equal",
        "searchsorted",
        "signbit",
        *_NONDIFFERENTIABLE_ARRAY_API_COMPOSITES,
    }
)
_NONDIFFERENTIABLE_PARAMETERS = frozenset(
    {
        ("empty_like", "x"),
        ("full_like", "x"),
        ("linalg.pinv", "rtol"),
        ("ones_like", "x"),
        ("repeat", "repeats"),
        ("searchsorted", "sorter"),
        ("take", "indices"),
        ("take_along_axis", "indices"),
        ("where", "condition"),
        ("zeros_like", "x"),
    }
)
_SPECIAL_DIFFERENTIABLE_PARAMETERS = frozenset(
    {
        ("diff", "append"),
        ("diff", "prepend"),
    }
)
_METADATA_LIVE_PARAMETERS = frozenset(
    {
        ("can_cast", "from_"),
        ("finfo", "type"),
        ("iinfo", "type"),
        ("result_type", "arrays_and_dtypes"),
    }
)
_PARTIAL_PARAMETERS = {
    ("repeat", "repeats"): "array-valued repeats are not traceable",
}
_SPECIAL_STAGED_PARAMETERS = frozenset(
    {
        ("asarray", "copy"),
        ("asarray", "device"),
        ("diff", "append"),
        ("diff", "prepend"),
        ("searchsorted", "sorter"),
    }
)
_STAGED_PARAMETER_ALIASES = {
    ("tile", "repetitions"): "reps",
}


def _operand_names(path: str, *, version: str) -> frozenset[str]:
    if path in _ARRAY_API_COMPOSITE_OPERANDS:
        return frozenset(_ARRAY_API_COMPOSITE_OPERANDS[path])
    spec = _FUNCTION_SPECS[path]
    names = official_parameter_names(path, version)
    positional = {names[position] for position in spec.positional_operands if position < len(names)}
    return frozenset((*spec.operands, *positional))


def _parameter_role(path: str, name: str, *, version: str) -> str:
    if (path, name) in _PARTIAL_PARAMETERS:
        return "unsupported"
    if (path, name) in _METADATA_LIVE_PARAMETERS:
        return "nondifferentiable"
    if path not in _FUNCTION_SPECS and path not in _ARRAY_API_COMPOSITES:
        return "static" if path in _ARRAY_API_META_FUNCTIONS else "unsupported"
    if (path, name) in _NONDIFFERENTIABLE_PARAMETERS:
        return "nondifferentiable"
    if path in _NONDIFFERENTIABLE_FUNCTIONS:
        return "nondifferentiable" if name in _operand_names(path, version=version) else "static"
    if (path, name) in _SPECIAL_DIFFERENTIABLE_PARAMETERS:
        return "differentiable"
    return "differentiable" if name in _operand_names(path, version=version) else "static"


def _fully_staged(path: str, *, version: str) -> bool:
    if path in _ARRAY_API_COMPOSITES:
        return path in _STAGED_ARRAY_API_COMPOSITES
    spec = _FUNCTION_SPECS[path]
    definition = get_registry().get_optional(spec.op)
    rule = None if definition is None else definition.abstract_schema
    if rule is None:
        return False
    operand_names = _operand_names(path, version=version)
    for name in official_parameter_names(path, version):
        if name in operand_names or (path, name) in _SPECIAL_STAGED_PARAMETERS:
            continue
        alias = _STAGED_PARAMETER_ALIASES.get((path, name), name)
        if alias not in rule.allowed_attrs and alias not in rule.positional_attrs:
            return False
    return True


def _signature(path: str, *, version: str) -> str:
    return materialize_array_api_profile(version).signatures[path]


def _static_parameters(*, version: str) -> dict[str, tuple[str, ...]]:
    profile = materialize_array_api_profile(version)
    return {
        path: tuple(
            name
            for name in official_parameter_names(path, version)
            if _parameter_role(path, name, version=version) == "static"
        )
        for path in profile.signatures
    }


def _expected_modes(path: str, *, version: str) -> tuple[str, ...]:
    if path in _ARRAY_API_META_FUNCTIONS:
        return ("dynamic", "staged", "serialized")
    if path not in _FUNCTION_SPECS and path not in _ARRAY_API_COMPOSITES:
        return ()
    declared_modes = operation_modes(path)
    if declared_modes == ("dynamic",):
        return declared_modes
    if _fully_staged(path, version=version):
        return declared_modes
    return ("dynamic",)


def _evidence_gaps(*, version: str) -> dict[str, tuple[str, ...]]:
    """Return every reason a callable has not earned a complete support claim."""
    profile = materialize_array_api_profile(version)
    static_parameters = _static_parameters(version=version)
    evidence_by_path: dict[str, list[OperationCase]] = {}
    for case in operation_evidence_cases(static_parameters, version):
        evidence_by_path.setdefault(case.path, []).append(case)
    metadata_by_path = {case.path: case for case in metadata_cases()}

    gaps: dict[str, tuple[str, ...]] = {}
    for path in profile.signatures:
        names = official_parameter_names(path, version)
        reasons: list[str] = []
        expected_modes = set(_expected_modes(path, version=version))
        if path in _ARRAY_API_META_FUNCTIONS:
            case = metadata_by_path.get(path)
            if case is None:
                reasons.append("no executable metadata evidence")
            else:
                if set(case.parameters) != set(names):
                    reasons.append("metadata parameters lack executable evidence")
                if set(case.modes) != expected_modes:
                    reasons.append("metadata lifetime evidence is incomplete")
            gaps[path] = tuple(reasons)
            continue

        cases = evidence_by_path.get(path, [])
        if not cases:
            reasons.append("no executable callable evidence")
            gaps[path] = tuple(reasons)
            continue
        if any(set(case.modes) != expected_modes for case in cases):
            reasons.append("claimed lifetimes lack executable evidence")

        if not any(case.variant == "baseline" for case in cases):
            reasons.append("no baseline callable evidence")
        for name in names:
            role = _parameter_role(path, name, version=version)
            if role in {"differentiable", "nondifferentiable"} and not any(
                input_indices(case_parameter_values(case, version)[name]) for case in cases
            ):
                reasons.append(f"{name} lacks live-parameter evidence")

        for name in static_parameters[path]:
            observed = {static_variant(case, name, version) for case in cases}
            required = static_variant_requirements(path, name, version)
            if not required <= observed:
                missing = ", ".join(sorted(required - observed))
                reasons.append(f"{name} lacks {missing} static-variant evidence")
        gaps[path] = tuple(reasons)
    return gaps


def build_support_profile(
    version: str = LATEST_ARRAY_API_VERSION,
) -> dict[str, object]:
    """Return one complete JSON-serializable Array API revision contract."""
    profile = materialize_array_api_profile(version)
    rows: list[dict[str, object]] = []
    evidence_gaps = _evidence_gaps(version=version)
    for path in sorted(profile.signatures):
        names = official_parameter_names(path, version)
        partial = [
            reason
            for (partial_path, _name), reason in _PARTIAL_PARAMETERS.items()
            if partial_path == path
        ]
        catalogued = path in _FUNCTION_SPECS or path in _ARRAY_API_COMPOSITES
        metadata = path in _ARRAY_API_META_FUNCTIONS
        evidence = evidence_gaps[path]
        complete = (catalogued or metadata) and not partial and not evidence
        modes: list[str] = []
        if complete:
            modes.extend(_expected_modes(path, version=version))
        note = (
            "; ".join(sorted({*partial, *evidence}))
            if partial or evidence
            else f"complete Array API {version} callable"
            if complete
            else "not supported by the Array API frontend"
        )
        rows.append(
            {
                "path": path,
                "kind": "function",
                "signature": _signature(path, version=version),
                "parameters": [
                    {"name": name, "role": _parameter_role(path, name, version=version)}
                    for name in names
                ],
                "modes": modes,
                "complete": complete,
                "note": note,
            }
        )
    return {
        "namespace": "array_api",
        "upstream_version": version,
        "callables": rows,
    }


__all__ = ["build_support_profile"]
