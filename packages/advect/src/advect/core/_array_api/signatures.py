"""Normalized signatures materialized from Advect's Array API profiles."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, cast

from advect.core._array_api.profiles import (
    LATEST_ARRAY_API_VERSION,
    materialize_array_api_profile,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def official_signatures(
    version: str = LATEST_ARRAY_API_VERSION,
) -> Mapping[str, str]:
    """Return the immutable official callable/signature map for one revision."""
    return materialize_array_api_profile(version).signatures


# The newest materialized surface remains convenient for superset registries.
# Runtime admission always consults the selected profile instead of this name.
OFFICIAL_SIGNATURES = official_signatures()


@dataclass(frozen=True, slots=True)
class OfficialParameter:
    """One parameter of a frozen official signature, in signature order."""

    name: str
    positional: bool
    has_default: bool
    default: object = None
    variadic: bool = False


@cache
def _signature_arguments(path: str, version: str) -> ast.arguments:
    signature = official_signatures(version)[path]
    function = cast(
        "ast.FunctionDef",
        ast.parse(f"def _function{signature}:\n    pass").body[0],
    )
    return function.args


def _parameter(name: str, *, positional: bool, default: ast.expr | None) -> OfficialParameter:
    return OfficialParameter(
        name=name,
        positional=positional,
        has_default=default is not None,
        default=None if default is None else ast.literal_eval(default),
    )


@cache
def official_parameters(
    path: str,
    version: str = LATEST_ARRAY_API_VERSION,
) -> tuple[OfficialParameter, ...]:
    """Return one revision's parameters with their frozen literal defaults."""
    arguments = _signature_arguments(path, version)
    positional = (*arguments.posonlyargs, *arguments.args)
    defaults = (None,) * (len(positional) - len(arguments.defaults)) + tuple(arguments.defaults)
    parameters = [
        _parameter(node.arg, positional=True, default=default)
        for node, default in zip(positional, defaults, strict=True)
    ]
    if arguments.vararg is not None:
        parameters.append(
            OfficialParameter(
                arguments.vararg.arg, positional=True, has_default=False, variadic=True
            )
        )
    parameters.extend(
        _parameter(node.arg, positional=False, default=default)
        for node, default in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True)
    )
    return tuple(parameters)


@cache
def official_parameter_names(
    path: str,
    version: str = LATEST_ARRAY_API_VERSION,
) -> tuple[str, ...]:
    """Derive parameter names from one revision's normalized signature."""
    return tuple(parameter.name for parameter in official_parameters(path, version))


@cache
def official_positional_parameter_names(
    path: str,
    version: str = LATEST_ARRAY_API_VERSION,
) -> tuple[str, ...]:
    """Return parameters accepted positionally by one revision."""
    return tuple(
        parameter.name
        for parameter in official_parameters(path, version)
        if parameter.positional and not parameter.variadic
    )


__all__ = [
    "OFFICIAL_SIGNATURES",
    "OfficialParameter",
    "official_parameter_names",
    "official_parameters",
    "official_positional_parameter_names",
    "official_signatures",
]
