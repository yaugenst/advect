"""Normalized signatures materialized from Advect's Array API profiles."""

from __future__ import annotations

import ast
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


@cache
def _signature_arguments(path: str, version: str) -> ast.arguments:
    signature = official_signatures(version)[path]
    function = cast(
        "ast.FunctionDef",
        ast.parse(f"def _function{signature}:\n    pass").body[0],
    )
    return function.args


def official_parameter_names(
    path: str,
    version: str = LATEST_ARRAY_API_VERSION,
) -> tuple[str, ...]:
    """Derive parameter names from one revision's normalized signature."""
    arguments = _signature_arguments(path, version)
    names = [argument.arg for argument in (*arguments.posonlyargs, *arguments.args)]
    if arguments.vararg is not None:
        names.append(arguments.vararg.arg)
    names.extend(argument.arg for argument in arguments.kwonlyargs)
    if arguments.kwarg is not None:
        names.append(arguments.kwarg.arg)
    return tuple(names)


def official_positional_parameter_names(
    path: str,
    version: str = LATEST_ARRAY_API_VERSION,
) -> tuple[str, ...]:
    """Return parameters accepted positionally by one revision."""
    arguments = _signature_arguments(path, version)
    return tuple(argument.arg for argument in (*arguments.posonlyargs, *arguments.args))


__all__ = [
    "OFFICIAL_SIGNATURES",
    "official_parameter_names",
    "official_positional_parameter_names",
    "official_signatures",
]
