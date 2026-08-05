"""Helpers for binding NumPy's live foreign signatures."""

from __future__ import annotations

import inspect
from typing import Any, cast


def normalize_required_positionals(
    func: object,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    positional: tuple[inspect.Parameter, ...] | None = None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Move required positional-or-keyword operands into handler positions."""
    if positional is None:
        try:
            signature = inspect.signature(cast("Any", func))
        except (TypeError, ValueError):
            return args, kwargs
        signature.bind(*args, **kwargs)
        positional = tuple(
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        )

    normalized = list(args)
    remaining = dict(kwargs)
    for parameter in positional[len(args) :]:
        if (
            parameter.kind is inspect.Parameter.POSITIONAL_ONLY
            or parameter.default is not inspect.Parameter.empty
            or parameter.name not in remaining
        ):
            break
        normalized.append(remaining.pop(parameter.name))
    return tuple(normalized), remaining


__all__ = ["normalize_required_positionals"]
