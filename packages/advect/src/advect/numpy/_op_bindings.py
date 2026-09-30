"""Canonical operation bindings for the NumPy backend.

This module is the NumPy plugin's source of truth for mapping backend-native
operation IDs to canonical IR operation IDs.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, cast

from advect.core._array_family_ops import (
    ARRAY_API_TO_CANONICAL,
    _canonical_array_family_op_name,
)
from advect.core._registry import get_registry

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "canonicalize_numpy_op",
    "decanonicalize_array_op",
    "frontend_lowering",
    "staged_numpy_op",
]

_GENERIC_STAGED_NUMPY = frozenset({"linalg.eigh", "linalg.qr", "linalg.slogdet", "linalg.svd"})


@functools.cache
def canonicalize_numpy_op(op_name: str) -> str:
    """Map a fully-qualified ``numpy.*`` op name to canonical op id."""
    if op_name.startswith("numpy."):
        return _canonical_array_family_op_name(op_name.removeprefix("numpy."))
    return op_name


def staged_numpy_op(name: str) -> str:
    """Resolve a NumPy spelling to a registered staged canonical operation."""
    leaf = ARRAY_API_TO_CANONICAL.get(name, name)
    op = _canonical_array_family_op_name(leaf)
    definition = get_registry().get_optional(op)
    rule = None if definition is None else definition.abstract_schema
    if rule is None or (rule.generic_only and name not in _GENERIC_STAGED_NUMPY):
        message = (
            f"NumPy function {name!r} has no abstract staging rule. "
            "Define it as an Advect primitive with def_abstract()."
        )
        raise NotImplementedError(message)
    return op


def decanonicalize_array_op(op_name: str) -> str:
    """Map canonical ``array.*`` / ``array_ext.*`` op names back to ``numpy.*``."""
    if op_name.startswith("array."):
        return f"numpy.{op_name.removeprefix('array.')}"
    if op_name.startswith("array_ext."):
        return f"numpy.{op_name.removeprefix('array_ext.')}"
    return op_name


def frontend_lowering[HandlerT](target: str) -> Callable[[HandlerT], HandlerT]:
    """Attach a non-conventional primitive target to an executable handler."""

    def decorate(handler: HandlerT) -> HandlerT:
        cast("Any", handler).__advect_lowering__ = target
        return handler

    return decorate
