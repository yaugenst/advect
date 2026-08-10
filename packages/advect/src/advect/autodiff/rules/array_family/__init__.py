"""Backend-neutral runtime support for built-in array operation rules."""

from __future__ import annotations

from advect.autodiff.rules.array_family.providers import (
    ArrayFamilyBackendProvider,
    resolve_array_family_backend_provider,
)

__all__ = [
    "ArrayFamilyBackendProvider",
    "resolve_array_family_backend_provider",
]
