"""Backend-neutral runtime support for built-in array operation rules."""

from __future__ import annotations

from advect.autodiff.rules.array_family.providers import (
    ArrayFamilyBackendProvider,
    get_array_family_backend_provider,
    register_array_family_backend_provider,
    resolve_array_family_backend_provider,
)

__all__ = [
    "ArrayFamilyBackendProvider",
    "get_array_family_backend_provider",
    "register_array_family_backend_provider",
    "resolve_array_family_backend_provider",
]
