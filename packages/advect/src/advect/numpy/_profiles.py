"""NumPy release bounds qualified by Advect's compatibility matrix."""

from __future__ import annotations

_VERSION_COMPONENTS = 2
_SUPPORTED_MINORS = frozenset({"2.0", "2.1", "2.2", "2.3", "2.4"})


def numpy_minor(version: str) -> str:
    """Normalize a NumPy release string and enforce the supported range."""
    parts = version.split(".")
    if len(parts) < _VERSION_COMPONENTS or not parts[0].isdigit() or not parts[1].isdigit():
        message = f"Could not parse NumPy version {version!r}"
        raise TypeError(message)
    minor = f"{int(parts[0])}.{int(parts[1])}"
    if minor not in _SUPPORTED_MINORS:
        message = (
            f"Advect supports NumPy >=2.0,<2.5; installed version {version!r} is outside that range"
        )
        raise TypeError(message)
    return minor


__all__ = ["numpy_minor"]
