"""Built-in fallback discovery for arrays recognized by ``array-api-compat``."""

from __future__ import annotations

import warnings

import array_api_compat

from advect.core._array_api_profiles import SUPPORTED_ARRAY_API_VERSIONS
from advect.core._array_namespace import _configure_array_namespace_fallback


def _resolve_namespace(value: object, *, api_version: str | None) -> object | None:
    """Return the native NumPy namespace or one upstream compatibility namespace."""
    if api_version is not None and api_version not in SUPPORTED_ARRAY_API_VERSIONS:
        return None
    if array_api_compat.is_numpy_array(value):
        try:
            return array_api_compat.array_namespace(value, use_compat=False)
        except TypeError:
            return None
    if bool(getattr(value, "requires_grad", False)):
        msg = (
            "Advect does not compose with an array provider's active autodiff tape. "
            "Pass an explicitly detached array or use that provider's autodiff system."
        )
        raise TypeError(msg)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    rf"The {api_version} version of the array API specification "
                    r"was requested but the returned namespace is actually version .*"
                ),
                category=UserWarning,
                module=r"array_api_compat\..*",
            )
            namespace = (
                array_api_compat.array_namespace(value, use_compat=True)
                if api_version is None
                else array_api_compat.array_namespace(
                    value,
                    api_version=api_version,
                    use_compat=True,
                )
            )
    except TypeError:
        return None
    if not callable(getattr(namespace, "asarray", None)) or not callable(
        getattr(namespace, "__array_namespace_info__", None)
    ):
        return None
    return namespace


def _can_donate(value: object) -> bool:
    """Admit mutable owned CuPy arrays to the staged donation path."""
    return array_api_compat.is_cupy_array(value)


_configure_array_namespace_fallback(_resolve_namespace, can_donate=_can_donate)
