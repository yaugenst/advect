"""Smoke tests for package imports and the public export inventory."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from importlib import import_module
from typing import TYPE_CHECKING

import advect as ad

if TYPE_CHECKING:
    import pytest


def test_import_and_version() -> None:
    # Package should import from source layout and expose a version string
    assert isinstance(ad.__version__, str)
    assert ad.__version__


def test_source_checkout_has_a_local_version_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_version(_distribution_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    assert ad.__version__  # Cache the installed value so monkeypatch restores it.
    monkeypatch.setattr(importlib.metadata, "version", missing_version)
    monkeypatch.delitem(vars(ad), "__version__")

    assert ad.__version__ == "0.0.0+local"


def test_public_autodiff_exports_resolve_from_root() -> None:
    assert set(ad._AUTODIFF_EXPORT_MODULES) <= set(ad.__all__)
    for name, module in ad._AUTODIFF_EXPORT_MODULES.items():
        leaf = import_module(f"advect.autodiff.api.{module}")
        assert getattr(ad, name) is getattr(leaf, name)


_OPTIONAL_PACKAGES = ("autograd", "jax", "pandas", "scipy", "torch", "xarray")
_LAZY_MODULES = ("advect.autodiff", "advect.scipy", "advect.xarray", "importlib.metadata")


def test_base_imports_leave_optional_dependencies_and_transforms_unloaded() -> None:
    # Only modules that the Advect imports add count, so neither site setup (an
    # editable-install finder, sitecustomize) nor what the required NumPy and
    # array-api-compat dependencies load themselves is blamed on Advect.
    script = f"""
import sys
import array_api_compat
import numpy
preloaded = set(sys.modules)
import advect
import advect.interop

loaded = sorted(
    name
    for name in set(sys.modules) - preloaded
    if name.partition(".")[0] in {_OPTIONAL_PACKAGES!r}
    or any(name == lazy or name.startswith(lazy + ".") for lazy in {_LAZY_MODULES!r})
)
assert not loaded, f"import advect loaded {{loaded}}"
"""
    completed = subprocess.run(  # noqa: S603 - fixed interpreter; script is test-owned
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
