"""Smoke tests for package imports and the public export inventory."""

from __future__ import annotations

import subprocess
import sys
from importlib import import_module

import advect as ad


def test_import_and_version() -> None:
    # Package should import from source layout and expose a version string
    assert isinstance(ad.__version__, str)
    assert ad.__version__


def test_public_autodiff_exports_resolve_from_root() -> None:
    assert set(ad._AUTODIFF_EXPORT_MODULES) <= set(ad.__all__)
    for name, module in ad._AUTODIFF_EXPORT_MODULES.items():
        leaf = import_module(f"advect.autodiff.api.{module}")
        assert getattr(ad, name) is getattr(leaf, name)


def test_root_import_keeps_autodiff_lazy() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import advect; "
                "assert not any(name == 'advect.autodiff' or "
                "name.startswith('advect.autodiff.') for name in sys.modules)"
            ),
        ],
        check=True,
    )
