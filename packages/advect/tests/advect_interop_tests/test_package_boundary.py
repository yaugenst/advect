"""Import and optional-dependency boundaries for framework interop."""

from __future__ import annotations

import subprocess
import sys
from importlib.metadata import metadata

import pytest

_FRAMEWORKS = ("autograd", "jax", "torch")


def _run(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed interpreter; script is test-owned
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )


def test_base_and_interop_package_imports_are_framework_free() -> None:
    completed = _run(
        f"""
import sys
import advect
import advect.interop

frameworks = set({_FRAMEWORKS!r})
loaded = frameworks.intersection(name.partition(".")[0] for name in sys.modules)
assert not loaded, loaded
"""
    )
    assert completed.returncode == 0, completed.stderr


def test_framework_dependencies_use_only_individual_extras() -> None:
    package_metadata = metadata("advect")
    extras = set(package_metadata.get_all("Provides-Extra") or ())
    assert set(_FRAMEWORKS) <= extras
    assert "interop" not in extras

    requirements = package_metadata.get_all("Requires-Dist") or ()
    for framework in _FRAMEWORKS:
        matches = [requirement for requirement in requirements if requirement.startswith(framework)]
        assert len(matches) == 1
        assert f"extra == '{framework}'" in matches[0]


@pytest.mark.parametrize("framework", _FRAMEWORKS)
def test_framework_module_reports_its_extra_when_dependency_is_missing(
    framework: str,
) -> None:
    completed = _run(
        f"""
import importlib
import importlib.abc
import sys

class BlockFramework(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == {framework!r} or fullname.startswith({framework!r} + "."):
            raise ModuleNotFoundError(name={framework!r})
        return None

sys.meta_path.insert(0, BlockFramework())
try:
    importlib.import_module("advect.interop.{framework}")
except ModuleNotFoundError as error:
    assert "advect[{framework}]" in str(error), error
else:
    raise AssertionError("optional framework import unexpectedly succeeded")
"""
    )
    assert completed.returncode == 0, completed.stderr
