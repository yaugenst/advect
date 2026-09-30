"""Import and optional-dependency boundaries for framework interop."""

from __future__ import annotations

import inspect
import subprocess
import sys
from importlib import import_module
from importlib.metadata import metadata

import pytest

import advect.interop._common as interop_common

_FRAMEWORKS = ("autograd", "jax", "torch")


@pytest.mark.parametrize("framework", _FRAMEWORKS)
def test_bridge_preserves_the_wrapped_signature(framework: str) -> None:
    if framework != "autograd":
        pytest.importorskip(framework)
    wrap = import_module(f"advect.interop.{framework}").wrap

    def operation(value: float, *, scale: float = 1.0) -> float:
        return scale * value

    assert inspect.signature(wrap(operation)) == inspect.signature(operation)


def test_framework_dependencies_use_only_individual_extras() -> None:
    package_metadata = metadata("advect")
    extras = set(package_metadata.get_all("Provides-Extra") or ())
    assert set(_FRAMEWORKS) <= extras
    assert not {"interop", "scientific"} & extras

    requirements = package_metadata.get_all("Requires-Dist") or ()
    for framework in _FRAMEWORKS:
        matches = [requirement for requirement in requirements if requirement.startswith(framework)]
        assert len(matches) == 1
        assert f"extra == '{framework}'" in matches[0]


def test_framework_modules_report_their_extra_when_the_dependency_is_missing() -> None:
    script = f"""
import importlib
import importlib.abc
import sys

frameworks = {_FRAMEWORKS!r}

class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.partition(".")[0]
        if root in frameworks:
            raise ModuleNotFoundError(name=root)
        return None

sys.meta_path.insert(0, BlockFrameworks())
failures = []
for framework in frameworks:
    try:
        importlib.import_module(f"advect.interop.{{framework}}")
    except ModuleNotFoundError as error:
        if f"advect[{{framework}}]" not in str(error):
            failures.append(f"{{framework}}: {{error}}")
    else:
        failures.append(f"{{framework}}: optional import unexpectedly succeeded")
assert not failures, failures
"""
    completed = subprocess.run(  # noqa: S603 - fixed interpreter; script is test-owned
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_dependency_errors_preserve_broken_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = ModuleNotFoundError("compiled extension missing", name="torch._C")

    def fail_import(_module_name: str) -> None:
        raise missing

    monkeypatch.setattr(interop_common, "import_module", fail_import)

    with pytest.raises(ModuleNotFoundError) as error:
        interop_common.require_dependency("torch")

    assert error.value is missing
