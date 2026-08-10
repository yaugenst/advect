"""Import-boundary checks for the backend-neutral core."""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path


def _is_type_checking_guard(test: ast.AST) -> bool:
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and isinstance(test.value, ast.Name):
        return test.value.id == "typing" and test.attr == "TYPE_CHECKING"
    return False


class _BackendImportVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.offenders: list[tuple[int, str]] = []
        self.in_type_checking_stack: list[bool] = [False]

    @property
    def in_type_checking(self) -> bool:
        return self.in_type_checking_stack[-1]

    def visit_If(self, node: ast.If) -> None:
        is_type_guard = _is_type_checking_guard(node.test)
        self.generic_visit(node.test)

        self.in_type_checking_stack.append(self.in_type_checking or is_type_guard)
        for stmt in node.body:
            self.visit(stmt)
        self.in_type_checking_stack.pop()

        for stmt in node.orelse:
            self.visit(stmt)

    def _record_if_banned(self, module: str, lineno: int) -> None:
        if self.in_type_checking:
            return
        if module == "numpy" or module.startswith("numpy."):
            self.offenders.append((lineno, module))
        if module == "cupy" or module.startswith("cupy."):
            self.offenders.append((lineno, module))
        if module in {"advect.numpy", "advect.cupy"} or module.startswith(
            ("advect.numpy.", "advect.cupy.")
        ):
            self.offenders.append((lineno, module))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._record_if_banned(alias.name, node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None:
            return
        self._record_if_banned(node.module, node.lineno)


def test_core_does_not_runtime_import_provider_frontends() -> None:
    core_dir = Path(__file__).resolve().parents[2] / "src" / "advect" / "core"
    protocol_files = sorted(core_dir.rglob("_array_protocol_*.py"))
    assert [path.relative_to(core_dir).as_posix() for path in protocol_files] == [
        "_array_protocol_helpers.py"
    ]

    offenders: list[str] = []
    for path in sorted(core_dir.rglob("*.py")):
        tree = ast.parse(path.read_text())
        visitor = _BackendImportVisitor()
        visitor.visit(tree)
        for lineno, module in visitor.offenders:
            relative_path = path.relative_to(core_dir).as_posix()
            offenders.append(f"{relative_path}:{lineno}: {module}")

    assert offenders == []


def test_rust_runtime_manifest_has_no_python_adapter_dependency() -> None:
    repository = Path(__file__).resolve().parents[4]
    manifest_path = repository / "packages" / "advect-runtime" / "Cargo.toml"
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    dependency_sections = (
        manifest.get("dependencies", {}),
        manifest.get("build-dependencies", {}),
        manifest.get("dev-dependencies", {}),
    )

    assert all("pyo3" not in dependencies for dependencies in dependency_sections)


def test_array_api_runtime_does_not_import_qualification_modules() -> None:
    core_dir = Path(__file__).resolve().parents[2] / "src" / "advect" / "core"
    array_api_dir = core_dir / "_array_api"
    banned_absolute_modules = {
        "advect.core._array_api.evidence",
        "advect.core._array_api.support",
    }

    for filename in (
        "frontend.py",
        "profiles.py",
        "providers.py",
        "results.py",
        "signatures.py",
    ):
        tree = ast.parse((array_api_dir / filename).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert banned_absolute_modules.isdisjoint(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0:
                    assert node.module not in banned_absolute_modules
                elif node.module is None:
                    assert {"evidence", "support"}.isdisjoint(alias.name for alias in node.names)
                else:
                    assert node.module not in {"evidence", "support"}
