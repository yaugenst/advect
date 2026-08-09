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


def test_core_stage_delegates_frontend_lifecycle_policy() -> None:
    core_dir = Path(__file__).resolve().parents[2] / "src" / "advect" / "core"
    source = (core_dir / "_stage.py").read_text()

    assert "array_factory._advect_stage_context" in source
    assert all(
        provider_detail not in source
        for provider_detail in (
            "_is_numpy_rng",
            "_reject_ambient_randomness",
            "_RNG_NAMES",
            "_RNG_ORIGINALS",
            "_RNG_PATCH_DEPTH",
            "_advect_validate_stage_capture",
            "numpy.random",
            "Generator",
            "RandomState",
        )
    )


def test_staged_frontend_boundary_uses_the_resolved_raw_namespace() -> None:
    core_dir = Path(__file__).resolve().parents[2] / "src" / "advect" / "core"
    source = (core_dir / "_eval_dispatch.py").read_text()

    assert 'getattr(namespace, "raw_namespace", namespace)' in source
    assert "_get_backend_key_from_namespace(raw_namespace)" in source
    assert "NumPy-authored node" in source
    assert "import numpy" not in source


def test_core_abstract_does_not_own_numpy_call_binding() -> None:
    core_dir = Path(__file__).resolve().parents[2] / "src" / "advect" / "core"
    path = core_dir / "_abstract.py"
    source = path.read_text()
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
    }
    abstract_array = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AbstractArray"
    )
    abstract_array_methods = {
        node.name: node
        for node in abstract_array.body
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
    }

    assert {
        "_abstract_average",
        "_abstract_compress",
        "_can_cast_dtype",
        "_functionalize_out",
        "_numpy_op",
    }.isdisjoint(functions)
    assert {"__array_function__", "__array_ufunc__"}.isdisjoint(abstract_array_methods)
    assert "_apply_foreign_array_method" not in functions
    assert "_record_abstract_op" in functions
    assert all(
        argument.arg != "backend"
        for function in functions.values()
        for argument in (*function.args.args, *function.args.kwonlyargs)
    )
    assert "numpy.can_cast" not in source
    assert "numpy.validate_staged_out" not in source
    assert '_advect_backend"] = "numpy"' not in source
    assert "advect.abstract_array_method" not in source

    method_sources = {
        name: ast.unparse(abstract_array_methods[name])
        for name in ("astype", "copy", "mean", "sum")
    }
    assert all(name not in method_sources["astype"] for name in ("casting", "order", "subok"))
    assert "order" not in method_sources["copy"]
    assert all(name not in method_sources["sum"] for name in ("initial", "out", "where"))
    assert all(name not in method_sources["mean"] for name in ("out", "where"))


def test_array_api_tracer_delegates_nested_foreign_protocols() -> None:
    core_dir = Path(__file__).resolve().parents[2] / "src" / "advect" / "core"
    source = (core_dir / "_array_api" / "frontend.py").read_text()
    tree = ast.parse(source)
    methods = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
    }

    assert "_as_numpy_nested_tracer" not in methods
    assert "numpy.wrap_traced" not in source
    assert "advect.foreign_array_ufunc" in source
    assert "advect.foreign_array_function" in source


def test_provider_revision_discovery_has_no_frontend_profile_hook() -> None:
    core_dir = Path(__file__).resolve().parents[2] / "src" / "advect" / "core"
    source = (core_dir / "_array_api" / "providers.py").read_text()

    assert ".array_api_version" not in source
    assert "__array_api_version__" in source


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
