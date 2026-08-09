"""Enforce source and API-reference documentation contracts."""

from __future__ import annotations

import ast
import inspect
import re
from collections import Counter
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

import advect

if TYPE_CHECKING:
    from collections.abc import Iterator

_REPOSITORY = Path(__file__).resolve().parents[4]
_SOURCE_ROOT = _REPOSITORY / "packages" / "advect" / "src" / "advect"
_API_ROOT = _REPOSITORY / "docs" / "api"
_API_DIRECTIVE = re.compile(r"^:::\s+(advect(?:\.[A-Za-z_]\w*)+)\s*$", re.MULTILINE)

_ROOT_REFERENCE_EXCLUSIONS = {
    "__version__": "package metadata is displayed by packaging tools, not API reference",
}
_PUBLIC_MODULE_DIRECTIVES = {
    "advect.numpy",
    "advect.pytree",
    "advect.scipy.ndimage",
    "advect.scipy.optimize",
    "advect.scipy.sparse.linalg",
    "advect.scipy.special",
    "advect.testing",
    "advect.xarray",
}
_INTEROP_OBJECT_DIRECTIVES = {
    "advect.interop.autograd.wrap",
    "advect.interop.jax.wrap",
    "advect.interop.torch.wrap",
}
_PRIMITIVE_HANDLE_DIRECTIVES = {
    "advect.core._primitive.Primitive.def_abstract",
    "advect.core._primitive.Primitive.def_jvp",
    "advect.core._primitive.Primitive.def_transpose",
}
_RUNTIME_PUBLIC_MODULES = (
    "advect.numpy",
    "advect.pytree",
    "advect.testing",
)
_SOURCE_PUBLIC_MODULES = {
    "advect.interop.autograd": _SOURCE_ROOT / "interop" / "autograd.py",
    "advect.interop.jax": _SOURCE_ROOT / "interop" / "jax.py",
    "advect.interop.torch": _SOURCE_ROOT / "interop" / "torch.py",
    "advect.scipy.ndimage": _SOURCE_ROOT / "scipy" / "ndimage.py",
    "advect.scipy.optimize": _SOURCE_ROOT / "scipy" / "optimize.py",
    "advect.scipy.sparse.linalg": _SOURCE_ROOT / "scipy" / "sparse" / "linalg.py",
    "advect.scipy.special": _SOURCE_ROOT / "scipy" / "special.py",
    "advect.xarray": _SOURCE_ROOT / "xarray" / "__init__.py",
}
_SOURCE_NONCALLABLE_EXPORTS = {
    "advect.scipy.optimize": {"RootSolver"},
    "advect.scipy.sparse.linalg": {"LinearOperator", "LinearSolver"},
}


def _production_modules() -> Iterator[Path]:
    yield from sorted((*_SOURCE_ROOT.rglob("*.py"), *_SOURCE_ROOT.rglob("*.pyi")))


def _module_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _literal_all(tree: ast.Module, *, module: str) -> set[str]:
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in statement.targets
        ):
            value = ast.literal_eval(statement.value)
            if isinstance(value, list) and all(isinstance(name, str) for name in value):
                return set(value)
    msg = f"{module} must define a literal string-list __all__"
    raise AssertionError(msg)


def _api_directives() -> Counter[str]:
    directives: Counter[str] = Counter()
    for path in sorted(_API_ROOT.rglob("*.md")):
        directives.update(_API_DIRECTIVE.findall(path.read_text(encoding="utf-8")))
    return directives


def test_every_production_module_has_a_docstring() -> None:
    missing = [
        path.relative_to(_SOURCE_ROOT).as_posix()
        for path in _production_modules()
        if not (ast.get_docstring(_module_tree(path), clean=False) or "").strip()
    ]

    assert missing == []


def test_public_reference_covers_the_root_surface_once() -> None:
    directives = _api_directives()
    excluded = set(_ROOT_REFERENCE_EXCLUSIONS)

    assert excluded < set(advect.__all__)
    assert all(_ROOT_REFERENCE_EXCLUSIONS.values())
    for name in set(advect.__all__) - excluded:
        assert directives[f"advect.{name}"] == 1, name
    for name in excluded:
        assert directives[f"advect.{name}"] == 0, name


def test_public_reference_does_not_duplicate_directives() -> None:
    duplicates = {target: count for target, count in _api_directives().items() if count > 1}

    assert duplicates == {}


def test_public_reference_covers_intended_modules_and_host_bridges_once() -> None:
    directives = _api_directives()

    for target in (
        _PUBLIC_MODULE_DIRECTIVES | _INTEROP_OBJECT_DIRECTIVES | _PRIMITIVE_HANDLE_DIRECTIVES
    ):
        assert directives[target] == 1, target


def test_root_public_callables_and_classes_have_docstrings() -> None:
    missing = [
        name
        for name in advect.__all__
        if (callable(value := getattr(advect, name)) or inspect.isclass(value))
        and not inspect.getdoc(value)
    ]

    assert missing == []


def test_required_base_module_exports_have_object_docstrings() -> None:
    missing: list[str] = []
    for module_name in _RUNTIME_PUBLIC_MODULES:
        module = import_module(module_name)
        for name in module.__all__:
            value = getattr(module, name)
            if (callable(value) or inspect.isclass(value)) and not inspect.getdoc(value):
                missing.append(f"{module_name}.{name}")

    assert missing == []


def test_optional_module_exports_have_source_docstrings() -> None:
    missing: list[str] = []
    for module_name, path in _SOURCE_PUBLIC_MODULES.items():
        tree = _module_tree(path)
        exports = _literal_all(tree, module=module_name)
        definitions = {
            statement.name: statement
            for statement in tree.body
            if isinstance(statement, (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef))
        }
        expected_noncallables = _SOURCE_NONCALLABLE_EXPORTS.get(module_name, set())
        assert exports - definitions.keys() == expected_noncallables, module_name
        missing.extend(
            f"{module_name}.{name}"
            for name in sorted(exports & definitions.keys())
            if not (ast.get_docstring(definitions[name], clean=False) or "").strip()
        )

    assert missing == []
