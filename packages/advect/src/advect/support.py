"""Runtime-derived support catalogs for Advect's public extensions."""

from __future__ import annotations

import inspect
from operator import attrgetter, itemgetter
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from advect.core._registry_types import OpDef
    from advect.numpy._support_contract import NumpySupportDeclaration

type Capability = Literal["yes", "no", "n/a", "composite"]
type VJPCapability = Literal["direct", "from JVP", "no", "n/a", "composite"]
type Backing = Literal["array_api", "numpy", "scipy", "advect", "composite", "metadata"]

_BINARY_ARITY = 2
_SINGLE_OUTPUT_ARITY = 1


def _derivative_status(definition: OpDef) -> tuple[Capability, VJPCapability]:
    if definition.non_differentiable_reason is not None:
        return "n/a", "n/a"
    jvp: Capability = "yes" if definition.jvp is not None else "no"
    if definition.vjp is not None:
        vjp: VJPCapability = "direct"
    elif definition.jvp is not None:
        vjp = "from JVP"
    else:
        vjp = "no"
    return jvp, vjp


def _has_abstract_semantics(definition: OpDef | None) -> bool:
    return definition is not None and (
        definition.abstract_evaluator is not None or definition.abstract_rule is not None
    )


def _primitive_capabilities(lowering: str) -> dict[str, object]:
    if lowering == "composite":
        return {
            "abstract": "composite",
            "jvp": "composite",
            "vjp": "composite",
        }
    if lowering in {"adapter", "metadata"}:
        return {"abstract": "n/a", "jvp": "n/a", "vjp": "n/a"}

    from advect.core._registry import get_registry  # noqa: PLC0415

    definition = get_registry().get_optional(lowering)
    if definition is None:
        message = f"Support catalog references unregistered primitive {lowering!r}"
        raise RuntimeError(message)
    jvp, vjp = _derivative_status(definition)
    return {
        "abstract": "yes" if _has_abstract_semantics(definition) else "no",
        "jvp": jvp,
        "vjp": vjp,
    }


def _function_row(
    *,
    callable_path: str,
    entrypoint: str,
    kind: str,
    lowering: str,
    backed_by: Backing,
    staged: bool,
) -> dict[str, object]:
    return {
        "backed_by": backed_by,
        "callable": callable_path,
        "dynamic": True,
        "entrypoint": entrypoint,
        "kind": kind,
        "lowering": lowering,
        "serialized": staged,
        "staged": staged,
        **_primitive_capabilities(lowering),
    }


def _array_api_extension() -> dict[str, object]:
    from advect.core._array_api.frontend import (  # noqa: PLC0415
        _ARRAY_API_COMPOSITES,
        _ARRAY_API_META_FUNCTIONS,
        _FUNCTION_SPECS,
        _NONDIFFERENTIABLE_ARRAY_API_COMPOSITES,
        _STAGED_ARRAY_API_COMPOSITES,
    )
    from advect.core._array_api.profiles import (  # noqa: PLC0415
        LATEST_ARRAY_API_VERSION,
        SUPPORTED_ARRAY_API_VERSIONS,
        materialize_array_api_profile,
    )
    from advect.core._array_api.support import (  # noqa: PLC0415
        _expected_modes,
        build_support_profile,
    )

    selected_profile = materialize_array_api_profile(LATEST_ARRAY_API_VERSION)

    def staged(path: str) -> bool:
        # Lifetimes come from the executable evidence contract, not from the
        # mere presence of an abstract rule.
        return "staged" in _expected_modes(path, version=LATEST_ARRAY_API_VERSION)

    rows = [
        _function_row(
            callable_path=path,
            entrypoint=f"array_namespace.{path}",
            kind="function",
            lowering=spec.op,
            backed_by="array_api",
            staged=staged(path),
        )
        for path, spec in _FUNCTION_SPECS.items()
        if selected_profile.admits(path)
    ]
    for path in _ARRAY_API_COMPOSITES:
        if not selected_profile.admits(path):
            continue
        row = _function_row(
            callable_path=path,
            entrypoint=f"array_namespace.{path}",
            kind="function",
            lowering="composite",
            backed_by="composite",
            staged=staged(path),
        )
        if path not in _STAGED_ARRAY_API_COMPOSITES:
            row["abstract"] = "no"
        if path in _NONDIFFERENTIABLE_ARRAY_API_COMPOSITES:
            row["jvp"] = "n/a"
            row["vjp"] = "n/a"
        rows.append(row)
    rows.extend(
        _function_row(
            callable_path=path,
            entrypoint=f"array_namespace.{path}",
            kind="metadata",
            lowering="metadata",
            backed_by="metadata",
            staged=staged(path),
        )
        for path in _ARRAY_API_META_FUNCTIONS
        if path not in _FUNCTION_SPECS
        and path not in _ARRAY_API_COMPOSITES
        and selected_profile.admits(path)
    )
    rows.sort(key=itemgetter("callable"))
    supported_profiles = []
    for version in SUPPORTED_ARRAY_API_VERSIONS:
        support_profile = build_support_profile(version)
        callables = cast("list[dict[str, object]]", support_profile["callables"])
        supported_profiles.append(
            {
                "callable_count": len(callables),
                "complete_callable_count": sum(row["complete"] is True for row in callables),
                "namespace": "array_api",
                "upstream_version": version,
            }
        )
    return {
        "available": True,
        "functions": rows,
        "published_range": {
            "maximum": SUPPORTED_ARRAY_API_VERSIONS[-1],
            "minimum": SUPPORTED_ARRAY_API_VERSIONS[0],
        },
        "selected_profile": LATEST_ARRAY_API_VERSION,
        "supported_profiles": supported_profiles,
        "version": LATEST_ARRAY_API_VERSION,
    }


def _resolve_numpy_function(path: str) -> object:
    import numpy as np  # noqa: PLC0415

    components = path.split(".")
    if not components or components[0] != "numpy":
        raise ValueError(path)
    target: Any = np
    for component in components[1:]:
        target = getattr(target, component)
    return target


def _backing(lowering: str, array_api_ops: frozenset[str]) -> Backing:
    if lowering == "composite":
        return "composite"
    if lowering == "metadata":
        return "metadata"
    if lowering in array_api_ops:
        return "array_api"
    if lowering.startswith("advect."):
        return "advect"
    if lowering.startswith("custom.scipy."):
        return "scipy"
    return "numpy"


def _numpy_forms(function_paths: Iterable[str]) -> Iterator[tuple[str, str, str]]:
    """Yield ``(kind, path, lowering)`` for each NumPy form this install executes."""
    import numpy as np  # noqa: PLC0415

    from advect.core._registry import get_registry  # noqa: PLC0415
    from advect.numpy._array_function.mutation import _FUNCTIONALIZERS  # noqa: PLC0415
    from advect.numpy._array_function.registry import (  # noqa: PLC0415
        _STATIC_ARRAY_FUNCTIONS,
        ARRAY_FUNCTION_HANDLERS,
    )
    from advect.numpy._composite_lowering import UFUNC_METHODS  # noqa: PLC0415
    from advect.numpy._op_bindings import canonicalize_numpy_op  # noqa: PLC0415
    from advect.numpy._supported_ufuncs import _SUPPORTED_UFUNCS  # noqa: PLC0415
    from advect.numpy._traced_array import TracedArray  # noqa: PLC0415

    handlers = ARRAY_FUNCTION_HANDLERS
    metadata_functions = frozenset({np.result_type, np.iscomplexobj, *_STATIC_ARRAY_FUNCTIONS})
    frontend_functions = frozenset({*handlers, *metadata_functions, *_FUNCTIONALIZERS})
    for path in function_paths:
        try:
            function = _resolve_numpy_function(path)
        except AttributeError:
            continue  # A declared legacy alias that the installed NumPy removed.
        if function not in frontend_functions:
            continue
        declared = getattr(handlers.get(cast("Any", function)), "__advect_lowering__", None)
        if function in metadata_functions:
            lowering = "metadata"
        elif isinstance(declared, str):
            lowering = declared
        else:
            candidate = canonicalize_numpy_op(path)
            lowering = candidate if get_registry().has(candidate) else "composite"
        yield "function", path, lowering

    for ufunc in _SUPPORTED_UFUNCS:
        name = ufunc.__name__
        yield "ufunc_call", f"numpy.{name}", canonicalize_numpy_op(f"numpy.{name}")
        for method in ("reduce", "accumulate"):
            target = UFUNC_METHODS.get((name, method))
            if target is not None:
                lowering = canonicalize_numpy_op(f"numpy.{target}")
                yield "ufunc_method", f"numpy.{name}.{method}", lowering
        if (
            ufunc.nin == _BINARY_ARITY
            and ufunc.nout == _SINGLE_OUTPUT_ARITY
            and ufunc.signature is None
        ):
            yield "ufunc_method", f"numpy.{name}.outer", "composite"

    for name, method in vars(TracedArray).items():
        lowering = getattr(method, "__advect_lowering__", None)
        if isinstance(lowering, str):
            yield "array_method", f"numpy.ndarray.{name}", lowering


def _numpy_rows(array_api_ops: frozenset[str]) -> list[dict[str, object]]:
    from advect.numpy._support_contract import numpy_support_declarations  # noqa: PLC0415

    declarations: dict[tuple[str, str], NumpySupportDeclaration] = {}
    for declaration in numpy_support_declarations():
        key = (declaration.kind, declaration.callable)
        if key in declarations:
            message = f"duplicate NumPy support declaration for {key!r}"
            raise RuntimeError(message)
        declarations[key] = declaration

    function_paths = [path for kind, path in declarations if kind == "function"]
    rows = []
    for kind, path, lowering in _numpy_forms(function_paths):
        declaration = declarations.get((kind, path))
        if declaration is None:
            continue
        row = _function_row(
            callable_path=path,
            entrypoint=path,
            kind=kind,
            lowering=lowering,
            backed_by=_backing(lowering, array_api_ops),
            staged="staged" in declaration.modes,
        )
        if not declaration.has_derivatives:
            row["jvp"] = "n/a"
            row["vjp"] = "n/a"
        rows.append(row)
    return rows


def _validate_numpy_bindings() -> None:
    from advect.core._registry import get_registry  # noqa: PLC0415
    from advect.numpy._array_function.registry import (  # noqa: PLC0415
        ARRAY_FUNCTION_HANDLERS,
    )
    from advect.numpy._traced_array import TracedArray  # noqa: PLC0415

    declared = {
        target
        for handler in ARRAY_FUNCTION_HANDLERS.values()
        if isinstance((target := getattr(handler, "__advect_lowering__", None)), str)
    }
    declared.update(
        target
        for method in vars(TracedArray).values()
        if isinstance((target := getattr(method, "__advect_lowering__", None)), str)
    )
    missing = sorted(
        target for target in declared if target != "composite" and not get_registry().has(target)
    )
    if missing:
        message = f"NumPy handlers declare unknown primitives: {missing}"
        raise RuntimeError(message)


def _numpy_extension(array_api_ops: frozenset[str]) -> dict[str, object]:
    import numpy as np  # noqa: PLC0415

    from advect.core._array_api.profiles import LATEST_ARRAY_API_VERSION  # noqa: PLC0415
    from advect.numpy._profiles import numpy_minor  # noqa: PLC0415
    from advect.numpy._supported_ufuncs import _SUPPORTED_UFUNCS  # noqa: PLC0415

    _validate_numpy_bindings()
    rows = _numpy_rows(array_api_ops)
    rows.sort(key=itemgetter("kind", "callable"))
    supported_method_count = sum(row["kind"] == "ufunc_method" for row in rows)
    return {
        "available": True,
        "array_api_version": min(np.__array_api_version__, LATEST_ARRAY_API_VERSION),
        "functions": rows,
        "minor": numpy_minor(np.__version__),
        "unsupported_ufunc_methods": len(_SUPPORTED_UFUNCS) * 5 - supported_method_count,
        "version": np.__version__,
    }


def _walk_public_functions(module: ModuleType) -> tuple[Callable[..., object], ...]:
    functions: list[Callable[..., object]] = []
    visited: set[str] = set()

    def visit(current: ModuleType) -> None:
        if current.__name__ in visited:
            return
        visited.add(current.__name__)
        for name in cast("Iterable[str]", getattr(current, "__all__", ())):
            value = getattr(current, name)
            if isinstance(value, ModuleType) and value.__name__.startswith("advect.scipy"):
                visit(value)
            elif inspect.isfunction(value):
                functions.append(cast("Callable[..., object]", value))

    visit(module)
    return tuple(sorted(functions, key=attrgetter("__module__")))


def _scipy_primitive_lowering(entrypoint: str) -> str:
    prefix = "advect.scipy."
    if not entrypoint.startswith(prefix):
        message = f"SciPy extension entry point is outside {prefix!r}: {entrypoint!r}"
        raise ValueError(message)
    return f"custom.scipy.{entrypoint.removeprefix(prefix)}"


def _scipy_function_row(function: Callable[..., object]) -> dict[str, object]:
    from advect.core._registry import get_registry  # noqa: PLC0415

    entrypoint = f"{function.__module__}.{function.__name__}"
    candidate = _scipy_primitive_lowering(entrypoint)
    declared = getattr(function, "__advect_lowering__", None)

    definition = get_registry().get_optional(candidate)
    if definition is not None:
        lowering = candidate
        stage_support = _has_abstract_semantics(definition)
    elif declared == "composite":
        # This executable marker is a full-lifetime contract for a public
        # composition whose component primitives carry the durable semantics.
        lowering = "composite"
        stage_support = True
    else:
        return _function_row(
            callable_path=entrypoint,
            entrypoint=entrypoint,
            kind="adapter",
            lowering="adapter",
            backed_by="scipy",
            staged=False,
        )

    return _function_row(
        callable_path=entrypoint.replace("advect.scipy", "scipy", 1),
        entrypoint=entrypoint,
        kind="function",
        lowering=lowering,
        backed_by="composite" if lowering == "composite" else "scipy",
        staged=stage_support,
    )


def _scipy_extension() -> dict[str, object]:
    try:
        import advect.scipy as scipy_extension  # noqa: PLC0415
    except ModuleNotFoundError as error:
        if error.name != "scipy":
            raise
        return {"available": False, "functions": [], "version": None}

    from importlib import metadata  # noqa: PLC0415 - costly import, needed only here

    rows = [_scipy_function_row(function) for function in _walk_public_functions(scipy_extension)]
    rows.sort(key=itemgetter("callable"))
    return {
        "available": True,
        "functions": rows,
        "version": metadata.version("scipy"),
    }


def _primitive_rows() -> list[dict[str, object]]:
    from advect.core._eval_dispatch import has_core_evaluator  # noqa: PLC0415
    from advect.core._registry import get_registry  # noqa: PLC0415
    from advect.numpy._eval import has_evaluator  # noqa: PLC0415

    rows: list[dict[str, object]] = []
    for definition in get_registry().definitions():
        if not definition.name.startswith(("advect.", "array.", "array_ext.", "custom.scipy.")):
            continue
        jvp, vjp = _derivative_status(definition)
        rows.append(
            {
                "abstract": _has_abstract_semantics(definition),
                "evaluator": (
                    has_core_evaluator(definition.name)
                    or has_evaluator(definition.name)
                    or definition.implementation is not None
                ),
                "jvp": jvp,
                "primitive": definition.name,
                "vjp": vjp,
            }
        )
    return rows


def support_catalog() -> dict[str, object]:
    """Return live primitive capabilities and supported functions by extension.

    Each mode marked true is an end-to-end support claim for the callable's
    declared frontend contract, rather than a statement that a handler exists.

    Examples
    --------
    >>> import advect as ad
    >>> catalog = ad.support_catalog()
    >>> catalog["schema_version"]
    3
    >>> sorted(catalog["extensions"])
    ['array_api', 'numpy', 'scipy']
    """
    from advect.core._array_api.frontend import _FUNCTION_SPECS  # noqa: PLC0415

    scipy = _scipy_extension()
    array_api_ops = frozenset(spec.op for spec in _FUNCTION_SPECS.values())
    return {
        "extensions": {
            "array_api": _array_api_extension(),
            "numpy": _numpy_extension(array_api_ops),
            "scipy": scipy,
        },
        "primitives": _primitive_rows(),
        "schema_version": 3,
    }


__all__ = ["support_catalog"]
