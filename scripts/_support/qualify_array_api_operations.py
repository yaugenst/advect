"""Execute catalogued Array API operations across Advect program lifetimes."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import array_api_strict
import numpy as np
from numpy.testing import assert_allclose

import advect as ad
from advect.autodiff._ephemeral import trace_call
from advect.core._array_api_evidence import (
    Device,
    DType,
    Input,
    operation_evidence_cases,
)
from advect.core._array_api_profiles import (
    LATEST_ARRAY_API_VERSION,
    SUPPORTED_ARRAY_API_VERSIONS,
)
from advect.core._array_api_support import _static_parameters
from advect.core._array_namespace import _clear_array_namespace_caches
from scripts._support.evidence import evidence_report_header

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from typing import Protocol

    from advect.core._array_api_evidence import OperationCase

    class _ArrayLike(Protocol):
        shape: Sequence[int]
        dtype: object
        device: object

        def __array_namespace__(self) -> object: ...


_LIFETIMES = ("dynamic", "staged", "serialized")
_PROVIDERS = ("array-api-strict", "numpy")


@dataclass(frozen=True, slots=True)
class _Provider:
    name: str
    namespace: object
    version: str


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--providers",
        default="array-api-strict",
        help="Comma-separated providers: array-api-strict,numpy",
    )
    parser.add_argument(
        "--array-api-version",
        choices=SUPPORTED_ARRAY_API_VERSIONS,
        default=LATEST_ARRAY_API_VERSION,
    )
    parser.add_argument(
        "--subset",
        choices=("all", "portable"),
        default="all",
        help="Run every declared staged case or the cross-provider portable subset",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _parse_providers(value: str) -> tuple[str, ...]:
    providers = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(providers).difference(_PROVIDERS))
    if not providers or unknown:
        msg = f"providers must be selected from {_PROVIDERS}; found {unknown or value!r}"
        raise ValueError(msg)
    if len(providers) != len(set(providers)):
        msg = f"providers must not contain duplicates: {providers}"
        raise ValueError(msg)
    return providers


def _provider(name: str, array_api_version: str = LATEST_ARRAY_API_VERSION) -> _Provider:
    if name == "array-api-strict":
        array_api_strict.set_array_api_strict_flags(api_version=array_api_version)
        api_version = getattr(array_api_strict, "__array_api_version__", None)
        if api_version != array_api_version:
            msg = f"Expected array-api-strict Array API {array_api_version}, found {api_version!r}"
            raise RuntimeError(msg)
        return _Provider(
            name=name,
            namespace=array_api_strict,
            version=str(getattr(array_api_strict, "__version__", "unknown")),
        )
    if name == "numpy":
        return _Provider(name=name, namespace=np, version=np.__version__)
    msg = f"Unknown provider {name!r}"
    raise ValueError(msg)


@contextmanager
def _restrict_provider_revision(
    provider: _Provider,
    array_api_version: str,
) -> Iterator[None]:
    """Make the reference provider behave like one revision-limited backend.

    ``array-api-strict`` can switch revisions in response to each protocol
    request. Dynamic negotiation would therefore always select Advect's newest
    profile, even when a qualification lane intends to exercise an older
    provider. This local protocol guard rejects newer requests before the
    reference implementation changes its global flags.
    """
    if provider.name != "array-api-strict":
        yield
        return

    configured_version = str(array_api_strict.__array_api_version__)
    _clear_array_namespace_caches()
    array_api_strict.set_array_api_strict_flags(api_version=array_api_version)
    sample = array_api_strict.asarray(0.0)
    array_type = type(sample)
    original = array_type.__array_namespace__

    def restricted_namespace(
        value: object,
        *,
        api_version: str | None = None,
    ) -> object:
        requested = array_api_version if api_version is None else api_version
        if requested != array_api_version:
            message = (
                f"reference provider is restricted to Array API {array_api_version}; "
                f"requested {requested}"
            )
            raise ValueError(message)
        return original(value, api_version=array_api_version)

    array_type.__array_namespace__ = restricted_namespace
    try:
        yield
    finally:
        array_type.__array_namespace__ = original
        array_api_strict.set_array_api_strict_flags(api_version=configured_version)
        _clear_array_namespace_caches()


def _resolve(namespace: object, path: str) -> Callable[..., object]:
    value = namespace
    for component in path.split("."):
        value = getattr(value, component)
    if not callable(value):
        msg = f"Array API path {path!r} did not resolve to a callable"
        raise TypeError(msg)
    return value


def _resolve_value(value: object, inputs: tuple[object, ...], namespace: object) -> object:
    if isinstance(value, Input):
        return inputs[value.index]
    if isinstance(value, DType):
        return getattr(namespace, value.name)
    if isinstance(value, Device):
        return cast("_ArrayLike", inputs[value.input_index]).device
    if isinstance(value, tuple):
        return tuple(_resolve_value(item, inputs, namespace) for item in value)
    if isinstance(value, list):
        return [_resolve_value(item, inputs, namespace) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_value(item, inputs, namespace) for key, item in value.items()}
    return value


def _materialize_inputs(case: OperationCase, namespace: object) -> tuple[object, ...]:
    asarray = _resolve(namespace, "asarray")
    return tuple(
        asarray(input_spec.data, dtype=getattr(namespace, input_spec.dtype))
        for input_spec in case.inputs
    )


def _invoke(
    case: OperationCase,
    namespace: object,
    inputs: tuple[object, ...],
) -> object:
    function = _resolve(namespace, case.path)
    args = cast("tuple[object, ...]", _resolve_value(case.args, inputs, namespace))
    kwargs = cast(
        "dict[str, object]",
        _resolve_value(dict(case.kwargs), inputs, namespace),
    )
    return function(*args, **kwargs)


def _transformed(case: OperationCase) -> Callable[..., object]:
    def function(*inputs: object) -> object:
        traced = next(
            (value for value in inputs if type(value).__module__.startswith("advect.")),
            inputs[0],
        )
        namespace = cast("_ArrayLike", traced).__array_namespace__()
        return _invoke(case, namespace, inputs)

    return function


def _array_spec(value: object) -> ad.ArraySpec:
    array = cast("_ArrayLike", value)
    return ad.ArraySpec(
        tuple(array.shape),
        array.dtype,
    )


def execute_lifetimes(
    case: OperationCase,
    provider: _Provider,
    *,
    array_api_version: str = LATEST_ARRAY_API_VERSION,
) -> tuple[object, dict[str, object]]:
    """Execute a case directly and through all three Advect lifetimes."""
    with _restrict_provider_revision(provider, array_api_version):
        inputs = _materialize_inputs(case, provider.namespace)
        expected = _invoke(case, provider.namespace, inputs)
        function = _transformed(case)

        trace = trace_call(
            function,
            args=inputs,
            kwargs={},
            argnums=tuple(range(len(inputs))),
            argnames=None,
        )
        try:
            dynamic = trace.output
        finally:
            trace.tape.release_payloads()

        outputs = {"dynamic": dynamic}
        if "staged" in case.modes:
            program = ad.stage(
                function,
                specs=tuple(_array_spec(value) for value in inputs),
                array_api_version=array_api_version,
            )
            outputs["staged"] = program(*inputs)
            if "serialized" in case.modes:
                restored = ad.StagedProgram.from_dict(program.to_dict())
                outputs["serialized"] = restored(*inputs)
        return expected, outputs


def _is_sequence(value: object) -> bool:
    return isinstance(value, (tuple, list))


def assert_output_matches(
    expected: object,
    actual: object,
    *,
    compare_values: bool,
) -> None:
    """Assert provider type metadata and, when meaningful, array values."""
    if _is_sequence(expected):
        if not _is_sequence(actual):
            msg = f"Expected a structured output, found {type(actual).__name__}"
            raise AssertionError(msg)
        expected_items = cast("tuple[object, ...] | list[object]", expected)
        actual_items = cast("tuple[object, ...] | list[object]", actual)
        if len(expected_items) != len(actual_items):
            msg = f"Output arity differs: expected {len(expected_items)}, got {len(actual_items)}"
            raise AssertionError(msg)
        for expected_item, actual_item in zip(expected_items, actual_items, strict=True):
            assert_output_matches(
                expected_item,
                actual_item,
                compare_values=compare_values,
            )
        return

    expected_array = cast("_ArrayLike", expected)
    actual_array = cast("_ArrayLike", actual)
    if type(actual_array) is not type(expected_array):
        msg = (
            "Output provider type differs: "
            f"expected {type(expected_array).__name__}, got {type(actual_array).__name__}"
        )
        raise AssertionError(msg)
    expected_shape = tuple(expected_array.shape)
    actual_shape = tuple(actual_array.shape)
    if actual_shape != expected_shape:
        msg = f"Output shape differs: expected {expected_shape}, got {actual_shape}"
        raise AssertionError(msg)
    expected_dtype = expected_array.dtype
    actual_dtype = actual_array.dtype
    if actual_dtype != expected_dtype:
        msg = f"Output dtype differs: expected {expected_dtype}, got {actual_dtype}"
        raise AssertionError(msg)
    if compare_values:
        assert_allclose(
            np.asarray(actual),
            np.asarray(expected),
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )


def _output_manifest(value: object) -> object:
    if _is_sequence(value):
        return [_output_manifest(item) for item in cast("tuple[object, ...] | list[object]", value)]
    array = cast("_ArrayLike", value)
    return {"dtype": str(array.dtype), "shape": list(array.shape)}


def _run_case(
    case: OperationCase,
    provider: _Provider,
    *,
    array_api_version: str,
) -> dict[str, object]:
    try:
        expected, outputs = execute_lifetimes(
            case,
            provider,
            array_api_version=array_api_version,
        )
        for output in outputs.values():
            assert_output_matches(
                expected,
                output,
                compare_values=case.compare_values,
            )
    except Exception as error:  # noqa: BLE001 - qualification reports every failure.
        return {
            "error": f"{type(error).__name__}: {error}",
            "path": case.path,
            "status": "failed",
        }
    return {
        "lifetimes": list(case.modes),
        "output": _output_manifest(expected),
        "path": case.path,
        "status": "qualified",
    }


def build_report(
    *,
    provider_names: tuple[str, ...],
    subset: str,
    array_api_version: str = LATEST_ARRAY_API_VERSION,
) -> dict[str, object]:
    """Execute the requested matrix and return deterministic evidence."""
    all_cases = operation_evidence_cases(
        _static_parameters(version=array_api_version),
        array_api_version,
    )
    cases = tuple(case for case in all_cases if subset == "all" or case.portable)
    if subset == "all" and any(name != "array-api-strict" for name in provider_names):
        msg = "The complete declared matrix is qualified only on array-api-strict"
        raise ValueError(msg)

    provider_reports = []
    for name in provider_names:
        provider = _provider(name, array_api_version)
        results = [_run_case(case, provider, array_api_version=array_api_version) for case in cases]
        failures = [result for result in results if result["status"] == "failed"]
        qualified = [result for result in results if result["status"] == "qualified"]
        provider_reports.append(
            {
                "cases": results,
                "failed": len(failures),
                "name": provider.name,
                "passed": not failures,
                "qualified": len(qualified),
                "reported_array_api_version": getattr(
                    provider.namespace,
                    "__array_api_version__",
                    None,
                ),
                "selected_array_api_version": array_api_version,
                "version": provider.version,
            }
        )

    return {
        **evidence_report_header(
            schema_version=1,
            report_kind="advect.array-api-operation-qualification",
        ),
        "api_version": array_api_version,
        "case_count": len(cases),
        "executable_case_count": len(cases),
        "lifetimes": list(_LIFETIMES),
        "passed": all(bool(report["passed"]) for report in provider_reports),
        "portable_case_count": sum(case.portable for case in all_cases),
        "providers": provider_reports,
        "subset": subset,
    }


def main() -> int:
    """Run the operation qualification matrix."""
    arguments = _arguments()
    report = build_report(
        provider_names=_parse_providers(arguments.providers),
        subset=arguments.subset,
        array_api_version=arguments.array_api_version,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(f"{rendered}\n", encoding="utf-8")
    return 0 if report["passed"] else 1
