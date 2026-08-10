"""Measure named Advect runtime-memory profiles in isolated child processes.

The parent process uses only the Python standard library. Provider imports
happen in one child per case and run, so allocator state cannot leak between
checkpoint, lifetime, or donation comparisons.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
import traceback
import tracemalloc
import weakref
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from scripts._support.evidence import evidence_environment, source_revision_is_recorded

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence


_MIB = 1024**2
_DEFAULT_BUDGET = 64 * _MIB
_DEFAULT_CAP = 512 * _MIB
_DYNAMIC_DEPTH = 12
_STENCIL_STEPS = 4
_CHECKPOINT_STEPS = 8
_CHECKPOINT_REGIONS = 4
_UPDATE_STEPS = 4
_ACCEPTANCE_RUNS = 5
_MAX_VARIATION = 0.05
_MAX_CALIBRATION_ERROR = 0.10
_CHECKPOINT_MEMORY_RATIO = 0.75
_CHECKPOINT_RUNTIME_RATIO = 1.35
_DONATION_MEMORY_RATIO = 0.80
_DONATION_RUNTIME_RATIO = 1.05
_ACCEPTANCE_TIMING_RUNS = 5
_ACCEPTANCE_TIMING_ITERATIONS = 10


@dataclass(frozen=True, slots=True)
class _Case:
    workload: str
    framework: str
    mode: str
    provider: str

    @property
    def name(self) -> str:
        return f"{self.workload}:{self.framework}:{self.mode}:{self.provider}"


@dataclass(frozen=True, slots=True)
class _AcceptanceProfile:
    name: str
    cases: tuple[_Case, ...]
    timed_cases: tuple[tuple[str, str], ...]


_ACCEPTANCE_PROFILES = {
    profile.name: profile
    for profile in (
        _AcceptanceProfile(
            name="cpu-runtime",
            cases=(
                _Case("allocation_probe", "python", "allocate", "host"),
                _Case("elementwise", "advect", "dynamic", "numpy"),
                _Case("stencil", "advect", "dynamic", "numpy"),
                _Case("checkpoint", "advect", "plain", "numpy"),
                _Case("checkpoint", "advect", "checkpoint", "numpy"),
                _Case("residual", "advect", "retained", "numpy"),
                _Case("linear_map", "advect", "reusable", "numpy"),
                _Case("captured_constant", "advect", "staged", "numpy"),
            ),
            timed_cases=(("checkpoint", "plain"), ("checkpoint", "checkpoint")),
        ),
        _AcceptanceProfile(
            name="cupy-donation",
            cases=(
                _Case("allocation_probe", "python", "allocate", "host"),
                _Case("functional_updates", "advect", "donation", "cupy"),
                _Case("functional_updates", "advect", "forced_fresh", "cupy"),
            ),
            timed_cases=(
                ("functional_updates", "donation"),
                ("functional_updates", "forced_fresh"),
            ),
        ),
    )
}


@dataclass(frozen=True, slots=True)
class _WorkerSpec:
    case: _Case
    byte_budget: int
    max_bytes: int
    sample_hold_seconds: float
    measurement: str
    timing_iterations: int


@dataclass(frozen=True, slots=True)
class _RssSample:
    monotonic_ns: int
    current_bytes: int
    high_water_bytes: int | None


class _ConstantRecordLike(Protocol):
    bytes: int


class _StagedProgramLike(Protocol):
    compile_seconds: float
    constants: tuple[_ConstantRecordLike, ...]

    def __call__(self, value: object) -> object: ...


def _staged_constant_bytes(program: _StagedProgramLike) -> int:
    return sum(record.bytes for record in program.constants)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        msg = f"expected a positive integer, got {value!r}"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        msg = f"expected a positive number, got {value!r}"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _parse_byte_size(value: str) -> int:
    """Parse an integer byte count or a binary KiB/MiB/GiB size."""
    normalized = value.strip().lower().replace("_", "")
    suffixes = {
        "gib": 1024**3,
        "gb": 1024**3,
        "mib": 1024**2,
        "mb": 1024**2,
        "kib": 1024,
        "kb": 1024,
        "b": 1,
    }
    multiplier = 1
    number = normalized
    for suffix, candidate in suffixes.items():
        if normalized.endswith(suffix):
            multiplier = candidate
            number = normalized[: -len(suffix)]
            break
    try:
        parsed = float(number)
    except ValueError as error:
        msg = f"invalid byte size {value!r}"
        raise argparse.ArgumentTypeError(msg) from error
    result = int(parsed * multiplier)
    if result < 1:
        msg = f"expected a positive byte size, got {value!r}"
        raise argparse.ArgumentTypeError(msg)
    return result


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=tuple(_ACCEPTANCE_PROFILES),
        help="select an exact acceptance case matrix",
    )
    parser.add_argument("--byte-budget", type=_parse_byte_size, default=_DEFAULT_BUDGET)
    parser.add_argument("--max-bytes", type=_parse_byte_size, default=_DEFAULT_CAP)
    parser.add_argument("--runs", type=_positive_int, default=5)
    parser.add_argument("--timing-runs", type=_positive_int, default=5)
    parser.add_argument("--timing-iterations", type=_positive_int, default=10)
    parser.add_argument("--sample-interval-ms", type=_positive_float, default=2.0)
    parser.add_argument("--sample-hold-ms", type=_positive_float, default=20.0)
    parser.add_argument("--no-timing", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="use one 4 MiB run, one timing iteration, and short sampling holds",
    )
    parser.add_argument(
        "--acceptance",
        action="store_true",
        help="enforce every invariant owned by the selected exact profile",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-rss-samples", action="store_true")
    parser.add_argument("--_worker-spec", dest="worker_spec", help=argparse.SUPPRESS)
    return parser.parse_args()


def _correctness_preflight_cases(cases: Sequence[_Case]) -> tuple[_Case, ...]:
    """Return every selected Advect scenario with a logical-result contract."""
    return tuple(
        case for case in cases if case.framework == "advect" and case.workload != "allocation_probe"
    )


def _elements_for_budget(
    byte_budget: int,
    *,
    live_array_factor: int,
    max_bytes: int,
    itemsize: int = 8,
) -> int:
    """Choose an element count from a provider-live-byte target and hard cap."""
    if byte_budget > max_bytes:
        msg = f"byte budget {byte_budget} exceeds explicit cap {max_bytes}"
        raise ValueError(msg)
    target = min(byte_budget, max_bytes)
    return max(1, target // (itemsize * live_array_factor))


def _live_array_factor(workload: str) -> int:
    return {
        "allocation_probe": 1,
        "elementwise": 4 * _DYNAMIC_DEPTH,
        "stencil": 6 * _STENCIL_STEPS,
        "checkpoint": 6 * _CHECKPOINT_STEPS,
        "functional_updates": 3,
        "residual": 4,
        "linear_map": 4 * _DYNAMIC_DEPTH,
        "captured_constant": 3,
    }[workload]


def _element_count(spec: _WorkerSpec) -> int:
    return _elements_for_budget(
        spec.byte_budget,
        live_array_factor=_live_array_factor(spec.case.workload),
        max_bytes=spec.max_bytes,
    )


def _read_proc_memory(pid: int) -> tuple[int, int | None] | None:
    """Return current and high-water RSS bytes from Linux procfs."""
    path = Path("/proc") / str(pid) / "status"
    try:
        contents = path.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    values: dict[str, int] = {}
    for line in contents.splitlines():
        key, separator, raw_value = line.partition(":")
        if not separator or key not in {"VmRSS", "VmHWM"}:
            continue
        fields = raw_value.split()
        if not fields:
            continue
        scale = 1024 if len(fields) == 1 or fields[1].lower() == "kb" else 1
        values[key] = int(fields[0]) * scale
    current = values.get("VmRSS")
    if current is None:
        return None
    return current, values.get("VmHWM")


def _resource_high_water_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _buffer_owner(value: object) -> object:
    owner = value
    seen: set[int] = set()
    while id(owner) not in seen:
        seen.add(id(owner))
        base = getattr(owner, "base", None)
        if base is None or not hasattr(base, "nbytes"):
            break
        owner = base
    return owner


def _buffer_record(value: object) -> tuple[tuple[object, ...], int, object] | None:
    """Return a provider allocation identity, byte size, and weakref owner."""
    nbytes = getattr(value, "nbytes", None)
    if not isinstance(nbytes, int) or nbytes < 0:
        return None

    data = getattr(value, "data", None)
    memory = getattr(data, "mem", None)
    pointer = getattr(memory, "ptr", None)
    allocation_bytes = getattr(memory, "size", None)
    if isinstance(pointer, int) and isinstance(allocation_bytes, int):
        device = getattr(getattr(value, "device", None), "id", None)
        return ("device", device, pointer), allocation_bytes, _buffer_owner(value)

    owner = _buffer_owner(value)
    owner_nbytes = getattr(owner, "nbytes", nbytes)
    size = int(owner_nbytes) if isinstance(owner_nbytes, int) else nbytes
    interface = getattr(owner, "__array_interface__", None)
    owner_pointer: int | None = None
    if isinstance(interface, dict):
        raw_data = interface.get("data")
        if isinstance(raw_data, tuple) and raw_data and isinstance(raw_data[0], int):
            owner_pointer = raw_data[0]
    key = ("host", type(owner).__module__, owner_pointer, id(owner))
    return key, size, owner


def _iter_array_values(value: object) -> Iterable[object]:
    if _buffer_record(value) is not None:
        yield value
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_array_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_array_values(item)


def _buffer_map(values: Iterable[object]) -> dict[tuple[object, ...], int]:
    records: dict[tuple[object, ...], int] = {}
    for value in values:
        for array in _iter_array_values(value):
            record = _buffer_record(array)
            if record is None:
                continue
            key, size, _owner = record
            records[key] = max(records.get(key, 0), size)
    return records


class _BufferTracker:
    """Weakly track provider allocations observed by benchmark evaluators."""

    def __init__(self) -> None:
        self._records: dict[tuple[object, ...], tuple[int, weakref.ReferenceType[object]]] = {}
        self._current_bytes = 0
        self._peak_bytes = 0

    def observe(self, value: object) -> None:
        for array in _iter_array_values(value):
            record = _buffer_record(array)
            if record is None:
                continue
            key, size, owner = record
            if key in self._records:
                continue
            try:
                owner_ref = weakref.ref(owner, lambda _ref, key=key: self._discard(key))
            except TypeError:
                continue
            self._records[key] = (size, owner_ref)
            self._current_bytes += size
            self._peak_bytes = max(self._peak_bytes, self._current_bytes)

    def _discard(self, key: tuple[object, ...]) -> None:
        record = self._records.pop(key, None)
        if record is not None:
            self._current_bytes -= record[0]

    def live_values(self) -> tuple[object, ...]:
        values: list[object] = []
        for _size, owner_ref in self._records.values():
            owner = owner_ref()
            if owner is not None:
                values.append(owner)
        return tuple(values)

    @property
    def peak_bytes(self) -> int:
        return self._peak_bytes


def _sync_provider(provider: object) -> None:
    cuda = getattr(provider, "cuda", None)
    runtime = getattr(cuda, "runtime", None)
    synchronize = getattr(runtime, "deviceSynchronize", None)
    if callable(synchronize):
        synchronize()


def _provider_pool_metrics(provider: object) -> dict[str, int | None]:
    get_pool = getattr(provider, "get_default_memory_pool", None)
    cuda = getattr(provider, "cuda", None)
    runtime = getattr(cuda, "runtime", None)
    mem_get_info = getattr(runtime, "memGetInfo", None)
    if not callable(get_pool):
        return {
            "provider_pool_used_bytes": None,
            "provider_pool_reserved_bytes": None,
            "device_free_bytes": None,
            "device_total_bytes": None,
        }
    pool = get_pool()
    used = getattr(pool, "used_bytes", None)
    reserved = getattr(pool, "total_bytes", None)
    free_bytes: int | None = None
    total_bytes: int | None = None
    if callable(mem_get_info):
        free_raw, total_raw = cast("tuple[int, int]", mem_get_info())
        free_bytes = free_raw
        total_bytes = total_raw
    used_bytes = cast("Callable[[], int] | None", used if callable(used) else None)
    reserved_bytes = cast(
        "Callable[[], int] | None",
        reserved if callable(reserved) else None,
    )
    return {
        "provider_pool_used_bytes": used_bytes() if used_bytes is not None else None,
        "provider_pool_reserved_bytes": (reserved_bytes() if reserved_bytes is not None else None),
        "device_free_bytes": free_bytes,
        "device_total_bytes": total_bytes,
    }


def _tape_values(tape: object) -> tuple[object, ...]:
    stats = cast("Mapping[str, object]", tape.stats())  # type: ignore[attr-defined]
    node_count = int(cast("int", stats["node_count"]))
    values: list[object] = []
    for node_id in range(node_count):
        try:
            values.append(tape.value(node_id))  # type: ignore[attr-defined]
        except RuntimeError:
            continue
    return tuple(values)


class _Reporter:
    """Emit child lifecycle markers with disjoint memory metrics."""

    def __init__(self, provider: object, *, hold_seconds: float) -> None:
        self._provider = provider
        self._hold_seconds = hold_seconds
        self._tracker = _BufferTracker()
        self._baseline_provider_bytes = 0
        self._baseline_traced_bytes = 0

    @property
    def tracker(self) -> _BufferTracker:
        return self._tracker

    def start(self, *, roots: Sequence[object]) -> None:
        self._tracker.observe(tuple(roots))
        gc.collect()
        _sync_provider(self._provider)
        tracemalloc.start()
        self._baseline_traced_bytes = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        self.mark("baseline", roots=roots, excluded_roots=roots)
        self._baseline_provider_bytes = self._tracker.peak_bytes

    def mark(
        self,
        phase: str,
        *,
        roots: Sequence[object] = (),
        excluded_roots: Sequence[object] = (),
        provider_cache_roots: Sequence[object] = (),
        tape: object | None = None,
        collect: bool = False,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        if collect:
            gc.collect()
        tape_values: tuple[object, ...] = ()
        native_stats: Mapping[str, object] | None = None
        if tape is not None:
            native_stats = cast("Mapping[str, object]", tape.stats())  # type: ignore[attr-defined]
            tape_values = _tape_values(tape)
        self._tracker.observe((tuple(roots), tuple(provider_cache_roots), tape_values))
        _sync_provider(self._provider)
        if self._hold_seconds:
            time.sleep(self._hold_seconds)
        _sync_provider(self._provider)

        live_values = (*self._tracker.live_values(), *roots, *tape_values)
        live_buffers = _buffer_map(live_values)
        excluded_buffers = _buffer_map(excluded_roots)
        provider_cache_buffers = _buffer_map(provider_cache_roots)
        owned_live_bytes = sum(
            size
            for key, size in live_buffers.items()
            if key not in excluded_buffers and key not in provider_cache_buffers
        )
        traced_current, traced_peak = tracemalloc.get_traced_memory()
        process_memory = _read_proc_memory(os.getpid())
        payload: dict[str, object] = {
            "event": "phase",
            "phase": phase,
            "monotonic_ns": time.monotonic_ns(),
            "rss_current_bytes": None if process_memory is None else process_memory[0],
            "rss_high_water_bytes": None if process_memory is None else process_memory[1],
            "resource_high_water_bytes": _resource_high_water_bytes(),
            "tracemalloc_current_delta_bytes": max(
                0,
                traced_current - self._baseline_traced_bytes,
            ),
            "tracemalloc_peak_delta_bytes": max(
                0,
                traced_peak - self._baseline_traced_bytes,
            ),
            "provider_live_bytes": sum(live_buffers.values()),
            "provider_owned_live_bytes": owned_live_bytes,
            "provider_cache_live_bytes": sum(provider_cache_buffers.values()),
            "provider_peak_delta_bytes": max(
                0,
                self._tracker.peak_bytes - self._baseline_provider_bytes,
            ),
            **_provider_pool_metrics(self._provider),
        }
        if native_stats is not None:
            payload["native_structural_bytes"] = native_stats.get("native_structural_bytes")
            payload["native_structural"] = native_stats.get("native_structural")
            payload["tape"] = {
                key: native_stats.get(key)
                for key in (
                    "node_count",
                    "edge_count",
                    "operation_count",
                    "retained_value_count",
                    "literal_count",
                    "residual_count",
                    "consumed",
                )
            }
        if extra is not None:
            payload.update(extra)
        print(json.dumps(payload, sort_keys=True), flush=True)

    def stop(self) -> None:
        tracemalloc.stop()


def _load_provider(name: str) -> object:
    if name == "numpy":
        return importlib.import_module("numpy")
    try:
        provider = importlib.import_module("cupy")
    except ImportError as error:
        msg = "CuPy is not installed"
        raise ModuleNotFoundError(msg) from error
    try:
        _sync_provider(provider)
    except Exception as error:
        msg = f"CuPy is installed but no usable CUDA device is available: {error}"
        raise RuntimeError(msg) from error
    return provider


def _make_input(provider: object, elements: int) -> object:
    dynamic_provider = cast("Any", provider)
    return dynamic_provider.linspace(-0.75, 0.75, elements, dtype=dynamic_provider.float64)


def _elementwise_loss(namespace: object, *, depth: int) -> Callable[[object], object]:
    dynamic_namespace = cast("Any", namespace)
    sin = cast("Callable[[object], object]", dynamic_namespace.sin)
    sum_value = cast("Callable[[object], object]", dynamic_namespace.sum)

    def loss(value: object) -> object:
        current = value
        for _ in range(depth):
            current = sin(current) * 0.95 + current * 0.05  # type: ignore[operator]
        return sum_value(current * current)  # type: ignore[operator]

    return loss


def _field_block(namespace: object, value: object, *, steps: int) -> object:
    dynamic_namespace = cast("Any", namespace)
    concatenate = cast("Callable[[tuple[object, ...]], object]", dynamic_namespace.concatenate)
    current = value
    for _ in range(steps):
        laplacian = (
            current[2:] - 2.0 * current[1:-1] + current[:-2]  # type: ignore[index,operator]
        )
        interior = current[1:-1] + 0.125 * laplacian  # type: ignore[index,operator]
        current = concatenate((current[:1], interior, current[-1:]))  # type: ignore[index]
    return current


def _stencil_loss(namespace: object, *, steps: int) -> Callable[[object], object]:
    sum_value = cast("Callable[[object], object]", cast("Any", namespace).sum)

    def loss(value: object) -> object:
        result = _field_block(namespace, value, steps=steps)
        return sum_value(result * result)  # type: ignore[operator]

    return loss


def _ones_like(provider: object, value: object) -> object:
    return cast("Callable[[object], object]", cast("Any", provider).ones_like)(value)


def _worker_environment(
    *,
    provider: object,
) -> dict[str, object]:
    environment: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "provider": getattr(provider, "__name__", type(provider).__name__),
        "provider_version": getattr(provider, "__version__", None),
    }
    ad = importlib.import_module("advect")
    native = importlib.import_module("advect.core._native")
    context = importlib.import_module("advect.core._context")
    environment.update(
        {
            "advect": getattr(ad, "__version__", None),
            "advect_native": native.native_build_info(),
            "advect_debug": context.is_debug(),
        }
    )
    return environment


def _linearize(
    loss: Callable[[object], object],
    value: object,
    *,
    reverse_only: bool,
) -> tuple[object, Any]:
    ephemeral = importlib.import_module("advect.autodiff._ephemeral")
    return ephemeral.linearize_call(
        loss,
        args=(value,),
        kwargs={},
        argnums=(0,),
        argnames=None,
        single_argnum=True,
        reverse_only=reverse_only,
    )


def _checkpoint_loss(
    provider: object,
    *,
    mode: str,
    calls: dict[str, int],
) -> Callable[[object], object]:
    ad = importlib.import_module("advect")
    region_steps = _CHECKPOINT_STEPS // _CHECKPOINT_REGIONS

    def region(field: object) -> object:
        calls["count"] += 1
        return _field_block(provider, field, steps=region_steps)

    active_region = ad.checkpoint(region) if mode == "checkpoint" else region
    sum_value = cast("Callable[[object], object]", cast("Any", provider).sum)

    def loss(field: object) -> object:
        result = field
        for _ in range(_CHECKPOINT_REGIONS):
            result = active_region(result)
        return sum_value(result * result)  # type: ignore[operator]

    return loss


def _residual_loss(
    provider: object,
    spec: _WorkerSpec,
    *,
    tracker_box: list[_BufferTracker],
    release_calls: dict[str, int],
) -> tuple[Callable[[object], object], int]:
    ad = importlib.import_module("advect")
    residual_elements = _elements_for_budget(
        spec.byte_budget,
        live_array_factor=1,
        max_bytes=spec.max_bytes,
    )

    @ad.primitive(
        name=f"bench.runtime_memory.residual_{os.getpid()}",
        residual=True,
    )
    def primitive(field: object) -> object:
        dynamic_provider = cast("Any", provider)
        scale = 2 * dynamic_provider.copy(field)
        payload = dynamic_provider.empty(residual_elements, dtype=dynamic_provider.float64)
        payload.fill(1.0)
        residual = (scale, payload)
        tracker_box[0].observe(residual)
        dynamic_field = cast("Any", field)

        def release(_residual: object) -> None:
            release_calls["count"] += 1

        return ad.PrimitiveResult(
            dynamic_field * dynamic_field,
            residual,
            release=release,
        )

    @primitive.def_jvp
    def jvp_rule(
        output: object,
        primals: tuple[object, ...],
        tangents: tuple[object | None, ...],
    ) -> object:
        del output
        tangent = tangents[0]
        if tangent is None:
            msg = "residual benchmark received no tangent"
            raise RuntimeError(msg)
        return 2 * primals[0] * tangent  # type: ignore[operator]

    @primitive.def_transpose
    def transpose_rule(
        cotangent: object,
        primals: tuple[object, ...],
        output: object,
        residual: object,
    ) -> tuple[object]:
        del primals, output
        scale, _payload = cast("tuple[object, object]", residual)
        return (cotangent * scale,)  # type: ignore[operator]

    sum_value = cast("Callable[[object], object]", cast("Any", provider).sum)

    def loss(field: object) -> object:
        return sum_value(primitive(field))

    return loss, residual_elements * 8


def _run_advect_dynamic_memory(spec: _WorkerSpec, provider: object) -> dict[str, object]:
    importlib.import_module("advect.numpy")
    elements = _element_count(spec)
    value = _make_input(provider, elements)
    calls = {"count": 0}
    residual_releases = {"count": 0}
    residual_tracker: _BufferTracker | None = None
    reporter_tracker_box: list[_BufferTracker] = []
    residual_payload_bytes = 0

    if spec.case.workload in {"elementwise", "linear_map"}:
        loss = _elementwise_loss(provider, depth=_DYNAMIC_DEPTH)
    elif spec.case.workload == "stencil":
        loss = _stencil_loss(provider, steps=_STENCIL_STEPS)
    elif spec.case.workload == "checkpoint":
        loss = _checkpoint_loss(
            provider,
            mode=spec.case.mode,
            calls=calls,
        )
    elif spec.case.workload == "residual":
        loss, residual_payload_bytes = _residual_loss(
            provider,
            spec,
            tracker_box=reporter_tracker_box,
            release_calls=residual_releases,
        )
    else:
        msg = f"unsupported Advect dynamic workload {spec.case.workload!r}"
        raise ValueError(msg)

    reporter = _Reporter(provider, hold_seconds=spec.sample_hold_seconds)
    if spec.case.workload == "residual":
        reporter_tracker_box.append(reporter.tracker)
        residual_tracker = reporter.tracker
    reporter.start(roots=(value,))
    output, linear = _linearize(
        loss,
        value,
        reverse_only=spec.case.workload not in {"linear_map", "residual"},
    )
    forward_calls = calls["count"]
    tape = linear._trace.tape  # noqa: SLF001 - benchmark lifetime inspection
    reporter.mark(
        "forward",
        roots=(value, output),
        excluded_roots=(value, output),
        tape=tape,
        extra={
            "forward_calls": forward_calls,
            "residual_release_count": residual_releases["count"],
            "residual_payload_bytes": residual_payload_bytes,
        },
    )
    cotangent = _ones_like(provider, output)
    if spec.case.workload in {"linear_map", "residual"}:
        gradient = linear.pullback(cotangent)
        reporter.mark(
            "reverse_retained",
            roots=(value, output, gradient),
            excluded_roots=(value, output, gradient),
            tape=tape,
            extra={
                "forward_calls": forward_calls,
                "recomputation_count": max(0, calls["count"] - forward_calls),
                "residual_release_count": residual_releases["count"],
                "residual_payload_bytes": residual_payload_bytes,
            },
        )
        linear.close()
    else:
        gradient = linear._consume_pullback(cotangent)  # noqa: SLF001
        reporter.mark(
            "reverse",
            roots=(value, output, gradient),
            excluded_roots=(value, output, gradient),
            tape=tape,
            extra={
                "forward_calls": forward_calls,
                "recomputation_count": max(0, calls["count"] - forward_calls),
                "residual_release_count": residual_releases["count"],
                "residual_payload_bytes": residual_payload_bytes,
            },
        )
    reporter.mark(
        "closed",
        roots=(value, output, gradient),
        excluded_roots=(value, output, gradient),
        tape=tape,
        collect=True,
        extra={
            "forward_calls": forward_calls,
            "recomputation_count": max(0, calls["count"] - forward_calls),
            "residual_release_count": residual_releases["count"],
            "residual_payload_bytes": residual_payload_bytes,
            "residual_tracker_active": residual_tracker is not None,
        },
    )
    reporter.stop()
    return {
        "elements": elements,
        "input_bytes": int(getattr(value, "nbytes", 0)),
        "provider_accounting": "all retained tape values and observed residuals",
        "residual_payload_bytes": residual_payload_bytes,
    }


def _copy_evaluator_metadata(source: object, target: object, *, force_fresh: bool) -> None:
    for name in (
        "__advect_owned_output__",
        "__advect_alias_positions__",
        "__advect_donation_positions__",
    ):
        if force_fresh and name == "__advect_donation_positions__":
            continue
        value = getattr(source, name, None)
        if value is not None:
            setattr(target, name, value)


def _stage_with_profile_binding(
    function: Callable[[object], object],
    *,
    elements: int,
    tracker: _BufferTracker | None,
    force_fresh: bool,
) -> _StagedProgramLike:
    ad = importlib.import_module("advect")
    stage_module = cast("Any", importlib.import_module("advect.core._stage"))
    original = stage_module.bind_native_node_evaluator

    def binder(op: str, attrs: Mapping[str, Any]) -> object:
        evaluator = original(op, attrs)
        if tracker is None:
            if force_fresh and hasattr(evaluator, "__advect_donation_positions__"):
                delattr(evaluator, "__advect_donation_positions__")
            return evaluator

        def observed(
            values: tuple[object, ...],
            context: object | None,
            donation_position: int | None,
        ) -> object:
            tracker.observe(values)
            result = evaluator(values, context, donation_position)
            tracker.observe(result)
            return result

        _copy_evaluator_metadata(evaluator, observed, force_fresh=force_fresh)
        return observed

    stage_module.bind_native_node_evaluator = binder
    try:
        return cast(
            "_StagedProgramLike",
            ad.stage(
                function,
                specs=(ad.ArraySpec((elements,), "float64"),),
            ),
        )
    finally:
        stage_module.bind_native_node_evaluator = original


def _functional_update_function() -> Callable[[object], object]:
    def update(field: object) -> object:
        result = field.copy()  # type: ignore[attr-defined]
        for _ in range(_UPDATE_STEPS):
            result[1:-1] += 0.125  # type: ignore[index,operator]
        return result

    return update


def _staged_provider_cache_roots(program: _StagedProgramLike) -> tuple[object, ...]:
    roots: list[object] = []
    state = cast("Any", program)._execution_state  # noqa: SLF001 - benchmark cache accounting
    for materialized in state.materialized_constants:
        roots.extend(materialized.values)
    return tuple(roots)


def _run_staged_memory(spec: _WorkerSpec, provider: object) -> dict[str, object]:
    importlib.import_module("advect.numpy")
    elements = _element_count(spec)
    value = _make_input(provider, elements)
    reporter = _Reporter(provider, hold_seconds=spec.sample_hold_seconds)

    if spec.case.workload == "functional_updates":
        function = _functional_update_function()
        force_fresh = spec.case.mode == "forced_fresh"
        program = _stage_with_profile_binding(
            function,
            elements=elements,
            tracker=reporter.tracker,
            force_fresh=force_fresh,
        )
        constant_bytes = 0
        caller_roots = (value,)
        reporter.start(roots=caller_roots)
    else:
        constant = cast("Callable[[object], object]", cast("Any", provider).copy)(value)
        caller_roots = (value, constant)
        reporter.start(roots=caller_roots)

        def function(field: object) -> object:
            return field * constant + 0.25  # type: ignore[operator]

        force_fresh = False
        program = _stage_with_profile_binding(
            function,
            elements=elements,
            tracker=reporter.tracker,
            force_fresh=False,
        )
        constant_bytes = int(getattr(constant, "nbytes", 0))
        reporter.mark(
            "compiled",
            roots=caller_roots,
            excluded_roots=caller_roots,
            extra={
                "force_fresh": force_fresh,
                "constant_bytes": constant_bytes,
                "constant_manifest_bytes": _staged_constant_bytes(program),
                "compile_seconds": program.compile_seconds,
            },
        )

    compile_seconds = program.compile_seconds
    output = program(value)
    provider_cache_roots = _staged_provider_cache_roots(program)
    result_roots = (*caller_roots, output)
    reporter.mark(
        "execute",
        roots=result_roots,
        excluded_roots=result_roots,
        provider_cache_roots=provider_cache_roots,
        extra={
            "force_fresh": force_fresh,
            "constant_bytes": constant_bytes,
            "constant_manifest_bytes": _staged_constant_bytes(program),
            "compile_seconds": compile_seconds,
        },
    )
    reporter.mark(
        "closed",
        roots=result_roots,
        excluded_roots=result_roots,
        provider_cache_roots=provider_cache_roots,
        collect=True,
        extra={
            "force_fresh": force_fresh,
            "constant_bytes": constant_bytes,
            "constant_manifest_bytes": _staged_constant_bytes(program),
            "compile_seconds": compile_seconds,
        },
    )
    reporter.stop()
    return {
        "elements": elements,
        "input_bytes": int(getattr(value, "nbytes", 0)),
        "provider_accounting": "instrumented staged evaluator inputs and outputs",
        "force_fresh": force_fresh,
        "compile_seconds": compile_seconds,
    }


def _run_allocation_probe(spec: _WorkerSpec) -> dict[str, object]:
    provider = object()
    allocation_size = min(spec.byte_budget, spec.max_bytes)
    reporter = _Reporter(provider, hold_seconds=spec.sample_hold_seconds)
    reporter.start(roots=())
    allocation = bytearray(allocation_size)
    page_size = max(1, getattr(os, "sysconf", lambda _name: 4096)("SC_PAGE_SIZE"))
    for offset in range(0, allocation_size, page_size):
        allocation[offset] = 1
    allocation[-1] = 1
    reporter.mark(
        "allocated",
        roots=(),
        extra={"expected_allocation_bytes": allocation_size},
    )
    del allocation
    reporter.mark(
        "closed",
        roots=(),
        collect=True,
        extra={"expected_allocation_bytes": allocation_size},
    )
    reporter.stop()
    return {
        "elements": allocation_size,
        "input_bytes": 0,
        "expected_allocation_bytes": allocation_size,
    }


def _assert_provider_allclose(provider: object, actual: object, expected: object) -> None:
    cast("Any", provider).testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-8)


def _run_correctness_preflight(spec: _WorkerSpec, provider: object) -> dict[str, object]:
    """Check one scenario outside every measured memory and timing worker."""
    ad = importlib.import_module("advect")
    importlib.import_module("advect.numpy")
    dynamic_provider = cast("Any", provider)
    elements = 16
    value = _make_input(provider, elements)
    original = dynamic_provider.copy(value)
    case = spec.case

    if case.workload in {"elementwise", "stencil"}:
        loss = (
            _elementwise_loss(provider, depth=_DYNAMIC_DEPTH)
            if case.workload == "elementwise"
            else _stencil_loss(provider, steps=_STENCIL_STEPS)
        )
        gradient = ad.grad(loss)(value)
        direction = dynamic_provider.linspace(0.2, 0.8, elements, dtype=dynamic_provider.float64)
        epsilon = 1e-5
        finite_difference = (
            loss(value + epsilon * direction) - loss(value - epsilon * direction)
        ) / (2 * epsilon)
        directional = dynamic_provider.sum(gradient * direction)
        _assert_provider_allclose(provider, directional, finite_difference)
    elif case.workload == "checkpoint":
        measured_loss = _checkpoint_loss(provider, mode=case.mode, calls={"count": 0})
        plain_loss = _checkpoint_loss(provider, mode="plain", calls={"count": 0})
        _assert_provider_allclose(provider, measured_loss(value), plain_loss(value))
        _assert_provider_allclose(
            provider,
            ad.grad(measured_loss)(value),
            ad.grad(plain_loss)(value),
        )
    elif case.workload == "functional_updates":
        function = _functional_update_function()
        donated = _stage_with_profile_binding(
            function,
            elements=elements,
            tracker=None,
            force_fresh=False,
        )
        forced_fresh = _stage_with_profile_binding(
            function,
            elements=elements,
            tracker=None,
            force_fresh=True,
        )
        expected = function(value)
        _assert_provider_allclose(provider, donated(value), expected)
        _assert_provider_allclose(provider, forced_fresh(value), expected)
    elif case.workload == "residual":
        releases = {"count": 0}
        loss, _payload_bytes = _residual_loss(
            provider,
            spec,
            tracker_box=[_BufferTracker()],
            release_calls=releases,
        )
        _assert_provider_allclose(provider, ad.grad(loss)(value), 2 * value)
        if releases["count"] != 1:
            msg = f"residual release ran {releases['count']} times, not once"
            raise AssertionError(msg)
    elif case.workload == "linear_map":
        loss = _elementwise_loss(provider, depth=_DYNAMIC_DEPTH)
        expected = ad.grad(loss)(value)
        output, linear = ad.linearize(loss, value)
        try:
            actual = linear.transpose()(_ones_like(provider, output))
        finally:
            linear.close()
        _assert_provider_allclose(provider, actual, expected)
    elif case.workload == "captured_constant":
        constant = dynamic_provider.copy(value)

        def function(field: object) -> object:
            return field * constant + 0.25  # type: ignore[operator]

        program = _stage_with_profile_binding(
            function,
            elements=elements,
            tracker=None,
            force_fresh=False,
        )
        restored = ad.StagedProgram.from_dict(program.to_dict())
        expected = function(value)
        _assert_provider_allclose(provider, program(value), expected)
        _assert_provider_allclose(provider, restored(value), expected)
    else:
        msg = f"no correctness preflight for {case.workload!r}"
        raise ValueError(msg)

    dynamic_provider.testing.assert_array_equal(value, original)
    _sync_provider(provider)
    return {
        "correct": True,
        "input_unchanged": True,
        "scenario": case.name,
    }


def _timed_callable(spec: _WorkerSpec, provider: object) -> Callable[[], object]:
    case = spec.case
    elements = _element_count(spec)
    value = _make_input(provider, elements)
    ad = importlib.import_module("advect")
    importlib.import_module("advect.numpy")
    recomputations = {"count": 0}
    if case.workload == "elementwise":
        call = ad.grad(_elementwise_loss(provider, depth=_DYNAMIC_DEPTH))
        return lambda: call(value)
    if case.workload == "stencil":
        call = ad.grad(_stencil_loss(provider, steps=_STENCIL_STEPS))
        return lambda: call(value)
    if case.workload == "checkpoint":
        loss = _checkpoint_loss(
            provider,
            mode=case.mode,
            calls=recomputations,
        )
        call = ad.grad(loss)

        def checkpoint_call() -> object:
            before = recomputations["count"]
            result = call(value)
            after = recomputations["count"]
            checkpoint_call.recomputations = max(  # type: ignore[attr-defined]
                0,
                after - before - _CHECKPOINT_REGIONS,
            )
            return result

        checkpoint_call.recomputations = 0  # type: ignore[attr-defined]
        return checkpoint_call
    if case.workload == "functional_updates":
        program = _stage_with_profile_binding(
            _functional_update_function(),
            elements=elements,
            tracker=None,
            force_fresh=case.mode == "forced_fresh",
        )
        return lambda: program(value)
    if case.workload == "captured_constant":
        constant = cast("Callable[[object], object]", cast("Any", provider).copy)(value)

        def function(field: object) -> object:
            return field * constant + 0.25  # type: ignore[operator]

        program = _stage_with_profile_binding(
            function,
            elements=elements,
            tracker=None,
            force_fresh=False,
        )
        return lambda: program(value)
    if case.workload in {"residual", "linear_map"}:
        if case.workload == "residual":
            loss, _payload_bytes = _residual_loss(
                provider,
                spec,
                tracker_box=[_BufferTracker()],
                release_calls={"count": 0},
            )
        else:
            loss = _elementwise_loss(provider, depth=_DYNAMIC_DEPTH)

        def linear_call() -> object:
            output, linear = ad.linearize(loss, value)
            try:
                return linear.pullback(_ones_like(provider, output))
            finally:
                linear.close()

        return linear_call
    msg = f"unsupported timing workload {case.workload!r}"
    raise ValueError(msg)


def _run_timing(spec: _WorkerSpec, provider: object) -> dict[str, object]:
    call = _timed_callable(spec, provider)
    for _ in range(3):
        call()
    _sync_provider(provider)
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        started = time.perf_counter_ns()
        for _ in range(spec.timing_iterations):
            call()
        _sync_provider(provider)
        elapsed_ns = time.perf_counter_ns() - started
    finally:
        if gc_was_enabled:
            gc.enable()
    return {
        "iterations": spec.timing_iterations,
        "seconds_per_call": elapsed_ns / (1e9 * spec.timing_iterations),
        "recomputation_count": int(getattr(call, "recomputations", 0)),
    }


def _run_worker(spec: _WorkerSpec) -> dict[str, object]:
    if spec.case.workload == "allocation_probe":
        if spec.measurement == "timing":
            return {"status": "not_applicable"}
        result = _run_allocation_probe(spec)
        environment = {
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
    else:
        provider = _load_provider(spec.case.provider)
        environment = _worker_environment(provider=provider)
        if spec.measurement == "correctness":
            result = _run_correctness_preflight(spec, provider)
        elif spec.measurement == "timing":
            result = _run_timing(spec, provider)
        elif spec.case.workload in {"functional_updates", "captured_constant"}:
            result = _run_staged_memory(spec, provider)
        else:
            result = _run_advect_dynamic_memory(spec, provider)
    return {
        "status": "ok",
        "event": "result",
        "case": asdict(spec.case),
        "measurement": spec.measurement,
        "environment": environment,
        **result,
    }


def _worker_main(raw_spec: str) -> int:
    payload = json.loads(raw_spec)
    case = _Case(**payload.pop("case"))
    spec = _WorkerSpec(case=case, **payload)
    try:
        result = _run_worker(spec)
    except (ImportError, ModuleNotFoundError) as error:
        result = {
            "status": "skipped",
            "event": "result",
            "case": asdict(spec.case),
            "measurement": spec.measurement,
            "reason": str(error),
        }
    except Exception as error:  # noqa: BLE001 - child must report structured diagnostics
        result = {
            "status": "error",
            "event": "result",
            "case": asdict(spec.case),
            "measurement": spec.measurement,
            "reason": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] in {"ok", "skipped"} else 1


def _sample_process(
    process: subprocess.Popen[str],
    *,
    interval_seconds: float,
) -> list[_RssSample]:
    samples: list[_RssSample] = []
    while process.poll() is None:
        memory = _read_proc_memory(process.pid)
        if memory is not None:
            samples.append(_RssSample(time.monotonic_ns(), memory[0], memory[1]))
        time.sleep(interval_seconds)
    memory = _read_proc_memory(process.pid)
    if memory is not None:
        samples.append(_RssSample(time.monotonic_ns(), memory[0], memory[1]))
    return samples


def _nearest_sample(
    samples: Sequence[_RssSample],
    monotonic_ns: int,
) -> _RssSample | None:
    if not samples:
        return None
    return min(samples, key=lambda sample: abs(sample.monotonic_ns - monotonic_ns))


def _annotate_memory_result(
    result: dict[str, object],
    *,
    markers: list[dict[str, object]],
    samples: Sequence[_RssSample],
    include_samples: bool,
) -> None:
    previous_ns = samples[0].monotonic_ns if samples else 0
    for marker in markers:
        marker_ns = int(cast("int", marker["monotonic_ns"]))
        nearest = _nearest_sample(samples, marker_ns)
        window = [sample for sample in samples if previous_ns <= sample.monotonic_ns <= marker_ns]
        marker["parent_rss_current_bytes"] = None if nearest is None else nearest.current_bytes
        marker["parent_rss_peak_since_previous_bytes"] = (
            None if not window else max(sample.current_bytes for sample in window)
        )
        previous_ns = marker_ns

    baseline = next((marker for marker in markers if marker["phase"] == "baseline"), None)
    baseline_ns = (
        samples[0].monotonic_ns
        if baseline is None and samples
        else int(cast("int", baseline["monotonic_ns"]))
        if baseline is not None
        else 0
    )
    active_samples = [sample for sample in samples if sample.monotonic_ns >= baseline_ns]
    baseline_sample = _nearest_sample(samples, baseline_ns)
    child_baseline_rss = None if baseline is None else baseline.get("rss_current_bytes")
    baseline_rss = (
        int(child_baseline_rss)
        if isinstance(child_baseline_rss, int)
        else 0
        if baseline_sample is None
        else baseline_sample.current_bytes
    )
    peak_rss = (
        baseline_rss
        if not active_samples
        else max(sample.current_bytes for sample in active_samples)
    )
    result["markers"] = markers
    result["parent_rss_baseline_bytes"] = baseline_rss
    result["parent_rss_peak_bytes"] = peak_rss
    result["peak_rss_delta_bytes"] = max(0, peak_rss - baseline_rss)
    if include_samples:
        result["rss_samples"] = [asdict(sample) for sample in samples]


def _run_isolated(
    spec: _WorkerSpec,
    *,
    sample_interval_seconds: float,
    include_samples: bool,
) -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "scripts.bench_runtime_memory",
        "--_worker-spec",
        json.dumps(asdict(spec), separators=(",", ":"), sort_keys=True),
    ]
    process = subprocess.Popen(  # noqa: S603 - fixed current interpreter and script
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    samples = (
        _sample_process(process, interval_seconds=sample_interval_seconds)
        if spec.measurement == "memory"
        else []
    )
    stdout, stderr = process.communicate()
    events: list[dict[str, object]] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        events.append(cast("dict[str, object]", json.loads(line)))
    result = next(
        (event for event in reversed(events) if event.get("event") == "result"),
        None,
    )
    if result is None:
        return {
            "status": "error",
            "case": asdict(spec.case),
            "measurement": spec.measurement,
            "reason": "worker emitted no result",
            "stderr": stderr,
            "returncode": process.returncode,
        }
    if stderr:
        result["stderr"] = stderr
    result["returncode"] = process.returncode
    if spec.measurement == "memory":
        markers = [event for event in events if event.get("event") == "phase"]
        _annotate_memory_result(
            result,
            markers=markers,
            samples=samples,
            include_samples=include_samples,
        )
    return result


def _numeric_summary(values: Sequence[float | int]) -> dict[str, float]:
    if not values:
        return {}
    median = float(statistics.median(values))
    minimum = float(min(values))
    maximum = float(max(values))
    variation = 0.0 if median == 0 else (maximum - minimum) / median
    median_absolute_deviation = float(
        statistics.median(abs(float(value) - median) for value in values)
    )
    median_absolute_variation = 0.0 if median == 0 else median_absolute_deviation / abs(median)
    return {
        "median": median,
        "min": minimum,
        "max": maximum,
        "range": maximum - minimum,
        "variation": variation,
        "median_absolute_deviation": median_absolute_deviation,
        "median_absolute_variation": median_absolute_variation,
    }


def _marker_metric(run: Mapping[str, object], key: str) -> float | None:
    markers = cast("Sequence[Mapping[str, object]]", run.get("markers", ()))
    values = [
        float(value)
        for marker in markers
        if (value := marker.get(key)) is not None and isinstance(value, (int, float))
    ]
    return max(values) if values else None


def _marker_phase_metric(
    run: Mapping[str, object],
    *,
    phases: frozenset[str],
    key: str,
) -> float | None:
    markers = cast("Sequence[Mapping[str, object]]", run.get("markers", ()))
    values = [
        float(value)
        for marker in markers
        if marker.get("phase") in phases
        and (value := marker.get(key)) is not None
        and isinstance(value, (int, float))
    ]
    return max(values) if values else None


def _marker_delta_from_baseline(run: Mapping[str, object], key: str) -> float | None:
    markers = cast("Sequence[Mapping[str, object]]", run.get("markers", ()))
    baseline = next((marker for marker in markers if marker.get("phase") == "baseline"), None)
    if baseline is None:
        return None
    raw_baseline = baseline.get(key)
    if not isinstance(raw_baseline, (int, float)):
        return None
    values = [
        float(value)
        for marker in markers
        if (value := marker.get(key)) is not None and isinstance(value, (int, float))
    ]
    return max(0.0, max(values, default=float(raw_baseline)) - float(raw_baseline))


def _marker_decrease_from_baseline(run: Mapping[str, object], key: str) -> float | None:
    markers = cast("Sequence[Mapping[str, object]]", run.get("markers", ()))
    baseline = next((marker for marker in markers if marker.get("phase") == "baseline"), None)
    if baseline is None:
        return None
    raw_baseline = baseline.get(key)
    if not isinstance(raw_baseline, (int, float)):
        return None
    values = [
        float(value)
        for marker in markers
        if (value := marker.get(key)) is not None and isinstance(value, (int, float))
    ]
    return max(0.0, float(raw_baseline) - min(values, default=float(raw_baseline)))


def _available_metrics(values: Iterable[float | None]) -> list[float]:
    return [value for value in values if value is not None]


def _summarize_case_runs(
    case: _Case,
    memory_runs: list[dict[str, object]],
    timing_runs: list[dict[str, object]],
) -> dict[str, object]:
    successful_memory = [run for run in memory_runs if run.get("status") == "ok"]
    successful_timing = [run for run in timing_runs if run.get("status") == "ok"]
    memory_summary = {
        "peak_rss_delta_bytes": _numeric_summary(
            [cast("int", run["peak_rss_delta_bytes"]) for run in successful_memory]
        ),
        "peak_tracemalloc_delta_bytes": _numeric_summary(
            _available_metrics(
                _marker_metric(run, "tracemalloc_peak_delta_bytes") for run in successful_memory
            )
        ),
        "peak_provider_delta_bytes": _numeric_summary(
            _available_metrics(
                _marker_metric(run, "provider_peak_delta_bytes") for run in successful_memory
            )
        ),
        "peak_provider_pool_used_delta_bytes": _numeric_summary(
            _available_metrics(
                _marker_delta_from_baseline(run, "provider_pool_used_bytes")
                for run in successful_memory
            )
        ),
        "peak_provider_pool_reserved_delta_bytes": _numeric_summary(
            _available_metrics(
                _marker_delta_from_baseline(run, "provider_pool_reserved_bytes")
                for run in successful_memory
            )
        ),
        "peak_device_used_delta_bytes": _numeric_summary(
            _available_metrics(
                _marker_decrease_from_baseline(run, "device_free_bytes")
                for run in successful_memory
            )
        ),
        "reverse_entry_provider_owned_bytes": _numeric_summary(
            _available_metrics(
                _marker_phase_metric(
                    run,
                    phases=frozenset({"forward"}),
                    key="provider_owned_live_bytes",
                )
                for run in successful_memory
            )
        ),
        "reverse_retained_provider_owned_bytes": _numeric_summary(
            _available_metrics(
                _marker_phase_metric(
                    run,
                    phases=frozenset({"reverse_retained"}),
                    key="provider_owned_live_bytes",
                )
                for run in successful_memory
            )
        ),
        "post_close_provider_owned_bytes": _numeric_summary(
            _available_metrics(
                _marker_phase_metric(
                    run,
                    phases=frozenset({"closed"}),
                    key="provider_owned_live_bytes",
                )
                for run in successful_memory
            )
        ),
        "provider_cache_live_bytes": _numeric_summary(
            _available_metrics(
                _marker_metric(run, "provider_cache_live_bytes") for run in successful_memory
            )
        ),
        "native_structural_bytes": _numeric_summary(
            _available_metrics(
                _marker_metric(run, "native_structural_bytes") for run in successful_memory
            )
        ),
        "recomputation_count": _numeric_summary(
            _available_metrics(
                _marker_metric(run, "recomputation_count") for run in successful_memory
            )
        ),
        "residual_release_count": _numeric_summary(
            _available_metrics(
                _marker_metric(run, "residual_release_count") for run in successful_memory
            )
        ),
        "compile_seconds": _numeric_summary(
            [
                cast("float", run["compile_seconds"])
                for run in successful_memory
                if isinstance(run.get("compile_seconds"), float)
            ]
        ),
    }
    timing_summary = _numeric_summary(
        [cast("float", run["seconds_per_call"]) for run in successful_timing]
    )
    statuses = [str(run.get("status")) for run in (*memory_runs, *timing_runs)]
    status = (
        "error"
        if "error" in statuses
        else "skipped"
        if statuses and all(item == "skipped" for item in statuses)
        else "ok"
    )
    return {
        "name": case.name,
        "case": asdict(case),
        "status": status,
        "memory": {
            "summary": memory_summary,
            "runs": memory_runs,
        },
        "timing": {
            "summary_seconds_per_call": timing_summary,
            "runs": timing_runs,
        },
    }


def _profile_contract_violations(
    payload: Mapping[str, object],
    *,
    profile: _AcceptanceProfile,
) -> list[str]:
    violations: list[str] = []
    config = cast("Mapping[str, object]", payload["config"])
    if int(cast("int", config["runs"])) != _ACCEPTANCE_RUNS:
        violations.append("acceptance requires exactly five memory runs")
    if int(cast("int", config["timing_runs"])) != _ACCEPTANCE_TIMING_RUNS:
        violations.append("acceptance requires exactly five timing runs")
    if int(cast("int", config["timing_iterations"])) != _ACCEPTANCE_TIMING_ITERATIONS:
        violations.append("acceptance requires exactly ten calls per timing worker")
    if bool(config.get("no_timing")):
        violations.append("acceptance profiles require separate timing workers")
    if int(cast("int", config["byte_budget"])) < _DEFAULT_BUDGET:
        violations.append("acceptance requires at least 64 MiB of target live provider data")
    cases = cast("Sequence[Mapping[str, object]]", payload["cases"])
    expected_case_names = {case.name for case in profile.cases}
    actual_case_names = {cast("str", case["name"]) for case in cases}
    if actual_case_names != expected_case_names:
        violations.append(
            "profile case matrix differs: "
            f"missing={sorted(expected_case_names - actual_case_names)}, "
            f"unexpected={sorted(actual_case_names - expected_case_names)}"
        )
    preflights = cast("Sequence[Mapping[str, object]]", payload.get("correctness_preflights", ()))
    expected_preflights = {
        name for name in expected_case_names if not name.startswith("allocation_probe:")
    }
    actual_preflights = {
        cast("str", item.get("scenario", ""))
        for item in preflights
        if item.get("status") == "ok" and item.get("correct") is True
    }
    if actual_preflights != expected_preflights:
        violations.append(
            "correctness preflights differ from the profile: "
            f"missing={sorted(expected_preflights - actual_preflights)}, "
            f"unexpected={sorted(actual_preflights - expected_preflights)}"
        )
    for preflight in preflights:
        if preflight.get("status") != "ok" or preflight.get("correct") is not True:
            case = cast("Mapping[str, object]", preflight.get("case", {}))
            violations.append(
                f"{case.get('workload', 'unknown')} correctness preflight failed: "
                f"{preflight.get('reason', 'no reason recorded')}"
            )
    return violations


def _phase_metric_is_present(
    run: Mapping[str, object],
    *,
    phase: str,
    key: str,
) -> bool:
    markers = cast("Sequence[Mapping[str, object]]", run.get("markers", ()))
    return any(
        marker.get("phase") == phase and isinstance(marker.get(key), (int, float))
        for marker in markers
    )


def _required_memory_metrics(
    run: Mapping[str, object],
    *,
    case: Mapping[str, str],
) -> list[tuple[str, bool]]:
    required: list[tuple[str, bool]] = [
        ("peak_rss_delta_bytes", isinstance(run.get("peak_rss_delta_bytes"), (int, float)))
    ]
    if case["provider"] == "cupy":
        required.append(
            (
                "provider_pool_reserved_delta",
                _marker_delta_from_baseline(run, "provider_pool_reserved_bytes") is not None,
            )
        )
    if case["framework"] == "advect":
        required.append(
            (
                "closed.provider_owned_live_bytes",
                _phase_metric_is_present(
                    run,
                    phase="closed",
                    key="provider_owned_live_bytes",
                ),
            )
        )
    workload_metric = {
        "checkpoint": ("closed", "recomputation_count"),
        "residual": ("closed", "residual_release_count"),
        "linear_map": ("reverse_retained", "provider_owned_live_bytes"),
        "captured_constant": ("execute", "provider_cache_live_bytes"),
    }.get(case["workload"])
    if workload_metric is not None:
        phase, key = workload_metric
        required.append(
            (
                f"{phase}.{key}",
                _phase_metric_is_present(run, phase=phase, key=key),
            )
        )
    if case["workload"] == "functional_updates":
        required.append(("input_bytes", isinstance(run.get("input_bytes"), int)))
    return required


def _run_contract_violations(
    case: Mapping[str, object],
    *,
    profile: _AcceptanceProfile,
) -> list[str]:
    """Require every profile run and every metric consumed by its verdict."""
    violations: list[str] = []
    case_meta = cast("Mapping[str, str]", case["case"])
    memory_runs = cast(
        "Sequence[Mapping[str, object]]",
        cast("Mapping[str, object]", case["memory"])["runs"],
    )
    timing_runs = cast(
        "Sequence[Mapping[str, object]]",
        cast("Mapping[str, object]", case["timing"])["runs"],
    )
    requires_timing = (case_meta["workload"], case_meta["mode"]) in profile.timed_cases
    expected_timing_runs = _ACCEPTANCE_TIMING_RUNS if requires_timing else 0

    for label, runs, expected in (
        ("memory", memory_runs, _ACCEPTANCE_RUNS),
        ("timing", timing_runs, expected_timing_runs),
    ):
        successful = sum(run.get("status") == "ok" for run in runs)
        if len(runs) != expected or successful != expected:
            violations.append(
                f"{case['name']} requires exactly {expected} successful {label} runs; "
                f"recorded={len(runs)}, successful={successful}"
            )

    for index, run in enumerate(memory_runs, start=1):
        if run.get("status") != "ok":
            continue
        for metric, present in _required_memory_metrics(run, case=case_meta):
            if not present:
                violations.append(f"{case['name']} memory run {index} is missing {metric}")

    for index, run in enumerate(timing_runs, start=1):
        if run.get("status") == "ok" and not isinstance(run.get("seconds_per_call"), (int, float)):
            violations.append(f"{case['name']} timing run {index} is missing seconds_per_call")
    return violations


def _case_measurement_violations(
    cases: Sequence[Mapping[str, object]],
    *,
    profile: _AcceptanceProfile,
) -> list[str]:
    violations: list[str] = []
    for case in cases:
        violations.extend(_run_contract_violations(case, profile=profile))
        if case["status"] != "ok":
            violations.append(f"{case['name']} did not complete successfully")
            continue
        memory = cast("Mapping[str, object]", case["memory"])
        summary = cast("Mapping[str, Mapping[str, float]]", memory["summary"])
        case_meta = cast("Mapping[str, str]", case["case"])
        primary_name = (
            "peak_provider_pool_reserved_delta_bytes"
            if case_meta["provider"] == "cupy"
            else "peak_rss_delta_bytes"
        )
        primary = summary[primary_name]
        if primary and primary["median_absolute_variation"] > _MAX_VARIATION:
            label = (
                "provider reserved-memory"
                if primary_name == "peak_provider_pool_reserved_delta_bytes"
                else "peak RSS"
            )
            violations.append(
                f"{case['name']} {label} median absolute variation is "
                f"{primary['median_absolute_variation']:.1%}, "
                f"above {_MAX_VARIATION:.0%}"
            )
        timing = cast("Mapping[str, object]", case.get("timing", {}))
        timing_summary = cast(
            "Mapping[str, float]",
            timing.get("summary_seconds_per_call", {}),
        )
        requires_timing = (case_meta["workload"], case_meta["mode"]) in profile.timed_cases
        if requires_timing:
            if not timing_summary:
                violations.append(f"{case['name']} has no timing evidence")
            elif timing_summary["median_absolute_variation"] > _MAX_VARIATION:
                violations.append(
                    f"{case['name']} timing median absolute variation is "
                    f"{timing_summary['median_absolute_variation']:.1%}, "
                    f"above {_MAX_VARIATION:.0%}"
                )
        if case_meta["framework"] == "advect":
            memory_runs = cast(
                "Sequence[Mapping[str, object]]",
                cast("Mapping[str, object]", case["memory"])["runs"],
            )
            timing_runs = cast(
                "Sequence[Mapping[str, object]]",
                cast("Mapping[str, object]", case["timing"])["runs"],
            )
            for measurement, runs in (("memory", memory_runs), ("timing", timing_runs)):
                for index, run in enumerate(runs, start=1):
                    if run.get("status") != "ok":
                        continue
                    environment = cast("Mapping[str, object]", run.get("environment", {}))
                    native = cast("Mapping[str, object]", environment.get("advect_native", {}))
                    if native.get("build_profile") != "release":
                        violations.append(
                            f"{case['name']} {measurement} run {index} did not use a release "
                            "native extension"
                        )
                    if environment.get("advect_debug") is not False:
                        violations.append(
                            f"{case['name']} {measurement} run {index} ran with Advect "
                            "diagnostics enabled"
                        )
    return violations


def _calibration_violations(
    cases: Sequence[Mapping[str, object]],
    *,
    expected_bytes: int,
) -> list[str]:
    violations: list[str] = []

    calibration = next(
        (
            case
            for case in cases
            if cast("Mapping[str, str]", case["case"])["workload"] == "allocation_probe"
        ),
        None,
    )
    if calibration is None:
        violations.append("acceptance requires the known allocation probe")
    else:
        summary = cast(
            "Mapping[str, Mapping[str, float]]",
            cast("Mapping[str, object]", calibration["memory"])["summary"],
        )
        measured = summary["peak_rss_delta_bytes"].get("median", 0.0)
        expected = float(expected_bytes)
        error = abs(measured - expected) / expected
        if error > _MAX_CALIBRATION_ERROR:
            violations.append(
                f"known allocation probe error is {error:.1%}, above the 10% calibration gate"
            )
    return violations


def _acceptance_violations(
    payload: Mapping[str, object],
    *,
    requested: bool,
    gated_checks: Sequence[Mapping[str, object]],
) -> list[str]:
    if not requested:
        return []
    violations: list[str] = []
    environment = cast("Mapping[str, object]", payload.get("environment", {}))
    if not source_revision_is_recorded(environment.get("source_revision")):
        violations.append(
            "source revision is unrecorded; set ADVECT_SOURCE_REVISION to the exact source state"
        )
    config = cast("Mapping[str, object]", payload["config"])
    profile_name = config.get("profile")
    if not isinstance(profile_name, str) or profile_name not in _ACCEPTANCE_PROFILES:
        violations.append("acceptance requires one named exact profile")
        return violations
    profile = _ACCEPTANCE_PROFILES[profile_name]
    cases = cast("Sequence[Mapping[str, object]]", payload["cases"])
    violations.extend(_profile_contract_violations(payload, profile=profile))
    violations.extend(_case_measurement_violations(cases, profile=profile))
    violations.extend(
        _calibration_violations(
            cases,
            expected_bytes=cast("int", config["byte_budget"]),
        )
    )
    violations.extend(
        f"{check['name']}: {check['reason']}"
        for check in gated_checks
        if check["status"] in {"failed", "unavailable"}
    )
    return violations


def _case_by_workload_mode(
    cases: Sequence[Mapping[str, object]],
    *,
    workload: str,
    mode: str,
) -> Mapping[str, object] | None:
    for case in cases:
        metadata = cast("Mapping[str, str]", case["case"])
        if metadata["workload"] == workload and metadata["mode"] == mode:
            return case
    return None


def _summary_median(case: Mapping[str, object], metric: str) -> float | None:
    memory = cast("Mapping[str, object]", case["memory"])
    summary = cast("Mapping[str, Mapping[str, float]]", memory["summary"])
    value = summary[metric].get("median")
    return None if value is None else float(value)


def _summary_stat(case: Mapping[str, object], metric: str, statistic: str) -> float | None:
    memory = cast("Mapping[str, object]", case["memory"])
    summary = cast("Mapping[str, Mapping[str, float]]", memory["summary"])
    value = summary[metric].get(statistic)
    return None if value is None else float(value)


def _timing_median(case: Mapping[str, object]) -> float | None:
    timing = cast("Mapping[str, object]", case["timing"])
    summary = cast("Mapping[str, float]", timing["summary_seconds_per_call"])
    value = summary.get("median")
    return None if value is None else float(value)


def _checkpoint_acceptance_check(
    plain: Mapping[str, object],
    checkpointed: Mapping[str, object],
) -> dict[str, object]:
    plain_rss = _summary_median(plain, "peak_rss_delta_bytes")
    checkpoint_rss = _summary_median(checkpointed, "peak_rss_delta_bytes")
    plain_runtime = _timing_median(plain)
    checkpoint_runtime = _timing_median(checkpointed)
    if (
        plain_rss is None
        or checkpoint_rss is None
        or plain_runtime is None
        or checkpoint_runtime is None
        or plain_rss <= 0
        or checkpoint_rss <= 0
        or plain_runtime <= 0
    ):
        return {
            "name": "checkpoint memory/runtime tradeoff",
            "status": "unavailable",
            "reason": "paired nonzero RSS and separate timing runs are required",
        }
    memory_ratio = checkpoint_rss / plain_rss
    runtime_ratio = checkpoint_runtime / plain_runtime
    passed = memory_ratio <= _CHECKPOINT_MEMORY_RATIO and runtime_ratio <= _CHECKPOINT_RUNTIME_RATIO
    return {
        "name": "checkpoint memory/runtime tradeoff",
        "status": "passed" if passed else "failed",
        "reason": ("requires at least 25% lower peak RSS and at most 35% additional runtime"),
        "checkpoint_over_plain_peak_rss": memory_ratio,
        "checkpoint_over_plain_runtime": runtime_ratio,
        "memory_threshold": _CHECKPOINT_MEMORY_RATIO,
        "runtime_threshold": _CHECKPOINT_RUNTIME_RATIO,
    }


def _donation_acceptance_check(
    donated: Mapping[str, object],
    forced_fresh: Mapping[str, object],
    *,
    provider: str,
) -> dict[str, object]:
    if provider != "cupy":
        return {
            "name": "staged donation device-memory/runtime",
            "status": "not_applicable",
            "reason": (
                "CPU allocation tracking is diagnostic; the donation gate requires "
                "CuPy device-memory measurements"
            ),
        }
    donated_memory = _summary_median(donated, "peak_provider_pool_reserved_delta_bytes")
    fresh_memory = _summary_median(forced_fresh, "peak_provider_pool_reserved_delta_bytes")
    donated_device = _summary_median(donated, "peak_device_used_delta_bytes")
    fresh_device = _summary_median(forced_fresh, "peak_device_used_delta_bytes")
    donated_runtime = _timing_median(donated)
    fresh_runtime = _timing_median(forced_fresh)
    fresh_runs = cast(
        "Sequence[Mapping[str, object]]",
        cast("Mapping[str, object]", forced_fresh["memory"])["runs"],
    )
    input_bytes = (
        float(cast("int", fresh_runs[0]["input_bytes"]))
        if fresh_runs and isinstance(fresh_runs[0].get("input_bytes"), int)
        else None
    )
    if (
        donated_memory is None
        or fresh_memory is None
        or donated_runtime is None
        or fresh_runtime is None
        or input_bytes is None
        or fresh_memory <= 0
        or fresh_runtime <= 0
    ):
        return {
            "name": "staged donation device-memory/runtime",
            "status": "unavailable",
            "reason": (
                "paired CuPy pool-reserved high-water, input size, and timing runs are required"
            ),
        }
    memory_ratio = donated_memory / fresh_memory
    runtime_ratio = donated_runtime / fresh_runtime
    avoided_bytes = fresh_memory - donated_memory
    passed = (
        memory_ratio <= _DONATION_MEMORY_RATIO
        and runtime_ratio <= _DONATION_RUNTIME_RATIO
        and avoided_bytes >= input_bytes
    )
    return {
        "name": "staged donation device-memory/runtime",
        "status": "passed" if passed else "failed",
        "reason": (
            "requires one full destination buffer avoided, at least 20% lower "
            "provider high-water usage, and at most 5% runtime change"
        ),
        "donation_over_forced_fresh_provider_memory": memory_ratio,
        "donation_over_forced_fresh_runtime": runtime_ratio,
        "donation_device_used_delta_bytes": donated_device,
        "forced_fresh_device_used_delta_bytes": fresh_device,
        "avoided_bytes": avoided_bytes,
        "required_avoided_bytes": input_bytes,
        "memory_threshold": _DONATION_MEMORY_RATIO,
        "runtime_threshold": _DONATION_RUNTIME_RATIO,
    }


def _paired_acceptance_checks(payload: Mapping[str, object]) -> list[dict[str, object]]:
    config = cast("Mapping[str, object]", payload["config"])
    provider = cast("str", config["provider"])
    cases = cast("Sequence[Mapping[str, object]]", payload["cases"])
    checks: list[dict[str, object]] = []
    plain = _case_by_workload_mode(cases, workload="checkpoint", mode="plain")
    checkpointed = _case_by_workload_mode(cases, workload="checkpoint", mode="checkpoint")
    if plain is not None and checkpointed is not None:
        checks.append(_checkpoint_acceptance_check(plain, checkpointed))
    donated = _case_by_workload_mode(cases, workload="functional_updates", mode="donation")
    forced_fresh = _case_by_workload_mode(
        cases,
        workload="functional_updates",
        mode="forced_fresh",
    )
    if donated is not None and forced_fresh is not None:
        checks.append(
            _donation_acceptance_check(
                donated,
                forced_fresh,
                provider=provider,
            )
        )
    return checks


def _zero_post_close_check(case: Mapping[str, object]) -> dict[str, object]:
    maximum = _summary_stat(case, "post_close_provider_owned_bytes", "max")
    if maximum is None:
        return {
            "name": f"{case['name']} post-close release",
            "status": "unavailable",
            "reason": "post-close provider-owned bytes were not measured",
        }
    return {
        "name": f"{case['name']} post-close release",
        "status": "passed" if maximum == 0 else "failed",
        "reason": "every run must release all provider values owned by the measured lifetime",
        "maximum_post_close_provider_owned_bytes": maximum,
    }


def _exact_summary_check(
    case: Mapping[str, object],
    *,
    metric: str,
    expected: float,
    label: str,
) -> dict[str, object]:
    minimum = _summary_stat(case, metric, "min")
    maximum = _summary_stat(case, metric, "max")
    if minimum is None or maximum is None:
        return {
            "name": label,
            "status": "unavailable",
            "reason": f"{metric} was not measured",
        }
    passed = minimum == expected and maximum == expected
    return {
        "name": label,
        "status": "passed" if passed else "failed",
        "reason": f"every run requires {metric} == {expected:g}",
        "minimum": minimum,
        "maximum": maximum,
        "expected": expected,
    }


def _lifetime_acceptance_checks(
    payload: Mapping[str, object],
    *,
    profile: _AcceptanceProfile,
) -> list[dict[str, object]]:
    cases = cast("Sequence[Mapping[str, object]]", payload["cases"])
    checks = [
        _zero_post_close_check(case)
        for case in cases
        if cast("Mapping[str, str]", case["case"])["framework"] == "advect"
    ]
    if profile.name == "cpu-runtime":
        residual = _case_by_workload_mode(cases, workload="residual", mode="retained")
        if residual is not None:
            checks.append(
                _exact_summary_check(
                    residual,
                    metric="residual_release_count",
                    expected=1.0,
                    label="residual release count",
                )
            )
        plain = _case_by_workload_mode(cases, workload="checkpoint", mode="plain")
        checkpointed = _case_by_workload_mode(
            cases,
            workload="checkpoint",
            mode="checkpoint",
        )
        if plain is not None:
            checks.append(
                _exact_summary_check(
                    plain,
                    metric="recomputation_count",
                    expected=0.0,
                    label="plain checkpoint control replay count",
                )
            )
        if checkpointed is not None:
            checks.append(
                _exact_summary_check(
                    checkpointed,
                    metric="recomputation_count",
                    expected=float(_CHECKPOINT_REGIONS),
                    label="checkpoint replay count",
                )
            )
        linear_map = _case_by_workload_mode(cases, workload="linear_map", mode="reusable")
        if linear_map is not None:
            retained = _summary_stat(
                linear_map,
                "reverse_retained_provider_owned_bytes",
                "min",
            )
            checks.append(
                {
                    "name": "linear map retains its reusable reverse state",
                    "status": "passed" if retained is not None and retained > 0 else "failed",
                    "reason": ("every retained reverse phase must own provider state before close"),
                    "minimum_reverse_retained_provider_owned_bytes": retained,
                }
            )
        captured = _case_by_workload_mode(cases, workload="captured_constant", mode="staged")
        if captured is not None:
            provider_cache = _summary_stat(captured, "provider_cache_live_bytes", "min")
            checks.append(
                {
                    "name": "staged captured constant is attributed to the provider cache",
                    "status": (
                        "passed" if provider_cache is not None and provider_cache > 0 else "failed"
                    ),
                    "reason": ("every warm staged program must report its materialized constant"),
                    "minimum_provider_cache_live_bytes": provider_cache,
                }
            )
    return checks


def _gated_acceptance_checks(
    payload: Mapping[str, object],
    *,
    profile: _AcceptanceProfile | None,
) -> list[dict[str, object]]:
    if profile is None:
        return []
    return [
        *_paired_acceptance_checks(payload),
        *_lifetime_acceptance_checks(payload, profile=profile),
    ]


def _format_bytes(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value) / _MIB:.1f}"


def _print_text(payload: Mapping[str, object]) -> None:
    config = cast("Mapping[str, object]", payload["config"])
    print(
        f"budget={_format_bytes(cast('int', config['byte_budget']))} MiB; "
        f"runs={config['runs']}; provider={config['provider']}"
    )
    print(
        "case                                             "
        "RSS MiB  Provider MiB  Trace MiB  Native KiB  Compile ms  Runtime ms"
    )
    for case in cast("Sequence[Mapping[str, object]]", payload["cases"]):
        if case["status"] != "ok":
            print(f"{case['name']!s:<49}{case['status']!s:>8}")
            continue
        memory = cast("Mapping[str, object]", case["memory"])
        summaries = cast("Mapping[str, Mapping[str, float]]", memory["summary"])
        timing = cast("Mapping[str, object]", case["timing"])
        timing_summary = cast("Mapping[str, float]", timing["summary_seconds_per_call"])
        rss = summaries["peak_rss_delta_bytes"].get("median")
        provider = summaries["peak_provider_delta_bytes"].get("median")
        traced = summaries["peak_tracemalloc_delta_bytes"].get("median")
        native = summaries["native_structural_bytes"].get("median")
        compile_seconds = summaries["compile_seconds"].get("median")
        runtime = timing_summary.get("median")
        print(
            f"{case['name']!s:<49}"
            f"{_format_bytes(rss):>8}"
            f"{_format_bytes(provider):>14}"
            f"{_format_bytes(traced):>11}"
            f"{('-' if native is None else f'{native / 1024:.1f}'):>12}"
            f"{('-' if compile_seconds is None else f'{compile_seconds * 1000:.2f}'):>12}"
            f"{('-' if runtime is None else f'{runtime * 1000:.2f}'):>12}"
        )
    violations = cast("Sequence[str]", payload["acceptance"]["violations"])  # type: ignore[index]
    checks = cast(
        "Sequence[Mapping[str, object]]",
        payload["acceptance"]["gated_checks"],  # type: ignore[index]
    )
    if checks:
        print("\ngated acceptance checks")
        for check in checks:
            print(f"- {check['name']}: {check['status']} ({check['reason']})")
    if violations:
        print("\nacceptance violations")
        for violation in violations:
            print(f"- {violation}")


def _build_payload(args: argparse.Namespace) -> dict[str, object]:
    profile = _ACCEPTANCE_PROFILES.get(args.profile) if args.profile is not None else None
    if profile is None:
        msg = "--profile is required"
        raise ValueError(msg)
    cases = profile.cases
    workloads = tuple(dict.fromkeys(case.workload for case in cases))
    providers = tuple(dict.fromkeys(case.provider for case in cases if case.framework == "advect"))
    if len(providers) != 1:
        msg = f"profile {profile.name!r} must select exactly one Advect provider"
        raise ValueError(msg)
    provider = providers[0]
    byte_budget = args.byte_budget
    max_bytes = args.max_bytes
    runs = args.runs
    timing_runs = args.timing_runs
    timing_iterations = args.timing_iterations
    hold_seconds = args.sample_hold_ms / 1000.0
    if args.smoke:
        byte_budget = min(byte_budget, 4 * _MIB)
        max_bytes = max(byte_budget, min(max_bytes, 16 * _MIB))
        runs = 1
        timing_runs = 1
        timing_iterations = 1
        hold_seconds = min(hold_seconds, 0.005)
    if byte_budget > max_bytes:
        msg = f"--byte-budget ({byte_budget}) exceeds --max-bytes ({max_bytes})"
        raise ValueError(msg)

    correctness_preflights = [
        _run_isolated(
            _WorkerSpec(
                case=case,
                byte_budget=min(byte_budget, 64 * 1024),
                max_bytes=min(max_bytes, _MIB),
                sample_hold_seconds=0.0,
                measurement="correctness",
                timing_iterations=1,
            ),
            sample_interval_seconds=args.sample_interval_ms / 1000.0,
            include_samples=False,
        )
        for case in _correctness_preflight_cases(cases)
    ]
    case_results: list[dict[str, object]] = []
    for case in cases:
        memory_runs = [
            _run_isolated(
                _WorkerSpec(
                    case=case,
                    byte_budget=byte_budget,
                    max_bytes=max_bytes,
                    sample_hold_seconds=hold_seconds,
                    measurement="memory",
                    timing_iterations=timing_iterations,
                ),
                sample_interval_seconds=args.sample_interval_ms / 1000.0,
                include_samples=args.include_rss_samples,
            )
            for _ in range(runs)
        ]
        timing_results: list[dict[str, object]] = []
        profile_requires_timing = (case.workload, case.mode) in profile.timed_cases
        if not args.no_timing and case.workload != "allocation_probe" and profile_requires_timing:
            timing_results = [
                _run_isolated(
                    _WorkerSpec(
                        case=case,
                        byte_budget=byte_budget,
                        max_bytes=max_bytes,
                        sample_hold_seconds=0.0,
                        measurement="timing",
                        timing_iterations=timing_iterations,
                    ),
                    sample_interval_seconds=args.sample_interval_ms / 1000.0,
                    include_samples=False,
                )
                for _ in range(timing_runs)
            ]
        case_results.append(_summarize_case_runs(case, memory_runs, timing_results))

    payload: dict[str, object] = {
        "schema_version": 2,
        "report_kind": "advect.runtime-memory",
        "command": [sys.executable, *sys.argv],
        "config": {
            "profile": profile.name,
            "workloads": list(workloads),
            "framework": "advect",
            "provider": provider,
            "byte_budget": byte_budget,
            "max_bytes": max_bytes,
            "runs": runs,
            "timing_runs": timing_runs,
            "timing_iterations": timing_iterations,
            "sample_interval_ms": args.sample_interval_ms,
            "sample_hold_ms": hold_seconds * 1000,
            "smoke": args.smoke,
            "no_timing": args.no_timing,
        },
        "environment": {
            **evidence_environment(),
            "linux_procfs": Path("/proc/self/status").exists(),
        },
        "correctness_preflights": correctness_preflights,
        "cases": case_results,
    }
    gated_checks = _gated_acceptance_checks(payload, profile=profile)
    violations = _acceptance_violations(
        payload,
        requested=args.acceptance,
        gated_checks=gated_checks,
    )
    payload["acceptance"] = {
        "requested": args.acceptance,
        "profile": profile.name,
        "valid": not violations,
        "violations": violations,
        "gated_checks": gated_checks,
    }
    return payload


def main() -> int:
    """Run the isolated memory corpus and emit text or JSON."""
    args = _arguments()
    if args.worker_spec is not None:
        return _worker_main(args.worker_spec)
    try:
        payload = _build_payload(args)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(payload, indent=2, sort_keys=True) if args.format == "json" else None
    if args.output is not None:
        output = rendered
        if output is None:
            output = json.dumps(payload, indent=2, sort_keys=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{output}\n", encoding="utf-8")
    elif rendered is not None:
        print(rendered)
    else:
        _print_text(payload)
    acceptance = cast("Mapping[str, object]", payload["acceptance"])
    return 0 if not acceptance["requested"] or acceptance["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
