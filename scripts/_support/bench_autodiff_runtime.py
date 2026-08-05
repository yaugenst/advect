"""Report a non-gating dynamic Advect/HIPS Autograd comparison."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, cast

import autograd.numpy as hips_numpy
import numpy as np
from autograd import grad as hips_grad, make_vjp as hips_make_vjp

import advect as ad
import advect.numpy
from advect.core._context import is_debug
from advect.core._native import native_build_info
from scripts._support.evidence import evidence_environment

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclass(frozen=True, slots=True)
class _Workload:
    name: str
    group: str
    comparison: str
    advect_loss: Callable[..., object]
    hips_loss: Callable[..., object]
    advect_args: tuple[object, ...]
    hips_args: tuple[object, ...]
    advect_argnums: int | tuple[int, ...]
    hips_argnum: int | tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _RunConfig:
    size: int
    warmup: int
    rounds: int
    block_size: int
    suite: str
    output_format: str
    warmed_replicates: int
    phases: bool
    scientific_mutation_size: int | None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        msg = f"expected a positive integer, got {value!r}"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=_positive_int, default=32)
    parser.add_argument("--warmup", type=_positive_int, default=100)
    parser.add_argument("--rounds", type=_positive_int, default=11)
    parser.add_argument("--block-size", type=_positive_int, default=1_000)
    parser.add_argument("--suite", choices=("core", "extended", "all"), default="core")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument(
        "--output",
        type=Path,
        help="write JSON output to this path instead of stdout",
    )
    parser.add_argument(
        "--warmed-replicates",
        type=_positive_int,
        default=1,
        help="warm and measure the suite this many times in the current process",
    )
    parser.add_argument(
        "--phases",
        action="store_true",
        help="also measure reusable VJP construction/release and repeated reverse application",
    )
    parser.add_argument(
        "--scientific-mutation-size",
        nargs="?",
        const=4_096,
        type=_positive_int,
        help="also time mutation accumulation at the latency size and this size (default: 4096)",
    )
    return parser.parse_args()


def _core_workloads(size: int) -> tuple[_Workload, ...]:
    values = np.linspace(-1.0, 1.0, size, dtype=np.float64)
    matrix = np.linspace(-0.5, 0.5, size * size, dtype=np.float64).reshape(size, size)
    matrix /= math.sqrt(size)
    hips_matrix = hips_numpy.array(matrix)

    def advect_elementwise(value: np.ndarray) -> object:
        return np.sum(np.sin(value) * value + value * value)

    def hips_elementwise(value: np.ndarray) -> object:
        return hips_numpy.sum(hips_numpy.sin(value) * value + value * value)

    def advect_matmul(value: np.ndarray) -> object:
        return np.sum((matrix @ value) ** 2)

    def hips_matmul(value: np.ndarray) -> object:
        return hips_numpy.sum((hips_matrix @ value) ** 2)

    def advect_stencil(value: np.ndarray) -> object:
        laplacian = value[2:] - 2 * value[1:-1] + value[:-2]
        return np.sum(laplacian**2)

    def hips_stencil(value: np.ndarray) -> object:
        laplacian = value[2:] - 2 * value[1:-1] + value[:-2]
        return hips_numpy.sum(laplacian**2)

    return (
        _Workload(
            "elementwise",
            "core",
            "same source-level kernel",
            advect_elementwise,
            hips_elementwise,
            (values,),
            (values,),
            0,
            0,
        ),
        _Workload(
            "matmul",
            "core",
            "same source-level kernel",
            advect_matmul,
            hips_matmul,
            (values,),
            (values,),
            0,
            0,
        ),
        _Workload(
            "stencil",
            "core",
            "same source-level kernel",
            advect_stencil,
            hips_stencil,
            (values,),
            (values,),
            0,
            0,
        ),
    )


def _extended_workloads(size: int) -> tuple[_Workload, ...]:
    values = np.sin(np.linspace(-1.2, 1.1, size, dtype=np.float64)) * 0.7
    other = np.cos(np.linspace(-0.8, 1.3, size, dtype=np.float64)) * 0.5
    scale = np.linspace(0.3, 1.2, size, dtype=np.float64)
    params = {"weight": values, "bias": other, "scale": scale}

    def advect_pytree(tree: dict[str, np.ndarray]) -> object:
        return np.sum(np.sin(tree["weight"]) * tree["scale"] + tree["bias"] ** 2)

    def hips_pytree(tree: dict[str, np.ndarray]) -> object:
        return hips_numpy.sum(hips_numpy.sin(tree["weight"]) * tree["scale"] + tree["bias"] ** 2)

    def advect_multiarg(left: np.ndarray, right: np.ndarray) -> object:
        return np.sum(np.sin(left) * right + left * left)

    def hips_multiarg(left: np.ndarray, right: np.ndarray) -> object:
        return hips_numpy.sum(hips_numpy.sin(left) * right + left * left)

    def advect_mutation_stencil(field: np.ndarray) -> object:
        result = field.copy()
        laplacian = result[2:] - 2.0 * result[1:-1] + result[:-2]
        result[1:-1] += 0.25 * laplacian
        return np.sum(result * result)

    def hips_mutation_stencil(field: np.ndarray) -> object:
        laplacian = field[2:] - 2.0 * field[1:-1] + field[:-2]
        interior = field[1:-1] + 0.25 * laplacian
        result = hips_numpy.concatenate((field[:1], interior, field[-1:]))
        return hips_numpy.sum(result * result)

    def advect_mutation_accumulation(field: np.ndarray) -> object:
        result = np.zeros_like(field)
        result[1:-1] += field[:-2]
        result[1:-1] += -2.0 * field[1:-1]
        result[1:-1] += field[2:]
        return np.sum(np.sin(result) + result * result)

    def hips_mutation_accumulation(field: np.ndarray) -> object:
        interior = field[:-2] - 2.0 * field[1:-1] + field[2:]
        result = hips_numpy.concatenate((hips_numpy.zeros(1), interior, hips_numpy.zeros(1)))
        return hips_numpy.sum(hips_numpy.sin(result) + result * result)

    return (
        _Workload(
            "pytree",
            "extended",
            "same nested mapping input",
            advect_pytree,
            hips_pytree,
            (params,),
            (params,),
            0,
            0,
        ),
        _Workload(
            "multiarg",
            "extended",
            "same two differentiated inputs",
            advect_multiarg,
            hips_multiarg,
            (values, other),
            (values, other),
            (0, 1),
            (0, 1),
        ),
        _Workload(
            "mutation_stencil",
            "extended",
            "functionalized += versus an exact pure HIPS rewrite",
            advect_mutation_stencil,
            hips_mutation_stencil,
            (values,),
            (values,),
            0,
            0,
        ),
        _Workload(
            "mutation_accumulation",
            "extended",
            "three functionalized += updates versus an exact pure HIPS rewrite",
            advect_mutation_accumulation,
            hips_mutation_accumulation,
            (values,),
            (values,),
            0,
            0,
        ),
    )


def _workloads(size: int, suite: str) -> tuple[_Workload, ...]:
    core = _core_workloads(size)
    extended = _extended_workloads(size)
    if suite == "core":
        return core
    if suite == "extended":
        return extended
    return core + extended


def _mutation_accumulation_workload(size: int) -> _Workload:
    return next(
        workload
        for workload in _extended_workloads(size)
        if workload.name == "mutation_accumulation"
    )


def _run_block(call: Callable[[], object], block_size: int) -> float:
    started = time.perf_counter_ns()
    for _ in range(block_size):
        call()
    return (time.perf_counter_ns() - started) / (1_000.0 * block_size)


def _time_pair(
    advect_call: Callable[[], object],
    hips_call: Callable[[], object],
    *,
    warmup: int,
    rounds: int,
    block_size: int,
) -> tuple[list[float], list[float]]:
    # Warm each implementation independently. Alternating whole blocks (AB/BA)
    # balances drift without perturbing interpreter and provider caches on every
    # measured invocation.
    for call in (advect_call, hips_call):
        for _ in range(warmup):
            call()

    advect_samples: list[float] = []
    hips_samples: list[float] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for iteration in range(rounds):
            ordered = (
                ((advect_call, advect_samples), (hips_call, hips_samples))
                if iteration % 2 == 0
                else ((hips_call, hips_samples), (advect_call, advect_samples))
            )
            for call, samples in ordered:
                samples.append(_run_block(call, block_size))
    finally:
        if gc_was_enabled:
            gc.enable()
    return advect_samples, hips_samples


def _summary(samples: Sequence[float]) -> dict[str, float]:
    return {"median_us": statistics.median(samples)}


def _assert_tree_allclose(actual: object, expected: object, *, workload: str) -> None:
    actual_leaves, actual_treedef = ad.pytree.tree_flatten(actual)
    expected_leaves, expected_treedef = ad.pytree.tree_flatten(expected)
    if actual_treedef != expected_treedef:
        msg = f"result structure mismatch for {workload}"
        raise AssertionError(msg)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        if actual_leaf is None or expected_leaf is None:
            if actual_leaf is not expected_leaf:
                msg = f"result leaf mismatch for {workload}"
                raise AssertionError(msg)
            continue
        np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=1e-10, atol=1e-12)


def _paired_summary(
    advect_samples: Sequence[float],
    hips_samples: Sequence[float],
) -> dict[str, object]:
    advect_result = _summary(advect_samples)
    hips_result = _summary(hips_samples)
    return {
        "advect": advect_result,
        "hips_autograd": hips_result,
        "advect_over_hips": advect_result["median_us"] / hips_result["median_us"],
    }


def _close_advect_pullback(pullback: object) -> None:
    close = getattr(pullback, "close", None)
    if not callable(close):
        msg = "Advect VJP pullback does not expose deterministic close()"
        raise TypeError(msg)
    close()


def _measure_phases(
    workload: _Workload,
    *,
    warmup: int,
    rounds: int,
    block_size: int,
) -> dict[str, object]:
    """Measure only public, semantically equivalent reusable-VJP phases.

    Construction includes releasing the just-created pullback. Isolating the
    raw trace from its required lifetime cleanup would need private,
    framework-specific hooks and would not be a sound cross-framework metric.
    """

    def advect_linearize() -> tuple[object, object]:
        return ad.linearize(
            workload.advect_loss,
            *workload.advect_args,
            argnums=workload.advect_argnums,
        )

    hips_linearize = hips_make_vjp(workload.hips_loss, argnum=workload.hips_argnum)
    cotangent = 1.0

    advect_value, advect_check_linear = advect_linearize()
    advect_check_pullback = advect_check_linear.transpose()
    hips_check_pullback, hips_value = hips_linearize(*workload.hips_args)
    try:
        _assert_tree_allclose(advect_value, hips_value, workload=f"{workload.name} VJP primal")
        _assert_tree_allclose(
            advect_check_pullback(cotangent),
            hips_check_pullback(cotangent),
            workload=f"{workload.name} VJP pullback",
        )
    finally:
        _close_advect_pullback(advect_check_linear)

    def advect_linearize_and_release() -> None:
        _, linear = advect_linearize()
        _close_advect_pullback(linear)

    def hips_linearize_and_release() -> None:
        pullback, _ = hips_linearize(*workload.hips_args)
        del pullback

    advect_build_samples, hips_build_samples = _time_pair(
        advect_linearize_and_release,
        hips_linearize_and_release,
        warmup=warmup,
        rounds=rounds,
        block_size=block_size,
    )

    advect_value, advect_linear = advect_linearize()
    advect_pullback = advect_linear.transpose()
    hips_pullback, hips_value = hips_linearize(*workload.hips_args)
    try:
        _assert_tree_allclose(advect_value, hips_value, workload=f"{workload.name} reused primal")
        _assert_tree_allclose(
            advect_pullback(cotangent),
            hips_pullback(cotangent),
            workload=f"{workload.name} reused pullback",
        )
        advect_reverse_samples, hips_reverse_samples = _time_pair(
            lambda: advect_pullback(cotangent),
            lambda: hips_pullback(cotangent),
            warmup=warmup,
            rounds=rounds,
            block_size=block_size,
        )
    finally:
        _close_advect_pullback(advect_linear)

    return {
        "linearize_and_release": {
            "comparison": "construct and release one reusable scalar-output VJP",
            **_paired_summary(advect_build_samples, hips_build_samples),
        },
        "reused_reverse": {
            "comparison": "apply one retained VJP to the same scalar cotangent",
            **_paired_summary(advect_reverse_samples, hips_reverse_samples),
        },
    }


def _measure_workload(
    workload: _Workload,
    *,
    warmup: int,
    rounds: int,
    block_size: int,
    phases: bool,
) -> dict[str, object]:
    advect_gradient = ad.grad(workload.advect_loss, argnums=workload.advect_argnums)
    hips_gradient = hips_grad(workload.hips_loss, argnum=workload.hips_argnum)

    _assert_tree_allclose(
        workload.advect_loss(*workload.advect_args),
        workload.hips_loss(*workload.hips_args),
        workload=workload.name,
    )
    actual = advect_gradient(*workload.advect_args)
    expected = hips_gradient(*workload.hips_args)
    _assert_tree_allclose(actual, expected, workload=workload.name)

    advect_samples, hips_samples = _time_pair(
        lambda: advect_gradient(*workload.advect_args),
        lambda: hips_gradient(*workload.hips_args),
        warmup=warmup,
        rounds=rounds,
        block_size=block_size,
    )
    result = {
        "name": workload.name,
        "group": workload.group,
        "comparison": workload.comparison,
        "correct": True,
        **_paired_summary(advect_samples, hips_samples),
    }
    if phases:
        result["phases"] = _measure_phases(
            workload,
            warmup=warmup,
            rounds=rounds,
            block_size=block_size,
        )
    return result


def _payload(
    config: _RunConfig,
    *,
    native_build: dict[str, str],
    advect_debug: bool,
) -> dict[str, object]:
    size = config.size
    warmup = config.warmup
    rounds = config.rounds
    block_size = config.block_size
    suite = config.suite
    phases = config.phases
    scientific_mutation_size = config.scientific_mutation_size

    measured = [
        (
            workload,
            _measure_workload(
                workload,
                warmup=warmup,
                rounds=rounds,
                block_size=block_size,
                phases=phases,
            ),
        )
        for workload in _workloads(size, suite)
    ]
    results = [result for _, result in measured]

    mutation_scaling: dict[str, object] | None = None
    if scientific_mutation_size is not None:
        latency_pair = next(
            (pair for pair in measured if pair[0].name == "mutation_accumulation"),
            None,
        )
        if latency_pair is None:
            latency_workload = _mutation_accumulation_workload(size)
            latency_pair = (
                latency_workload,
                _measure_workload(
                    latency_workload,
                    warmup=warmup,
                    rounds=rounds,
                    block_size=block_size,
                    phases=phases,
                ),
            )
            measured.append(latency_pair)

        scaling_measurements = [{"size": size, "result": latency_pair[1]}]
        if scientific_mutation_size != size:
            scientific_workload = _mutation_accumulation_workload(scientific_mutation_size)
            scientific_pair = (
                scientific_workload,
                _measure_workload(
                    scientific_workload,
                    warmup=warmup,
                    rounds=rounds,
                    block_size=block_size,
                    phases=phases,
                ),
            )
            measured.append(scientific_pair)
            scaling_measurements.append(
                {"size": scientific_mutation_size, "result": scientific_pair[1]}
            )
        mutation_scaling = {
            "comparison": latency_pair[0].comparison,
            "measurements": scaling_measurements,
        }

    ratios = [cast("float", result["advect_over_hips"]) for result in results]
    geometric_mean = math.prod(ratios) ** (1.0 / len(ratios))
    group_geometric_means: dict[str, float] = {}
    for group in {cast("str", result["group"]) for result in results}:
        group_ratios = [
            cast("float", result["advect_over_hips"])
            for result in results
            if result["group"] == group
        ]
        group_geometric_means[group] = math.prod(group_ratios) ** (1.0 / len(group_ratios))
    payload: dict[str, object] = {
        "schema_version": 1,
        "report_kind": "advect.ecosystem-comparison",
        "size": size,
        "warmup": warmup,
        "rounds": rounds,
        "block_size": block_size,
        "suite": suite,
        "workloads": results,
        "geometric_mean_advect_over_hips": geometric_mean,
        "group_geometric_means_advect_over_hips": group_geometric_means,
        "comparison": {
            "gating": False,
            "purpose": "historical ecosystem context",
        },
        "environment": {
            **evidence_environment(),
            "numpy": np.__version__,
            "autograd": version("autograd"),
            "advect": ad.__version__,
            "advect_native": native_build,
            "advect_debug": advect_debug,
            "timing": "warmed alternating framework-isolated blocks",
        },
    }
    if mutation_scaling is not None:
        payload["mutation_accumulation_scaling"] = mutation_scaling
    return payload


def _aggregate_warmed_replicates(payloads: Sequence[dict[str, object]]) -> dict[str, object]:
    """Aggregate same-process, independently warmed measurements without gating them."""
    if not payloads:
        message = "at least one warmed replicate is required"
        raise ValueError(message)
    workloads_by_replicate = [
        {
            cast("str", result["name"]): result
            for result in cast("list[dict[str, object]]", payload["workloads"])
        }
        for payload in payloads
    ]
    names = set(workloads_by_replicate[0])
    if any(set(workloads) != names for workloads in workloads_by_replicate[1:]):
        message = "warmed replicates measured different workloads"
        raise ValueError(message)
    run_evidence = [
        {
            "geometric_mean_advect_over_hips": payload["geometric_mean_advect_over_hips"],
            "workload_ratios_advect_over_hips": {
                name: workloads[name]["advect_over_hips"] for name in sorted(workloads)
            },
        }
        for payload, workloads in zip(payloads, workloads_by_replicate, strict=True)
    ]

    representative = payloads[0]
    representative_workloads = {
        cast("str", result["name"]): result
        for result in cast("list[dict[str, object]]", representative["workloads"])
    }

    def aggregate_result(
        result: dict[str, object],
        replicated: Sequence[dict[str, object]],
    ) -> None:
        for implementation in ("advect", "hips_autograd"):
            result[implementation] = {
                "median_us": statistics.median(
                    cast("dict[str, float]", item[implementation])["median_us"]
                    for item in replicated
                )
            }
        result["advect_over_hips"] = statistics.median(
            cast("float", item["advect_over_hips"]) for item in replicated
        )
        if "phases" not in result:
            return
        phases = cast("dict[str, dict[str, object]]", result["phases"])
        replicated_phases = [
            cast("dict[str, dict[str, object]]", item["phases"]) for item in replicated
        ]
        if any(set(item) != set(phases) for item in replicated_phases):
            message = "warmed replicates measured different workload phases"
            raise ValueError(message)
        for phase_name, phase in phases.items():
            aggregate_result(
                phase,
                [item[phase_name] for item in replicated_phases],
            )

    for name in names:
        result = representative_workloads[name]
        replicated = [workloads[name] for workloads in workloads_by_replicate]
        aggregate_result(result, replicated)

    representative_result_ids = {id(result) for result in representative_workloads.values()}
    scaling_by_replicate = [
        cast("dict[str, object]", payload.get("mutation_accumulation_scaling", {}))
        for payload in payloads
    ]
    representative_scaling = scaling_by_replicate[0]
    if representative_scaling:
        representative_measurements = cast(
            "list[dict[str, object]]",
            representative_scaling["measurements"],
        )
        measurements_by_replicate = [
            {
                cast("int", item["size"]): cast("dict[str, object]", item["result"])
                for item in cast("list[dict[str, object]]", scaling["measurements"])
            }
            for scaling in scaling_by_replicate
        ]
        sizes = set(measurements_by_replicate[0])
        if any(set(items) != sizes for items in measurements_by_replicate[1:]):
            message = "warmed replicates measured different mutation scaling sizes"
            raise ValueError(message)
        for measurement in representative_measurements:
            result = cast("dict[str, object]", measurement["result"])
            if id(result) in representative_result_ids:
                continue
            size = cast("int", measurement["size"])
            aggregate_result(
                result,
                [items[size] for items in measurements_by_replicate],
            )

    representative["geometric_mean_advect_over_hips"] = statistics.median(
        cast("float", payload["geometric_mean_advect_over_hips"]) for payload in payloads
    )
    group_means = [
        cast("dict[str, float]", payload["group_geometric_means_advect_over_hips"])
        for payload in payloads
    ]
    representative["group_geometric_means_advect_over_hips"] = {
        group: statistics.median(item[group] for item in group_means) for group in group_means[0]
    }
    representative["warmed_replicates"] = {
        "count": len(payloads),
        "process_isolation": False,
        "runs": run_evidence,
    }
    return representative


def _print_text(payload: dict[str, object]) -> None:
    print(f"n={payload['size']}; rounds={payload['rounds']}; calls/block={payload['block_size']}")
    print(f"suite={payload['suite']}")
    environment = cast("dict[str, object]", payload["environment"])
    native_build = cast("dict[str, str]", environment["advect_native"])
    print(
        f"native={native_build['version']} ({native_build['build_profile']}); comparison=non-gating"
    )
    replicate_evidence = cast("dict[str, object]", payload["warmed_replicates"])
    run_count = cast("int", replicate_evidence["count"])
    if run_count > 1:
        print(f"\nmedians across {run_count} warmed same-process replicates")
    else:
        print("\nworkload timings")
    print("workload                  Advect us    HIPS us    Advect/HIPS")
    for raw_result in cast("list[dict[str, object]]", payload["workloads"]):
        advect_result = cast("dict[str, float]", raw_result["advect"])
        hips_result = cast("dict[str, float]", raw_result["hips_autograd"])
        print(
            f"{cast('str', raw_result['name']):<24}"
            f"{advect_result['median_us']:>9.2f}"
            f"{hips_result['median_us']:>11.2f}"
            f"{cast('float', raw_result['advect_over_hips']):>13.3f}x"
        )
    print(f"geometric mean: {cast('float', payload['geometric_mean_advect_over_hips']):.3f}x HIPS")
    group_means = cast("dict[str, float]", payload["group_geometric_means_advect_over_hips"])
    for group, ratio in sorted(group_means.items()):
        print(f"{group} geometric mean: {ratio:.3f}x HIPS")

    reported: list[tuple[str, dict[str, object]]] = []
    seen_results: set[int] = set()
    for raw_result in cast("list[dict[str, object]]", payload["workloads"]):
        reported.append((cast("str", raw_result["name"]), raw_result))
        seen_results.add(id(raw_result))

    raw_scaling = payload.get("mutation_accumulation_scaling")
    if raw_scaling is not None:
        scaling = cast("dict[str, object]", raw_scaling)
        measurements = cast("list[dict[str, object]]", scaling["measurements"])
        print("\nmutation accumulation scaling")
        print("size                      Advect us    HIPS us    Advect/HIPS")
        for measurement in measurements:
            result = cast("dict[str, object]", measurement["result"])
            advect_result = cast("dict[str, float]", result["advect"])
            hips_result = cast("dict[str, float]", result["hips_autograd"])
            measurement_size = cast("int", measurement["size"])
            print(
                f"{measurement_size:<24}"
                f"{advect_result['median_us']:>9.2f}"
                f"{hips_result['median_us']:>11.2f}"
                f"{cast('float', result['advect_over_hips']):>13.3f}x"
            )
            if id(result) not in seen_results:
                reported.append((f"mutation_accumulation[n={measurement_size}]", result))
                seen_results.add(id(result))

    phase_rows = [
        (label, cast("dict[str, dict[str, object]]", result["phases"]))
        for label, result in reported
        if "phases" in result
    ]
    if phase_rows:
        print("\nphase decomposition")
        print(
            "workload / phase                                           "
            "Advect us    HIPS us    Advect/HIPS"
        )
        for label, phases in phase_rows:
            for phase_name in ("linearize_and_release", "reused_reverse"):
                phase = phases[phase_name]
                advect_result = cast("dict[str, float]", phase["advect"])
                hips_result = cast("dict[str, float]", phase["hips_autograd"])
                print(
                    f"{f'{label} / {phase_name}':<58}"
                    f"{advect_result['median_us']:>9.2f}"
                    f"{hips_result['median_us']:>11.2f}"
                    f"{cast('float', phase['advect_over_hips']):>13.3f}x"
                )


def main() -> int:
    """Run the benchmark and emit the selected output format."""
    args = _arguments()
    config = _RunConfig(
        size=args.size,
        warmup=args.warmup,
        rounds=args.rounds,
        block_size=args.block_size,
        suite=args.suite,
        output_format=args.format,
        warmed_replicates=args.warmed_replicates,
        phases=args.phases,
        scientific_mutation_size=args.scientific_mutation_size,
    )
    native_build = native_build_info()
    advect_debug = is_debug()
    payload = _aggregate_warmed_replicates(
        [
            _payload(
                config,
                native_build=native_build,
                advect_debug=advect_debug,
            )
            for _ in range(config.warmed_replicates)
        ],
    )
    if args.output is not None:
        if config.output_format != "json":
            message = "--output requires --format json"
            raise ValueError(message)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
    elif config.output_format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_text(payload)
    return 0
