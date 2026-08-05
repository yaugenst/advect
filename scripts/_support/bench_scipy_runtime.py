"""Benchmark every Advect SciPy operation added by the filter-coverage slice."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage as scipy_ndimage, special as scipy_special

import advect as ad
from advect.scipy import ndimage as advect_ndimage, special as advect_special
from scripts._support.evidence import evidence_environment

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


_NEW_SPECIAL_NAMES = (
    "erfc",
    "erfcx",
    "erfinv",
    "log_expit",
    "log_ndtr",
    "ndtri",
    "softmax",
    "log_softmax",
)


@dataclass(frozen=True, slots=True)
class _Case:
    name: str
    family: str
    value: np.ndarray
    args: tuple[object, ...] = ()
    kwargs: Mapping[str, object] | None = None
    operand_name: str | None = None
    fixed_input: np.ndarray | None = None

    def call(self, module: object, value: object) -> object:
        function = getattr(module, self.name)
        kwargs = {} if self.kwargs is None else dict(self.kwargs)
        if self.operand_name is not None:
            kwargs[self.operand_name] = value
            value = self.fixed_input
        return function(value, *self.args, **kwargs)


@dataclass(frozen=True, slots=True)
class _Config:
    warmup: int
    rounds: int
    target_seconds: float
    max_block_size: int
    derivative_limit: float


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


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=_positive_int, default=256)
    parser.add_argument("--warmup", type=_positive_int, default=3)
    parser.add_argument("--rounds", type=_positive_int, default=9)
    parser.add_argument("--target-ms", type=_positive_float, default=10.0)
    parser.add_argument("--max-block-size", type=_positive_int, default=100)
    parser.add_argument("--derivative-limit", type=_positive_float, default=8.0)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", type=Path, help="write JSON evidence to this path")
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when a measured JVP or reverse gradient exceeds the configured limit",
    )
    return parser.parse_args()


def _cases(size: int) -> tuple[_Case, ...]:
    rng = np.random.default_rng(20260801)
    field = rng.normal(size=(size, size))
    kernel = rng.normal(size=(3, 3))
    kernel1d = rng.normal(size=5)
    structure = rng.normal(scale=0.1, size=(3, 3))
    plateau = np.zeros_like(field)
    plateau[:, size // 2 :] = 1
    common: list[_Case] = [
        _Case("gaussian_filter", "linear", field, (1.25,), {"radius": 4}),
        _Case("gaussian_filter1d", "linear", field, (1.25,), {"axis": 1, "radius": 4}),
        _Case("uniform_filter", "linear", field, (), {"size": 5}),
        _Case("uniform_filter1d", "linear", field, (5,), {"axis": 1}),
        _Case("convolve", "linear", field, (kernel,), {"mode": "reflect"}),
        _Case("correlate", "linear", field, (kernel,), {"mode": "reflect"}),
        _Case("convolve1d", "linear", field, (kernel1d,), {"axis": 1}),
        _Case("correlate1d", "linear", field, (kernel1d,), {"axis": 1}),
        _Case("laplace", "linear", field),
        _Case("gaussian_laplace", "linear", field, (1.25,), {"radius": 4}),
        _Case("sobel", "linear", field, (), {"axis": 1}),
        _Case("prewitt", "linear", field, (), {"axis": 1}),
        _Case("maximum_filter", "selection", plateau, (), {"size": 31}),
        _Case("minimum_filter", "selection", field, (), {"size": 31}),
        _Case("maximum_filter1d", "selection", field, (3,), {"axis": 1}),
        _Case("minimum_filter1d", "selection", field, (31,), {"axis": 1}),
        _Case(
            "grey_dilation",
            "selection",
            structure,
            operand_name="structure",
            fixed_input=field,
        ),
        _Case(
            "grey_erosion",
            "selection",
            field,
            (),
            {"structure": structure, "mode": "constant", "cval": -0.25},
        ),
        _Case("grey_opening", "selection", field, (), {"size": (3, 3)}),
        _Case("grey_closing", "selection", field, (), {"size": (3, 3)}),
        _Case("morphological_gradient", "selection", field, (), {"size": (3, 3)}),
        _Case("morphological_laplace", "selection", field, (), {"size": (3, 3)}),
        _Case("white_tophat", "selection", field, (), {"size": (3, 3)}),
        _Case("black_tophat", "selection", field, (), {"size": (3, 3)}),
        _Case("median_filter", "rank", field, (), {"size": 3}),
        _Case("rank_filter", "rank", field, (4,), {"size": 3}),
        _Case("percentile_filter", "rank", field, (65.0,), {"size": 3}),
    ]
    linear = np.linspace(-3.0, 3.0, field.size).reshape(field.shape)
    bounded = np.linspace(-0.9, 0.9, field.size).reshape(field.shape)
    probabilities = np.linspace(0.01, 0.99, field.size).reshape(field.shape)
    common.extend(
        [
            _Case("erfc", "special", linear),
            _Case("erfcx", "special", linear),
            _Case("erfinv", "special", bounded),
            _Case("log_expit", "special", linear),
            _Case("log_ndtr", "special", linear),
            _Case("ndtri", "special", probabilities),
            _Case("softmax", "special", field, (), {"axis": 1}),
            _Case("log_softmax", "special", field, (), {"axis": 1}),
        ]
    )
    ndimage_names = {case.name for case in common if case.family != "special"}
    if ndimage_names != set(advect_ndimage.__all__):
        msg = "benchmark ndimage inventory has drifted from the public module"
        raise RuntimeError(msg)
    if {case.name for case in common if case.family == "special"} != set(_NEW_SPECIAL_NAMES):
        msg = "benchmark special inventory has drifted"
        raise RuntimeError(msg)
    return tuple(common)


def _modules(case: _Case) -> tuple[object, object]:
    if case.family == "special":
        return scipy_special, advect_special
    return scipy_ndimage, advect_ndimage


def _measure(function: Callable[[], object], config: _Config) -> dict[str, float | int]:
    function()
    started = time.perf_counter_ns()
    function()
    elapsed = max(1, time.perf_counter_ns() - started)
    block_size = min(
        config.max_block_size,
        max(1, round(config.target_seconds * 1e9 / elapsed)),
    )
    for _ in range(config.warmup):
        for _ in range(block_size):
            function()
    samples = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(config.rounds):
            started = time.perf_counter_ns()
            for _ in range(block_size):
                function()
            samples.append((time.perf_counter_ns() - started) / (1_000 * block_size))
    finally:
        if gc_was_enabled:
            gc.enable()
    return {
        "median_us": statistics.median(samples),
        "minimum_us": min(samples),
        "block_size": block_size,
    }


def _ratio(numerator: Mapping[str, float | int], denominator: Mapping[str, float | int]) -> float:
    return float(numerator["median_us"]) / float(denominator["median_us"])


def _benchmark_case(case: _Case, config: _Config) -> dict[str, object]:
    scipy_module, advect_module = _modules(case)
    value = case.value
    tangent = np.ones_like(value)

    def reference() -> object:
        return case.call(scipy_module, value)

    def operation(argument: object) -> object:
        return case.call(advect_module, argument)

    def loss(argument: object) -> object:
        return np.sum(operation(argument))

    dynamic_jvp = ad.jvp(operation)
    dynamic_gradient = ad.value_and_grad(loss)
    spec = ad.ArraySpec(value.shape, value.dtype)
    staged_forward = ad.stage(operation, specs=(spec,))
    staged_loss = ad.stage(loss, specs=(spec,))
    staged_gradient = ad.value_and_grad(staged_loss)

    expected = reference()
    np.testing.assert_allclose(operation(value), expected, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(staged_forward(value), expected, rtol=2e-13, atol=2e-13)
    dynamic_result = dynamic_gradient(value)
    staged_result = staged_gradient(value)
    np.testing.assert_allclose(staged_result[0], dynamic_result[0], rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(staged_result[1], dynamic_result[1], rtol=2e-12, atol=2e-12)

    timings = {
        "scipy_forward": _measure(reference, config),
        "advect_forward": _measure(lambda: operation(value), config),
        "dynamic_jvp": _measure(lambda: dynamic_jvp(value, tangents=tangent), config),
        "dynamic_value_and_grad": _measure(lambda: dynamic_gradient(value), config),
        "staged_forward": _measure(lambda: staged_forward(value), config),
        "staged_value_and_grad": _measure(lambda: staged_gradient(value), config),
    }
    baseline = timings["scipy_forward"]
    ratios = {
        name: _ratio(timing, baseline)
        for name, timing in timings.items()
        if name != "scipy_forward"
    }
    artifact = staged_gradient.to_dict()
    return {
        "name": case.name,
        "family": case.family,
        "timings": timings,
        "ratios_over_scipy_forward": ratios,
        "compile_ms": {
            "forward": staged_forward.compile_seconds * 1_000,
            "value_and_grad": (staged_loss.compile_seconds + staged_gradient.compile_seconds)
            * 1_000,
        },
        "gradient_graph_nodes": len(artifact["program"]["graph"]["nodes"]),
        "gradient_artifact_bytes": len(json.dumps(artifact, separators=(",", ":"))),
    }


def _violations(results: Sequence[dict[str, object]], limit: float) -> list[str]:
    violations = []
    derivative_modes = ("dynamic_jvp", "dynamic_value_and_grad", "staged_value_and_grad")
    for result in results:
        ratios = result["ratios_over_scipy_forward"]
        for mode in derivative_modes:
            ratio = ratios[mode]
            if ratio > limit:
                violations.append(f"{result['name']} {mode} {ratio:.2f}x exceeds {limit:.2f}x")
    return violations


def _print_text(payload: dict[str, object]) -> None:
    print("operation                    scipy us   fwd x   jvp x  dyn grad x  staged grad x")
    print("-" * 82)
    for result in payload["results"]:
        timings = result["timings"]
        ratios = result["ratios_over_scipy_forward"]
        print(
            f"{result['name']:<28}"
            f"{timings['scipy_forward']['median_us']:>9.1f}"
            f"{ratios['advect_forward']:>8.2f}"
            f"{ratios['dynamic_jvp']:>8.2f}"
            f"{ratios['dynamic_value_and_grad']:>12.2f}"
            f"{ratios['staged_value_and_grad']:>15.2f}"
        )
    print()
    if payload["violations"]:
        print("derivative limit violations")
        for violation in payload["violations"]:
            print(f"- {violation}")
    else:
        print(f"all derivative ratios are within {payload['derivative_limit']:.2f}x")


def main() -> int:
    args = _arguments()
    config = _Config(
        warmup=args.warmup,
        rounds=args.rounds,
        target_seconds=args.target_ms / 1_000,
        max_block_size=args.max_block_size,
        derivative_limit=args.derivative_limit,
    )
    results = [_benchmark_case(case, config) for case in _cases(args.size)]
    violations = _violations(results, config.derivative_limit)
    payload = {
        "environment": evidence_environment(),
        "shape": [args.size, args.size],
        "warmup": args.warmup,
        "rounds": args.rounds,
        "target_ms": args.target_ms,
        "derivative_limit": config.derivative_limit,
        "results": results,
        "violations": violations,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if args.format == "json" and args.output is None:
        print(json.dumps(payload, indent=2))
    elif args.format == "text":
        _print_text(payload)
    if args.check and violations:
        print(f"ERROR: {len(violations)} derivative limit violation(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
