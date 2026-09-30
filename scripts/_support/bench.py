"""Standard-library timing and reporting shared by the benchmark commands."""

from __future__ import annotations

import gc
import importlib
import json
import platform
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path


def timed_blocks(
    call: Callable[[], object],
    *,
    rounds: int,
    block_size: int,
    synchronize: Callable[[], object] | None = None,
) -> list[float]:
    """Return microseconds per call of *rounds* blocks timed with collection off."""
    samples: list[float] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(rounds):
            started = time.perf_counter_ns()
            for _ in range(block_size):
                call()
            if synchronize is not None:
                synchronize()
            samples.append((time.perf_counter_ns() - started) / (1_000 * block_size))
    finally:
        if gc_was_enabled:
            gc.enable()
    return samples


def advect_worker_environment() -> dict[str, object]:
    """Describe the interpreter and the Advect native build a worker measured."""
    native = importlib.import_module("advect.core._native")
    context = importlib.import_module("advect.core._context")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "advect_native": native.native_build_info(),
        "advect_debug": context.is_debug(),
    }


def emit_report(
    report: Mapping[str, object],
    *,
    fmt: str,
    output: Path | None,
    print_text: Callable[[Mapping[str, object]], None],
) -> None:
    """Write JSON to *output*; print text, or JSON when nothing was written."""
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{rendered}\n", encoding="utf-8")
    if fmt == "text":
        print_text(report)
    elif output is None:
        print(rendered)
