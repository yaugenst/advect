"""Argument types, timing and report emission shared by the evidence commands."""

from __future__ import annotations

import argparse
import gc
import json
import math
from typing import TYPE_CHECKING

import pytest
from hypothesis import example, given, strategies as st
from scripts._support import bench, cli

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_NON_FINITE = ("nan", "inf", "-inf", "1e400")
_ARGUMENTS = (
    st.text()
    | st.sampled_from((*_NON_FINITE, "1e300 GiB", "0", "-1", "0.5", "1", "1.5"))
    | st.from_regex(r"\s*-?[0-9._e]+\s*(kib|mib|gib|kb|mb|gb|b)?", fullmatch=True)
)
_DOMAINS: dict[Callable[[str], object], Callable[[object], bool]] = {
    cli.positive_int: lambda value: isinstance(value, int) and value >= 1,
    cli.nonnegative_int: lambda value: isinstance(value, int) and value >= 0,
    cli.positive_float: lambda value: (
        isinstance(value, float) and math.isfinite(value) and value > 0
    ),
    cli.fraction: lambda value: isinstance(value, float) and 0 < value < 1,
    cli.byte_size: lambda value: isinstance(value, int) and value >= 1,
}


@pytest.mark.parametrize("parse", tuple(_DOMAINS), ids=lambda parse: parse.__name__)
@given(raw=_ARGUMENTS)
@example(raw="nan")
@example(raw="inf")
@example(raw="1e400")
@example(raw="1e300gib")
def test_argument_types_return_a_finite_domain_value_or_an_argparse_error(
    parse: Callable[[str], object],
    raw: str,
) -> None:
    try:
        parsed = parse(raw)
    except (argparse.ArgumentTypeError, ValueError):
        return
    assert _DOMAINS[parse](parsed)


@pytest.mark.parametrize(
    ("parse", "raw"),
    [
        *((cli.positive_float, raw) for raw in _NON_FINITE),
        *((cli.byte_size, raw) for raw in (*_NON_FINITE, "1e300 GiB")),
    ],
)
def test_non_finite_numbers_are_rejected(parse: Callable[[str], object], raw: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="finite"):
        parse(raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1024", 1024),
        ("1 KiB", 1024),
        ("1.5MiB", 1_572_864),
        ("2_gib", 2 * 1024**3),
    ],
)
def test_byte_size_accepts_binary_units(raw: str, expected: int) -> None:
    assert cli.byte_size(raw) == expected


def test_timed_blocks_time_each_block_with_collection_disabled() -> None:
    events: list[str] = []

    def call() -> None:
        events.append("call" if gc.isenabled() else "call-without-gc")

    samples = bench.timed_blocks(
        call, rounds=3, block_size=2, synchronize=lambda: events.append("sync")
    )

    assert len(samples) == 3
    assert all(sample >= 0 for sample in samples)
    assert events == ["call-without-gc", "call-without-gc", "sync"] * 3
    assert gc.isenabled()


def test_report_output_is_json_and_stdout_follows_the_format(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "nested" / "report.json"
    report = {"b": 1, "a": [2.5]}
    texts: list[object] = []

    bench.emit_report(report, fmt="text", output=output, print_text=texts.append)
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert texts == [report]
    assert capsys.readouterr().out == ""

    bench.emit_report(report, fmt="json", output=output, print_text=texts.append)
    assert capsys.readouterr().out == ""

    bench.emit_report(report, fmt="json", output=None, print_text=texts.append)
    assert json.loads(capsys.readouterr().out) == report
    assert texts == [report]
