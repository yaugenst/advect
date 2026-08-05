"""Deterministic sampling and exact outcomes for Array API qualification."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Protocol

_RESULT_ENV = "ADVECT_ARRAY_API_PYTEST_RESULTS"
_RESULTS: dict[str, str] = {}


class _Report(Protocol):
    nodeid: str
    failed: bool
    skipped: bool
    passed: bool
    when: str


class _Item(Protocol):
    nodeid: str


def pytest_runtest_setup(item: _Item) -> None:
    """Give baseline and transformed runs the same ``.example()`` draws."""
    digest = hashlib.sha256(str(item.nodeid).encode()).digest()
    random.seed(int.from_bytes(digest[:8], byteorder="big"))


def pytest_runtest_logreport(report: _Report) -> None:
    """Retain one terminal outcome for each collected test node."""
    node_id = str(report.nodeid)
    if report.failed:
        _RESULTS[node_id] = "failed"
    elif report.skipped and _RESULTS.get(node_id) != "failed":
        _RESULTS[node_id] = "skipped"
    elif report.when == "call" and report.passed and node_id not in _RESULTS:
        _RESULTS[node_id] = "passed"


def pytest_sessionfinish(session: object, exitstatus: int) -> None:
    """Write machine-readable outcomes even when the baseline has failures."""
    del session
    destination = os.environ.get(_RESULT_ENV)
    if destination is None:
        return
    payload = {
        "exitstatus": int(exitstatus),
        "results": dict(sorted(_RESULTS.items())),
    }
    Path(destination).write_text(
        f"{json.dumps(payload, sort_keys=True)}\n",
        encoding="utf-8",
    )
