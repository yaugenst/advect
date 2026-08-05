#!/usr/bin/env python3
"""Measure Advect runtime memory in isolated child processes."""

from __future__ import annotations

from scripts._support.bench_runtime_memory import main

if __name__ == "__main__":
    raise SystemExit(main())
