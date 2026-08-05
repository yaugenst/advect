#!/usr/bin/env python3
"""Run the pinned official Array API surface through Advect trace and stage."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._support.run_array_api_conformance import main

if __name__ == "__main__":
    raise SystemExit(main())
