# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Validate Advect release distributions and write their provenance."""

from __future__ import annotations

from _support.release_artifacts import main

if __name__ == "__main__":
    raise SystemExit(main())
